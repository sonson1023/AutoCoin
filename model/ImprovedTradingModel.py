import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import ta
from ta.trend import SMAIndicator, MACD, ADXIndicator, CCIIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import MFIIndicator, ChaikinMoneyFlowIndicator
from dotenv import load_dotenv
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# .env 파일 로드
load_dotenv()


class ImprovedTradingModel:
    def __init__(self, symbol, interval='1h', lookback_period=100):
        self.symbol = symbol
        self.interval = interval
        self.lookback_period = lookback_period
        self.scaler = StandardScaler()

        # 트레이딩 파라미터
        self.RISK_PER_TRADE = 0.02  # 거래당 리스크 2%
        self.STOP_LOSS = 0.02  # 2% 손절
        self.TAKE_PROFIT = 0.04  # 4% 익절
        self.ENTRY_THRESHOLD = 0.7  # 진입 확신도 임계값

    def prepare_features(self, df):
        # 기본 기술적 지표
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['ema_20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()

        # RSI
        rsi = RSIIndicator(close=df['close'])
        df['rsi'] = rsi.rsi()

        # Stochastic
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

        # ADX (추세 강도)
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
        df['adx'] = adx.adx()

        # CCI (Commodity Channel Index)
        df['cci'] = CCIIndicator(high=df['high'], low=df['low'], close=df['close']).cci()

        # 볼륨 지표
        df['mfi'] = MFIIndicator(high=df['high'], low=df['low'], close=df['close'],
                                 volume=df['volume']).money_flow_index()
        df['cmf'] = ChaikinMoneyFlowIndicator(high=df['high'], low=df['low'],
                                              close=df['close'], volume=df['volume']).chaikin_money_flow()

        # 가격 변화율
        df['price_change'] = df['close'].pct_change()
        df['price_change_ma5'] = df['price_change'].rolling(5).mean()

        # 거래량 지표
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma5'] = df['volume'].rolling(5).mean()
        df['volume_ma20'] = df['volume'].rolling(20).mean()

        # 이동평균 교차 시그널
        df['ema_cross'] = np.where(df['ema_20'] > df['sma_50'], 1, 0)

        # 변동성 지표
        df['atr'] = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'],
                                                   close=df['close']).average_true_range()

        # NaN 값 제거
        df.dropna(inplace=True)

        # 스케일링
        feature_columns = df.select_dtypes(include=[np.float64, np.int64]).columns
        df[feature_columns] = self.scaler.fit_transform(df[feature_columns])

        return df

    def create_labels(self, df, threshold=0.002, window=12):
        # n시간 후의 수익률 예측
        df['future_return'] = df['close'].shift(-window) / df['close'] - 1

        # 상승/하락 구분을 세분화
        conditions = [
            (df['future_return'] > threshold * 2),  # 강한 상승
            (df['future_return'] > threshold),  # 약한 상승
            (df['future_return'] < -threshold),  # 하락
        ]
        choices = [2, 1, 0]
        df['label'] = np.select(conditions, choices, default=1)

        df.dropna(inplace=True)
        return df

    def prepare_data(self, df):
        df = self.prepare_features(df)
        df = self.create_labels(df)

        # 학습에 사용할 피처 선택
        feature_columns = [col for col in df.columns if
                           col not in ['label', 'future_return', 'open', 'high', 'low', 'close', 'volume']]

        X = df[feature_columns]
        y = df['label']

        return X, y

    def train_model(self, X_train, y_train, X_val, y_val):
        model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            min_child_weight=2,
            gamma=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=1.5,
            random_state=42
        )

        # 기본 학습
        model.fit(X_train, y_train)

        return model

    def calculate_max_drawdown(self, equity_curve):
        peak = equity_curve[0]
        max_dd = 0

        for value in equity_curve:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd

        return max_dd

    def backtest(self, X_test, y_test, model, df, initial_balance=10000):
        balance = initial_balance
        position = 0
        entry_price = 0
        returns = []
        equity_curve = [initial_balance]
        trades = []

        prices = df['close'].loc[y_test.index]

        # 예측 확률값 사용
        predictions_proba = model.predict_proba(X_test)

        for i in range(len(X_test)):
            current_price = prices.iloc[i]
            current_time = prices.index[i]

            # 포지션이 있을 때
            if position > 0:
                profit_pct = (current_price - entry_price) / entry_price

                # 손절/익절 확인
                if profit_pct <= -self.STOP_LOSS or profit_pct >= self.TAKE_PROFIT:
                    returns.append(profit_pct)
                    balance = position * (1 + profit_pct)
                    trades.append({
                        'type': 'exit',
                        'time': current_time,
                        'price': current_price,
                        'profit_pct': profit_pct,
                        'balance': balance
                    })
                    position = 0

            # 새로운 진입 시그널
            elif predictions_proba[i][1] > self.ENTRY_THRESHOLD and position == 0:
                position_size = balance * self.RISK_PER_TRADE / self.STOP_LOSS
                position = min(position_size, balance)
                entry_price = current_price
                trades.append({
                    'type': 'entry',
                    'time': current_time,
                    'price': current_price,
                    'position_size': position,
                    'balance': balance
                })

            # 자본금 곡선 업데이트
            current_equity = balance if position == 0 else position * (1 + (current_price - entry_price) / entry_price)
            equity_curve.append(current_equity)

        # 성과 지표 계산
        total_return = (equity_curve[-1] - initial_balance) / initial_balance
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
        max_drawdown = self.calculate_max_drawdown(equity_curve)
        win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0
        profit_factor = abs(sum(r for r in returns if r > 0) / sum(r for r in returns if r < 0)) if returns else 0

        avg_trade_duration = np.mean([
            (t2['time'] - t1['time']).total_seconds() / 3600
            for t1, t2 in zip(
                [t for t in trades if t['type'] == 'entry'],
                [t for t in trades if t['type'] == 'exit']
            )
        ]) if len(trades) >= 2 else 0

        results = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'num_trades': len(returns),
            'avg_trade_duration': avg_trade_duration,
            'equity_curve': equity_curve,
            'trades': trades,
            'returns': returns
        }

        return results

    def plot_results(self, df, results):
        # Plotly를 사용한 인터랙티브 차트
        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=('Price & Trades', 'Equity Curve', 'Drawdown'),
            row_heights=[0.5, 0.25, 0.25]
        )

        # 가격 차트
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price'
        ), row=1, col=1)

        # 매수/매도 포인트
        entry_trades = [t for t in results['trades'] if t['type'] == 'entry']
        exit_trades = [t for t in results['trades'] if t['type'] == 'exit']

        fig.add_trace(go.Scatter(
            x=[t['time'] for t in entry_trades],
            y=[t['price'] for t in entry_trades],
            mode='markers',
            name='Buy',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[t['time'] for t in exit_trades],
            y=[t['price'] for t in exit_trades],
            mode='markers',
            name='Sell',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ), row=1, col=1)

        # 자본금 곡선
        fig.add_trace(go.Scatter(
            x=df.index,
            y=results['equity_curve'][:-1],
            name='Equity',
            line=dict(color='blue')
        ), row=2, col=1)

        # Drawdown 차트
        equity_series = pd.Series(results['equity_curve'][:-1])
        running_max = equity_series.expanding().max()
        drawdown = (equity_series - running_max) / running_max

        fig.add_trace(go.Scatter(
            x=df.index,
            y=drawdown,
            name='Drawdown',
            line=dict(color='red')
        ), row=3, col=1)

        fig.update_layout(
            title='Trading Results',
            xaxis_title='Date',
            yaxis_title='Price',
            height=1000
        )

        return fig


def main():
    # Binance API 설정
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_SECRET_KEY')
    client = Client(api_key, api_secret)

    # 데이터 수집
    symbol = 'BTCUSDT'
    training_days = 180
    start_time = (datetime.now() - timedelta(days=training_days)).strftime("%d %b %Y %H:%M:%S")
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, start_time)

    # DataFrame 생성
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'taker_buy_base', 'taker_buy_quote', 'ignore'])

    # 데이터 전처리
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)

    # 모든 숫자형 컬럼을 float로 변환
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'close_time',
                       'quote_asset_volume', 'number_of_trades', 'taker_buy_base',
                       'taker_buy_quote', 'ignore']

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print(f"수집된 데이터 크기: {len(df)} 행")
    print("\n데이터 타입 확인:")
    print(df.dtypes)

    # 모델 초기화
    model = ImprovedTradingModel(symbol)

    # 데이터 준비
    print("데이터 전처리 중...")
    X, y = model.prepare_data(df)

    # 학습/검증/테스트 데이터 분할
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.15)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size + val_size]
    y_val = y[train_size:train_size + val_size]
    X_test = X[train_size + val_size:]
    y_test = y[train_size + val_size:]

    # 모델 학습
    print("모델 학습 중...")
    trained_model = model.train_model(X_train, y_train, X_val, y_val)

    # 백테스팅
    print("백테스팅 수행 중...")
    results = model.backtest(X_test, y_test, trained_model, df[train_size + val_size:])

    # 결과 출력
    print("\n=== 백테스팅 결과 ===")
    print(f"총 수익률: {results['total_return'] * 100:.2f}%")
    print(f"샤프 비율: {results['sharpe_ratio']:.4f}")
    print(f"최대 낙폭: {results['max_drawdown'] * 100:.2f}%")
    print(f"승률: {results['win_rate'] * 100:.2f}%")
    print(f"수익 팩터: {results['profit_factor']:.2f}")
    print(f"총 거래 횟수: {results['num_trades']}")
    print(f"평균 거래 시간: {results['avg_trade_duration']:.1f}시간")

    # 결과 시각화
    print("\n차트 생성 중...")
    fig = model.plot_results(df[train_size + val_size:], results)
    fig.show()

    # 특성 중요도 시각화
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': trained_model.feature_importances_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(12, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()

    # 모델 저장
    model_filename = f'improved_trading_model_{training_days}days.json'
    trained_model.save_model(model_filename)
    print(f"\n모델 저장 완료: {model_filename}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"오류 발생: {str(e)}")