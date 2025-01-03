import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from dotenv import load_dotenv
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta



# .env 파일 로드
load_dotenv()

# API 키 가져오기
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_SECRET_KEY')

# days 변수 정의
training_days = 180

class TradingModel:
    def __init__(self, symbol, interval='1h', lookback_period=100):
        self.symbol = symbol
        self.interval = interval
        self.lookback_period = lookback_period

    def prepare_features(self, df):
        # 기술적 지표 계산
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()

        # RSI
        rsi = RSIIndicator(close=df['close'])
        df['rsi'] = rsi.rsi()

        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()

        # 가격 변화율
        df['price_change'] = df['close'].pct_change()

        # 거래량 변화율
        df['volume_change'] = df['volume'].pct_change()

        # NaN 값 제거
        df.dropna(inplace=True)

        return df

    def create_labels(self, df, threshold=0.001):
        # 다음 시간의 가격 변화를 기준으로 레이블 생성
        df['future_return'] = df['close'].shift(-1) / df['close'] - 1
        df['label'] = np.where(df['future_return'] > threshold, 1, 0)
        df.dropna(inplace=True)
        return df

    def prepare_data(self, df):
        df = self.prepare_features(df)
        df = self.create_labels(df)

        # 학습에 사용할 피처 선택
        feature_columns = ['sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
                           'bb_upper', 'bb_lower', 'price_change', 'volume_change']

        X = df[feature_columns]
        y = df['label']

        return X, y

    def backtest(self, X_test, y_test, predictions, df, initial_balance=10000):
        balance = initial_balance
        position = 0
        returns = []
        equity_curve = [initial_balance]
        positions = []  # 포지션 상태 추적
        entry_prices = []  # 진입 가격
        exit_prices = []  # 청산 가격
        entry_times = []  # 진입 시간
        exit_times = []  # 청산 시간

        # 테스트 데이터의 종가 가져오기
        prices = df['close'].loc[y_test.index]

        for i in range(len(X_test)):
            current_price = prices.iloc[i]
            current_time = prices.index[i]

            if predictions[i] == 1 and position == 0:
                # 매수 신호
                position = balance
                entry_price = current_price
                entry_prices.append(entry_price)
                entry_times.append(current_time)
                positions.append(1)  # 롱 포지션
            elif predictions[i] == 0 and position > 0:
                # 매도 신호
                returns.append((current_price - entry_price) / entry_price)
                balance = position * (1 + returns[-1])
                position = 0
                exit_prices.append(current_price)
                exit_times.append(current_time)
                positions.append(-1)  # 청산

            equity_curve.append(
                balance if position == 0 else balance * (1 + (current_price - entry_price) / entry_price))

        # 성과 지표 계산
        if len(returns) > 0:
            total_return = (balance - initial_balance) / initial_balance
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
            win_rate = sum(1 for r in returns if r > 0) / len(returns)
        else:
            total_return = 0
            sharpe_ratio = 0
            win_rate = 0

        # 시각화 데이터 저장
        self.backtest_results = {
            'equity_curve': equity_curve,
            'returns': returns,
            'entry_prices': entry_prices,
            'exit_prices': exit_prices,
            'entry_times': entry_times,
            'exit_times': exit_times,
            'prices': prices,
            'positions': positions
        }

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'num_trades': len(returns),
            'win_rate': win_rate
        }

    def plot_backtest_results(self):
        """백테스팅 결과를 시각화합니다."""
        # Plotly를 사용한 인터랙티브 차트
        fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,  # shared_xaxis -> shared_xaxes로 수정
                            vertical_spacing=0.03,
                            subplot_titles=('Price & Trades', 'Equity Curve'),
                            row_heights=[0.7, 0.3])

        # 가격 차트
        fig.add_trace(go.Scatter(
            x=self.backtest_results['prices'].index,
            y=self.backtest_results['prices'],
            name='Price',
            line=dict(color='blue')
        ), row=1, col=1)

        # 매수 포인트
        fig.add_trace(go.Scatter(
            x=self.backtest_results['entry_times'],
            y=self.backtest_results['entry_prices'],
            mode='markers',
            name='Buy',
            marker=dict(color='green', size=10, symbol='triangle-up')
        ), row=1, col=1)

        # 매도 포인트
        fig.add_trace(go.Scatter(
            x=self.backtest_results['exit_times'],
            y=self.backtest_results['exit_prices'],
            mode='markers',
            name='Sell',
            marker=dict(color='red', size=10, symbol='triangle-down')
        ), row=1, col=1)

        # 자본금 곡선
        fig.add_trace(go.Scatter(
            x=self.backtest_results['prices'].index,
            y=self.backtest_results['equity_curve'][:-1],
            name='Equity',
            line=dict(color='green')
        ), row=2, col=1)

        fig.update_layout(
            title='Backtest Results',
            xaxis_title='Date',
            yaxis_title='Price',
            height=800
        )

        fig.show()


def main():
    # Binance API 설정
    client = Client(api_key, api_secret)

    # 데이터 수집
    symbol = 'BTCUSDT'
    # klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, "1 year ago UTC")

    start_time = (datetime.now() - timedelta(days=training_days)).strftime("%d %b %Y %H:%M:%S")
    klines = client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, start_time)

    # DataFrame 생성
    df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                       'close_time', 'quote_asset_volume', 'number_of_trades',
                                       'taker_buy_base', 'taker_buy_quote', 'ignore'])

    # 데이터 전처리
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df.astype(float)

    # 모델 생성 및 학습
    model = TradingModel(symbol)
    X, y = model.prepare_data(df)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # XGBoost 모델 학습
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)

    # 모델 저장 시 days 정보를 파일명에 포함
    model_filename = f'trained_model_{training_days}days.model'
    xgb_model.save_model(model_filename)

    # 예측 및 백테스팅
    predictions = xgb_model.predict(X_test)
    results = model.backtest(X_test, y_test, predictions, df)

    print("백테스팅 결과:")
    print(f"총 수익률: {results['total_return'] * 100:.2f}%")
    print(f"샤프 비율: {results['sharpe_ratio']:.2f}")
    print(f"거래 횟수: {results['num_trades']}")
    print(f"승률: {results['win_rate'] * 100:.2f}%")

    # 결과 시각화
    model.plot_backtest_results()

    # 특성 중요도 시각화
    plt.figure(figsize=(10, 6))
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)

    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance')
    plt.show()


if __name__ == "__main__":
    main()