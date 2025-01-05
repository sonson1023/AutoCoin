import pandas as pd
import numpy as np
from binance.client import Client
from sklearn.model_selection import train_test_split
import xgboost as xgb
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from datetime import datetime, timedelta
import time
import json
from collections import deque
import asyncio
import websockets
from dotenv import load_dotenv
import os
from sklearn import preprocessing
import threading
from typing import Optional

from v2.ConfigLoader import ConfigLoader
from v2.GenerateModel import TradingModel
from v2.RealTimePaperTrading import RealTimePaperTrading

load_dotenv()
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_SECRET_KEY')


class TradingSystem:
    def __init__(self, symbol: str, interval: str = '1h',
                 initial_balance: float = 10000,
                 retraining_interval: int = 24,
                 training_days: int = 60,
                 leverage: int = 5,
                 paper_trading: bool = True):

        self.symbol = symbol
        self.interval = interval
        self.initial_balance = initial_balance
        self.retraining_interval = retraining_interval
        self.training_days = training_days
        self.leverage = min(max(leverage, 1), 5)
        self.paper_trading = paper_trading

        # API 연결 및 계정 상태 확인
        try:
            self.client = Client(api_key, api_secret)

            # Futures 계정 상태 확인
            futures_account = self.client.futures_account()
            print("Futures account successfully connected")

            if not paper_trading:
                # Futures API 권한 확인
                try:
                    self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
                    self.client.futures_change_margin_type(symbol=self.symbol, marginType='ISOLATED')
                    print(f"Leverage set to {self.leverage}x")
                except Exception as e:
                    print(f"Error setting futures configuration: {e}")
                    print("Switching to paper trading mode")
                    self.paper_trading = True
        except Exception as e:
            print(f"Error connecting to Binance API: {e}")
            print("Please check your API permissions and Futures account status")
            raise
        self.symbol = symbol
        self.interval = interval
        self.initial_balance = initial_balance
        self.retraining_interval = retraining_interval
        self.training_days = training_days

        self.client = Client(api_key, api_secret)
        self.model: Optional[xgb.XGBClassifier] = None
        self.trading_bot = None
        self.is_running = False
        self.leverage = min(max(leverage, 1), 5)  # 레버리지는 1-5 사이로 제한

        # Binance Futures 설정
        self.client.futures_change_leverage(symbol=self.symbol, leverage=self.leverage)
        self.client.futures_change_margin_type(symbol=self.symbol, marginType='ISOLATED')

        # 초기 모델 학습
        self.train_model()

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """기술적 지표 계산"""
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['rsi'] = RSIIndicator(close=df['close']).rsi()

        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()

        bb = BollingerBands(close=df['close'])
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_lower'] = bb.bollinger_lband()

        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()

        df.dropna(inplace=True)
        return df

    def create_labels(self, df: pd.DataFrame, threshold: float = 0.001) -> pd.DataFrame:
        """레이블 생성"""
        df['future_return'] = df['close'].shift(-1) / df['close'] - 1
        df['label'] = np.where(df['future_return'] > threshold, 1, 0)
        df.dropna(inplace=True)
        return df

    def fetch_training_data(self) -> pd.DataFrame:
        """선물 거래 학습 데이터 수집"""
        start_time = (datetime.now() - timedelta(days=self.training_days)).strftime("%d %b %Y %H:%M:%S")
        klines = self.client.futures_historical_klines(
            self.symbol,
            self.interval,
            start_time
        )

        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                                           'close_time', 'quote_asset_volume', 'number_of_trades',
                                           'taker_buy_base', 'taker_buy_quote', 'ignore'])

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        df = df.astype(float)

        return df

    def train_model(self):
        """모델 학습"""
        print(f"\nStarting model training at {datetime.now()}")

        try:
            # 데이터 준비
            df = self.fetch_training_data()
            df = self.prepare_features(df)
            df = self.create_labels(df)

            # 피처 선택
            feature_columns = ['sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
                               'bb_upper', 'bb_lower', 'price_change', 'volume_change']

            X = df[feature_columns]
            y = df['label']

            # 데이터 분할
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

            # 모델 학습
            self.model = xgb.XGBClassifier()
            self.model.fit(X_train, y_train)

            # 모델 성능 평가
            accuracy = self.model.score(X_test, y_test)
            print(f"Model training completed. Test accuracy: {accuracy:.4f}")

        except Exception as e:
            print(f"Error during model training: {e}")

    async def run_trading_bot(self):
        """트레이딩 봇 실행"""
        if self.model is None:
            print("Model not trained. Please train the model first.")
            return

        self.trading_bot = RealTimePaperTrading(
            symbol=self.symbol,
            model=self.model,
            initial_balance=self.initial_balance,
            interval=self.interval
        )

        await self.trading_bot._connect_websocket()

    def periodic_retraining(self):
        """주기적 재학습 실행"""
        while self.is_running:
            time.sleep(self.retraining_interval * 3600)  # 시간을 초로 변환
            print(f"\nScheduled retraining at {datetime.now()}")

            # 이전 모델 백업
            old_model = self.model

            # 새로운 모델 학습
            self.train_model()

            # 학습 실패시 이전 모델 복원
            if self.model is None:
                print("Retraining failed. Restoring previous model.")
                self.model = old_model

    def start(self):
        """시스템 시작"""
        self.is_running = True

        # 재학습 스레드 시작
        retraining_thread = threading.Thread(target=self.periodic_retraining)
        retraining_thread.daemon = True
        retraining_thread.start()

        # 트레이딩 봇 시작
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self.run_trading_bot())
        finally:
            loop.close()

    def stop(self):
        """시스템 중지"""
        self.is_running = False
        if self.trading_bot:
            self.trading_bot.stop_trading()


def train_and_evaluate_model(self):
    """모델 학습 및 백테스팅 수행"""
    MAX_ATTEMPTS = 10  # 최대 재시도 횟수
    MIN_WIN_RATE = 0.65  # 최소 승률 기준 (65%)

    for attempt in range(MAX_ATTEMPTS):
        print(f"\n시도 #{attempt + 1}")

        # 기존 모델 학습 코드
        df = self.fetch_training_data()
        df = self.prepare_features(df)
        df = self.create_labels(df)

        feature_columns = ['sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
                           'bb_upper', 'bb_lower', 'price_change', 'volume_change']

        X = df[feature_columns]
        y = df['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # 모델 파라미터 랜덤화 (다양한 모델 생성을 위해)
        params = {
            'max_depth': np.random.randint(3, 10),
            'learning_rate': np.random.uniform(0.01, 0.3),
            'n_estimators': np.random.randint(50, 200),
            'min_child_weight': np.random.randint(1, 7),
            'subsample': np.random.uniform(0.6, 1.0),
            'colsample_bytree': np.random.uniform(0.6, 1.0),
        }

        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X_train, y_train)

        # 백테스팅 수행
        total_return, sharpe_ratio, trade_count, win_rate = self.backtest()

        print(f"백테스팅 결과:")
        print(f"총 수익률: {total_return:.2f}%")
        print(f"샤프 비율: {sharpe_ratio:.2f}")
        print(f"거래 횟수: {trade_count}")
        print(f"승률: {win_rate:.2f}%")

        # 승률이 65% 이상인 경우 모델 저장 및 종료
        if win_rate >= MIN_WIN_RATE:
            print(f"\n목표 승률 달성! ({win_rate:.2f}% >= {MIN_WIN_RATE * 100}%)")
            self.save_model()
            return True

        print(f"\n목표 승률 미달. ({win_rate:.2f}% < {MIN_WIN_RATE * 100}%)")
        print("모델 재학습 중...")

    print(f"\n{MAX_ATTEMPTS}번의 시도 후에도 목표 승률을 달성하지 못했습니다.")
    return False


def save_model(self):
    """승률 조건을 만족한 모델 저장"""
    model_path = f'trained_model_{self.training_days}_days.model'
    self.model.save_model(model_path)
    print(f"모델이 저장되었습니다: {model_path}")


def main():
    # 설정 로드
    config_loader = ConfigLoader()
    trading_config = config_loader.get_trading_config()
    model_config = config_loader.get_model_config()

    # TradingModel 객체 생성
    trading_model = TradingModel(
        symbol=trading_config['symbol'],
        interval=trading_config['interval'],
        training_days=trading_config['training_days']
    )

    # 목표 승률로 모델 학습
    best_model, results = trading_model.train_model_with_target_winrate(
        min_win_rate=model_config['min_win_rate'],
        max_attempts=model_config['max_attempts']
    )

    if results['win_rate'] >= model_config['min_win_rate']:
        print("\n트레이딩 시작 가능한 모델이 준비되었습니다.")

        # 실시간 트레이딩 시작
        trader = RealTimePaperTrading(
            symbol=trading_config['symbol'],
            model=best_model,
            initial_balance=trading_config['initial_balance'],
            interval=trading_config['interval']  # 동일한 interval 사용
        )

        try:
            trader.start_trading()
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping trading bot...")
            trader.stop_trading()
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            trader.stop_trading()
    else:
        print("\n목표 승률을 달성하지 못했습니다. 프로그램을 종료합니다.")
        return


if __name__ == "__main__":
    main()
