import pandas as pd
import numpy as np
from binance.client import Client
from threading import Thread
from sklearn.preprocessing import StandardScaler
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


# .env 파일 로드
load_dotenv()

# API 키 가져오기
api_key = os.getenv('BINANCE_API_KEY')
api_secret = os.getenv('BINANCE_SECRET_KEY')


class RealTimePaperTrading:
    def __init__(self, symbol, model, initial_balance=10000, interval='1h'):
        self.symbol = symbol.lower()  # Binance websocket requires lowercase
        self.interval = interval
        self.model = model
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity_curve = []

        # Historical data buffer for technical indicators
        self.price_buffer = deque(maxlen=100)
        self.current_price = None

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        # Binance client
        self.client = Client(api_key, api_secret)

        # Initialize historical data
        self._initialize_historical_data()

        # WebSocket connection
        self.ws = None
        self.running = False

    def _initialize_historical_data(self):
        """Initialize price buffer with historical data"""
        klines = self.client.get_historical_klines(
            self.symbol.upper(),
            self.interval,
            str(datetime.now() - timedelta(days=5))
        )

        for k in klines:
            self.price_buffer.append({
                'timestamp': datetime.fromtimestamp(k[0] / 1000),
                'open': float(k[1]),
                'high': float(k[2]),
                'low': float(k[3]),
                'close': float(k[4]),
                'volume': float(k[5])
            })

    def prepare_features(self, data):
        """Calculate technical indicators for real-time data"""
        df = pd.DataFrame(data)

        # Calculate technical indicators
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

        return df.iloc[-1:][['sma_20', 'sma_50', 'rsi', 'macd', 'macd_signal',
                             'bb_upper', 'bb_lower', 'price_change', 'volume_change']]

    async def process_market_data(self, msg):
        """Process incoming market data"""
        try:
            msg = json.loads(msg)
            if 'k' in msg:
                # Update current price
                self.current_price = float(msg['k']['c'])

                # Update price buffer if candle is closed
                if msg['k']['x']:
                    new_candle = {
                        'timestamp': datetime.fromtimestamp(msg['k']['t'] / 1000),
                        'open': float(msg['k']['o']),
                        'high': float(msg['k']['h']),
                        'low': float(msg['k']['l']),
                        'close': float(msg['k']['c']),
                        'volume': float(msg['k']['v'])
                    }
                    self.price_buffer.append(new_candle)

                    # Generate trading signals
                    self.generate_trading_signals()

                    # Update performance metrics
                    self.update_metrics()

                    print(f"\nTimestamp: {datetime.now()}")
                    print(f"Current Price: {self.current_price}")
                    print(f"Current Balance: {self.current_balance}")
                    print(f"Position: {self.position}")
                    print(f"Total Trades: {self.total_trades}")
                    print(f"Win Rate: {self.get_win_rate():.2f}%")
        except Exception as e:
            print(f"Error processing market data: {e}")

    def generate_trading_signals(self):
        """Generate trading signals based on the model"""
        features = self.prepare_features(self.price_buffer)
        if not features.isnull().values.any():  # Check if we have valid features
            prediction = self.model.predict(features)
            self.execute_trade(prediction[0])

    def execute_trade(self, signal):
        """Execute paper trades based on signals"""
        if signal == 1 and self.position == 0:  # Buy signal
            self.position = self.current_balance / self.current_price
            self.entry_price = self.current_price
            self.trades.append({
                'type': 'buy',
                'price': self.current_price,
                'timestamp': datetime.now(),
                'balance': self.current_balance
            })
            print(f"\nBUY Signal - Entry Price: {self.current_price}")

        elif signal == 0 and self.position > 0:  # Sell signal
            profit = (self.current_price - self.entry_price) * self.position
            self.current_balance += profit
            self.trades.append({
                'type': 'sell',
                'price': self.current_price,
                'timestamp': datetime.now(),
                'balance': self.current_balance,
                'profit': profit
            })

            # Update trade statistics
            self.total_trades += 1
            if profit > 0:
                self.winning_trades += 1
            else:
                self.losing_trades += 1

            self.position = 0
            print(f"\nSELL Signal - Exit Price: {self.current_price}")
            print(f"Trade Profit: {profit:.2f}")

    def get_win_rate(self):
        """Calculate win rate"""
        if self.total_trades == 0:
            return 0
        return (self.winning_trades / self.total_trades) * 100

    def update_metrics(self):
        """Update performance metrics"""
        current_equity = self.current_balance
        if self.position > 0:
            current_equity += (self.current_price - self.entry_price) * self.position
        self.equity_curve.append({
            'timestamp': datetime.now(),
            'equity': current_equity
        })

    async def _connect_websocket(self):
        """Connect to Binance WebSocket"""
        stream_name = f"{self.symbol}@kline_{self.interval}"
        url = f"wss://stream.binance.com:9443/ws/{stream_name}"

        async with websockets.connect(url) as websocket:
            self.ws = websocket
            self.running = True
            print(f"Started paper trading for {self.symbol}")
            print(f"Initial balance: {self.initial_balance}")

            while self.running:
                try:
                    message = await websocket.recv()
                    await self.process_market_data(message)
                except Exception as e:
                    print(f"WebSocket error: {e}")
                    break

    def start_trading(self):
        """Start real-time paper trading"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._connect_websocket())
        finally:
            loop.close()

    def stop_trading(self):
        """Stop paper trading and print final results"""
        self.running = False
        if self.ws:
            asyncio.get_event_loop().run_until_complete(self.ws.close())

        print("\nPaper Trading Results:")
        print(f"Initial Balance: {self.initial_balance}")
        print(f"Final Balance: {self.current_balance}")
        print(f"Total Return: {((self.current_balance - self.initial_balance) / self.initial_balance) * 100:.2f}%")
        print(f"Total Trades: {self.total_trades}")
        print(f"Win Rate: {self.get_win_rate():.2f}%")


# def main():
#     # Load your trained model
#     model = xgb.XGBClassifier()
#     model.load_model('trained_model.json')  # Load your pre-trained model
#
#     # Initialize paper trading system
#     trader = RealTimePaperTrading(
#         symbol='BTCUSDT',
#         model=model,
#         initial_balance=10000,
#         interval='1h'
#     )
#
#     try:
#         trader.start_trading()
#     except KeyboardInterrupt:
#         trader.stop_trading()

def main():
    # Load your trained model
    try:
        model = xgb.XGBClassifier()
        # 먼저 기본 모델 생성
        model.fit(np.random.rand(10, 9), np.random.randint(2, size=10))
        # 그 다음 저장된 모델 파라미터 로드
        booster = xgb.Booster()
        booster.load_model('trained_model.model')  # .json이 아닌 .model 확장자 사용
        model._Booster = booster
        model._le = preprocessing.LabelEncoder().fit([0, 1])  # 레이블 인코더 초기화
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize paper trading system

    # Initialize paper trading system
    trader = RealTimePaperTrading(
        symbol='BTCUSDT',
        model=model,
        initial_balance=10000,
        interval='5m'
    )

    try:
        trader.start_trading()
        # Let it run for a specified time or until keyboard interrupt
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping trading bot...")
        trader.stop_trading()
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        trader.stop_trading()


if __name__ == "__main__":
    main()