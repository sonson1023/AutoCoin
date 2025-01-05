import json
import os
from typing import Dict, Any


class ConfigLoader:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """설정 파일 로드"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            self._validate_config(config)
            return config
        except FileNotFoundError:
            print(f"설정 파일을 찾을 수 없습니다: {self.config_path}")
            raise
        except json.JSONDecodeError:
            print("설정 파일 형식이 잘못되었습니다.")
            raise

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """설정값 유효성 검사"""
        # 필수 키 확인
        required_keys = {
            'trading': ['symbol', 'interval', 'initial_balance', 'training_days'],
            'model': ['min_win_rate', 'max_attempts'],
            'indicators': ['sma_short', 'sma_long', 'rsi_period']
        }

        for section, keys in required_keys.items():
            if section not in config:
                raise ValueError(f"설정 파일에 {section} 섹션이 없습니다.")
            for key in keys:
                if key not in config[section]:
                    raise ValueError(f"설정 파일의 {section} 섹션에 {key} 설정이 없습니다.")

        # interval 유효성 검사
        valid_intervals = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d']
        if config['trading']['interval'] not in valid_intervals:
            raise ValueError(f"잘못된 interval 값입니다. 가능한 값: {', '.join(valid_intervals)}")

    def get_config(self) -> Dict[str, Any]:
        """전체 설정 반환"""
        return self.config

    def get_trading_config(self) -> Dict[str, Any]:
        """트레이딩 관련 설정만 반환"""
        return self.config['trading']

    def get_model_config(self) -> Dict[str, Any]:
        """모델 관련 설정만 반환"""
        return self.config['model']

    def get_indicators_config(self) -> Dict[str, Any]:
        """지표 관련 설정만 반환"""
        return self.config['indicators']