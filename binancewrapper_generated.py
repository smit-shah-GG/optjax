import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
from binance.client import Client
from binance.exceptions import BinanceAPIException
from binance.websockets import BinanceSocketManager
import pandas as pd
import jax.numpy as jnp

from enhanced_network import PPOAgent, PPOConfig, TransactionLog


class BinanceTradingWrapper:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str = "BTCUSDT",
        window_size: int = 30,
        trade_interval: str = "1h",
        test_mode: bool = True,
    ):
        """
        Initialize the Binance trading wrapper.

        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            symbol: Trading pair symbol (default: BTCUSDT)
            window_size: Number of historical candles to consider (default: 30)
            trade_interval: Trading interval (default: 1h)
            test_mode: Whether to run in test mode (default: True)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbol = symbol
        self.window_size = window_size
        self.trade_interval = trade_interval
        self.test_mode = test_mode

        # Initialize Binance client
        self.client = Client(api_key, api_secret, testnet=test_mode)
        self.bm = BinanceSocketManager(self.client)

        # Initialize PPO agent
        config = PPOConfig(window_size=window_size)
        self.agent = PPOAgent(config)

        # Trading state
        self.current_position = 0
        self.trades_history: List[TransactionLog] = []
        self.running = False
        self.last_update_time = 0

        # Cache for technical indicators
        self.price_history: List[float] = []
        self.technical_cache: Dict[str, float] = {}

    def _calculate_technical_indicators(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate technical indicators from price data."""
        if len(prices) < self.window_size:
            return {}

        df = pd.DataFrame({"close": prices})

        # Calculate indicators
        df["SMA_20"] = df["close"].rolling(window=20).mean()
        df["SMA_50"] = df["close"].rolling(window=50).mean()
        df["RSI"] = self._calculate_rsi(df["close"], 14)
        df["MACD"], df["Signal"] = self._calculate_macd(df["close"])
        df["BB_upper"], df["BB_lower"] = self._calculate_bollinger_bands(df["close"])

        # Get latest values
        latest = df.iloc[-1]
        return {
            "sma_20": latest["SMA_20"],
            "sma_50": latest["SMA_50"],
            "rsi": latest["RSI"],
            "macd": latest["MACD"],
            "macd_signal": latest["Signal"],
            "bb_upper": latest["BB_upper"],
            "bb_lower": latest["BB_lower"],
        }

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int = 20, std: int = 2
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        rolling_std = prices.rolling(window=period).std()
        upper_band = sma + (rolling_std * std)
        lower_band = sma - (rolling_std * std)
        return upper_band, lower_band

    def _prepare_observation(
        self, price_history: List[float], position_info: dict
    ) -> jnp.ndarray:
        """Prepare observation for the agent."""
        # Price history normalization
        price_array = np.array(price_history[-self.window_size :])
        price_normalized = (price_array - np.mean(price_array)) / (
            np.std(price_array) + 1e-8
        )

        # Technical indicators
        indicators = self._calculate_technical_indicators(pd.Series(price_history))
        technical_features = np.array(
            [
                indicators.get("sma_20", 0),
                indicators.get("sma_50", 0),
                indicators.get("rsi", 50),
                indicators.get("macd", 0),
                indicators.get("macd_signal", 0),
                indicators.get("bb_upper", 0),
            ]
        )

        # Portfolio state
        portfolio_features = np.array(
            [
                position_info["current_price"],
                position_info["position_size"],
                position_info["unrealized_pnl"],
                position_info["available_balance"],
                position_info["total_balance"],
                position_info["leverage"],
                position_info["margin_ratio"],
            ]
        )

        # Combine all features
        observation = np.concatenate(
            [price_normalized, technical_features, portfolio_features]
        )

        return jnp.array(observation)

    def _get_position_info(self) -> dict:
        """Get current position information from Binance."""
        try:
            account = self.client.futures_account()
            positions = self.client.futures_position_information(symbol=self.symbol)
            position = next((p for p in positions if p["symbol"] == self.symbol), None)

            return {
                "current_price": float(
                    self.client.futures_mark_price(symbol=self.symbol)["markPrice"]
                ),
                "position_size": float(position["positionAmt"]) if position else 0,
                "unrealized_pnl": (
                    float(position["unRealizedProfit"]) if position else 0
                ),
                "available_balance": float(account["availableBalance"]),
                "total_balance": float(account["totalWalletBalance"]),
                "leverage": float(position["leverage"]) if position else 1,
                "margin_ratio": (
                    float(account["totalMaintMargin"])
                    / float(account["totalMarginBalance"])
                    if float(account["totalMarginBalance"]) > 0
                    else 0
                ),
            }
        except BinanceAPIException as e:
            print(f"Error getting position info: {e}")
            return {
                "current_price": 0,
                "position_size": 0,
                "unrealized_pnl": 0,
                "available_balance": 0,
                "total_balance": 0,
                "leverage": 1,
                "margin_ratio": 0,
            }

    def _execute_trade(self, action: int, position_info: dict) -> bool:
        """Execute trade on Binance based on agent's action."""
        try:
            current_position = position_info["position_size"]
            price = position_info["current_price"]

            # Define position size (you may want to adjust this based on your risk management)
            quantity = abs(
                self.agent.config.max_position_size
                * position_info["available_balance"]
                / price
            )

            if action == 1 and current_position <= 0:  # Buy
                order = self.client.futures_create_order(
                    symbol=self.symbol, side="BUY", type="MARKET", quantity=quantity
                )
                print(f"Buy order executed: {order}")
                return True

            elif action == 2 and current_position >= 0:  # Sell
                order = self.client.futures_create_order(
                    symbol=self.symbol, side="SELL", type="MARKET", quantity=quantity
                )
                print(f"Sell order executed: {order}")
                return True

            return False

        except BinanceAPIException as e:
            print(f"Error executing trade: {e}")
            return False

    def start_trading(self):
        """Start the trading loop."""
        self.running = True

        def process_message(msg):
            if msg["e"] == "kline":
                # Update price history
                close_price = float(msg["k"]["c"])
                self.price_history.append(close_price)

                # Maintain window size
                if len(self.price_history) > self.window_size * 2:
                    self.price_history = self.price_history[-self.window_size * 2 :]

                # Check if we have enough data and if it's time to update
                current_time = time.time()
                if (
                    len(self.price_history) >= self.window_size
                    and current_time - self.last_update_time
                    >= self._get_interval_seconds()
                ):

                    # Get position info
                    position_info = self._get_position_info()

                    # Prepare observation
                    obs = self._prepare_observation(self.price_history, position_info)

                    # Get action from agent
                    action, _, _, _ = self.agent.get_action_and_value(
                        self.agent.state,
                        obs,
                        jnp.array([0]),  # dummy key since we're not training
                    )

                    # Execute trade
                    if self._execute_trade(int(action[0]), position_info):
                        # Log transaction
                        self.trades_history.append(
                            TransactionLog(
                                timestamp=int(current_time),
                                action=["HOLD", "BUY", "SELL"][int(action[0])],
                                price=close_price,
                                shares=position_info["position_size"],
                                cash=position_info["available_balance"],
                                portfolio_value=position_info["total_balance"],
                            )
                        )

                    self.last_update_time = current_time

        # Start websocket
        self.conn_key = self.bm.start_kline_socket(
            self.symbol, process_message, interval=self.trade_interval
        )
        self.bm.start()

    def stop_trading(self):
        """Stop the trading loop."""
        self.running = False
        self.bm.stop_socket(self.conn_key)
        self.bm.close()

    def _get_interval_seconds(self) -> int:
        """Convert trading interval to seconds."""
        interval_map = {
            "1m": 60,
            "3m": 180,
            "5m": 300,
            "15m": 900,
            "30m": 1800,
            "1h": 3600,
            "2h": 7200,
            "4h": 14400,
            "6h": 21600,
            "8h": 28800,
            "12h": 43200,
            "1d": 86400,
        }
        return interval_map.get(self.trade_interval, 3600)

    def get_trading_summary(self) -> dict:
        """Get summary of trading performance."""
        if not self.trades_history:
            return {}

        trades_df = pd.DataFrame([t._asdict() for t in self.trades_history])

        # Calculate metrics
        initial_value = self.trades_history[0].portfolio_value
        final_value = self.trades_history[-1].portfolio_value
        total_return = (final_value - initial_value) / initial_value * 100

        buy_trades = trades_df[trades_df["action"] == "BUY"]
        sell_trades = trades_df[trades_df["action"] == "SELL"]

        return {
            "total_trades": len(buy_trades) + len(sell_trades),
            "total_return": total_return,
            "current_position": self.current_position,
            "current_portfolio_value": final_value,
            "start_time": datetime.fromtimestamp(self.trades_history[0].timestamp),
            "end_time": datetime.fromtimestamp(self.trades_history[-1].timestamp),
        }
