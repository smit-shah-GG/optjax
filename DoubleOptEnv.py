import polars as pl
from gymnax.environments import environment
from gymnax.environments import spaces

import os
from datetime import datetime
from functools import partial
from typing import Any, NamedTuple, Sequence
import pandas as pd
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from jax import lax
from jax import vmap


@struct.dataclass
class EnvState(environment.EnvState):
    price: float
    cash: float
    shares: int
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    initial_price: float = 100
    initial_cash: float = 1000000
    price_std: float = 0.1
    max_steps: int = 100


class TradingEnv(environment.Environment[EnvState, EnvParams]):
    def __init__(self, token: str = "BTCUSDT", window_size: int = 30):
        super().__init__()
        self.window_size = window_size
        data_dir = "/kaggle/input/btcusdt/"
        self.price_data = self.load_price_data(data_dir)
        self.data_len = len(self.price_data)

        # Calculate technical indicators
        self.technical_indicators = self.calculate_indicators(self.price_data)

    def load_price_data(self, data_dir):
        # List all the Parquet files in the directory
        parquet_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".parquet")
        ]

        # Load and concatenate all the files
        df_list = [pl.read_parquet(file)["close"].to_pandas() for file in parquet_files]

        # Combine all the DataFrames into one
        combined_df = pd.concat(df_list, ignore_index=True)

        # Convert to a NumPy array for fast access in the environment
        return combined_df.to_numpy()

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def reset(self, key):
        # Reset the environment with the initial state
        state = EnvState(
            price=self.price_data[0],  # Use the first price from the combined data
            cash=self.default_params.initial_cash,
            shares=0,
            time=0,
        )
        obs = self.get_observation(state)
        return obs, state, key

    def step(self, key, state: EnvState, action: int):
        action = jnp.asarray(action).item()

        done = state.time + 1 >= self.data_len
        result = (self.get_observation(state), state, 0.0, done, key)

        # Calculate portfolio value before action
        old_portfolio_value = state.cash + state.shares * state.price

        # Get new price
        new_price = self.price_data[state.time + 1]

        # Execute trade with transaction costs
        transaction_cost = 0.001  # 0.1% transaction fee

        def hold(cash, shares):
            return jnp.array([cash, shares], dtype=jnp.float32)

        def buy(cash, shares):
            cost = new_price * (1 + transaction_cost)
            max_shares = jnp.floor(cash / cost)
            shares_to_buy = jnp.minimum(1, max_shares)
            total_cost = shares_to_buy * cost
            return jnp.array(
                [cash - total_cost, shares + shares_to_buy], dtype=jnp.float32
            )

        def sell(cash, shares):
            proceeds = new_price * (1 - transaction_cost)
            new_cash = jnp.where(shares > 0, cash + proceeds, cash)
            new_shares = jnp.where(shares > 0, shares - 1, shares)
            return jnp.array([new_cash, new_shares], dtype=jnp.float32)

        cash, shares = jax.lax.switch(
            action, [hold, buy, sell], state.cash, state.shares
        )

        # Calculate new portfolio value and log return
        new_portfolio_value = cash + shares * new_price
        reward = jnp.log(new_portfolio_value / old_portfolio_value)

        # Update state
        new_state = EnvState(
            price=new_price,
            cash=cash,
            shares=shares,
            time=state.time + 1,
        )

        done = new_state.time >= self.default_params.max_steps
        obs = self.get_observation(new_state)
        return obs, new_state, reward, done, key

    @property
    def name(self) -> str:
        """Environment name."""
        return "Crypto Environment Trial 1"

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3

    @property
    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        low = jnp.array([0, 0, 0])  # Minimum values for price, cash, shares
        high = jnp.array([jnp.inf, jnp.inf, jnp.inf])  # Maximum values
        return spaces.Box(low=low, high=high, shape=(3,), dtype=jnp.float32)

    @property
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(3)  # 0: hold, 1: buy, 2: sell

    def calculate_indicators(self, prices):
        # Calculate various technical indicators
        def calculate_ma(prices, window):
            return jnp.convolve(prices, jnp.ones(window) / window, mode="valid")

        ma7 = calculate_ma(prices, 7)
        ma25 = calculate_ma(prices, 25)

        # Calculate RSI
        def calculate_rsi(prices, period=14):
            deltas = jnp.diff(prices)
            gains = jnp.where(deltas > 0, deltas, 0)
            losses = jnp.where(deltas < 0, -deltas, 0)

            avg_gain = calculate_ma(gains, period)
            avg_loss = calculate_ma(losses, period)

            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi

        rsi = calculate_rsi(prices)

        # Combine indicators
        indicators = {"ma7": ma7, "ma25": ma25, "rsi": rsi}
        return indicators

    def get_observation(self, state: EnvState):
        # Get historical prices for the window
        end_idx = state.time + 1
        start_idx = max(0, end_idx - self.window_size)
        price_window = self.price_data[start_idx:end_idx]

        # Pad if necessary
        if len(price_window) < self.window_size:
            pad_size = self.window_size - len(price_window)
            price_window = jnp.pad(price_window, (pad_size, 0), mode="edge")

        # Get technical indicators
        # Only get indicators if we have enough data
        ma7 = self.technical_indicators["ma7"][
            min(state.time, len(self.technical_indicators["ma7"]) - 1)
        ]
        ma25 = self.technical_indicators["ma25"][
            min(state.time, len(self.technical_indicators["ma25"]) - 1)
        ]
        rsi = self.technical_indicators["rsi"][
            min(state.time, len(self.technical_indicators["rsi"]) - 1)
        ]

        # Normalize features
        price_window = (price_window - jnp.mean(price_window)) / (
            jnp.std(price_window) + 1e-8
        )

        # Include position information and account value
        portfolio_value = state.cash + state.shares * state.price
        position_size = (
            state.shares * state.price / portfolio_value if portfolio_value > 0 else 0
        )

        return jnp.concatenate(
            [
                price_window,
                jnp.array(
                    [
                        ma7 / state.price - 1,  # MA7 relative to current price
                        ma25 / state.price - 1,  # MA25 relative to current price
                        rsi / 100,  # Normalize RSI to [0,1]
                        position_size,  # Current position size as fraction of portfolio
                        (
                            state.cash / portfolio_value if portfolio_value > 0 else 0
                        ),  # Cash ratio
                    ]
                ),
            ]
        )
