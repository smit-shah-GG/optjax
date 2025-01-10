import polars as pl
from gymnax.environments import environment
from gymnax.environments import spaces
from typing import Tuple
import os
from datetime import datetime
from functools import partial
from typing import Any, NamedTuple, Sequence
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax import lax


@struct.dataclass
class EnvState(environment.EnvState):
    price: float
    cash: float
    shares: int
    time: int
    historical_prices: jnp.ndarray  # Rolling window of prices
    returns: jnp.ndarray  # Rolling window of returns


@struct.dataclass
class EnvParams(environment.EnvParams):
    initial_cash: float = 1000000
    max_steps: int = 500  # Increased for more learning opportunities
    window_size: int = 30
    transaction_cost: float = 0.001
    num_envs: int = 1  # Support for parallel environments
    random_seed: int = 42  # Reproducibility


class TradingEnv(environment.Environment[EnvState, EnvParams]):
    def __init__(self, token: str = "BTCUSDT", window_size: int = 30):
        super().__init__()
        self.window_size = window_size
        data_dir = "data/token_data/" + token + "/"
        self.price_data = self.load_price_data(data_dir)
        self.data_len = len(self.price_data)

        # Pre-calculate all technical indicators
        self.technical_indicators = self.calculate_all_indicators(self.price_data)

    def load_price_data(self, data_dir):
        parquet_files = [f for f in os.listdir(data_dir) if f.endswith(".parquet")]
        parquet_files = sorted([os.path.join(data_dir, f) for f in parquet_files])

        df_list = [pl.read_parquet(file)["close"].to_pandas() for file in parquet_files]
        combined_df = pd.concat(df_list, ignore_index=True)

        return jnp.array(combined_df.values)

    def calculate_all_indicators(self, prices):
        """Pre-calculate all technical indicators for the entire price series."""

        def calculate_ma(prices, window):
            """Calculate moving average using JAX operations."""
            weights = jnp.ones(window) / window
            ma = jnp.convolve(prices, weights, mode="valid")
            # Pad the beginning to maintain array length
            pad_width = len(prices) - len(ma)
            ma = jnp.pad(ma, (pad_width, 0), mode="edge")
            return ma

        def calculate_rsi(prices, period=14):
            """Calculate RSI using JAX operations."""
            deltas = jnp.diff(prices, prepend=prices[0])
            gains = jnp.maximum(deltas, 0)
            losses = -jnp.minimum(deltas, 0)

            avg_gains = calculate_ma(gains, period)
            avg_losses = calculate_ma(losses, period)

            rs = avg_gains / (avg_losses + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi

        def calculate_volatility(prices, window=30):
            """Calculate rolling volatility."""
            returns = jnp.diff(jnp.log(prices), prepend=jnp.log(prices[0]))
            vol = jnp.sqrt(calculate_ma(returns**2, window))
            return vol

        def calculate_momentum(prices, window=14):
            """Calculate momentum indicator."""
            momentum = (prices / jnp.roll(prices, window) - 1) * 100
            momentum = momentum.at[:window].set(0)  # Replace initial NaN values
            return momentum

        # Calculate all indicators
        indicators = {
            "ma7": calculate_ma(prices, 7),
            "ma25": calculate_ma(prices, 25),
            "ma99": calculate_ma(prices, 99),
            "rsi": calculate_rsi(prices),
            "volatility": calculate_volatility(prices),
            "momentum": calculate_momentum(prices),
        }

        return indicators

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(window_size=self.window_size)

    def get_initial_historical_prices(self, start_idx):
        """Get initial historical prices for the rolling window."""
        end_idx = start_idx + 1
        start_window_idx = max(0, end_idx - self.window_size)

        # Get price window
        price_window = self.price_data[start_window_idx:end_idx]

        # Pad if necessary
        if len(price_window) < self.window_size:
            price_window = jnp.pad(
                price_window, (self.window_size - len(price_window), 0), mode="edge"
            )

        # Calculate returns
        returns = jnp.diff(jnp.log(price_window), prepend=jnp.log(price_window[0]))

        return price_window, returns

    def reset(self, key):
        """Reset the environment with a random starting point."""
        key, subkey = jax.random.split(key)
        start_idx = jax.random.randint(
            subkey, (1,), 0, max(1, self.data_len - self.default_params.max_steps)
        )[0]

        # Get initial prices and returns
        historical_prices, returns = self.get_initial_historical_prices(start_idx)

        # Create initial state
        state = EnvState(
            price=self.price_data[start_idx],
            cash=self.default_params.initial_cash,
            shares=0,
            time=start_idx,
            historical_prices=historical_prices,
            returns=returns,
        )

        obs = self.get_observation(state)
        return obs, state, key

    def update_historical_data(
        self, state: EnvState, new_price: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Update historical prices and returns with new data."""
        # Update historical prices
        historical_prices = jnp.roll(state.historical_prices, -1)
        historical_prices = historical_prices.at[-1].set(new_price)

        # Update returns
        returns = jnp.roll(state.returns, -1)
        new_return = jnp.log(new_price) - jnp.log(state.price)
        returns = returns.at[-1].set(new_return)

        return historical_prices, returns

    def reset_at(self, key, params=None):
        """Reset method for parallel environments."""
        params = params or self.default_params
        keys = jax.random.split(key, params.num_envs)
        reset_fn = jax.vmap(self.reset)
        obs, states, keys = reset_fn(keys)
        return obs, states, keys

    def step_at(self, keys, states, actions):
        """Step method for parallel environments."""
        step_fn = jax.vmap(self.step, in_axes=(0, 0, 0))
        obs, new_states, rewards, dones, new_keys = step_fn(keys, states, actions)
        return obs, new_states, rewards, dones, new_keys

    def get_observation(self, state: EnvState) -> jnp.ndarray:
        """Enhanced observation with better normalization."""

        # Existing observation logic, with added normalization
        def robust_normalize(x, epsilon=1e-8):
            return (x - jnp.mean(x)) / (jnp.std(x) + epsilon)

        normalized_prices = robust_normalize(state.historical_prices)

        technical_indicators = jnp.array(
            [
                robust_normalize(self.technical_indicators["ma7"])[state.time],
                robust_normalize(self.technical_indicators["ma25"])[state.time],
                robust_normalize(self.technical_indicators["ma99"])[state.time],
                self.technical_indicators["rsi"][state.time] / 100,
                robust_normalize(self.technical_indicators["volatility"])[state.time],
                self.technical_indicators["momentum"][state.time] / 100,
            ]
        )

        portfolio_value = state.cash + state.shares * state.price
        portfolio_state = jnp.array(
            [
                jnp.clip(state.shares * state.price / portfolio_value, 0, 1),
                jnp.clip(state.cash / portfolio_value, 0, 1),
                jnp.tanh(jnp.std(state.returns) * jnp.sqrt(252)),  # Bounded volatility
                jnp.tanh(jnp.mean(state.returns) * 252),  # Bounded return
                jnp.clip(state.shares / 10, 0, 1),  # Normalized shares
                jnp.clip(state.cash / self.default_params.initial_cash, 0, 1),
                jnp.clip(portfolio_value / self.default_params.initial_cash, 0, 2),
            ]
        )

        return jnp.concatenate(
            [normalized_prices, technical_indicators, portfolio_state]
        )

    def step(self, key, state: EnvState, action: int):
        """Execute one step in the environment with risk-adjusted rewards."""
        action = jnp.asarray(action).item()
        old_portfolio_value = state.cash + state.shares * state.price

        # Get new price
        new_price = self.price_data[state.time + 1]

        # Execute trade
        def hold(cash, shares):
            return jnp.array([cash, shares], dtype=jnp.float32)

        def buy(cash, shares):
            cost = new_price * (1 + self.default_params.transaction_cost)
            max_shares = jnp.floor(cash / cost)
            shares_to_buy = jnp.minimum(1, max_shares)  # Buy 1 share at a time
            total_cost = shares_to_buy * cost
            return jnp.array(
                [cash - total_cost, shares + shares_to_buy], dtype=jnp.float32
            )

        def sell(cash, shares):
            proceeds = new_price * (1 - self.default_params.transaction_cost)
            new_cash = jnp.where(shares > 0, cash + proceeds, cash)
            new_shares = jnp.where(shares > 0, shares - 1, shares)
            return jnp.array([new_cash, new_shares], dtype=jnp.float32)

        cash, shares = jax.lax.switch(
            action, [hold, buy, sell], state.cash, state.shares
        )

        # Update historical data
        historical_prices, returns = self.update_historical_data(state, new_price)

        # Calculate base reward (portfolio return)
        new_portfolio_value = cash + shares * new_price
        base_reward = jnp.log(new_portfolio_value / old_portfolio_value)

        # Add trading penalty and risk adjustment
        trade_penalty = jnp.where(
            action != 0, -0.0001, 0.0
        )  # Smaller penalty for trading
        vol = jnp.std(returns) * jnp.sqrt(252)  # Annualized volatility
        risk_adjustment = -0.1 * vol  # Penalize high volatility

        # Combine and clip reward components
        reward = base_reward + trade_penalty + risk_adjustment
        reward = jnp.clip(reward, -1.0, 1.0)  # Clip for stability

        # Update state
        new_state = EnvState(
            price=new_price,
            cash=cash,
            shares=shares,
            time=state.time + 1,
            historical_prices=historical_prices,
            returns=returns,
        )

        done = new_state.time >= min(
            self.data_len - 1, state.time + self.default_params.max_steps
        )

        obs = self.get_observation(new_state)

        return obs, new_state, reward, done, key

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return 3  # hold, buy, sell

    @property
    def observation_space(self) -> spaces.Box:
        """Observation space of the environment."""
        # Space for: price history (window_size) + technical indicators (6) + portfolio state (7)
        obs_dim = self.window_size + 6 + 7
        return spaces.Box(
            low=-jnp.inf * jnp.ones(obs_dim),
            high=jnp.inf * jnp.ones(obs_dim),
            shape=(obs_dim,),
        )

    @property
    def action_space(self) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(3)
