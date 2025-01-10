from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import environment, spaces
from jax import lax, random


@dataclass(frozen=True)
class EnvState(environment.EnvState):
    time_absolute: int
    time_relative: int
    wallet_balance: float
    available_balance: float
    coins_short: float
    coins_long: float
    average_price_short: float
    average_price_long: float
    position_value_short: float
    position_value_long: float
    initial_margin_short: float
    initial_margin_long: float
    margin_short: float
    margin_long: float
    equity: float
    unrealized_pnl_short: float
    unrealized_pnl_long: float
    state_queue: chex.Array
    reset_queue: chex.Array
    liquidation: bool
    episode_maxstep_achieved: bool


@dataclass(frozen=True)
class EnvParams(environment.EnvParams):
    dataset_name: str = "dataset"
    leverage: float = 2.0
    episode_max_len: int = 168
    lookback_window_len: int = 168
    train_start: List[int] = (7200, 10200, 13200, 16200, 19200)
    train_end: List[int] = (9700, 12700, 15700, 18700, 21741 - 1)
    test_start: List[int] = (9700, 12700, 15700, 18700)
    test_end: List[int] = (10200, 13200, 16200, 19200)
    order_size: float = 50.0
    initial_capital: float = 1000.0
    open_fee: float = 0.06e-2
    close_fee: float = 0.06e-2
    maintenance_margin_percentage: float = 0.012
    initial_random_allocated: float = 0.0
    regime: str = "training"
    record_stats: bool = False


class JaxCryptoEnv(environment.Environment):
    def __init__(self):
        super().__init__()
        self.price_array, self.tech_array_total = DiskDataLoader(
            dataset_name="dataset"
        ).load_dataset()
        # Fix: Calculate observation_dim during initialization
        self.lookback_window_len = EnvParams().lookback_window_len
        self.observation_dim = (
            self.tech_array_total.shape[1] + 2
        ) * self.lookback_window_len

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    # Fix: Make step_env pure by removing instance variable access
    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: int, params: EnvParams
    ) -> Tuple[chex.Array, EnvState, float, bool, Dict]:
        # Get current price from price array
        current_price = self.price_array[state.time_absolute, 0]
        price_bid = current_price * (1 - params.open_fee)
        price_ask = current_price * (1 + params.open_fee)

        # Fix: Handle action updates using pure functions
        def handle_action_0(carry):
            state, _ = carry
            return state, 0.0

        def handle_action_1(carry):
            state, _ = carry
            # Buy logic with pure operations
            buy_amount = jnp.where(
                (state.coins_long >= 0) & (state.available_balance > params.order_size),
                params.order_size / price_ask,
                0.0,
            )

            new_average_price_long = jnp.where(
                buy_amount > 0,
                (state.position_value_long + buy_amount * price_ask)
                / (state.coins_long + buy_amount),
                state.average_price_long,
            )

            close_short_amount = jnp.where(
                -state.coins_short > 0,
                jnp.minimum(-state.coins_short, params.order_size / price_ask),
                0.0,
            )

            realized_pnl = close_short_amount * (state.average_price_short - price_ask)

            return (
                state.replace(
                    coins_long=state.coins_long + buy_amount,
                    average_price_long=new_average_price_long,
                    coins_short=jnp.minimum(state.coins_short + close_short_amount, 0),
                    wallet_balance=state.wallet_balance + realized_pnl,
                ),
                realized_pnl,
            )

        def handle_action_2(carry):
            state, _ = carry
            # Sell logic with pure operations
            sell_amount = jnp.where(
                state.coins_long > 0,
                jnp.minimum(state.coins_long, params.order_size / price_ask),
                0.0,
            )

            realized_pnl = sell_amount * (price_bid - state.average_price_long)

            short_amount = jnp.where(
                (-state.coins_short >= 0)
                & (state.available_balance > params.order_size),
                params.order_size / price_ask,
                0.0,
            )

            new_average_price_short = jnp.where(
                short_amount > 0,
                (state.position_value_short + short_amount * price_bid)
                / (-state.coins_short + short_amount),
                state.average_price_short,
            )

            return (
                state.replace(
                    coins_long=jnp.maximum(state.coins_long - sell_amount, 0),
                    coins_short=state.coins_short - short_amount,
                    average_price_short=new_average_price_short,
                    wallet_balance=state.wallet_balance + realized_pnl,
                ),
                realized_pnl,
            )

        def handle_action_3(carry):
            state, _ = carry
            # Close all positions with pure operations
            realized_pnl_long = state.coins_long * (
                price_bid - state.average_price_long
            )
            realized_pnl_short = -state.coins_short * (
                state.average_price_short - price_ask
            )
            total_pnl = realized_pnl_long + realized_pnl_short

            return (
                state.replace(
                    coins_long=0.0,
                    coins_short=0.0,
                    initial_margin_long=0.0,
                    initial_margin_short=0.0,
                    wallet_balance=state.wallet_balance + total_pnl,
                ),
                total_pnl,
            )

        # Fix: Use pure function for action handling
        new_state, realized_pnl = lax.switch(
            action,
            [handle_action_0, handle_action_1, handle_action_2, handle_action_3],
            (state, 0.0),
        )

        # Update state calculations with pure operations
        position_value_short = -new_state.coins_short * new_state.average_price_short
        position_value_long = new_state.coins_long * new_state.average_price_long

        initial_margin_short = position_value_short / params.leverage
        initial_margin_long = position_value_long / params.leverage

        fee_to_close = (
            jnp.abs(position_value_short + position_value_long) * params.close_fee
        )

        margin_short = (
            initial_margin_short
            + params.maintenance_margin_percentage * position_value_short
        )
        margin_long = (
            initial_margin_long
            + params.maintenance_margin_percentage * position_value_long
        )

        total_margin = margin_short + margin_long + fee_to_close
        available_balance = jnp.maximum(new_state.wallet_balance - total_margin, 0)

        unrealized_pnl_short = -new_state.coins_short * (
            new_state.average_price_short - price_ask
        )
        unrealized_pnl_long = new_state.coins_long * (
            price_bid - new_state.average_price_long
        )

        next_equity = (
            new_state.wallet_balance + unrealized_pnl_short + unrealized_pnl_long
        )

        # Fix: Use pure function for observation update
        new_obs = self._get_observation_step(
            new_state.time_absolute,
            available_balance,
            unrealized_pnl_short + unrealized_pnl_long,
            new_state.state_queue,
            self.tech_array_total,
        )

        # Calculate done conditions with pure operations
        liquidation = -unrealized_pnl_long - unrealized_pnl_short > total_margin
        episode_maxstep_achieved = new_state.time_relative == params.episode_max_len - 1
        done = liquidation | episode_maxstep_achieved

        # Calculate reward with pure operations
        reward = (
            unrealized_pnl_short
            + unrealized_pnl_long
            - (new_state.unrealized_pnl_short + new_state.unrealized_pnl_long)
        ) / params.initial_capital

        # Update final state
        final_state = new_state.replace(
            time_absolute=new_state.time_absolute + 1,
            time_relative=new_state.time_relative + 1,
            position_value_short=position_value_short,
            position_value_long=position_value_long,
            initial_margin_short=initial_margin_short,
            initial_margin_long=initial_margin_long,
            margin_short=margin_short,
            margin_long=margin_long,
            available_balance=available_balance,
            unrealized_pnl_short=unrealized_pnl_short,
            unrealized_pnl_long=unrealized_pnl_long,
            equity=next_equity,
            liquidation=liquidation,
            episode_maxstep_achieved=episode_maxstep_achieved,
            state_queue=new_obs,
        )

        info = {
            "equity": next_equity,
            "wallet_balance": new_state.wallet_balance,
            "margin_short": margin_short,
            "margin_long": margin_long,
            "realized_pnl": realized_pnl,
        }

        return new_obs, final_state, reward, done, info

    # Fix: Make observation methods pure and jit-compatible
    @partial(jax.jit, static_argnums=(0,))
    def _get_observation_step(
        self,
        current_time: int,
        available_balance: float,
        unrealized_pnl: float,
        state_queue: chex.Array,
        tech_array: chex.Array,
    ) -> chex.Array:
        input_array = tech_array[current_time]
        current_observation = jnp.concatenate(
            [
                input_array[:2],
                jnp.array([available_balance, unrealized_pnl]),
                input_array[2:],
            ]
        )

        return jnp.roll(state_queue, -1, axis=0).at[-1].set(current_observation)

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        # Fix: Use pure operations for regime selection
        regime_idx = {"training": 0, "evaluation": 1, "backtesting": 2}[params.regime]

        def get_time(regime_key, interval_lists):
            train_start, train_end, test_start, test_end = interval_lists
            interval_key, time_key = jax.random.split(regime_key)

            interval = jax.random.randint(interval_key, (1,), 0, len(train_start))[0]

            start = jnp.where(
                regime_idx == 0, train_start[interval], test_start[interval]
            )

            end = jnp.where(regime_idx == 0, train_end[interval], test_end[interval])

            return jax.random.randint(
                time_key, (1,), start, end - params.episode_max_len
            )[0]

        time_absolute = get_time(
            key,
            (params.train_start, params.train_end, params.test_start, params.test_end),
        )

        # Initialize state with pure operations
        initial_state = EnvState(
            time_absolute=time_absolute,
            time_relative=0,
            wallet_balance=params.initial_capital,
            available_balance=params.initial_capital,
            coins_short=0.0,
            coins_long=0.0,
            average_price_short=self.price_array[time_absolute, 0],
            average_price_long=self.price_array[time_absolute, 0],
            position_value_short=0.0,
            position_value_long=0.0,
            initial_margin_short=0.0,
            initial_margin_long=0.0,
            margin_short=0.0,
            margin_long=0.0,
            equity=params.initial_capital,
            unrealized_pnl_short=0.0,
            unrealized_pnl_long=0.0,
            state_queue=jnp.zeros(
                (params.lookback_window_len, self.tech_array_total.shape[1] + 2)
            ),
            reset_queue=jnp.zeros(
                (params.lookback_window_len * 4, self.tech_array_total.shape[1] + 2)
            ),
            liquidation=False,
            episode_maxstep_achieved=False,
        )

        # Get initial observation using pure operations
        obs = self._get_observation_reset(initial_state, params, self.tech_array_total)

        return obs, initial_state

    @partial(jax.jit, static_argnums=(0,))
    def _get_observation_reset(
        self, state: EnvState, params: EnvParams, tech_array: chex.Array
    ) -> chex.Array:
        def scan_fn(carry, time_idx):
            queue = carry
            current_time = (
                state.time_absolute - params.lookback_window_len * 4 + time_idx
            )

            input_array = tech_array[current_time]
            current_observation = jnp.concatenate(
                [
                    input_array[:2],
                    jnp.array(
                        [
                            state.available_balance,
                            state.unrealized_pnl_short + state.unrealized_pnl_long,
                        ]
                    ),
                    input_array[2:],
                ]
            )

            return jnp.roll(queue, -1, axis=0).at[-1].set(current_observation), None

        init_queue = jnp.zeros(
            (params.lookback_window_len * 4, tech_array.shape[1] + 2)
        )
        final_queue, _ = lax.scan(
            scan_fn, init_queue, jnp.arange(params.lookback_window_len * 4)
        )

        return final_queue[-params.lookback_window_len :]

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            low=jnp.full((self.observation_dim,), -jnp.inf),
            high=jnp.full((self.observation_dim,), jnp.inf),
        )

    def action_space(self, params: EnvParams) -> spaces.Discrete:
        return spaces.Discrete(4)


class DiskDataLoader:
    def __init__(self, dataset_name: str = "dataset"):
        self.dataset_name = dataset_name
        self.price_shape = (25000, 6)
        self.tech_shape = (25000, 40)

    @partial(jax.jit, static_argnums=(0,))
    def load_dataset(self) -> Tuple[chex.Array, chex.Array]:
        def load_data():
            # Load data from disk using numpy (implement actual loading logic)
            price_data = np.load(f"{self.dataset_name}_price.npy")
            tech_data = np.load(f"{self.dataset_name}_tech.npy")

            # Ensure correct shapes
            price_data = price_data.reshape(self.price_shape)
            tech_data = tech_data.reshape(self.tech_shape)

            return jnp.array(price_data), jnp.array(tech_data)

        return jax.pure_callback(
            load_data, (jnp.ones(self.price_shape), jnp.ones(self.tech_shape))
        )


# Helper functions
# Modify create_env to return the necessary components separately
def create_env():
    """Create environment components with default parameters."""
    price_array, tech_array_total = DiskDataLoader(
        dataset_name="dataset"
    ).load_dataset()
    params = EnvParams()

    # Return the data arrays and parameters separately instead of the env instance
    return price_array, tech_array_total, params


# Example usage with proper JAX patterns
def run_episode(
    env: JaxCryptoEnv, params: EnvParams, rng: chex.PRNGKey, policy_fn: callable
) -> Tuple[float, Dict]:
    """Run a single episode with pure operations."""

    def episode_step(carry, _):
        state, rng, total_reward = carry

        # Split RNG key for action and step
        rng, action_key, step_key = jax.random.split(rng, 3)

        # Get action from policy
        action = policy_fn(state.state_queue, action_key)

        # Step environment
        obs, next_state, reward, done, info = env.step_env(
            step_key, state, action, params
        )

        # Update total reward
        total_reward += reward

        return (next_state, rng, total_reward), (reward, info, done)

    # Reset environment
    rng, reset_key = jax.random.split(rng)
    obs, init_state = env.reset_env(reset_key, params)

    # Run episode
    final_carry, episode_history = lax.scan(
        episode_step, (init_state, rng, 0.0), jnp.arange(params.episode_max_len)
    )

    _, _, total_reward = final_carry
    return total_reward, episode_history


# Example usage
if __name__ == "__main__":
    # Create environment components
    price_array, tech_array, params = create_env()

    # Initialize environment state directly
    rng = random.PRNGKey(0)

    # Create the environment instance (not jitted)
    env = JaxCryptoEnv()

    # Reset environment
    rng, reset_key = random.split(rng)
    obs, state = env.reset_env(reset_key, params)

    # Run one episode
    done = False
    total_reward = 0

    while not done:
        # Sample random action
        rng, action_key = random.split(rng)
        action = random.randint(action_key, (1,), 0, 4)[0]

        # Step environment
        rng, step_key = random.split(rng)
        obs, state, reward, done, info = env.step_env(step_key, state, action, params)
        total_reward += reward

    print(f"Episode finished with total reward: {total_reward}")
