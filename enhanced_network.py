import polars as pl
from gymnax.environments import environment
from gymnax.environments import spaces

import os
from datetime import datetime
from functools import partial
from typing import Any, NamedTuple, Sequence, Tuple
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
from MoreTechEnv import TradingEnv


class PPOState(struct.PyTreeNode):
    train_state: TrainState
    key: jnp.ndarray

    @classmethod
    def create(cls, *, apply_fn, params, tx, key):
        return cls(
            train_state=TrainState.create(
                apply_fn=apply_fn,
                params=params,
                tx=tx,
            ),
            key=key,
        )


class TradingMetrics(NamedTuple):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    positions_held: float
    num_trades: int


class PriceEncoder(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, price_data, training=True):
        # Price data shape: (batch_size, window_size)
        x = price_data

        # First dense layer
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(1.0))(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        # Second dense layer
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(1.0))(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        return x


class TechnicalEncoder(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, technical_data, training=True):
        # Technical data shape: (batch_size, num_indicators)
        x = technical_data

        # First dense layer with smaller hidden dimension
        x = nn.Dense(self.hidden_dim // 2, kernel_init=nn.initializers.orthogonal(1.0))(
            x
        )
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        # Second dense layer matching hidden dimension
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(1.0))(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        return x


class PortfolioEncoder(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, portfolio_data, training=True):
        # Portfolio data shape: (batch_size, num_portfolio_features)
        x = portfolio_data

        # First dense layer with smaller hidden dimension
        x = nn.Dense(self.hidden_dim // 2, kernel_init=nn.initializers.orthogonal(1.0))(
            x
        )
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        # Second dense layer matching hidden dimension
        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(1.0))(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        return x


class EnhancedActorCritic(nn.Module):
    action_dim: int
    hidden_dim: int = 512
    num_layers: int = 3

    @nn.compact
    def __call__(self, obs, training=True):
        # First, ensure obs is 2D
        obs = obs.reshape(-1, obs.shape[-1]) if obs.ndim == 1 else obs

        # Calculate the expected sizes based on your environment
        window_size = 30
        num_technical = 6
        num_portfolio = 7
        total_features = window_size + num_technical + num_portfolio

        # Verify the input shape
        assert (
            obs.shape[-1] == total_features
        ), f"Expected {total_features} features, got {obs.shape[-1]}"

        # Split observation into components
        start_idx = 0
        price_history = obs[:, start_idx : start_idx + window_size]
        start_idx += window_size

        technical_indicators = obs[:, start_idx : start_idx + num_technical]
        start_idx += num_technical

        portfolio_state = obs[:, start_idx : start_idx + num_portfolio]

        # Encode each component
        price_encoded = PriceEncoder(hidden_dim=self.hidden_dim)(price_history)
        technical_encoded = TechnicalEncoder(hidden_dim=self.hidden_dim)(
            technical_indicators
        )
        portfolio_encoded = PortfolioEncoder(hidden_dim=self.hidden_dim)(
            portfolio_state
        )

        # Rest of the implementation remains the same
        combined = jnp.concatenate(
            [price_encoded, technical_encoded, portfolio_encoded], axis=-1
        )

        x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(1.0))(
            combined
        )
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        for _ in range(self.num_layers - 1):
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(1.0))(
                x
            )
            x = nn.gelu(x)
            x = nn.LayerNorm()(x)

        risk_features = nn.Dense(
            self.hidden_dim // 2, kernel_init=nn.initializers.orthogonal(1.0)
        )(x)
        risk_features = nn.gelu(risk_features)

        var = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(risk_features)
        vol = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(risk_features)
        sharpe = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(risk_features)

        features = jnp.concatenate([x, var, vol, sharpe], axis=-1)

        actor_features = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.orthogonal(1.0)
        )(features)
        actor_features = nn.gelu(actor_features)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=nn.initializers.orthogonal(0.01)
        )(actor_features)

        critic1 = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.orthogonal(1.0)
        )(features)
        critic1 = nn.gelu(critic1)
        value1 = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic1)

        critic2 = nn.Dense(
            self.hidden_dim, kernel_init=nn.initializers.orthogonal(1.0)
        )(features)
        critic2 = nn.gelu(critic2)
        value2 = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.0))(critic2)

        value = jnp.squeeze((value1 + value2) / 2, axis=-1)

        return action_logits, value


class TransactionLog(NamedTuple):
    timestamp: int
    action: str
    price: float
    shares: int
    cash: float
    portfolio_value: float


@struct.dataclass
class PPOConfig:
    # PPO hyperparameters
    num_envs: int = 16
    num_steps: int = 256
    num_minibatches: int = 4
    update_epochs: int = 8
    gamma: float = 0.98
    gae_lambda: float = 0.95
    clip_eps: float = 0.4
    ent_coef: float = 0.3
    vf_coef: float = 0.25
    max_grad_norm: float = 1.0
    lr: float = 5e-4
    total_timesteps: int = int(5e6)
    anneal_lr: bool = False
    debug: bool = True

    # Computed fields
    num_updates: int = struct.field(default=0)
    minibatch_size: int = struct.field(default=0)

    # Architecture
    hidden_dim: int = 512
    window_size: int = 30

    # Trading specific
    initial_cash: float = 1000000
    max_position_size: float = 0.3
    transaction_cost: float = 0.0005


class PPOAgent:
    def __init__(self, config: PPOConfig):
        self.config = config
        self.env = TradingEnv(token="BTCUSDT", window_size=config.window_size)
        self.action_dim = self.env.num_actions
        self.transaction_log = []

        # Initialize network
        self.network = EnhancedActorCritic(
            action_dim=self.action_dim, hidden_dim=config.hidden_dim, num_layers=3
        )

        key = jax.random.PRNGKey(0)
        dummy_obs = jnp.zeros((1, self.env.observation_space.shape[0]))
        params = self.network.init(key, dummy_obs)

        # Setup optimizer with cosine decay
        tx = optax.chain(
            optax.clip_by_global_norm(config.max_grad_norm),
            optax.adam(config.lr, eps=1e-5),
        )

        self.state = PPOState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=tx,
            key=key,
        )

    def log_transaction(self, timestamp, action, state):
        """Log transaction details"""
        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        portfolio_value = float(state.cash + state.shares * state.price)
        portfolio_value = round(portfolio_value, 2)

        transaction = TransactionLog(
            timestamp=timestamp,
            action=action_map[action],
            price=float(state.price),
            shares=int(state.shares),
            cash=float(state.cash),
            portfolio_value=float(portfolio_value),
        )
        self.transaction_log.append(transaction)

    def save_transaction_log(self, filename="transaction_log.txt"):
        dirr = "logs"
        if not os.path.exists(dirr):
            os.makedirs(dirr)

        filepath = os.path.join(dirr, filename)

        with open(filepath, "w") as f:
            f.write("Timestamp | Action | Price | Shares | Cash | Portfolio Value\n")
            f.write("-" * 70 + "\n")
            for tx in self.transaction_log:
                f.write(
                    f"{tx.timestamp} | {tx.action:5} | {tx.price:8.2f} | {tx.shares:6} | {tx.cash:10.2f} | {tx.portfolio_value:10.2f}\n"
                )

    @partial(jax.jit, static_argnums=(0,))
    def get_action_and_value(self, state: PPOState, obs: jnp.ndarray, key: jnp.ndarray):
        logits, value = self.network.apply(state.train_state.params, obs)

        # Add numerical stability
        logits = jnp.clip(logits, -20, 20)
        logits = logits - jax.scipy.special.logsumexp(logits, axis=-1, keepdims=True)

        key, subkey = jax.random.split(key)
        pi = distrax.Categorical(logits=logits)
        action = pi.sample(seed=subkey)
        action = jnp.clip(action, 0, self.action_dim - 1)
        log_prob = pi.log_prob(action)

        return action, log_prob, value, key

    @partial(jax.jit, static_argnums=(0,))
    def train_step(
        self,
        state: PPOState,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        log_probs: jnp.ndarray,
        values: jnp.ndarray,
        rewards: jnp.ndarray,
        dones: jnp.ndarray,
        advantages: jnp.ndarray,
        returns: jnp.ndarray,
    ):
        def loss_fn(params):
            # Wrap the network.apply function to handle batches
            apply_fn = vmap(self.network.apply, in_axes=(None, 0))
            logits, new_values = apply_fn(params, obs)

            pis = [distrax.Categorical(logits=logit) for logit in logits]
            new_log_probs = jnp.array(
                [pi.log_prob(action) for pi, action in zip(pis, actions)]
            )

            ratio = jnp.clip(jnp.exp(new_log_probs - log_probs), 1e-5, 1e5)
            ratio = jnp.nan_to_num(ratio, nan=1.0, posinf=1.0, neginf=1.0)

            # Policy loss
            policy_loss = -jnp.minimum(
                ratio * advantages,
                jnp.clip(ratio, 1 - self.config.clip_eps, 1 + self.config.clip_eps)
                * advantages,
            ).mean()

            # Value loss with dual critics
            value_pred_clipped = values[:-1] + jnp.clip(
                new_values - values[:-1], -self.config.clip_eps, self.config.clip_eps
            )
            value_losses = jnp.square(new_values - returns)
            value_losses_clipped = jnp.square(value_pred_clipped - returns)
            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

            # Entropy bonus
            entropies = jnp.array([pi.entropy() for pi in pis])
            entropy = entropies.mean()

            total_loss = (
                policy_loss
                + self.config.vf_coef * value_loss
                - self.config.ent_coef * entropy
            )

            return total_loss, (policy_loss, value_loss, entropy)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (policy_loss, value_loss, entropy)), grads = grad_fn(
            state.train_state.params
        )

        grads = jax.tree.map(
            lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), grads
        )

        new_state = state.replace(
            train_state=state.train_state.apply_gradients(grads=grads)
        )

        metrics = {
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
        }

        return new_state, metrics

    @partial(jax.jit, static_argnums=(0,))
    def create_optimizer(self, config: PPOConfig) -> optax.GradientTransformation:
        """
        JAX-optimized optimizer creation using lax.cond.
        """

        def create_annealing_optimizer():
            return optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(learning_rate=self._cosine_decay_schedule, eps=1e-5),
            )

        def create_fixed_optimizer():
            return optax.chain(
                optax.clip_by_global_norm(config.max_grad_norm),
                optax.adam(config.lr, eps=1e-5),
            )

        return lax.cond(
            config.anneal_lr, create_annealing_optimizer, create_fixed_optimizer
        )

    def _cosine_decay_schedule(self, count):
        decay_steps = (
            self.config.num_updates
            * self.config.num_minibatches
            * self.config.update_epochs
        )
        alpha = 0.0
        decay = 0.5 * (1 + jnp.cos(jnp.pi * count / decay_steps))
        return self.config.lr * decay + alpha * self.config.lr

    @partial(jax.jit, static_argnums=(0,))
    def compute_gae(
        self,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        dones: jnp.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> jnp.ndarray:
        """
        JAX-optimized GAE computation using scan.
        Replaces explicit loops with JAX's scan operation.
        """
        # Prepare deltas
        values_except_last = values[:-1]
        next_values = values[1:]
        deltas = rewards + gamma * next_values * (1 - dones) - values_except_last

        # Reverse sequences for scan
        reversed_deltas = jnp.flip(deltas)
        reversed_dones = jnp.flip(dones)

        def gae_step(carry: float, x: Tuple[float, float]) -> Tuple[float, float]:
            delta, done = x
            gae = delta + gamma * gae_lambda * (1 - done) * carry
            return gae, gae

        # Compute advantages using scan
        _, advantages = lax.scan(
            gae_step, init=jnp.zeros(256), xs=(reversed_deltas, reversed_dones)
        )

        # Flip back to original order
        advantages = jnp.flip(advantages)

        # Normalize advantages
        advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

        return advantages

    @partial(jax.jit, static_argnums=(0,))
    def calculate_trading_metrics(
        self,
        prices: jnp.ndarray,
        actions: jnp.ndarray,
        portfolio_values: jnp.ndarray,
        initial_cash: float,
    ) -> TradingMetrics:
        # Ensure all arrays are the same length by taking the minimum length
        min_length = min(len(prices), len(actions), len(portfolio_values))
        prices = prices[:min_length]
        actions = actions[:min_length]
        portfolio_values = portfolio_values[:min_length]

        returns = (portfolio_values[1:] - portfolio_values[:-1]) / portfolio_values[:-1]

        total_return = (portfolio_values[-1] - initial_cash) / initial_cash
        avg_return = jnp.mean(returns)
        std_return = jnp.std(returns)
        risk_free_rate = 0.02 / 252  # daily risk-free rate
        sharpe_ratio = (
            jnp.sqrt(252) * (avg_return - risk_free_rate) / (std_return + 1e-8)
        )

        # Use JAX-friendly max calculation (running maximum)
        def running_max(carry, x):
            return jnp.maximum(carry, x), jnp.maximum(carry, x)

        _, peak = lax.scan(running_max, portfolio_values[0], portfolio_values[1:])
        peak = jnp.concatenate([portfolio_values[:1], peak])
        drawdowns = (peak - portfolio_values) / peak
        max_drawdown = jnp.max(drawdowns) * 100

        # Calculate the number of trades by detecting position changes
        position_changes = jnp.diff(actions.squeeze())
        trades = position_changes != 0
        num_trades = jnp.sum(trades)

        # Track returns for individual trades
        def calculate_trade_returns(carry, x):
            current_position, entry_price, cumulative_return = carry
            price, action = x

            # Check if position changed
            position_changed = action != current_position

            # Calculate raw return
            raw_return = (price - entry_price) / (entry_price + 1e-8)

            # Determine if we should apply the return
            should_apply_return = jnp.logical_and(
                position_changed, current_position != 0
            )

            # Apply sign based on position direction
            trade_return = jnp.where(
                should_apply_return,
                jnp.where(current_position < 0, -raw_return, raw_return),
                0.0,
            )

            # Update entry price
            new_entry_price = jnp.where(
                jnp.logical_and(position_changed, action != 0),
                price,
                jnp.where(position_changed, 0.0, entry_price),
            )

            # Update current position
            new_position = jnp.where(position_changed, action, current_position)

            new_cumulative_return = cumulative_return + trade_return

            return (new_position, new_entry_price, new_cumulative_return), trade_return

        # Initialize carry state
        init_carry = (jnp.array(0), jnp.array(0.0), jnp.array(0.0))

        # Prepare data for scan
        xs = (
            prices[:-1],
            actions.squeeze()[:-1],
        )  # Use all but last element to match lengths

        # Run the scan
        _, trade_returns = lax.scan(calculate_trade_returns, init_carry, xs)

        # Calculate win rate, profit factor, and average trade return
        non_zero_returns = jnp.where(trade_returns > 0, trade_returns, 0)
        positive_returns = jnp.sum(jnp.maximum(non_zero_returns, 0))
        negative_returns = jnp.abs(jnp.sum(jnp.minimum(non_zero_returns, 0)))

        win_rate = jnp.where(
            len(non_zero_returns) > 0, jnp.mean(non_zero_returns > 0) * 100, 0.0
        )

        profit_factor = positive_returns / (negative_returns + 1e-8)
        avg_trade_return = jnp.where(
            len(non_zero_returns) > 0, jnp.mean(non_zero_returns) * 100, 0.0
        )

        # Calculate average position duration
        position_durations = jnp.where(actions[:-1] != 0, 1, 0)
        avg_position_duration = jnp.mean(position_durations)
        return TradingMetrics(
            total_return=total_return.astype(float),
            sharpe_ratio=sharpe_ratio.astype(float),
            max_drawdown=max_drawdown.astype(float),
            win_rate=win_rate.astype(float),
            profit_factor=profit_factor.astype(float),
            avg_trade_return=avg_trade_return.astype(float),
            positions_held=avg_position_duration.astype(float),
            num_trades=num_trades.astype(int),
        )

    def train(self, num_updates: int):
        obs, state, key = self.env.reset(self.state.key)

        self.transaction_log = []
        timestamp = 0
        running_returns = []
        running_std = jnp.array(1.0)

        def collect_trajectory():
            nonlocal obs, state, key, timestamp
            observations = []
            actions = []
            log_probs = []
            values = []
            rewards = []
            dones = []
            states = []

            for _ in range(self.config.num_steps):
                # Get action from policy
                action, log_prob, value, new_key = self.get_action_and_value(
                    self.state, obs, key
                )
                key = new_key

                # Environment step
                next_obs, next_state, reward, done, new_key = self.env.step(
                    key, state, int(action[0])
                )
                key = new_key

                # Log transaction
                self.log_transaction(timestamp, int(action[0]), next_state)
                timestamp += 1

                # Store trajectory
                observations.append(obs)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                dones.append(done)
                states.append(state)

                obs = next_obs
                state = next_state

                if done:
                    obs, state, key = self.env.reset(key)

            return (
                jnp.array(observations),
                jnp.array(actions),
                jnp.array(log_probs),
                jnp.array(values),
                jnp.array(rewards),
                jnp.array(dones),
                states,
            )

        portfolio_values = [float(self.config.initial_cash)]
        prices = [obs[0]]
        action_history = []

        for update in range(num_updates):
            # Collect trajectory
            observations, actions, log_probs, values, rewards, dones, states = (
                collect_trajectory()
            )

            # Get final value
            _, _, final_value, _ = self.get_action_and_value(self.state, obs, key)

            # Update tracking
            running_returns.extend(rewards)
            if len(running_returns) > 1000:
                running_returns = running_returns[-1000:]
            running_std = jnp.maximum(jnp.std(jnp.array(running_returns)), 1e-8)

            # Update portfolio values and other tracking
            portfolio_values.extend([s.cash + s.shares * s.price for s in states])
            prices.extend([s.price for s in states])
            action_history.extend(actions)

            # Normalize rewards
            normalized_rewards = rewards / running_std

            # Calculate advantages and returns
            advantages = self.compute_gae(
                normalized_rewards,
                jnp.concatenate([values, jnp.array([final_value])]),
                dones,
                self.config.gamma,
                self.config.gae_lambda,
            )
            returns = advantages + values

            # Update policy
            for epoch in range(self.config.update_epochs):
                self.state, metrics = self.train_step(
                    self.state,
                    observations,
                    actions,
                    log_probs,
                    jnp.concatenate([values, jnp.array([final_value])]),
                    normalized_rewards,
                    dones,
                    advantages,
                    returns,
                )

            # Calculate metrics
            trading_metrics = self.calculate_trading_metrics(
                jnp.array(prices),
                jnp.array(action_history),
                jnp.array(portfolio_values),
                self.config.initial_cash,
            )

            # Print progress
            print(f"\nUpdate {update}:")
            print(f"Running reward std: {float(running_std):}")
            print(f"Total Return: {trading_metrics.total_return:}")
            print(f"Number of Trades: {trading_metrics.num_trades}")
            print(f"Policy Loss: {metrics['policy_loss']:}")
            print(f"Value Loss: {metrics['value_loss']:}")
            print(f"Entropy: {metrics['entropy']:}")

            # Save transaction log
            self.save_transaction_log(f"transaction_log_update_{update}.txt")

        return trading_metrics


config = PPOConfig()
agent = PPOAgent(config)

# Train the agent
agent.train(num_updates=100)
