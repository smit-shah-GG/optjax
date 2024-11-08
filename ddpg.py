import random
from collections import deque
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn
from flax.training.train_state import TrainState

from MoreTechEnv import TradingEnv


class Actor(nn.Module):
    """Actor network that maps states to actions."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)  # Output layer with 3 actions (buy, sell, hold)
        return nn.softmax(x)  # Use softmax for discrete actions


class Critic(nn.Module):
    """Critic network that maps state-action pairs to Q-values."""

    @nn.compact
    def __call__(self, state, action):
        # Convert action to one-hot if it's a scalar
        if len(action.shape) == 1:
            action = jax.nn.one_hot(action, 3)  # 3 is the number of actions

        # Concatenate state and action
        x = jnp.concatenate([state, action], axis=-1)

        # Hidden layers
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)  # Output single Q-value
        return x


# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self) -> int:
        return len(self.buffer)


# DDPG Agent
class DDPGAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 64,
        buffer_size: int = 100000,
    ):
        # Initialize parameters
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Create networks
        self.actor = Actor()
        self.critic = Critic()

        # Initialize network parameters
        self.key = jax.random.PRNGKey(0)
        dummy_state = jnp.ones((1, state_dim))
        dummy_action = jnp.zeros((1,), dtype=jnp.int32)

        # Initialize actor
        self.key, actor_key = jax.random.split(self.key)
        actor_params = self.actor.init(actor_key, dummy_state)
        self.actor_state = TrainState.create(apply_fn=self.actor.apply, params=actor_params, tx=optax.adam(learning_rate))
        self.target_actor_params = actor_params

        # Initialize critic
        self.key, critic_key = jax.random.split(self.key)
        critic_params = self.critic.init(critic_key, dummy_state, dummy_action)
        self.critic_state = TrainState.create(
            apply_fn=self.critic.apply,
            params=critic_params,
            tx=optax.adam(learning_rate),
        )
        self.target_critic_params = critic_params

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # JIT compile update functions
        self.update_critic = jax.jit(self._update_critic)
        self.update_actor = jax.jit(self._update_actor)
        self.update_target = jax.jit(self._update_target)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select action using the current policy."""
        state = jnp.expand_dims(state, 0)
        action_probs = self.actor.apply(self.actor_state.params, state)
        return jnp.argmax(action_probs[0])  # Returns scalar action index

    def _update_critic(
        self,
        critic_state: TrainState,
        actor_params: Dict,
        target_actor_params: Dict,
        target_critic_params: Dict,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> Tuple[TrainState, jnp.ndarray]:
        """Update critic network."""

        # Calculate target Q-values
        next_actions = self.actor.apply(target_actor_params, next_states)
        next_q = self.critic.apply(target_critic_params, next_states, next_actions)
        target_q = rewards + self.gamma * next_q * (1 - dones)

        # Update critic
        def critic_loss_fn(critic_params):
            q = self.critic.apply(critic_params, states, actions)
            loss = jnp.mean((q - target_q) ** 2)
            return loss

        grad_fn = jax.value_and_grad(critic_loss_fn)
        loss, grads = grad_fn(critic_state.params)
        critic_state = critic_state.apply_gradients(grads=grads)

        return critic_state, loss

    def _update_actor(
        self,
        actor_state: TrainState,
        critic_params: Dict,
        states: np.ndarray,
    ) -> Tuple[TrainState, jnp.ndarray]:
        """Update actor network."""

        def actor_loss_fn(actor_params):
            actions = self.actor.apply(actor_params, states)
            q = self.critic.apply(critic_params, states, actions)
            loss = -jnp.mean(q)
            return loss

        grad_fn = jax.value_and_grad(actor_loss_fn)
        loss, grads = grad_fn(actor_state.params)
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, loss

    def _update_target(
        self,
        target_params: Dict,
        params: Dict,
    ) -> Dict:
        """Soft update target networks."""
        return jax.tree.map(
            lambda target, source: (1 - self.tau) * target + self.tau * source,
            target_params,
            params,
        )

    def train(self) -> Tuple[float, float]:
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0, 0.0

        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # Convert to jax arrays
        states = jnp.array(states)
        actions = jnp.array(actions)
        rewards = jnp.array(rewards)
        next_states = jnp.array(next_states)
        dones = jnp.array(dones)

        # Update critic
        self.critic_state, critic_loss = self.update_critic(
            self.critic_state,
            self.actor_state.params,
            self.target_actor_params,
            self.target_critic_params,
            states,
            actions,
            rewards,
            next_states,
            dones,
        )

        # Update actor
        self.actor_state, actor_loss = self.update_actor(
            self.actor_state,
            self.critic_state.params,
            states,
        )

        # Update target networks
        self.target_actor_params = self.update_target(
            self.target_actor_params,
            self.actor_state.params,
        )
        self.target_critic_params = self.update_target(
            self.target_critic_params,
            self.critic_state.params,
        )

        return float(critic_loss), float(actor_loss)


# Training loop function
def train_ddpg(
    env,
    agent: DDPGAgent,
    num_episodes: int,
    max_steps: int = 1000,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
):
    """Train the DDPG agent."""
    rewards_history = []
    epsilon = epsilon_start

    for episode in range(num_episodes):
        key = jax.random.PRNGKey(episode)
        obs, state, key = env.reset(key)
        episode_reward = 0
        critic_losses = []
        actor_losses = []

        for step in range(max_steps):
            # Select action
            if random.random() < epsilon:
                action = random.randint(0, env.num_actions - 1)
            else:
                action = agent.select_action(obs)

            # Take action in environment
            next_obs, next_state, reward, done, key = env.step(key, state, action)

            # Store transition in replay buffer
            agent.replay_buffer.push(obs, action, reward, next_obs, done)

            # Train agent
            if len(agent.replay_buffer) >= agent.batch_size:
                critic_loss, actor_loss = agent.train()
                critic_losses.append(critic_loss)
                actor_losses.append(actor_loss)

            episode_reward += reward
            obs = next_obs
            state = next_state

            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Log progress
        rewards_history.append(episode_reward)
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
        avg_actor_loss = np.mean(actor_losses) if actor_losses else 0

        if episode % 10 == 0:
            print(f"Episode {episode}")
            print(f"Average Reward: {np.mean(rewards_history[-10:]):.2f}")
            print(f"Epsilon: {epsilon:.3f}")
            print(f"Average Critic Loss: {avg_critic_loss:.3f}")
            print(f"Average Actor Loss: {avg_actor_loss:.3f}")
            print("-" * 50)

    return rewards_history


env = TradingEnv(token="BTCUSDT", window_size=30)
state_dim = env.observation_space.shape[0]  # Size of observation space
action_dim = env.num_actions  # Number of possible actions (3 in your case)

agent = DDPGAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=1e-3,
    gamma=0.99,
    tau=0.005,
    batch_size=64,
    buffer_size=100000,
)
rewards_history = train_ddpg(
    env=env,
    agent=agent,
    num_episodes=1000,
    max_steps=1000,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=0.995,
)
