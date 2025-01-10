
import time
from typing import NamedTuple
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training.train_state import TrainState
from datetime import datetime
import pickle
from functools import partial

class TransitionBatch(NamedTuple):
    observations: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    next_observations: jnp.ndarray
    values: jnp.ndarray

class ActorCritic(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, x):
        # Shared layers
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)

        # Actor head (policy)
        actor = nn.Dense(32)(x)
        actor = nn.relu(actor)
        actor = nn.Dense(self.action_dim)(actor)
        action_probs = nn.softmax(actor)

        # Critic head (value)
        critic = nn.Dense(32)(x)
        critic = nn.relu(critic)
        value = nn.Dense(1)(critic)

        return action_probs, value

class A2CAgent:
    def __init__(self, observation_dim, action_dim, learning_rate=0.001, gamma=0.99,
                 entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        # Initialize network and optimizer
        self.network = ActorCritic(action_dim=action_dim)
        self.key = jax.random.PRNGKey(0)

        # Initialize parameters
        dummy_obs = jnp.zeros((1, observation_dim))
        params = self.network.init(self.key, dummy_obs)

        # Create train state
        tx = optax.adam(learning_rate)
        self.state = TrainState.create(
            apply_fn=self.network.apply,
            params=params,
            tx=tx,
        )

        # Create directories for logging and saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = f"logs/a2c_{timestamp}"
        self.model_dir = f"models/a2c_{timestamp}"
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)

        self.best_reward = float('-inf')

    @partial(jax.jit, static_argnums=(0,))
    def get_action_and_value(self, params, observation):
        action_probs, value = self.network.apply(params, observation)
        action = jax.random.categorical(self.key, jnp.log(action_probs))
        return action, action_probs, value.squeeze()

    @partial(jax.jit, static_argnums=(0,))
    def compute_loss(self, params, batch: TransitionBatch):
        # Get predictions
        action_probs, values = self.network.apply(params, batch.observations)
        _, next_values = self.network.apply(params, batch.next_observations)

        # Compute returns and advantages
        next_values = next_values.squeeze()
        values = values.squeeze()
        returns = batch.rewards + self.gamma * next_values * (1 - batch.dones)
        advantages = returns - values

        # Policy loss
        log_probs = jnp.log(action_probs + 1e-8)
        actions_one_hot = jax.nn.one_hot(batch.actions, action_probs.shape[-1])
        policy_loss = -jnp.mean(jnp.sum(log_probs * actions_one_hot, axis=-1) * advantages)

        # Value loss
        value_loss = jnp.mean(jnp.square(returns - values))

        # Entropy loss for exploration
        entropy = -jnp.mean(jnp.sum(action_probs * log_probs, axis=-1))

        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        return total_loss, (policy_loss, value_loss, entropy)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(self, state, batch: TransitionBatch):
        grad_fn = jax.value_and_grad(self.compute_loss, has_aux=True)
        (loss, (policy_loss, value_loss, entropy)), grads = grad_fn(state.params, batch)

        # Clip gradients
        grads = jax.tree_map(lambda x: jnp.clip(x, -self.max_grad_norm, self.max_grad_norm), grads)

        # Update parameters
        state = state.apply_gradients(grads=grads)

        return state, loss, policy_loss, value_loss, entropy

    def train(self, env, num_episodes=1000, batch_size=32):
        log_file = open(os.path.join(self.log_dir, "training_log.txt"), "w")

        for episode in range(num_episodes):
            observations = []
            actions = []
            rewards = []
            dones = []
            values = []

            obs, state, key = env.reset(self.key)
            episode_reward = 0

            while True:
                action, action_probs, value = self.get_action_and_value(self.state.params, obs)
                next_obs, next_state, reward, done, key = env.step(key, state, action)

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                values.append(value)

                episode_reward += reward

                if done:
                    break

                obs = next_obs
                state = next_state

            # Convert to batches and train
            if len(observations) >= batch_size:
                observations = jnp.array(observations)
                actions = jnp.array(actions)
                rewards = jnp.array(rewards)
                dones = jnp.array(dones)
                values = jnp.array(values)
                next_observations = jnp.roll(observations, -1, axis=0)

                batch = TransitionBatch(
                    observations=observations,
                    actions=actions,
                    rewards=rewards,
                    dones=dones,
                    next_observations=next_observations,
                    values=values
                )

                self.state, loss, policy_loss, value_loss, entropy = self.train_step(
                    self.state, batch)

                # Logging
                log_msg = (f"Episode {episode}, Reward: {episode_reward:.2f}, "
                          f"Loss: {loss:.4f}, Policy Loss: {policy_loss:.4f}, "
                          f"Value Loss: {value_loss:.4f}, Entropy: {entropy:.4f}")
                print(log_msg)
                log_file.write(log_msg + "\n")
                log_file.flush()

                # Save best model
                if episode_reward > self.best_reward:
                    self.best_reward = episode_reward
                    self.save_model(os.path.join(self.model_dir, "best_model.pkl"))

        log_file.close()

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.state, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.state = pickle.load(f)

from a2cenv import TradingEnv

# Initialize environment and agent
env = TradingEnv(token="BTCUSDT")
observation_dim = env.observation_space.shape[0]
action_dim = env.num_actions

# Create and train agent
agent = A2CAgent(
    observation_dim=observation_dim,
    action_dim=action_dim,
    learning_rate=0.001,
    gamma=0.99,
    entropy_coef=0.01,
    value_coef=0.5,
    max_grad_norm=0.5
)

# Train the agent
agent.train(env, num_episodes=1000, batch_size=32)
