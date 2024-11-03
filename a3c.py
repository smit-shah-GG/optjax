import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from typing import Tuple, List, NamedTuple
import multiprocessing as mp
from multiprocessing import get_context
import numpy as np
from functools import partial
import time
from MoreTechEnv import TradingEnv


# Experience tuple for storing transitions
class Experience(NamedTuple):
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    value: float


# Define the Actor-Critic Network
class ActorCriticNetwork(nn.Module):
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
        actor = nn.softmax(actor)

        # Critic head (value)
        critic = nn.Dense(32)(x)
        critic = nn.relu(critic)
        critic = nn.Dense(1)(critic)

        return actor, critic


class A3CWorker(mp.Process):
    def __init__(
        self,
        worker_id: int,
        param_queue: mp.Queue,
        experience_queue: mp.Queue,
        env_class,
        env_kwargs,
        max_steps: int = 5,
        discount_factor: float = 0.99,
    ):
        # Use 'spawn' method explicitly
        ctx = get_context("spawn")
        super().__init__()
        self.worker_id = worker_id
        self.param_queue = param_queue
        self.experience_queue = experience_queue
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.max_steps = max_steps
        self.discount_factor = discount_factor
        self.running = True

    def run_episode(self, params, rng):
        """Run one episode and collect experience."""
        experiences = []

        # Reset environment
        obs, state, rng = self.env.reset(rng)
        done = False
        episode_reward = 0
        steps = 0

        while not done and steps < self.max_steps:
            # Get action probabilities and value estimate
            action_probs, value = self.network.apply(params, obs[None, ...])
            action_probs = np.array(action_probs[0])
            value = float(value[0, 0])

            # Sample action using numpy
            action = np.random.choice(self.env.num_actions, p=action_probs)

            # Take action in environment
            next_obs, next_state, reward, done, rng = self.env.step(
                rng, state, int(action)
            )

            # Store experience
            experience = Experience(
                state=np.array(obs),
                action=int(action),
                reward=float(reward),
                next_state=np.array(next_obs),
                done=bool(done),
                value=float(value),
            )
            experiences.append(experience)

            # Update for next iteration
            obs = next_obs
            state = next_state
            episode_reward += reward
            steps += 1

        return experiences, episode_reward, rng

    def run(self):
        """Main worker loop."""
        # Initialize environment
        self.env = self.env_class(**self.env_kwargs)

        # Initialize network
        self.network = ActorCriticNetwork(action_dim=self.env.num_actions)

        # Initialize RNG using numpy
        rng = np.random.RandomState(self.worker_id).randint(0, 2**32)
        rng = jax.random.PRNGKey(rng)

        while self.running:
            # Get latest parameters
            try:
                params = self.param_queue.get_nowait()
            except:
                time.sleep(0.1)
                continue

            # Collect experience
            experiences, episode_reward, rng = self.run_episode(params, rng)

            if experiences:
                # Send experience to trainer
                self.experience_queue.put((experiences, episode_reward, self.worker_id))


class A3CTrainer:
    def __init__(
        self,
        env_class,
        env_kwargs,
        num_workers: int = 4,
        learning_rate: float = 0.001,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ):
        # Create dummy environment for initialization
        dummy_env = env_class(**env_kwargs)
        self.action_dim = dummy_env.num_actions
        self.obs_dim = dummy_env.observation_space.shape[0]

        # Initialize network and training state
        self.network = ActorCriticNetwork(action_dim=self.action_dim)
        self.optimizer = optax.adam(learning_rate)

        # Initialize parameters
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((1, self.obs_dim))
        self.params = self.network.init(key, dummy_input)

        self.train_state = TrainState.create(
            apply_fn=self.network.apply, params=self.params, tx=self.optimizer
        )

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        # Initialize multiprocessing queues
        self.param_queue = mp.Queue()
        self.experience_queue = mp.Queue()

        # Initialize workers
        self.workers = []
        for i in range(num_workers):
            worker = A3CWorker(
                worker_id=i,
                param_queue=self.param_queue,
                experience_queue=self.experience_queue,
                env_class=env_class,
                env_kwargs=env_kwargs,
            )
            self.workers.append(worker)

    @partial(jax.jit, static_argnums=(0,))
    def compute_returns_and_advantages(
        self,
        rewards: jnp.ndarray,
        values: jnp.ndarray,
        dones: jnp.ndarray,
        next_value: float,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Compute returns and advantages using GAE."""
        advantages = jnp.zeros_like(rewards)
        returns = jnp.zeros_like(rewards)
        next_advantage = 0.0
        next_return = next_value

        for t in reversed(range(len(rewards))):
            returns = returns.at[t].set(
                rewards[t] + discount_factor * next_return * (1 - dones[t])
            )
            next_return = returns[t]

            td_error = (
                rewards[t] + discount_factor * next_value * (1 - dones[t]) - values[t]
            )

            advantages = advantages.at[t].set(
                td_error
                + discount_factor * gae_lambda * next_advantage * (1 - dones[t])
            )
            next_advantage = advantages[t]

        return returns, advantages

    @partial(jax.jit, static_argnums=(0,))
    def compute_loss(
        self,
        params,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        returns: jnp.ndarray,
        advantages: jnp.ndarray,
    ):
        """Compute actor and critic losses."""
        # Get policy and value predictions
        action_probs, values = self.network.apply(params, states)
        values = values.squeeze()

        # Compute policy loss
        selected_action_probs = action_probs[jnp.arange(len(actions)), actions]
        policy_loss = -jnp.mean(jnp.log(selected_action_probs + 1e-10) * advantages)

        # Compute value loss
        value_loss = 0.5 * jnp.mean((returns - values) ** 2)

        # Compute entropy bonus
        entropy = -jnp.sum(action_probs * jnp.log(action_probs + 1e-10), axis=1)
        entropy_loss = -jnp.mean(entropy)

        # Combine losses
        total_loss = (
            policy_loss
            + self.value_loss_coef * value_loss
            + self.entropy_coef * entropy_loss
        )

        return total_loss, (policy_loss, value_loss, entropy_loss)

    @partial(jax.jit, static_argnums=(0,))
    def train_step(
        self,
        train_state: TrainState,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        returns: jnp.ndarray,
        advantages: jnp.ndarray,
    ):
        """Perform one training step."""
        grad_fn = jax.value_and_grad(self.compute_loss, has_aux=True)
        (loss, loss_info), grads = grad_fn(
            train_state.params, states, actions, returns, advantages
        )

        train_state = train_state.apply_gradients(grads=grads)

        return train_state, loss, loss_info

    def process_experiences(self, experiences):
        """Process a batch of experiences."""
        states = jnp.array([e.state for e in experiences])
        actions = jnp.array([e.action for e in experiences])
        rewards = jnp.array([e.reward for e in experiences])
        next_states = jnp.array([e.next_state for e in experiences])
        dones = jnp.array([e.done for e in experiences])
        values = jnp.array([e.value for e in experiences])

        # Get value estimate for last state
        _, next_value = self.network.apply(
            self.train_state.params, experiences[-1].next_state[None, ...]
        )
        next_value = next_value[0, 0]

        # Calculate returns and advantages
        returns, advantages = self.compute_returns_and_advantages(
            rewards, values, dones, next_value
        )

        return states, actions, returns, advantages

    def train(self, num_updates: int):
        """Main training loop."""
        # Start workers
        for worker in self.workers:
            worker.start()

        # Training metrics
        episode_rewards = []
        losses = []

        # Initial parameter broadcast
        for _ in range(len(self.workers)):
            self.param_queue.put(self.train_state.params)

        update_count = 0
        try:
            while update_count < num_updates:
                # Get experience from workers
                experiences, episode_reward, worker_id = self.experience_queue.get()

                # Process experiences
                states, actions, returns, advantages = self.process_experiences(
                    experiences
                )

                # Perform update
                self.train_state, loss, (policy_loss, value_loss, entropy_loss) = (
                    self.train_step(
                        self.train_state, states, actions, returns, advantages
                    )
                )

                # Send updated parameters to worker
                self.param_queue.put(self.train_state.params)

                # Store metrics
                episode_rewards.append(episode_reward)
                losses.append(float(loss))

                # Log progress
                update_count += 1
                if update_count % 100 == 0:
                    avg_reward = np.mean(episode_rewards[-100:])
                    avg_loss = np.mean(losses[-100:])
                    print(f"Update {update_count}")
                    print(f"Average Reward: {avg_reward:.4f}")
                    print(f"Average Loss: {avg_loss:.4f}")
                    print(f"Policy Loss: {float(policy_loss):.4f}")
                    print(f"Value Loss: {float(value_loss):.4f}")
                    print(f"Entropy Loss: {float(entropy_loss):.4f}")
                    print("--------------------")

        finally:
            # Clean up
            for worker in self.workers:
                worker.running = False
                worker.terminate()
                worker.join()

        return episode_rewards, losses


# Usage example
if __name__ == "__main__":
    # Set multiprocessing start method to 'spawn'
    mp.set_start_method("spawn")

    # Create trainer
    trainer = A3CTrainer(
        env_class=TradingEnv,
        env_kwargs={"token": "BTCUSDT"},
        num_workers=4,
        learning_rate=0.001,
        value_loss_coef=0.5,
        entropy_coef=0.01,
    )

    # Train the agent
    episode_rewards, losses = trainer.train(num_updates=10000)
