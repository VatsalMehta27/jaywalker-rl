from gymnasium import Env
import numpy as np
import torch
import tqdm
from src.agents.agent import TrainingResult
from src.agents.reinforce.network import REINFORCEBaselineNetwork
from src.agents.agent import Agent
from src.utils.utils import customized_weights_init
from torch.distributions import Categorical


class REINFORCEAgent(Agent):
    def __init__(self, env: Env, params: dict):
        super().__init__(env, params)

        self.action_dim = params["action_dim"]
        self.state_dim = params["state_dim"]
        self.action_space = params["action_space"]

        self.policy = REINFORCEBaselineNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=params["hidden_dim"],
        )
        self.policy.apply(customized_weights_init)

        self.device = torch.device(params["device"])
        self.policy.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=params["learning_rate"]
        )

        self.gamma = params["gamma"]
        self.timeout = params["timeout"]

        # Running mean and variance for normalization
        self.state_mean = np.zeros(self.state_dim)
        self.state_var = np.ones(self.state_dim)
        self.state_count = 0  # Number of states observed

    def normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Normalize the state using running mean and variance."""
        state_flat = self.transform_state(state)  # Flatten the state to 1D
        return (state_flat - self.state_mean) / (np.sqrt(self.state_var) + 1e-8)

    def update_normalization(self, state: np.ndarray):
        """Update running mean and variance using Welford's method."""
        state_flat = self.transform_state(state)  # Flatten the state to 1D
        self.state_count += 1
        delta = state_flat - self.state_mean
        self.state_mean += delta / self.state_count
        delta2 = state_flat - self.state_mean
        self.state_var += delta * delta2

    def get_action(self, state: np.ndarray) -> int:
        """Get an action using the policy."""
        _, action, _ = self._evaluate_state(state)

        return action

    def get_greedy_action(self, state: np.ndarray) -> int:
        """Get a greedy action (highest probability) from the policy."""
        normalized_state = self.normalize_state(state)
        state_tensor = torch.tensor(normalized_state).float().view(1, -1)
        _, action_probs = self.policy(state_tensor)
        return self.action_space[torch.argmax(action_probs).item()]

    def _evaluate_state(self, state: np.ndarray) -> tuple[float, int, float]:
        """Evaluate a state and return the state value, action, and log probability."""
        normalized_state = self.normalize_state(state)
        state_tensor = (
            torch.tensor(normalized_state).float().view(1, -1).to(self.device)
        )

        state_value, action_probs = self.policy(state_tensor)

        distribution = Categorical(action_probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return state_value, action.item(), log_prob

    def _rollout(self):
        """Perform one rollout to collect data."""
        state_value_list, log_probs_list, rewards_list, returns_list = ([], [], [], [])

        state, _ = self.env.reset()
        done = False

        while not done:
            # Update normalization with the current state
            self.update_normalization(state)

            state_value, action, log_prob = self._evaluate_state(state)
            next_state, reward, terminated, _, _ = self.env.step(action)

            truncated = self.env.time_steps >= self.timeout
            done = terminated or truncated

            state = next_state

            state_value_list.append(state_value)
            log_probs_list.append(log_prob)
            rewards_list.append(reward)

        returns_list = self._discount_rewards(rewards_list)

        batch = (
            state_value_list,
            log_probs_list,
            returns_list,
        )

        return batch

    def _discount_rewards(self, rewards):
        """Discount rewards to compute returns."""
        discounted = []
        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted.insert(0, G)

        return discounted

    def _update(self, batch_data):
        """Update the policy and value network."""
        policy_loss = []
        value_loss = []

        state_values, log_probs, returns = batch_data

        for state_val, log_prob, r in zip(state_values, log_probs, returns):
            delta = r - state_val
            value_loss.append(-(self.gamma * delta * state_val))
            policy_loss.append(-(self.gamma * delta * log_prob))

        total_loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return batch_data[-1][0], total_loss.item()

    def train(self, episodes) -> TrainingResult:
        """Train the agent for a specified number of episodes."""
        train_returns = []
        train_losses = []
        train_timesteps = []

        # Training loop
        ep_bar = tqdm.trange(episodes)
        for ep in ep_bar:
            # Collect one episode
            batch = self._rollout()

            # Update the policy using the collected episode
            G, loss = self._update(batch)

            # Save the return and loss
            train_returns.append(G)
            train_losses.append(loss)
            train_timesteps.append(self.env.time_steps)

            # Add description
            ep_bar.set_description(f"Episode: {ep} | Return: {G} | Loss: {loss:.2f}")

        return TrainingResult(
            returns=np.array(train_returns),
            timesteps=np.array(train_timesteps),
            loss=np.array(train_losses),
        )

    def save(self, filepath: str) -> None:
        """
        Save the policy network's state_dict and optimizer state.
        """
        torch.save(
            {
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "state_mean": self.state_mean,
                "state_var": self.state_var,
                "state_count": self.state_count,
            },
            filepath,
        )
        print(f"Model and optimizer states saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load the policy network's state_dict and optimizer state.
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state_mean = checkpoint.get("state_mean", self.state_mean)
        self.state_var = checkpoint.get("state_var", self.state_var)
        self.state_count = checkpoint.get("state_count", self.state_count)

        self.policy.to(self.device)
        print(f"Model and optimizer states loaded from {filepath}")
