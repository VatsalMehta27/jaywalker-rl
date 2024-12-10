from gymnasium import Env
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm
from src.agents.acer.network import ActorCriticNetwork
from src.agents.acer.replay_buffer import TrajectoryBuffer
from src.agents.agent import Agent, TrainingResult


class ACERAgent(Agent):
    def __init__(self, env: Env, params: dict):
        super().__init__(env, params)

        # Environment and training parameters
        self.device = params["device"]
        self.timeout = params["timeout"]
        self.replay_ratio = params["replay_ratio"]
        self.gamma = params["gamma"]
        self.clip = params["clip"]
        self.kl_beta = params["kl_beta"]
        self.early_stop = params["early_stop"]

        # Network and optimizer
        self.actor_critic = ActorCriticNetwork(
            params["state_dim"], params["action_dim"], params["hidden_dim"]
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=params["learning_rate"]
        )
        self.loss_fn = nn.MSELoss()

        self.trajectory_buffer = TrajectoryBuffer(params["replay_buffer_size"])

    def _evaluate_state(self, state: dict):
        """Evaluate the state to get value and action probabilities."""
        state_tensor = (
            torch.tensor(self.transform_state(state), dtype=torch.float32)
            .view(1, -1)
            .to(self.device)
        )
        state_value, action_probs = self.actor_critic(state_tensor)

        return state_value.squeeze(), action_probs.squeeze()

    def _rollout(self):
        """Perform one episode rollout and collect data."""
        trajectory = []
        state, _ = self.env.reset()
        done = False

        while not done:
            _, action_probs = self._evaluate_state(state)
            action = Categorical(probs=action_probs).sample().item()

            next_state, reward, terminated, _, _ = self.env.step(action)
            done = terminated or (self.env.time_steps >= self.timeout)

            trajectory.append((state, action, reward, next_state, done, action_probs))
            state = next_state

        self.trajectory_buffer.add(trajectory)

        return trajectory

    def _process_trajectory(self, trajectory):
        """Process a trajectory for training."""
        q_retrace = torch.tensor(0.0, device=self.device)
        losses = []
        G = torch.tensor(0.0, device=self.device)

        for state, action, reward, _, done, old_action_probs in reversed(trajectory):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)
            done = torch.tensor(done, dtype=torch.float32, device=self.device)

            G = reward + self.gamma * G * (1 - done)
            q_retrace = reward + self.gamma * q_retrace * (1 - done)

            cur_state_values, cur_action_probs = self._evaluate_state(state)
            v_value = torch.sum(cur_action_probs * cur_state_values)

            # Importance sampling weights
            importance_weights = cur_action_probs / (old_action_probs.detach() + 1e-8)
            rho_i = torch.clamp(importance_weights[action], max=self.clip)

            # Actor loss
            log_prob = torch.log(cur_action_probs[action] + 1e-8)
            actor_loss = rho_i * (q_retrace - v_value.detach()) * log_prob

            # Bias correction
            correction = (
                (1 - self.clip / (importance_weights + 1e-8)).clamp(min=0.0)
                * cur_state_values.detach()
                * old_action_probs.detach()
                * torch.log(old_action_probs.detach() + 1e-8)
            ).sum()

            # Critic loss
            q_value = cur_state_values[action]
            critic_loss = self.loss_fn(q_value, q_retrace)

            # KL divergence
            kl_divergence = torch.sum(
                cur_action_probs
                * (
                    torch.log(cur_action_probs + 1e-8)
                    - torch.log(old_action_probs.detach() + 1e-8)
                )
            )

            # Total loss
            total_loss = (
                actor_loss + critic_loss + correction + self.kl_beta * kl_divergence
            )

            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward(retain_graph=False)  # Ensure no graph retention
            self.optimizer.step()

            losses.append(total_loss.item())

        return G.item(), np.mean(losses)

    def train(self, episodes) -> TrainingResult:
        all_returns = []
        all_losses = []
        trajectory_lengths = []

        past_return = None
        same_return = 0

        with tqdm(range(episodes), desc="Training", unit="episode") as pbar:
            for _ in pbar:
                # On-policy update
                trajectory = self._rollout()
                G, avg_loss = self._process_trajectory(trajectory)

                all_returns.append(G)
                all_losses.append(avg_loss)
                trajectory_lengths.append(len(trajectory))

                # Replay updates
                for _ in range(np.random.poisson(self.replay_ratio)):
                    trajectory = self.trajectory_buffer.sample_trajectory()
                    G, avg_loss = self._process_trajectory(trajectory)

                    all_returns.append(G)
                    all_losses.append(avg_loss)

                average_return = np.mean(all_returns[-10:])

                if average_return == past_return:
                    same_return += 1

                    if same_return >= self.early_stop:
                        break
                else:
                    same_return = 0

                past_return = average_return

                pbar.set_postfix(average_return=average_return)

        return TrainingResult(
            returns=np.array(all_returns),
            timesteps=np.array(trajectory_lengths),
            loss=np.array(all_losses),
        )

    def get_action_probs(self, state: dict):
        _, action_probs = self._evaluate_state(state)

        return action_probs

    def get_action(self, state: dict) -> int:
        _, action_probs = self._evaluate_state(state)
        action = Categorical(probs=action_probs).sample().item()

        return action

    def get_greedy_action(self, state: dict) -> int:
        _, action_probs = self._evaluate_state(state)
        action = torch.argmax(action_probs).item()

        return action

    def save(self, file_path: str):
        """
        Save the actor-critic network and optimizer to the specified file path.

        Args:
            file_path (str): Path to save the model and optimizer.
        """
        torch.save(
            {
                "actor_critic_state_dict": self.actor_critic.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            file_path,
        )
        print(f"Model and optimizer successfully saved to {file_path}")

    def load(self, file_path: str):
        """
        Load the actor-critic network and optimizer from the specified file path.

        Args:
            file_path (str): Path from which to load the model and optimizer.
        """
        checkpoint = torch.load(file_path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.actor_critic.to(self.device)
        print(f"Model and optimizer successfully loaded from {file_path}")
