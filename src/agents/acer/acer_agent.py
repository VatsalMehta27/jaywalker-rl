from gymnasium import Env
import numpy as np
import torch
from torch import nn
from src.agents.acer.network import ActorCriticNetwork
from src.agents.acer.replay_buffer import TrajectoryBuffer
from src.agents.agent import Agent, TrainingResult


class ACERAgent(Agent):
    def __init__(self, env: Env, params: dict):
        super().__init__(env, params)

        # environment parameters
        self.device = params["device"]
        self.timeout = params["timeout"]
        self.replay_ratio = params["replay_ratio"]
        self.gamma = params["gamma"]
        self.clip = params["clip"]
        self.kl_beta = params["kl_beta"]

        # executable actions
        self.action_space = params["action_space"]

        self.actor_critic = ActorCriticNetwork(
            params["state_dim"], params["action_dim"], params["hidden_dim"]
        )

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=params["learning_rate"]
        )
        # loss
        self.loss = nn.MSELoss()

        self.trajectory_buffer = TrajectoryBuffer(params["replay_buffer_size"])

    def _evaluate_state(self, state: np.ndarray) -> tuple[float, int, float]:
        normalized_state = self.normalize_state(state)
        state_tensor = (
            torch.tensor(normalized_state).float().view(1, -1).to(self.device)
        )

        state_value, action_probs = self.actor_critic(state_tensor)

        return state_value.view(-1), action_probs.view(-1)

    def _rollout(self):
        """Perform one rollout to collect data."""
        trajectory = []

        state, _ = self.env.reset()
        done = False
        state = state["world_grid"]

        while not done:
            # Update normalization with the current state
            self.update_normalization(state)

            _, action_probs = self._evaluate_state(state)
            # print(action_probs)
            action = action_probs.multinomial(1).item()
            next_state, reward, terminated, _, _ = self.env.step(action)
            next_state = next_state["world_grid"]

            truncated = self.env.time_steps >= self.timeout
            done = terminated or truncated

            trajectory.append((state, action, reward, next_state, done, action_probs))

            state = next_state

        self.trajectory_buffer.add(trajectory)

        return trajectory

    def get_action(self, state: np.ndarray) -> int:
        _, action_probs = self._evaluate_state(state)
        action = action_probs.multinomial(1).item()

        return action

    def get_greedy_action(self, state: np.ndarray) -> int:
        _, action_probs = self._evaluate_state(state)

        return torch.argmax(action_probs)

    def load(self, filepath: str) -> None:
        return super().load(filepath)

    def save(self, filepath: str) -> None:
        return super().save(filepath)

    def _acer(self, run_on_policy=True):
        if run_on_policy:
            trajectory = self._rollout()
        else:
            trajectory = self.trajectory_buffer.sample_trajectory()

        q_retrace = torch.tensor(0.0, device=self.device)
        losses = []
        G = torch.tensor(0.0, device=self.device)

        for state, action, reward, _, done, action_probs in reversed(trajectory):
            # Convert data to tensors
            reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
            done = torch.tensor(done, device=self.device, dtype=torch.float32)
            action_probs = torch.tensor(
                action_probs, device=self.device, dtype=torch.float32
            )

            G = reward + self.gamma * G * (1 - done)
            q_retrace = reward + self.gamma * q_retrace * (1 - done)

            cur_state_values, cur_action_probs = self._evaluate_state(state)
            v_value = torch.sum(
                cur_state_values * cur_action_probs
            )  # Avoid using `dot`

            # Compute importance weights
            importance_weights = cur_action_probs / action_probs
            rho_i = torch.clamp(importance_weights[action], max=1.0)

            # Actor loss
            log_prob_action = torch.log(cur_action_probs[action] + 1e-8)  # Avoid log(0)
            actor_loss = rho_i * (q_retrace - v_value) * log_prob_action

            # Bias correction
            bias_correction = (
                torch.clamp(1 - self.clip / (importance_weights + 1e-8), min=0.0)
                * (cur_state_values - v_value.detach())
                * action_probs
                * torch.log(action_probs + 1e-8)
            )

            # Critic loss
            critic_loss = self.loss(
                cur_state_values,  # .gather(0, torch.tensor([action])),
                q_retrace.unsqueeze(0),
            )

            # KL divergence
            kl_divergence = torch.sum(
                cur_action_probs
                * (torch.log(cur_action_probs + 1e-8) - torch.log(action_probs + 1e-8))
            )

            # Total loss
            total_loss = (
                actor_loss
                + bias_correction.sum()
                + critic_loss
                + self.kl_beta * kl_divergence
            )

            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            losses.append(total_loss.detach().item())

        return G.item(), losses, len(trajectory)

    def train(self, episodes) -> TrainingResult:
        all_returns = []
        all_losses = []
        trajectory_lengths = []

        for _ in range(episodes):
            G, losses, trajectory_length = self._acer(run_on_policy=True)
            all_returns.append(G)
            all_losses.append(np.mean(losses))
            trajectory_lengths.append(trajectory_length)

            n = np.random.poisson(self.replay_ratio)

            for _ in range(n):
                G, losses, trajectory_length = self._acer(run_on_policy=False)
                all_returns.append(G)
                all_losses.append(np.mean(losses))

        return TrainingResult(
            returns=np.array(all_returns),
            timesteps=np.array(trajectory_lengths),
            loss=np.array(all_losses),
        )
