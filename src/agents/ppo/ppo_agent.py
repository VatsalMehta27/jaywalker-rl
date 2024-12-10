from gymnasium import Env
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm
from src.agents.ppo.network import NeuralNetwork
from src.agents.agent import Agent, TrainingResult
import numpy as np


class PPOAgent(Agent):
    def __init__(self, env: Env, params: dict):
        super().__init__(env, params)

        self.action_space = params["action_space"]

        self.actor = NeuralNetwork(
            state_dim=self.state_dim,
            output_dim=self.action_dim,
            num_layers=params["num_layers"],
            hidden_dim=params["hidden_dim"],
        )
        self.critic = NeuralNetwork(
            state_dim=self.state_dim,
            output_dim=1,
            num_layers=params["num_layers"],
            hidden_dim=params["hidden_dim"],
        )

        self.device = torch.device(params["device"])
        self.actor.to(self.device)
        self.critic.to(self.device)

        # optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=params["learning_rate"]
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=params["learning_rate"]
        )

        # loss
        self.loss = nn.MSELoss()

        self.clip = params["clip"]
        self.batch_size = params["batch_size"]
        self.gamma = params["gamma"]
        self.epochs_per_iteration = params["epochs_per_iteration"]
        self.timeout = params["timeout"]

    def get_action(self, state):
        action, _ = self._get_action_prob(state)

        return action

    def _get_action_prob(self, state):
        state_tensor = torch.tensor(self.transform_state(state)).float().view(1, -1)

        action_logits = self.actor(state_tensor)

        distribution = Categorical(logits=action_logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.detach().item(), log_prob.detach()

    def get_greedy_action(self, state):
        state_tensor = torch.tensor(self.transform_state(state)).float().view(1, -1)

        action_logits = self.actor(state_tensor)

        return torch.argmax(action_logits).item()

    def _get_values(self, states: torch.tensor):
        V_values = self.critic(states)

        return V_values

    def _get_batch_log_prob(self, states: torch.tensor, actions: torch.tensor):
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions.squeeze(dim=1))

        return log_probs.unsqueeze(dim=1)

    def _calculate_returns(self, rewards: list[list[float]]) -> list[float]:
        discounted = []

        for episode in reversed(rewards):
            G = 0

            for step_reward in reversed(episode):
                G = step_reward + self.gamma * G
                discounted.insert(0, G)

        return discounted

    def _rollout(self) -> dict:
        state_list, actions_list, log_probs_list, rewards_list, timesteps_list = (
            [],
            [],
            [],
            [],
            [],
        )

        state, _ = self.env.reset()
        done = False
        episode_rewards = []

        for _ in range(self.batch_size):
            state_list.append(self.transform_state(state))

            action, log_prob = self._get_action_prob(state)
            state, reward, terminated, _, _ = self.env.step(action)
            truncated = self.env.time_steps >= self.timeout

            done = terminated or truncated

            episode_rewards.append(reward)
            actions_list.append(action)
            log_probs_list.append(log_prob)

            if done:
                rewards_list.append(episode_rewards)
                episode_rewards = []

                timesteps_list.append(self.env.time_steps)

                state, _ = self.env.reset()

        if episode_rewards:
            rewards_list.append(episode_rewards)
            episode_rewards = []

            timesteps_list.append(self.env.time_steps)

        batch_tensor = {
            "states": torch.tensor(np.array(state_list), dtype=torch.float32).to(
                self.device
            ),
            "actions": torch.tensor(actions_list).long().view(-1, 1).to(self.device),
            "log_probs": torch.tensor(log_probs_list, dtype=torch.float32)
            .view(-1, 1)
            .to(self.device),
            "returns": torch.tensor(
                self._calculate_returns(rewards_list), dtype=torch.float32
            )
            .view(-1, 1)
            .to(self.device),
            "timesteps": torch.tensor(timesteps_list, dtype=torch.long)
            .view(-1, 1)
            .to(self.device),
        }

        return batch_tensor

    def train(self, training_steps):
        train_returns = []
        train_losses = []
        train_timesteps = []

        for _ in tqdm(
            range(0, training_steps, self.batch_size), desc="Training Progress"
        ):
            batch = self._rollout()

            V_values = self._get_values(batch["states"])
            advantage = batch["returns"] - V_values.detach()
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            batch_losses = []

            for _ in range(self.epochs_per_iteration):
                V_values = self._get_values(batch["states"])
                current_log_probs = self._get_batch_log_prob(
                    batch["states"], batch["actions"]
                )

                ratios = torch.exp(current_log_probs - batch["log_probs"])

                clipped_loss = torch.min(
                    ratios * advantage,
                    torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantage,
                )

                actor_loss = (-clipped_loss).mean()
                critic_loss = self.loss(V_values, batch["returns"])

                batch_losses.append((actor_loss + critic_loss).detach().item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            train_returns.extend(batch["returns"].flatten().tolist())
            train_losses.append(np.mean(batch_losses))
            train_timesteps.extend(batch["timesteps"].flatten().tolist())

        return TrainingResult(
            returns=np.array(train_returns),
            timesteps=np.array(train_timesteps),
            loss=np.array(train_losses),
        )

    def load(self, filepath):
        """
        Loads the actor and critic models along with their optimizers.

        Parameters:
            filepath (str): Base path where the models and optimizers are saved.
        """
        # Load actor model and optimizer
        actor_checkpoint = torch.load(f"{filepath}_actor.pth", map_location=self.device)
        self.actor.load_state_dict(actor_checkpoint["model_state_dict"])
        self.actor_optimizer.load_state_dict(actor_checkpoint["optimizer_state_dict"])
        self.actor.to(self.device)

        # Load critic model and optimizer
        critic_checkpoint = torch.load(
            f"{filepath}_critic.pth", map_location=self.device
        )
        self.critic.load_state_dict(critic_checkpoint["model_state_dict"])
        self.critic_optimizer.load_state_dict(critic_checkpoint["optimizer_state_dict"])
        self.critic.to(self.device)

    def save(self, filepath):
        """
        Saves the actor and critic models along with their optimizers.

        Parameters:
            filepath (str): Base path where the models and optimizers will be saved.
        """
        # Save actor model and optimizer
        torch.save(
            {
                "model_state_dict": self.actor.state_dict(),
                "optimizer_state_dict": self.actor_optimizer.state_dict(),
            },
            f"{filepath}_actor.pth",
        )

        # Save critic model and optimizer
        torch.save(
            {
                "model_state_dict": self.critic.state_dict(),
                "optimizer_state_dict": self.critic_optimizer.state_dict(),
            },
            f"{filepath}_critic.pth",
        )
