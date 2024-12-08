from gymnasium import Env
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm import tqdm
from src.agents.ppo.network import NeuralNetwork
from src.agents.agent import Agent, TrainingResult
import numpy as np

from src.utils.utils import customized_weights_init


class PPOAgent(Agent):
    def __init__(self, env: Env, params: dict):
        super().__init__(env, params)

        # executable actions
        self.action_space = params["action_space"]

        self.actor_critic = NeuralNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_layers=params["num_layers"],
            hidden_dim=params["hidden_dim"],
        )
        print(self.actor_critic)
        self.actor_critic.apply(customized_weights_init)

        self.device = torch.device(params["device"])
        self.actor_critic.to(self.device)

        # optimizer
        self.actor_critic_optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=params["learning_rate"]
        )

        # loss
        self.loss = nn.MSELoss()

        self.clip = params["clip"]
        self.batch_size = params["batch_size"]
        self.gamma = params["gamma"]
        self.epochs_per_iteration = params["epochs_per_iteration"]
        self.entropy_weight = params["entropy_weight"]

    def get_action(self, state: np.ndarray) -> int:
        action, _ = self._get_action_prob(state)

        return action

    def _get_action_prob(self, state: np.ndarray) -> tuple[int, float, float]:
        state_tensor = torch.tensor(state).float().view(1, -1)

        _, action_probs = self.actor_critic(state_tensor)

        distribution = Categorical(probs=action_probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        entropy = distribution.entropy()

        return self.action_space[action.item()], log_prob.detach(), entropy.detach()

    def get_greedy_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state).float().view(1, -1)

        _, action_probs = self.actor_critic(state_tensor)

        return self.action_space[torch.argmax(action_probs.detach()).item()]

    def _get_batch_log_prob(self, states: torch.tensor, actions: torch.tensor):
        _, action_probs = self.actor_critic(states)

        distribution = Categorical(probs=action_probs)
        log_probs = distribution.log_prob(actions.squeeze(dim=1))
        entropies = distribution.entropy()

        return log_probs.unsqueeze(dim=1), entropies

    def train(self, training_steps) -> TrainingResult:
        train_returns = []
        train_actor_losses = []
        train_critic_losses = []

        for _ in tqdm(
            range(0, training_steps, self.batch_size), desc="Training Progress"
        ):
            batch_data_tensor = self._rollout(self.batch_size)

            # get the transition data
            states_tensor = batch_data_tensor["state"]
            actions_tensor = batch_data_tensor["action"]
            log_probs_tensor = batch_data_tensor["log_prob"]
            # entropies_tensor = batch_data_tensor["entropy"]
            returns_tensor = batch_data_tensor["return"]

            # print(states_tensor.shape)
            # print(actions_tensor.shape)
            # print(log_probs_tensor.shape)
            # print(entropies_tensor.shape)
            # print(returns_tensor.shape)

            train_returns.extend(returns_tensor.numpy().flatten())

            with torch.no_grad():
                V_values, _ = self.actor_critic(states_tensor)
                advantage = returns_tensor - V_values.detach()
                # target_val = returns_tensor + gamma * next_state_values

                normalized_advantage = (advantage - advantage.mean()) / (
                    advantage.std() + 1e-8
                )

            for _ in range(self.epochs_per_iteration):
                V_values, _ = self.actor_critic(states_tensor)

                current_log_probs, _ = self._get_batch_log_prob(
                    states_tensor, actions_tensor
                )

                ratios = torch.exp(current_log_probs - log_probs_tensor)
                clipped_loss = torch.min(
                    ratios * normalized_advantage,
                    torch.clamp(ratios, 1 - self.clip, 1 + self.clip)
                    * normalized_advantage,
                )

                actor_loss = (-clipped_loss).mean()
                critic_loss = self.loss(returns_tensor, V_values)

                total_loss = (
                    actor_loss + 0.5 * critic_loss
                    # + self.entropy_weight * -torch.mean(current_entropies)
                )

                self.actor_critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(), max_norm=0.5
                )
                self.actor_critic_optimizer.step()

                train_actor_losses.append(actor_loss.detach().item())
                train_critic_losses.append(critic_loss.detach().item())

        return TrainingResult(
            returns=np.array(train_returns),
            timesteps=np.array(train_actor_losses),
            loss=np.array(train_critic_losses),
        )

    def _rollout(self, num_steps):
        state_list, actions_list, log_probs_list, entropies_list, returns_list = (
            [],
            [],
            [],
            [],
            [],
        )

        state, _ = self.env.reset()
        state = state["world_grid"]

        episode_rewards = []

        for _ in range(num_steps):
            action, log_prob, entropy = self._get_action_prob(state)
            next_state, reward, done, _, _ = self.env.step(action)
            next_state = next_state["world_grid"]

            state_list.append(np.asarray(state))
            actions_list.append(np.asarray(action))
            log_probs_list.append(np.asarray(log_prob))
            entropies_list.append(np.asarray(entropy))
            episode_rewards.append(reward)

            if done:
                state, _ = self.env.reset()
                state = state["world_grid"]

                returns_list.extend(
                    self._discount_rewards(episode_rewards, terminal=True)
                )
                episode_rewards = []
            else:
                state = next_state

        if episode_rewards:
            returns_list.extend(self._discount_rewards(episode_rewards, terminal=False))

        batch = (
            np.array(state_list),
            np.array(actions_list),
            np.array(log_probs_list),
            np.array(entropies_list),
            np.array(returns_list),
        )

        return self._batch_to_tensor(batch)

    def _discount_rewards(self, rewards):
        discounted = []
        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            discounted.insert(0, G)

        return discounted

    # auxiliary functions
    def _arr_to_tensor(self, arr):
        arr = np.array(arr)
        arr_tensor = torch.from_numpy(arr).float().to(self.device)
        return arr_tensor

    def _batch_to_tensor(self, batch_data):
        # store the tensor
        batch_data_tensor = {
            "state": [],
            "action": [],
            "log_prob": [],
            "entropy": [],
            "return": [],
        }

        # get the numpy arrays
        state_arr, action_arr, log_prob_arr, entropy_arr, return_arr = batch_data
        batch_size = self.params["batch_size"]
        # convert to tensors
        batch_data_tensor["state"] = torch.tensor(
            state_arr.reshape(batch_size, -1), dtype=torch.float32
        ).to(self.device)
        batch_data_tensor["action"] = (
            torch.tensor(action_arr).long().view(-1, 1).to(self.device)
        )
        batch_data_tensor["log_prob"] = (
            torch.tensor(log_prob_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        )
        batch_data_tensor["entropy"] = (
            torch.tensor(entropy_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        )
        batch_data_tensor["return"] = (
            torch.tensor(return_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        )

        return batch_data_tensor

    def save(self, filepath: str) -> None:
        pass

    def load(self, filepath: str) -> None:
        pass
