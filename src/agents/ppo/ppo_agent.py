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

        self.action_dim = params["action_dim"]
        self.state_dim = params["state_dim"]

        # executable actions
        self.action_space = params["action_space"]

        self.actor = NeuralNetwork(
            state_dim=self.state_dim,
            output_dim=self.action_dim,
            num_layers=params["num_layers"],
            hidden_dim=params["hidden_dim"],
        )
        print(self.actor)
        self.actor.apply(customized_weights_init)

        self.critic = NeuralNetwork(
            state_dim=self.state_dim,
            output_dim=1,
            num_layers=params["num_layers"],
            hidden_dim=params["hidden_dim"],
        )
        print(self.critic)
        self.critic.apply(customized_weights_init)

        self.device = torch.device(params["device"])
        self.actor.to(self.device)
        self.critic.to(self.device)

        # optimizer
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=params["learning_rate"]
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=params["learning_rate"]
        )

        # loss
        self.loss = nn.MSELoss()

        self.clip = params["clip"]
        self.batch_size = params["batch_size"]
        self.gamma = params["gamma"]
        self.epochs_per_iteration = params["epochs_per_iteration"]

    def get_action(self, state: np.ndarray) -> int:
        action, _ = self._get_action_prob(state)
        return action

    def _get_action_prob(self, state: np.ndarray) -> tuple[int, float]:
        state_tensor = torch.tensor(state).float().view(1, -1)

        logits = self.actor(state_tensor)
        # distribution = Categorical(logits=logits)
        distribution = Categorical(probs=torch.softmax(logits, dim=-1))
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return self.action_space[action.item()], log_prob.detach()

    def get_greedy_action(self, state: np.ndarray) -> int:
        state_tensor = torch.tensor(state).float().view(1, -1)

        logits = self.actor(state_tensor).detach()

        return self.action_space[logits.max(dim=1)[1].item()]

    def _get_batch_log_prob(self, states: torch.tensor, actions: torch.tensor):
        logits = self.actor(states)
        # distribution = Categorical(logits=logits)
        distribution = Categorical(probs=torch.softmax(logits, dim=-1))
        log_probs = distribution.log_prob(actions.squeeze(dim=1))

        return log_probs.unsqueeze(dim=1)

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
            returns_tensor = batch_data_tensor["return"]

            # print(states_tensor.shape)
            # print(actions_tensor.shape)
            # print(log_probs_tensor.shape)
            # print(returns_tensor.shape)

            train_returns.extend(returns_tensor.numpy().flatten())

            # with torch.no_grad():

            for _ in range(self.epochs_per_iteration):
                V_values = self.critic(states_tensor)
                # TODO: Try GAE?
                advantage = returns_tensor - V_values.detach()

                normalized_advantage = (advantage - advantage.mean()) / (
                    advantage.std() + 1e-8
                )
                current_log_probs = self._get_batch_log_prob(
                    states_tensor, actions_tensor
                )

                ratios = torch.exp(current_log_probs - log_probs_tensor)
                clipped_loss = torch.min(
                    ratios * normalized_advantage,
                    torch.clamp(ratios, 1 - self.clip, 1 + self.clip)
                    * normalized_advantage,
                )

                actor_loss = (-clipped_loss).mean()
                critic_loss = self.loss(V_values, returns_tensor)

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=0.5)
                self.critic_optimizer.step()

                train_actor_losses.append(actor_loss.detach().item())
                train_critic_losses.append(critic_loss.detach().item())

        return TrainingResult(
            returns=np.array(train_returns),
            timesteps=np.array(train_actor_losses),
            loss=np.array(train_critic_losses),
        )

    def _rollout(self, num_steps):
        state_list, actions_list, log_probs_list, returns_list = (
            [],
            [],
            [],
            [],
        )

        state, _ = self.env.reset()
        state = state["world_grid"]

        episode_rewards = []

        for _ in range(num_steps):
            action, prob = self._get_action_prob(state)
            next_state, reward, done, _, _ = self.env.step(action)
            next_state = next_state["world_grid"]

            state_list.append(np.asarray(state))
            actions_list.append(np.asarray(action))
            log_probs_list.append(np.asarray(prob))
            episode_rewards.append(reward)

            if done:
                state, _ = self.env.reset()
                state = state["world_grid"]

                returns_list.extend(self._discount_rewards(episode_rewards))
                episode_rewards = []
            else:
                state = next_state

        if episode_rewards:
            returns_list.extend(self._discount_rewards(episode_rewards))

        batch = (
            np.array(state_list),
            np.array(actions_list),
            np.array(log_probs_list),
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
            "return": [],
        }

        # get the numpy arrays
        state_arr, action_arr, log_prob_arr, reward_arr = batch_data
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
        batch_data_tensor["return"] = (
            torch.tensor(reward_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        )

        return batch_data_tensor

    def save(self, filepath: str) -> None:
        pass

    def load(self, filepath: str) -> None:
        pass
