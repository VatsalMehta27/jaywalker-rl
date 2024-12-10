import numpy as np
import torch
from torch import nn
import tqdm
from src.agents.agent import Agent, TrainingResult
from src.agents.dqn.dqn import DQN
from src.agents.dqn.replay_buffer import ReplayBuffer
from src.utils.epsilon_schedules.exponential_schedule import ExponentialSchedule
from src.utils.epsilon_schedules.linear_schedule import LinearSchedule


# customized weight initialization
def customized_weights_init(m):
    # compute the gain
    gain = nn.init.calculate_gain("relu")
    # init the linear layer
    if isinstance(m, nn.Linear):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)


class DQNAgent(Agent):
    def __init__(self, env, params):
        super().__init__(env, params)

        if params["scheduler_type"] == "linear":
            self.epsilon_scheduler = LinearSchedule(
                start_value=params["epsilon_start_value"],
                end_value=params["epsilon_end_value"],
                duration=params["epsilon_duration"],
            )
        elif params["scheduler_type"] == "exponential":
            self.epsilon_scheduler = ExponentialSchedule(
                start_value=params["epsilon_start_value"],
                end_value=params["epsilon_end_value"],
                duration=params["epsilon_duration"],
            )

        # create value network
        self.behavior_policy_net = DQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_layers=params["num_layers"],
            hidden_dim=params["hidden_dim"],
        )

        # create target network
        self.target_policy_net = DQN(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            num_layers=params["num_layers"],
            hidden_dim=params["hidden_dim"],
        )

        # initialize target network with behavior network
        self.behavior_policy_net.apply(customized_weights_init)
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

        # send the agent to a specific device: cpu or gpu
        self.device = torch.device(params["device"])
        self.behavior_policy_net.to(self.device)
        self.target_policy_net.to(self.device)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.behavior_policy_net.parameters(), lr=params["learning_rate"]
        )
        # loss
        self.loss = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(params["replay_buffer_size"])
        self.replay_buffer.populate(self.env, params["replay_buffer_prepopulate_size"])

        self.epsilon = params["epsilon_start_value"]

    def get_action(self, state):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space, 1)[0]
            return action

        return self.get_greedy_action(state)

    def get_action_probs(self, state):
        state_tensor = torch.tensor(self.transform_state(state)).float().view(1, -1)
        q_values = self.behavior_policy_net(state_tensor)

        return q_values

    def get_greedy_action(self, state):
        state_tensor = torch.tensor(self.transform_state(state)).float().view(1, -1)

        with torch.no_grad():
            q_values = self.behavior_policy_net(state_tensor)
            action = torch.argmax(q_values).item()

        return self.action_space[int(action)]

    def update_behavior_policy(self, batch_data):
        # convert batch data to tensor and put them on device
        batch_data_tensor = self._batch_to_tensor(batch_data)

        # get the transition data
        state_tensor = batch_data_tensor["state"]
        actions_tensor = batch_data_tensor["action"]
        next_state_tensor = batch_data_tensor["next_state"]
        rewards_tensor = batch_data_tensor["reward"]
        dones_tensor = batch_data_tensor["done"]

        # compute the q value estimation using the behavior network
        q_value_estimation = self.behavior_policy_net(state_tensor).gather(
            1, actions_tensor
        )

        # compute the TD target using the target network
        with torch.no_grad():
            max_q_s_prime = (
                self.target_policy_net(next_state_tensor).max(dim=1)[0].reshape(-1, 1)
            )
            td_target = (
                rewards_tensor
                + (1 - dones_tensor) * self.params["gamma"] * max_q_s_prime
            )

        # compute the loss
        td_loss = self.loss(q_value_estimation, td_target) / self.params["batch_size"]

        # minimize the loss
        self.optimizer.zero_grad()
        td_loss.backward()
        self.optimizer.step()

        return td_loss.item()

    # update update target policy
    def update_target_policy(self):
        # hard update
        self.target_policy_net.load_state_dict(self.behavior_policy_net.state_dict())

    def _batch_to_tensor(self, batch_data):
        # store the tensor
        batch_data_tensor = {
            "state": [],
            "action": [],
            "reward": [],
            "next_state": [],
            "done": [],
        }

        # get the numpy arrays
        state_arr, action_arr, reward_arr, next_state_arr, done_arr = batch_data
        batch_size = self.params["batch_size"]
        # convert to tensors
        batch_data_tensor["state"] = torch.tensor(
            state_arr.reshape(batch_size, -1), dtype=torch.float32
        ).to(self.device)
        batch_data_tensor["action"] = (
            torch.tensor(action_arr).long().view(-1, 1).to(self.device)
        )
        batch_data_tensor["reward"] = (
            torch.tensor(reward_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        )
        batch_data_tensor["next_state"] = torch.tensor(
            next_state_arr.reshape(batch_size, -1), dtype=torch.float32
        ).to(self.device)
        batch_data_tensor["done"] = (
            torch.tensor(done_arr, dtype=torch.float32).view(-1, 1).to(self.device)
        )

        return batch_data_tensor

    def train(self, training_steps):
        episode_timestep = 0
        rewards = []

        train_returns = []
        train_loss = []
        train_timesteps = []

        # reset the environment
        state, _ = self.env.reset()

        # start training
        pbar = tqdm.trange(training_steps)
        last_best_return = 0

        for total_timestep in pbar:
            # scheduled epsilon at time step t
            self.epsilon = self.epsilon_scheduler.get_value(total_timestep)
            # get one epsilon-greedy action
            action = self.get_action(state)

            # step in the environment
            next_state, reward, done, _, _ = self.env.step(action)

            # add to the buffer
            self.replay_buffer.add(state, action, reward, next_state, done)
            rewards.append(reward)

            # check termination
            if done:
                # compute the return
                G = 0
                for r in reversed(rewards):
                    G = r + self.params["gamma"] * G

                if G > last_best_return:
                    torch.save(
                        self.behavior_policy_net.state_dict(),
                        f"./checkpoints/{self.params['model_name']}",
                    )
                    last_best_return = G

                # store the return and timesteps
                train_returns.append(G)
                train_timesteps.append(episode_timestep)

                # reset the environment
                episode_timestep, rewards = 0, []
                state, _ = self.env.reset()
            else:
                # increment
                state = next_state
                episode_timestep += 1

            # update the behavior model
            if np.mod(total_timestep, self.params["freq_update_behavior_policy"]) == 0:
                sample_batch = self.replay_buffer.sample_batch(
                    self.params["batch_size"]
                )
                loss = self.update_behavior_policy(sample_batch)
                train_loss.append(loss)

            # update the target model
            if np.mod(total_timestep, self.params["freq_update_target_policy"]) == 0:
                self.update_target_policy()

        # Convert collected metrics to numpy arrays
        train_returns = np.array(train_returns, dtype=np.int)
        train_timesteps = np.array(train_timesteps, dtype=np.int)
        train_loss = np.array(train_loss, dtype=np.int)

        return TrainingResult(
            returns=train_returns,
            timesteps=train_timesteps,
            loss=train_loss,
        )

    def save(self, filepath: str) -> None:
        """
        Save the agent's policy networks, optimizer state, and epsilon scheduler.
        """
        torch.save(
            {
                "behavior_policy_net": self.behavior_policy_net.state_dict(),
                "target_policy_net": self.target_policy_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon_scheduler": {
                    "scheduler_type": self.params["scheduler_type"],
                    "start_value": self.epsilon_scheduler.start_value,
                    "end_value": self.epsilon_scheduler.end_value,
                    "duration": self.epsilon_scheduler.duration,
                },
                "params": self.params,
            },
            filepath,
        )
        print(f"Agent's state has been saved to {filepath}")

    def load(self, filepath: str) -> None:
        """
        Load the agent's policy networks, optimizer state, and epsilon scheduler.
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.behavior_policy_net.load_state_dict(checkpoint["behavior_policy_net"])
        self.target_policy_net.load_state_dict(checkpoint["target_policy_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load epsilon scheduler
        scheduler_data = checkpoint.get("epsilon_scheduler", {})
        if scheduler_data.get("scheduler_type") == "linear":
            self.epsilon_scheduler = LinearSchedule(
                start_value=scheduler_data["start_value"],
                end_value=scheduler_data["end_value"],
                duration=scheduler_data["duration"],
            )
        elif scheduler_data.get("scheduler_type") == "exponential":
            self.epsilon_scheduler = ExponentialSchedule(
                start_value=scheduler_data["start_value"],
                end_value=scheduler_data["end_value"],
                duration=scheduler_data["duration"],
            )

        # Update device
        self.behavior_policy_net.to(self.device)
        self.target_policy_net.to(self.device)
        print(f"Agent's state has been loaded from {filepath}")
