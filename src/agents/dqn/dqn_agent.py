import torch
from torch import nn
import numpy as np
from gymnasium import Env
import tqdm


from agents.dqn.replay_buffer import ReplayBuffer
from agents.dqn.dqn import DQN
from utils.epsilon_schedules.exponential_schedule import ExponentialSchedule
from utils.epsilon_schedules.linear_schedule import LinearSchedule


# customized weight initialization
def customized_weights_init(m):
    # compute the gain
    gain = nn.init.calculate_gain("relu")
    # init the convolutional layer
    if isinstance(m, nn.Conv2d):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)
    # init the linear layer
    if isinstance(m, nn.Linear):
        # init the params using uniform
        nn.init.xavier_uniform_(m.weight, gain=gain)
        nn.init.constant_(m.bias, 0)


class DQNAgent:
    # initialize the agent
    def __init__(
        self,
        env: Env,
        params,
    ):
        self.env = env

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

        # save the parameters
        self.params = params

        # environment parameters
        self.action_dim = params["action_dim"]
        self.obs_dim = params["state_dim"]

        # executable actions
        self.action_space = params["action_space"]

        # create value network
        self.behavior_policy_net = DQN(
            state_dim=params["state_dim"],
            action_dim=params["action_dim"],
            num_layers=params["num_layers"],
            hidden_dim=params["hidden_dim"],
        )

        # create target network
        self.target_policy_net = DQN(
            state_dim=params["state_dim"],
            action_dim=params["action_dim"],
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

    def get_action(self, state, eps):
        if np.random.random() < eps:
            action = np.random.choice(self.action_space, 1)[0]
            return action

        return self.get_greedy_action(state)

    def get_greedy_action(self, state, rollout=False):
        state = self._arr_to_tensor(state).flatten().view(1, -1)

        with torch.no_grad():
            q_values = self.behavior_policy_net(state)
            action = q_values.max(dim=1)[1].item()
            if rollout:
                print(q_values)
                print(action)

        return self.action_space[int(action)]

    # update behavior policy
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

    def run(self):
        episode_timestep = 0
        train_rewards = []
        rewards = []
        train_returns = []
        train_loss = []

        # reset the environment
        state, _ = self.env.reset()
        state = state["world_grid"]

        # start training
        pbar = tqdm.trange(self.params["total_training_time_step"])
        last_best_return = 0

        for total_timestep in pbar:
            # scheduled epsilon at time step t
            eps_t = self.epsilon_scheduler.get_value(total_timestep)
            # get one epsilon-greedy action
            action = self.get_action(state, eps_t)

            # step in the environment
            next_state, reward, done, _, _ = self.env.step(action)
            next_state = next_state["world_grid"]

            # add to the buffer
            self.replay_buffer.add(
                state, self.action_space.index(action), reward, next_state, done
            )
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

                # store the return
                train_returns.append(G)
                train_rewards.append(rewards)
                episode_idx = len(train_returns)

                # print the information
                pbar.set_description(
                    f"Ep={episode_idx} | "
                    f"G={np.mean(train_returns[-10:]) if train_returns else 0:.2f} | "
                    f"Eps={eps_t}"
                )

                # reset the environment
                episode_timestep, rewards = 0, []
                state, _ = self.env.reset()
                state = state["world_grid"]
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

        return train_rewards, train_returns, train_loss
