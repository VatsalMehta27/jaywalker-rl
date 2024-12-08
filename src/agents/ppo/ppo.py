"""
The file contains the PPO class to train with.
NOTE: All "ALG STEP"s are following the numbers from the original PPO pseudocode.
                It can be found here: https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg
"""

import gymnasium as gym
import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical

from src.agents.ppo.network import NeuralNetwork


class PPO:
    """
    This is the PPO class we will use as our model in main.py
    """

    def __init__(self, env, **hyperparameters):
        """
        Initializes the PPO model, including hyperparameters.

        Parameters:
                policy_class - the policy class to use for our actor/critic networks.
                env - the environment to train on.
                hyperparameters - all extra arguments passed into PPO that should be hyperparameters.

        Returns:
                None
        """
        # Make sure the environment is compatible with our code
        assert type(env.observation_space) is gym.spaces.Box
        # assert type(env.action_space) == gym.spaces.Box

        # Initialize hyperparameters for training with PPO
        self._init_hyperparameters(hyperparameters)

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
        self.act_dim = env.action_space.n

        # Initialize actor and critic networks
        self.actor = NeuralNetwork(
            self.obs_dim, self.act_dim, hidden_dim=64
        )  # ALG STEP 1
        self.critic = NeuralNetwork(self.obs_dim, 1, hidden_dim=64)

        self.loss = nn.MSELoss()

        # Initialize optimizers for actor and critic
        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        # This logger will help us with printing out summaries of each iteration
        self.logger = {
            "delta_t": time.time_ns(),
            "t_so_far": 0,  # timesteps so far
            "i_so_far": 0,  # iterations so far
            "batch_lens": [],  # episodic lengths in batch
            "batch_rews": [],  # episodic returns in batch
            "actor_losses": [],  # losses of actor network in current iteration
        }

    def learn(self, total_timesteps):
        """
        Train the actor and critic networks. Here is where the main PPO algorithm resides.

        Parameters:
                total_timesteps - the total number of timesteps to train for

        Return:
                None
        """
        print(
            f"Learning... Running {self.max_timesteps_per_episode} timesteps per episode, ",
            end="",
        )
        print(
            f"{self.timesteps_per_batch} timesteps per batch for a total of {total_timesteps} timesteps"
        )
        t_so_far = 0  # Timesteps simulated so far
        i_so_far = 0  # Iterations ran so far

        while t_so_far < total_timesteps:  # ALG STEP 2
            # Autobots, roll out (just kidding, we're collecting our batch simulations here)
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = (
                self.rollout()
            )  # ALG STEP 3

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger["t_so_far"] = t_so_far
            self.logger["i_so_far"] = i_so_far

            # Calculate advantage at k-th iteration
            V_values, _ = self.evaluate(batch_obs, batch_acts)
            advantage = batch_rtgs - V_values.detach()  # ALG STEP 5

            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            for _ in range(self.n_updates_per_iteration):  # ALG STEP 6 & 7
                V_values, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses.
                # surr1 = ratios * advantage
                # surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantage
                clipped_loss = torch.min(
                    ratios * advantage,
                    torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * advantage,
                )
                # Calculate actor and critic losses.
                # NOTE: we take the negative min of the surrogate losses because we're trying to maximize
                # the performance function, but Adam minimizes the loss. So minimizing the negative
                # performance function maximizes it.
                actor_loss = (-clipped_loss).mean()
                critic_loss = self.loss(V_values, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

                # Log actor loss
                self.logger["actor_losses"].append(actor_loss.detach())

            # Print a summary of our training so far
            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), "./ppo_actor.pth")
                torch.save(self.critic.state_dict(), "./ppo_critic.pth")

    def rollout(self):
        """
        Too many transformers references, I'm sorry. This is where we collect the batch of data
        from simulation. Since this is an on-policy algorithm, we'll need to collect a fresh batch
        of data each time we iterate the actor/critic networks.

        Parameters:
                None

        Return:
                batch_obs - the observations collected this batch. Shape: (number of timesteps, dimension of observation)
                batch_acts - the actions collected this batch. Shape: (number of timesteps, dimension of action)
                batch_log_probs - the log probabilities of each action taken this batch. Shape: (number of timesteps)
                batch_rtgs - the Rewards-To-Go of each timestep in this batch. Shape: (number of timesteps)
                batch_lens - the lengths of each episode this batch. Shape: (number of episodes)
        """
        batch_states = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = []
        batch_lens = []

        episode_rewards = []
        state, _ = self.env.reset()
        done = False

        episode_timesteps = 0

        for _ in range(self.timesteps_per_batch):
            if (
                self.render
                and (self.logger["i_so_far"] % self.render_every_i == 0)
                and len(batch_lens) == 0
            ):
                self.env.render()

            batch_states.append(state.flatten())

            action, log_prob = self.get_action(state.flatten())
            state, rew, terminated, _, _ = self.env.step(action)
            truncated = self.env.time_steps >= self.max_timesteps_per_episode

            done = terminated | truncated

            episode_rewards.append(rew)
            batch_acts.append(action)
            batch_log_probs.append(log_prob)

            # If the environment tells us the episode is terminated, break
            if done:
                batch_lens.append(episode_timesteps + 1)
                batch_rews.append(episode_rewards)
                episode_timesteps = 0
                episode_rewards = []

                state, _ = self.env.reset()

        if episode_rewards:
            batch_lens.append(episode_timesteps + 1)
            batch_rews.append(episode_rewards)

        # Reshape data as tensors in the shape specified in function description, before returning
        batch_states = torch.tensor(np.array(batch_states), dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.compute_rtgs(batch_rews)  # ALG STEP 4

        # Log the episodic returns and episodic lengths in this batch.
        self.logger["batch_rews"] = batch_rews
        self.logger["batch_lens"] = batch_lens

        return batch_states, batch_acts, batch_log_probs, batch_rtgs, batch_lens

    def compute_rtgs(self, batch_rews):
        discounted = []

        for ep in reversed(batch_rews):
            G = 0

            for rew in reversed(ep):
                G = rew + self.gamma * G
                discounted.insert(0, G)

        return torch.tensor(discounted, dtype=torch.float)

    def get_action(self, state):
        state_tensor = torch.tensor(state).float().view(1, -1)

        action_logits = self.actor(state_tensor)

        distribution = Categorical(logits=action_logits)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)

        return action.detach().item(), log_prob.detach()

    def evaluate(self, states, actions):
        states_tensor = torch.tensor(states).float()
        actions_tensor = torch.tensor(actions).float().view(1, -1)

        V_values = self.critic(states_tensor).squeeze()

        logits = self.actor(states)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions_tensor.squeeze(dim=1))

        return V_values, log_probs.unsqueeze(dim=1)

    def _init_hyperparameters(self, hyperparameters):
        """
        Initialize default and custom values for hyperparameters

        Parameters:
                hyperparameters - the extra arguments included when creating the PPO model, should only include
                                                        hyperparameters defined below with custom values.

        Return:
                None
        """
        # Initialize default values for hyperparameters
        # Algorithm hyperparameters
        self.timesteps_per_batch = 4800  # Number of timesteps to run per batch
        self.max_timesteps_per_episode = 1600  # Max number of timesteps per episode
        self.n_updates_per_iteration = (
            5  # Number of times to update actor/critic per iteration
        )
        self.lr = 0.005  # Learning rate of actor optimizer
        self.gamma = (
            0.95  # Discount factor to be applied when calculating Rewards-To-Go
        )
        self.clip = 0.2  # Recommended 0.2, helps define the threshold to clip the ratio during SGA

        # Miscellaneous parameters
        self.render = True  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 10  # How often we save in number of iterations
        self.seed = (
            None  # Sets the seed of our program, used for reproducibility of results
        )

        # Change any default values to custom values for specified hyperparameters
        for param, val in hyperparameters.items():
            exec("self." + param + " = " + str(val))

        # Sets the seed if specified
        if self.seed is not None:
            # Check if our seed is valid first
            assert type(self.seed) is int

            # Set the seed
            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self):
        """
        Print to stdout what we've logged so far in the most recent batch.

        Parameters:
                None

        Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger["delta_t"]
        self.logger["delta_t"] = time.time_ns()
        delta_t = (self.logger["delta_t"] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger["t_so_far"]
        i_so_far = self.logger["i_so_far"]
        avg_ep_lens = np.mean(self.logger["batch_lens"])
        avg_ep_rews = np.mean(
            [np.sum(ep_rews) for ep_rews in self.logger["batch_rews"]]
        )
        avg_actor_loss = np.mean(
            [losses.float().mean() for losses in self.logger["actor_losses"]]
        )

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(
            f"-------------------- Iteration #{i_so_far} --------------------",
            flush=True,
        )
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print("------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger["batch_lens"] = []
        self.logger["batch_rews"] = []
        self.logger["actor_losses"] = []
