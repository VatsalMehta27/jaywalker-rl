from gymnasium import Env
from collections import defaultdict
import numpy as np


class QLearning(object):
    def __init__(
        self, env: Env, alpha: float, epsilon: float, gamma: float, timeout: int
    ):
        # define the parameters
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # environment
        self.env: Env = env

        self.actions = [
            a
            for a in range(
                self.env.action_space.start,
                self.env.action_space.start + self.env.action_space.n,
            )
        ]
        self.action_num = self.env.action_space.n

        self.Q = defaultdict(lambda: [0 for _ in range(self.action_num)])

        # define the mapping of state/action to index
        self.action_mapping = {i: action for i, action in enumerate(self.actions)}

        # define the timeout
        self.timeout = timeout

    @staticmethod
    def argmax(arr) -> int:
        """Argmax that breaks ties randomly"""
        arr = np.asarray(arr)
        max_indices = np.flatnonzero(arr == arr.max())

        return np.random.choice(max_indices)

    def behavior_policy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_num)

        return self.greedy_policy(state)

    def greedy_policy(self, state):
        return self.argmax(self.Q[state])

    def update(self, s: tuple, a: int, r: int, s_prime: tuple):
        self.Q[s][a] = self.Q[s][a] + self.alpha * (
            r + self.gamma * max(self.Q[s_prime]) - self.Q[s][a]
        )

    @staticmethod
    def nested_tuple(lst: list[list]) -> tuple[tuple]:
        return tuple(tuple(i) for i in lst)

    def run(self, episodes):
        rewards = []
        timesteps = []

        for _ in range(episodes):
            state, _ = self.env.reset()
            state = self.nested_tuple(state["world_grid"])
            done = False

            while not done:
                if self.env.time_steps >= self.timeout:
                    reward = -1000
                    print("timeout")
                    break

                action_index = self.behavior_policy(state)
                action = self.action_mapping[action_index]

                next_state, reward, done, _ = self.env.step(action)
                next_state = self.nested_tuple(next_state["world_grid"])

                self.update(state, action_index, reward, next_state)

                state = next_state

            rewards.append(reward)
            timesteps.append(self.env.time_steps)

        return rewards, timesteps
