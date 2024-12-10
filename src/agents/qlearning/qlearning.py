import pickle
from gymnasium import Env
from collections import defaultdict
import numpy as np
from tqdm import tqdm

from src.agents.agent import Agent, TrainingResult


class QLearningAgent(Agent):
    def __init__(self, env: Env, params: dict):
        super().__init__(env, params)
        # define the parameters
        self.alpha = params["alpha"]
        self.epsilon = params["epsilon"]
        self.gamma = params["gamma"]

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
        self.action_mapping = {action: i for i, action in enumerate(self.actions)}

        # define the timeout
        self.timeout = params["timeout"]

    @staticmethod
    def nested_tuple(array: np.ndarray) -> tuple[tuple]:
        if array.ndim == 1:
            return tuple(array)

        return tuple(QLearningAgent.nested_tuple(subarr) for subarr in array)

    def get_action(self, state: dict) -> int:
        if np.random.random() < self.epsilon:
            return self.actions[np.random.choice(self.action_num)]

        return self.get_greedy_action(state)

    def get_action_probs(self, state):
        state = self.nested_tuple(self.transform_state(state))

        return self.Q[state]

    def get_greedy_action(self, state: dict) -> int:
        state = self.nested_tuple(self.transform_state(state))

        return self.actions[self.argmax(self.Q[state])]

    def update(self, s: dict, a: int, r: int, s_prime: dict) -> None:
        s = self.nested_tuple(self.transform_state(s))
        s_prime = self.nested_tuple(self.transform_state(s_prime))

        self.Q[s][a] = self.Q[s][a] + self.alpha * (
            r + self.gamma * max(self.Q[s_prime]) - self.Q[s][a]
        )

    def train(self, episodes):
        returns = []
        timesteps = []

        for _ in tqdm(range(episodes)):
            state, _ = self.env.reset()
            done = False

            ep_rewards = []

            while not done:
                action = self.get_action(state)
                action_index = self.action_mapping[action]

                next_state, reward, done, _, _ = self.env.step(action)
                done = done or self.env.time_steps >= self.timeout

                ep_rewards.append(reward)

                self.update(state, action_index, reward, next_state)

                state = next_state

            G = 0
            for r in reversed(ep_rewards):
                G = self.gamma * G + r

            returns.append(G)
            timesteps.append(self.env.time_steps)

        return TrainingResult(
            returns=np.array(returns), timesteps=np.array(timesteps), loss=np.array([])
        )

    def save(self, filepath: str = "qlearning_qvalue.pkl") -> None:
        with open(filepath, "wb") as file:
            pickle.dump(dict(self.Q), file)

        print(f"Saved to {filepath}!")

    def load(self, filepath: str = "qlearning_qvalue.pkl") -> None:
        if filepath.endswith(".pkl"):
            with open(filepath, "rb") as file:
                q_values = pickle.load(file)
                self.Q = defaultdict(
                    lambda: [0 for _ in range(self.action_num)], q_values
                )

            print(f"Loaded values from {filepath}!")
            return

        raise Exception("Invalid file type.")
