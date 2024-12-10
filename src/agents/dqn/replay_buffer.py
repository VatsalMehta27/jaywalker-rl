import numpy as np


class ReplayBuffer(object):
    """Implement the Replay Buffer as a class, which contains:
    - self._data_buffer (list): a list variable to store all transition tuples.
    - add: a function to add new transition tuple into the buffer
    - sample_batch: a function to sample a batch training data from the Replay Buffer
    """

    def __init__(self, buffer_size):
        """Args:
        buffer_size (int): size of the replay buffer
        """
        # total size of the replay buffer
        self.total_size = buffer_size

        # create a list to store the transitions
        self._data_buffer = []
        self._next_idx = 0

    def __len__(self):
        return len(self._data_buffer)

    def add(self, state, act, reward, next_state, done):
        # create a tuple
        transition = (state, act, reward, next_state, done)

        # interesting implementation
        if self._next_idx >= len(self._data_buffer):
            self._data_buffer.append(transition)
        else:
            self._data_buffer[self._next_idx] = transition

        # increase the index
        self._next_idx = (self._next_idx + 1) % self.total_size

    def _encode_sample(self, indices):
        """Function to fetch the state, action, reward, next state, and done arrays.

        Args:
            indices (list): list contains the index of all sampled transition tuples.
        """
        # lists for transitions
        state_list, actions_list, rewards_list, next_state_list, dones_list = (
            [],
            [],
            [],
            [],
            [],
        )

        # collect the data
        for idx in indices:
            # get the single transition
            data = self._data_buffer[idx]
            obs, act, reward, next_obs, d = data
            # store to the list
            state_list.append(np.asarray(obs))
            actions_list.append(np.asarray(act))
            rewards_list.append(np.asarray(reward))
            next_state_list.append(np.asarray(next_obs))
            dones_list.append(np.asarray(d))

        # return the sampled batch data as numpy arrays
        return (
            np.array(state_list),
            np.array(actions_list),
            np.array(rewards_list),
            np.array(next_state_list),
            np.array(dones_list),
        )

    def sample_batch(self, batch_size):
        """Args:
        batch_size (int): size of the sampled batch data.
        """
        # sample indices with replaced
        # indices = [
        #     np.random.randint(0, len(self._data_buffer)) for _ in range(batch_size)
        # ]
        indices = np.random.choice(len(self._data_buffer), batch_size, replace=False)
        return self._encode_sample(indices)

    def populate(self, env, num_steps):
        """Populate this replay memory with `num_steps` from the random policy.

        :param env: Gymnasium environment
        :param num_steps: Number of steps to populate the replay memory
        """
        state, _ = env.reset()

        for _ in range(num_steps):
            action = env.action_space.sample()
            next_state, reward, done, _, _ = env.step(action)
            self.add(state, action, reward, next_state, done)

            if done:
                state, _ = env.reset()
            else:
                state = next_state
