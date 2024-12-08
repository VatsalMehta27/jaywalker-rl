import numpy as np


class TrajectoryBuffer(object):
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

    def add(self, trajectory):
        if self._next_idx >= len(self._data_buffer):
            self._data_buffer.append(trajectory)
        else:
            self._data_buffer[self._next_idx] = trajectory

        # increase the index
        self._next_idx = (self._next_idx + 1) % self.total_size

    def sample_trajectory(self):
        """Args:
        batch_size (int): size of the sampled batch data.
        """
        index = np.random.randint(len(self._data_buffer))
        return self._data_buffer[index]
