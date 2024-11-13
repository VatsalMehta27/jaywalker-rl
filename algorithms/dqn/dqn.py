import torch
from torch import nn


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, num_layers=3, hidden_dim=256):
        """Deep Q-Network PyTorch model.

        Args:
            - state_dim: Dimensionality of states
            - action_dim: Dimensionality of actions
            - num_layers: Number of total linear layers
            - hidden_dim: Number of neurons in the hidden layers
        """

        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        network_layers = []

        network_layers.extend([nn.Linear(self.state_dim, self.hidden_dim), nn.ReLU()])

        for _ in range(self.num_layers - 2):
            network_layers.extend(
                [nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()]
            )

        network_layers.extend([nn.Linear(self.hidden_dim, self.action_dim)])

        self.network = nn.Sequential(*network_layers)

    def forward(self, states) -> torch.Tensor:
        """Q function mapping from states to action-values.

        :param states: (*, S) torch.Tensor where * is any number of additional
                dimensions, and S is the dimensionality of state-space
        :rtype: (*, A) torch.Tensor where * is the same number of additional
                dimensions as the `states`, and A is the dimensionality of the
                action-space. This represents the Q values Q(s, .)
        """
        return self.network(states)
