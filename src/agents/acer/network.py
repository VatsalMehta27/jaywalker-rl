from torch import nn


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.input = nn.Linear(self.state_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.actor_output = nn.Linear(self.hidden_dim, self.action_dim)
        self.actor_softmax = nn.Softmax(dim=-1)

        self.critic_output = nn.Linear(self.hidden_dim, self.action_dim)

    def forward(self, x):
        hidden_values = self.input(x)
        hidden_values = self.relu(hidden_values)
        hidden_values = self.hidden(hidden_values)
        hidden_values = self.relu(hidden_values)

        action_probs = self.actor_output(hidden_values)
        action_probs = self.actor_softmax(action_probs)

        state_value = self.critic_output(hidden_values)

        return state_value, action_probs
