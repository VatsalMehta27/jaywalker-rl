from torch import nn


class REINFORCEBaselineNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.actor_input = nn.Linear(self.state_dim, self.hidden_dim)
        self.actor_relu = nn.ReLU()
        self.actor_output = nn.Linear(self.hidden_dim, self.action_dim)
        self.actor_softmax = nn.Softmax(dim=-1)

        self.critic_input = nn.Linear(self.state_dim, self.hidden_dim)
        self.critic_relu = nn.ReLU()
        self.critic_output = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        action_probs = self.actor_input(x)
        action_probs = self.actor_relu(action_probs)
        action_probs = self.actor_output(action_probs)
        action_probs = self.actor_softmax(action_probs)

        state_value = self.critic_input(x)
        state_value = self.critic_relu(state_value)
        state_value = self.critic_output(state_value)

        return state_value, action_probs
