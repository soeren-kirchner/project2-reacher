import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=300, fc2_units=400):
        """Initialize parameters and build model."""
        super(Actor, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(state_size, fc1_units),
            nn.BatchNorm1d(fc1_units),
            nn.ReLU(),
            nn.Linear(fc1_units, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, action_size),
            nn.Tanh()
        )

        # Initialize weights
        self.seq[0].weight.data.uniform_(*hidden_init(self.seq[0]))
        self.seq[3].weight.data.uniform_(*hidden_init(self.seq[3]))
        self.seq[5].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        return self.seq(state)

    def add_parameter_noise(self, scalar=.1):
        for layer in [0, 3, 5]:
            self.seq[layer].weight.data += torch.randn_like(self.seq[layer].weight.data) * scalar


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, fcs1_units=300, fc2_units=400):
        """Initialize parameters and build model. """
        super(Critic, self).__init__()

        self.seq1 = nn.Sequential(
            nn.Linear(state_size, fcs1_units),
            nn.BatchNorm1d(fcs1_units),
            nn.ReLU()
        )

        self.seq2 = nn.Sequential(
            nn.Linear(fcs1_units+action_size, fc2_units),
            nn.ReLU(),
            nn.Linear(fc2_units, 1)
        )
        
        # Initialize weights
        self.seq1[0].weight.data.uniform_(*hidden_init(self.seq1[0]))
        self.seq2[0].weight.data.uniform_(*hidden_init(self.seq2[0]))
        self.seq2[2].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        return self.seq2(torch.cat((self.seq1(state), action), dim=1))
