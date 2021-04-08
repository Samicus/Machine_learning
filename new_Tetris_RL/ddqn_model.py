import numpy as np
import pandas as pd
import torch
from torch import nn
import math


class DeepQNetwork(nn.Module):
    def __init__(self, num_states, num_actions):

        super(DeepQNetwork, self).__init__()
        self.output_size = num_actions
        self.input_layer_size = num_states


        layer_1_width = 64
        layer_2_width = 64
        layer_3_width = 32

        # Layers
        self.fc1 = nn.Linear(self.input_layer_size, layer_1_width)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(layer_1_width, layer_2_width)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(layer_2_width, layer_3_width)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(layer_3_width, self.output_size)

        # Initialize bias parameters
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)
        nn.init.zeros_(self.fc3.bias)
        nn.init.zeros_(self.fc_out.bias)

        # Initialize final layer uniformly
        nn.init.uniform_(self.fc_out.weight, a=-1e-6, b=1e-6)

    def forward(self, observation):
        # Layer 1
        h = self.fc1(observation)
        h = self.relu1(h)

        # Layer 2
        h = self.fc2(h)
        h = self.relu2(h)

        # Layer 3
        h = self.fc3(h)
        h = self.relu3(h)

        # Final Layer
        q = self.fc_out(h)

        return q


class DeepDoubleQNetwork(object):
    def __init__(self, device, num_states, lr):
        self.device = device
        self.num_states = num_states
        self.num_actions = 2
        self.lr = lr

        # Creating the two deep Q-networks
        self.online_model = DeepQNetwork(self.num_states, self.num_actions).to(
            device=self.device
        )
        self.offline_model = DeepQNetwork(self.num_states, self.num_actions).to(
            device=self.device
        )

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), self.lr)

        # Define loss function
        self.mse = nn.MSELoss(reduction="mean").to(device=self.device)

    def calc_loss(self, q_online_curr, q_target, a):
        """
        Calculate loss for given batch
        :param q_online_curr: batch of q values at current state. Shape (N, num actions)
        :param q_target: batch of temporal difference targets. Shape (N,)
        :param a: batch of actions taken at current state. Shape (N,)
        :return:
        """

        # Get batch size
        batch_size = q_online_curr.shape[0]

        # verify input dimensions
        assert q_online_curr.shape == (batch_size, self.num_actions)
        assert q_target.shape == (batch_size,)
        assert a.shape == (batch_size,)

        # Select only the Q-values corresponding to the actions taken (loss should only be applied for these)
        q_online_curr_allactions = q_online_curr
        q_online_curr = q_online_curr[
            torch.arange(batch_size), a
        ]  # New shape: (batch_size,)
        for j in [0, 3, 4]:
            assert q_online_curr_allactions[j, a[j]] == q_online_curr[j]

        # Make sure that gradient is not back-propagated through Q target
        assert not q_target.requires_grad

        loss = self.mse(q_online_curr, q_target)
        assert loss.shape == ()

        return loss

    def update_target_network(self):
        """
        Update target network parameters, by copying from online network.
        """
        online_params = self.online_model.state_dict()
        self.offline_model.load_state_dict(online_params)