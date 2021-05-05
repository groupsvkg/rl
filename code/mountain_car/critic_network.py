import numpy as np

import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, env):
        super().__init__()
        input_dimension = env.observation_space.shape[0] + \
            env.action_space.shape[0]

        self.q = nn.Sequential(
            nn.Linear(input_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.Linear(64, 1),
            nn.Identity()
        )

    def forward(self, state, action):
        q = self.q(torch.cat([state, action], dim=-1))
        return torch.squeeze(q, -1)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
