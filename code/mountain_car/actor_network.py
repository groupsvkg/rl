import numpy as np

import torch
import torch.nn as nn


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        input_dimension = env.observation_space.shape[0]
        # self.policy = nn.Sequential(
        #     nn.Linear(input_dimension, 400),
        #     nn.ReLU(),
        #     nn.Linear(400, 300),
        #     nn.ReLU(),
        #     nn.Linear(300, 1),
        #     nn.Tanh()
        # )
        self.policy = nn.Sequential(
            nn.Linear(input_dimension, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, state):
        return self.policy(state)

    def action(self, state):
        with torch.no_grad():
            return self.policy(state).numpy()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def unfreeze(self):
        for p in self.parameters():
            p.requires_grad = True
