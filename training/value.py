
import gymnasium as gym
import numpy as np 
import torch
import torch.nn as nn 
import torch.optim


class ValueFunction(nn.Module):
    """
    Value function network that estimates the expected return from a state.
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Single value output
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the value network.
        Returns estimated value of the state.
        """
        return self.network(state)