import gymnasium as gym 
import numpy as np  
import torch
import torch.nn as nn 
import torch.optim as optim 
from typing import List, Tuple, Dict

class Policy(nn.Module):
    """ Policy maps actions to 
        We are using a stochaistic policy in our case 
    """
    def __init__(self, state_dim:int, action_dim:int, hidden_dim:int=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(), 
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        ) 
    def forward(self, state:torch.Tensor):
        return self.network(state)
    



