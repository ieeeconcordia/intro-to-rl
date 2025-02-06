from typing import Tuple, List, Dict
import numpy as np
import torch
import torch.optim as optim

from policy import Policy
from value import ValueFunction

class Agent:
    def __init__(self, 
                 state_dim:int, 
                 action_dim:int,
                 learning_rate: float=0.001,
                 hidden_dim:int=64,
                 gamma: float=0.99,
    ):
        self.policy = Policy(state_dim=state_dim,
                             action_dim=action_dim,
                             hidden_dim=hidden_dim)
        
        self.value_fn = ValueFunction(state_dim=state_dim,
                             hidden_dim=hidden_dim)


        # Setup optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_fn.parameters(), lr=learning_rate)
        
        # Store parameters
        self.gamma = gamma

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action using the current policy.
        
        Args:
            state: Current state of the environment
            
        Returns:
            Selected action
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state)
        
        # Get action probabilities from policy
        action_probs = self.policy(state_tensor)
        
        # Sample action from probability distribution
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        
        return action.item()
    
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> Dict[str, float]:
        """
        Update both the policy and value function using the collected experience.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode ended
            
        Returns:
            Dictionary containing training statistics
        """
        # Convert inputs to tensors
        state_tensor = torch.FloatTensor(state)
        next_state_tensor = torch.FloatTensor(next_state)
        
        # Compute value estimates
        current_value = self.value_fn(state_tensor)
        next_value = 0 if done else self.value_fn(next_state_tensor).item()
        
        # Compute TD target and advantage
        td_target = reward + self.gamma * next_value
        advantage = td_target - current_value.item()
        
        # Get action probabilities and compute log probability of taken action
        action_probs = self.policy(state_tensor)
        action_dist = torch.distributions.Categorical(action_probs)
        action_log_prob = action_dist.log_prob(torch.tensor(action))
        
        # Compute policy (actor) loss using advantage
        policy_loss = -action_log_prob * advantage
        
        # Compute value function (critic) loss
        value_loss = (current_value - torch.tensor([td_target])) ** 2
        
        # Update policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # Update value function
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # Return statistics for monitoring
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'advantage': advantage,
            'value_estimate': current_value.item()
        }
def train_episode(
    env,
    agent: Agent,
    render: bool = False,
    max_steps: int = 1000
) -> Tuple[float, int, List[Dict[str, float]]]:
    """
    Train the agent for one episode.
    
    Args:
        env: The environment to train in
        agent: The Actor-Critic agent
        render: Whether to render the environment
        max_steps: Maximum steps per episode
        
    Returns:
        Tuple of (total_reward, num_steps, list of training statistics)
    """
    state, _ = env.reset()
    total_reward = 0
    stats_list = []
    
    for step in range(max_steps):
        if render:
            env.render()
            
        # Select and take action
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update agent
        update_stats = agent.update(state, action, reward, next_state, done)
        stats_list.append(update_stats)
        
        # Update tracking variables
        state = next_state
        total_reward += reward
        
        if done:
            break
            
    return total_reward, step + 1, stats_list


def main():
    """Example usage of the ActorCriticAgent"""
    from env import Env
    
    # Create environment and agent
    env = Env(render_mode="human")
    agent = Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=0.001,
        gamma=0.99
    )
    
    # Training loop
    num_episodes = 100
    for episode in range(num_episodes):
        total_reward, steps, stats = train_episode(
            env, 
            agent,
            render=(episode % 10 == 0)  # Render every 10th episode
        )
        print(f"Episode {episode}: Reward = {total_reward:.2f}, Steps = {steps}")
        
    env.close()

if __name__ == "__main__":
    main()