from agent import Agent
from env import Env
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt

import numpy as np
import time

def train_episode(
    env,
    agent: Agent,
    render: bool = False,
    max_steps: int = 100,
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
    from env import Env
    # Create environment and agent
    env = Env(render_mode="human")
    agent = Agent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        learning_rate=0.001,
        gamma=0.99
    )
    
    time = []
    rewards = []
    num_time_steps = 0
    num_episodes = 100
    for episode in range(num_episodes):

        total_reward, steps, stats = train_episode(
            env, 
            agent,
            render=(episode % 10 == 0)  # Render every 10th episode
        )
    
        num_time_steps+=steps
        time.append(num_time_steps)
        rewards.append(total_reward)
        
        print(f"Episode {episode}: Reward = {total_reward:.2f}, Steps = {steps}")

        
    env.close()

if __name__ == "__main__":
    main()