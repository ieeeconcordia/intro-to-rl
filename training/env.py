from typing import Tuple, Dict, Any, Optional
import gymnasium as gym
import numpy as np

class Env:
    """
    Wrapper environment, we use wrapper to improve upon the existing environment.  
    """
    def __init__(self, render_mode: Optional[str] = "human"):
        """
        Args:
            render_mode: One of None, 'human', 'rgb_array' see gymnasimu documentation 
        """
        # Initialize the base environment
        self.env = gym.make(
            "LunarLander-v3", 
            render_mode=render_mode,
            gravity=-10.0,
            enable_wind=False, 
            wind_power=15.0, 
            turbulence_power=1.5
            
            )
        
        # Store spaces for easy access
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # Store dimensions as properties
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        
        # Initialize internal state tracking
        self._current_state = None
        self._episode_reward = 0.0
        self._steps_taken = 0

    def reset(self, *, seed: Optional[int] = None, 
             options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to an initial state.
        
        Args:
            seed: Optional seed for reproducibility
            options: Optional configuration for reset behavior
            
        Returns:
            A tuple of (initial_observation, info_dict)
        """
        # Reset the internal state tracking
        self._episode_reward = 0.0
        self._steps_taken = 0
        
        # Reset the underlying environment
        initial_state, info = self.env.reset(seed=seed, options=options)
        self._current_state = initial_state
        
        return initial_state, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take an action in the environment 
        
        Args:
            action: The action to take (must be valid according to action_space)
            
        Returns:
            A tuple of (next_state, reward, terminated, truncated, info) where:
            - next_state: The new environment state
            - reward: The reward obtained from the action
            - terminated: Whether the episode ended naturally
            - truncated: Whether the episode was artificially terminated
            - info: Additional information about the step
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action} for action space {self.action_space}")
        
        # Take the step in the environment
        next_state, reward, terminated, truncated, info = self.env.step(action)
        
        # Update internal state tracking
        self._current_state = next_state
        self._episode_reward += reward
        self._steps_taken += 1
        
        # Add episode statistics to info
        info.update({
            'episode_reward': self._episode_reward,
            'steps_taken': self._steps_taken
        })
        
        return next_state, reward, terminated, truncated, info

    def render(self) -> Optional[np.ndarray]:
        """
        Render the environment according to the render_mode specified in __init__.
        
        Returns:
            None for 'human' mode, np.ndarray for 'rgb_array' mode
        """
        return self.env.render()

    def close(self):
        """
        Clean up resources used by the environment.
        """
        self.env.close()

    @property
    def unwrapped(self):
        """
        Returns the base environment without any wrappers.
        """
        return self.env.unwrapped
    

def main():
    wrapped_env = Env(render_mode="human")
    state, info = wrapped_env.reset()
    cumulative_reward = 0
    episode_over = False
    NUM_STEPS = 1000
    for step in range(NUM_STEPS):
        action = wrapped_env.env.action_space.sample()
        state, reward, terminated, truncated,info = wrapped_env.step(action)
        cumulative_reward += reward
        episode_over = terminated or truncated 
        if episode_over:
            print(f"Episode over: reward is {cumulative_reward}")
            state, info = wrapped_env.reset()
        
    wrapped_env.close()

    
    
if __name__ == "__main__":
    main()