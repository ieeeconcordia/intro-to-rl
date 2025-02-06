import gymnasium as gym

env = gym.make(
    "LunarLander-v3", 
    render_mode="rgb_array"
    )
observation, info = env.reset()
cumulative_reward = 0
episode_over = False
PRINT_FREQUENCY = 10
NUM_STEP = 1000


def simple_policy():
    pass 

for step in range(NUM_STEP):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    cumulative_reward += reward

    if step % PRINT_FREQUENCY == 0:
        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
        print(f"Position (x,y): {observation[0]:.2f}, {observation[1]:.2f}")
        print(f"Velocity (x,y): {observation[2]:.2f}, {observation[3]:.2f}")
        print(f"Angle: {observation[4]:.2f}, Angular velocity: {observation[5]:.2f}")
        print(f"Contact with the ground {observation[6]}, {observation[7]}")
        print("\n+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")

    episode_over = terminated or truncated
    if episode_over :
        print("Episode over ")
        print(f"Cumulative reward: ",cumulative_reward)
        observation , info = env.reset()

print(f"Final reward is {cumulative_reward}")
env.close()