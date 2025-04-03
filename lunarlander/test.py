import gymnasium as gym

# Create the Lunar Lander environment with rendering enabled
# Use 'human' mode for displaying the graphics
env = gym.make("LunarLander-v2", render_mode="human")

# Run a few episodes with random actions
for episode in range(5):
    # Use a fixed seed for reproducibility
    observation, info = env.reset(seed=42)
    terminated = False
    truncated = False
    total_reward = 0
    step = 0

    while not terminated and not truncated:
        # Render the environment. The window might open automatically.
        env.render()

        # Choose a random action from the environment's action space
        action = env.action_space.sample()

        # Take the action and observe the result
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step += 1

        # Optional: Add a small delay to make it easier to watch
        # time.sleep(0.01)

        if terminated or truncated:
            print(f"Episode {episode+1} finished after {step} steps. Total Reward: {total_reward}")

# Close the environment and the rendering window
env.close()

print("\nTo run this, you need to install gymnasium and the Box2D dependency:")
print("pip install gymnasium[box2d]")
