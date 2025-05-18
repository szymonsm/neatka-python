import os
import imageio
import numpy as np
import lunar_lander
import gym

def make_movie(net, output_filename, fps=30, max_steps=1000, seed=None):
    """
    Create a reproducible movie of the LunarLander agent controlled by the given network.

    Args:
        net: Neural network (either NEAT or KAN)
        output_filename: Path to save the movie
        fps: Frames per second
        max_steps: Maximum number of steps
        seed: Random seed for reproducibility

    Returns:
        float: Total reward achieved
    """
    # Create the LunarLander environment with render_mode=rgb_array for video capturing
    env = gym.make("LunarLander-v2", render_mode="rgb_array")

    # Seed environment for reproducibility
    if seed is not None:
        reset_result = env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
        np.random.seed(seed)
        observation = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    else:
        observation = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()

    # Initialize variables
    frames = []
    total_reward = 0.0
    done = False
    steps = 0

    # Capture frames of the episode
    while not done and steps < max_steps:
        frame = env.render()
        frames.append(frame)

        # Get action from network
        action = lunar_lander.get_action(net, observation)

        # Step the environment
        step_result = env.step(action)
        if len(step_result) == 4:
            observation, reward, done, info = step_result
        else:
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        total_reward += reward
        steps += 1

    env.close()

    # Save the frames as a movie
    if frames:
        os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
        print(f"Writing {len(frames)} frames to {output_filename}...")
        imageio.mimsave(output_filename, frames, fps=fps)
        print(f"Video saved with total reward: {total_reward:.2f}")
    else:
        print("No frames captured - unable to create movie")

    return total_reward