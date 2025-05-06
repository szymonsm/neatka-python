"""
Module for creating movies of LunarLander agents in action.
"""
import os
import imageio
import datetime
import numpy as np
import lunar_lander
import gym

def make_movie(net, output_filename, fps=30, max_steps=1000):
    """
    Create a movie of the LunarLander agent controlled by the given network.
    
    Args:
        net: Neural network (either NEAT or KAN)
        output_filename: Path to save the movie
        fps: Frames per second (default: 30)
        max_steps: Maximum number of steps (default: 1000)
        
    Returns:
        float: Total reward achieved
    """
    # Create the LunarLander environment with render_mode=rgb_array for video capturing
    env = gym.make("LunarLander-v2", render_mode="rgb_array")
    
    # Reset the environment
    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]
    
    # Initialize variables
    frames = []
    total_reward = 0.0
    done = False
    steps = 0
    
    # Capture frames of the episode
    while not done and steps < max_steps:
        # Render the environment and capture the frame
        frame = env.render()
        frames.append(frame)
        
        # Get action from network
        action = lunar_lander.get_action(net, observation)
        
        # Step the environment
        step_result = env.step(action)
        
        # Handle different gym API versions
        if len(step_result) == 4:
            # Old gym API: observation, reward, done, info
            observation, reward, done, info = step_result
        else:
            # New gym API: observation, reward, terminated, truncated, info
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated
        
        # Update reward and step count
        total_reward += reward
        steps += 1
    
    # Close the environment
    env.close()
    
    # Save the frames as a movie
    if frames:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_filename)), exist_ok=True)
        
        print(f"Writing {len(frames)} frames to {output_filename}...")
        imageio.mimsave(output_filename, frames, fps=fps)
        print(f"Video saved with total reward: {total_reward:.2f}")
    else:
        print("No frames captured - unable to create movie")
    
    return total_reward