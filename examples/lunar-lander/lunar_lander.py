"""
Module with utility functions for the LunarLander-v2 environment.
"""
import gym
import numpy as np

def create_env():
    """Create and return the LunarLander environment"""
    return gym.make("LunarLander-v2")

def get_action(net, observation):
    """
    Get the action from a neural network.
    
    Args:
        net: Neural network (either NEAT or KAN)
        observation: Environment observation
        
    Returns:
        int: Action index (0-3)
    """
    # Ensure observation is properly formatted
    if isinstance(observation, tuple):
        observation = observation[0]
    
    # Run the network to get action
    outputs = net.activate(observation)
    return np.argmax(outputs)

def run_simulation(net, env=None, render=False, max_steps=1000):
    """
    Run a simulation episode with the given network.
    
    Args:
        net: Neural network (either NEAT or KAN)
        env: Environment instance (will create if None)
        render: Whether to render the simulation (default: False)
        max_steps: Maximum number of steps (default: 1000)
        
    Returns:
        float: Total reward achieved
    """
    # Create environment if not provided
    if env is None:
        env = create_env()
        local_env = True
    else:
        local_env = False
    
    # Reset the environment
    observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]
    
    total_reward = 0.0
    done = False
    steps = 0
    
    # Run the episode
    while not done and steps < max_steps:
        # Get action from network
        action = get_action(net, observation)
        
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
        
        # Render if requested
        if render:
            env.render()
        
        # Update reward
        total_reward += reward
        steps += 1
    
    # Clean up if we created the environment
    if local_env:
        env.close()
    
    return total_reward