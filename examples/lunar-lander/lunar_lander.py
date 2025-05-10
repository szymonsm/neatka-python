import gym
import numpy as np
import random

def create_env(seed=None):
    """Create and return the LunarLander environment with an optional seed."""
    env = gym.make("LunarLander-v2")
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env

def get_action(net, observation):
    """
    Get the action from a neural network.
    """
    if isinstance(observation, tuple):
        observation = observation[0]
    outputs = net.activate(observation)
    return np.argmax(outputs)

def run_simulation(net, env=None, render=False, max_steps=1000, seed=None):
    """
    Run a simulation episode with the given network and optional reproducibility seed.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    if env is None:
        env = create_env(seed=seed)
        local_env = True
    else:
        local_env = False
        if seed is not None:
            env.reset(seed=seed)

    observation = env.reset()[0] if isinstance(env.reset(), tuple) else env.reset()

    total_reward = 0.0
    done = False
    steps = 0

    while not done and steps < max_steps:
        action = get_action(net, observation)
        step_result = env.step(action)
        
        if len(step_result) == 4:
            observation, reward, done, info = step_result
        else:
            observation, reward, terminated, truncated, info = step_result
            done = terminated or truncated

        if render:
            env.render()

        total_reward += reward
        steps += 1

    if local_env:
        env.close()

    return total_reward
