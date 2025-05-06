import neat
import numpy as np
import gym

import run_neat_base


def eval_network(net, net_input):
    activation = net.activate(net_input)
    return np.argmax(activation)


def eval_single_genome(genome, genome_config, environment_name="LunarLander-v2"):
    # Create a local environment instance for this worker process
    local_env = gym.make(environment_name)
    
    net = neat.nn.FeedForwardNetwork.create(genome, genome_config)
    total_reward = 0.0

    for i in range(run_neat_base.n):
        # Handle different gym API versions
        observation = local_env.reset()
        
        # If observation is a tuple (newer gym versions), get the first element
        if isinstance(observation, tuple):
            observation = observation[0]
        
        # Print observation shape for debugging
        # print(f"Observation shape: {observation.shape}, values: {observation}")
            
        action = eval_network(net, observation)

        done = False
        t = 0

        while not done:
            # Step through the environment
            step_result = local_env.step(action)
            
            # Handle different gym API versions (step returns different format in different versions)
            if len(step_result) == 4:
                # Old gym API: observation, reward, done, info
                observation, reward, done, info = step_result
            else:
                # New gym API: observation, reward, terminated, truncated, info
                observation, reward, terminated, truncated, info = step_result
                done = terminated or truncated
                
            action = eval_network(net, observation)
            total_reward += reward
            t += 1

            if done:
                break

    local_env.close()  # Clean up the environment
    return total_reward / run_neat_base.n


def main():
    # Create a partial function that includes the environment name
    from functools import partial
    eval_function = partial(eval_single_genome, environment_name="LunarLander-v2")
    
    run_neat_base.run(eval_network,
                      eval_function,
                      environment_name="LunarLander-v2")


if __name__ == '__main__':
    main()