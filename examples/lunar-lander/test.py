"""
Test the performance of the best genome produced by evolve.py.
Supports both feedforward and KAN networks.
"""

import os
import pickle
import traceback
import argparse
import random
import numpy as np
import time
import gym

import neat
import lunar_lander
from movie import make_movie
import results_manager
import visualize

# Conditionally import KAN modules
try:
    from neat.nn.kan import KANNetwork
    from neat.kan_genome import KANGenome
    KAN_AVAILABLE = True
except ImportError:
    KAN_AVAILABLE = False

def load_genome(genome_path):
    """Load a genome from the specified path."""
    try:
        with open(genome_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading genome from {genome_path}: {e}")
        traceback.print_exc()
        return None

def test_genome(genome_path=None, view=False, seed=None, net_type='feedforward',
                create_movie=True, generate_plots=True, verbose=True, episodes=5):
    """
    Test a genome and visualize its performance.
    
    Args:
        genome_path: Path to the genome file (default: None, will search in results directory)
        view: Whether to display plots during execution (default: False)
        seed: Random seed for reproducibility (default: 999)
        net_type: Type of network to test ('feedforward' or 'kan')
        create_movie: Whether to create a simulation movie (default: True)
        generate_plots: Whether to generate visualization plots (default: True)
        verbose: Whether to print detailed information (default: True)
        episodes: Number of episodes to test (default: 5)
    
    Returns:
        float: Average reward across test episodes
    """
    # Validate network type
    if net_type.lower() == 'kan' and not KAN_AVAILABLE:
        raise ImportError("KAN network type requested but KAN modules not available")
        
    # Load the appropriate config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, f'config-{net_type.lower()}')
    
    # Use appropriate genome type based on network type
    if net_type.lower() == 'feedforward':
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    else:
        config = neat.Config(KANGenome, neat.DefaultReproduction,
                             neat.DefaultSpeciesSet, neat.DefaultStagnation,
                             config_path)
    
    # Determine which genome to load
    if genome_path is None:
        # Check if there's a results directory with a winner
        results_dir, run_id, _ = results_manager.setup_results_directory(
            config, experiment_name=f"lunar-lander-{net_type.lower()}", seed=seed)
        genome_path = os.path.join(results_dir, f'winner-{net_type.lower()}.pkl')
        
        if not os.path.exists(genome_path):
            # Fall back to the default location
            genome_path = f'winner-{net_type.lower()}'
    
    # Load the genome
    genome = load_genome(genome_path)
    if genome is None:
        return
    
    if verbose:
        print('Loaded genome:')
        print(genome)
    
    # Set up test results directory
    local_results_path = os.path.dirname(genome_path)
    test_dir = os.path.join(local_results_path, 'test_results')
    os.makedirs(test_dir, exist_ok=True)
    
    # Create the appropriate network type
    if net_type.lower() == 'feedforward':
        net = neat.nn.FeedForwardNetwork.create(genome, config)
    else:
        net = KANNetwork.create(genome, config)
    
    # Set the random seed for reproducibility
    if seed is None:
        seed = 999
    random.seed(seed)
    np.random.seed(seed)
    
    # Define node names for visualization
    node_names = {
        -1: 'x_pos', -2: 'y_pos', -3: 'x_vel', -4: 'y_vel', 
        -5: 'angle', -6: 'angular_vel', -7: 'leg1', -8: 'leg2', 
        0: 'nothing', 1: 'left', 2: 'main', 3: 'right'
    }
    
    # Print detailed analysis of the genome
    if generate_plots:
        analysis_path = os.path.join(test_dir, 'analysis.txt')
        with open(analysis_path, 'w') as f:
            # Redirect stdout to the file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            print(f"{net_type.capitalize()} Network Analysis:")
            visualize.analyze_genome(genome, config.genome_config, node_names, net_type)
            
            # Restore stdout
            sys.stdout = original_stdout
        
        # Generate network visualizations
        if verbose:
            print("Visualizing network...")
        net_path = os.path.join(test_dir, "network")
        visualize.draw_net(config, genome, view=view, node_names=node_names,
                          filename=net_path, fmt='png', net_type=net_type)
        
        if verbose:
            print("Visualizing network (pruned)...")
        pruned_path = os.path.join(test_dir, "network-pruned")
        visualize.draw_net(config, genome, view=view, node_names=node_names,
                          filename=pruned_path, prune_unused=True, fmt='png', net_type=net_type)
        
        # Create KAN-specific visualizations if applicable
        if net_type.lower() == 'kan':
            try:
                if verbose:
                    print("Plotting KAN splines...")
                splines_path = os.path.join(test_dir, 'splines.png')
                visualize.plot_kan_splines(genome, config.genome_config, 
                                        filename=splines_path, view=view)
                if verbose:
                    print("KAN splines plotted successfully.")
            except Exception as e:
                print(f"Error plotting KAN splines: {str(e)}")
                if verbose:
                    traceback.print_exc()
        
        # Try to load and visualize statistics if available
        stats_path = os.path.join(os.path.dirname(genome_path), 'statistics.pkl')
        if os.path.exists(stats_path):
            try:
                with open(stats_path, 'rb') as f:
                    stats = pickle.load(f)
                
                if verbose:
                    print("Plotting fitness statistics...")
                fitness_path = os.path.join(test_dir, 'fitness.png')
                visualize.plot_stats(stats, ylog=False, view=view, filename=fitness_path)
                
                if verbose:
                    print("Plotting species statistics...")
                species_path = os.path.join(test_dir, 'speciation.png')
                visualize.plot_species(stats, view=view, filename=species_path)
            except Exception as e:
                print(f"Error plotting statistics: {str(e)}")
                if verbose:
                    traceback.print_exc()
    
    # Run the simulation - test the agent
    rewards = []
    env = lunar_lander.create_env()
    
    # Store data for the best episode
    best_reward = float('-inf')
    best_episode = None
    
    for i in range(episodes):
        if verbose:
            print(f"\nRunning test episode {i+1}/{episodes}")
        
        # Record episode data for plotting
        states = []
        actions = []
        rewards_step = []
        
        # Reset the environment
        observation = env.reset()
        if isinstance(observation, tuple):
            observation = observation[0]
        
        states.append(observation)
        
        done = False
        total_reward = 0.0
        step = 0
        
        while not done:
            # Get action from network
            action = lunar_lander.get_action(net, observation)
            actions.append(action)
            
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
            
            states.append(observation)
            rewards_step.append(reward)
            
            total_reward += reward
            step += 1
            
            if verbose and step % 20 == 0:
                print(f"  Step {step}, Reward: {total_reward:.2f}")
            
            if done:
                if verbose:
                    print(f"Episode finished after {step} steps with reward {total_reward:.2f}")
                break
        
        rewards.append(total_reward)
        
        # Save the best episode data
        if total_reward > best_reward:
            best_reward = total_reward
            best_episode = {
                'states': np.array(states),
                'actions': np.array(actions),
                'rewards': np.array(rewards_step),
                'total_reward': total_reward,
                'steps': step
            }
    
    # Close the environment
    env.close()
    
    # Calculate statistics
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    
    if verbose:
        print("\nTest Results:")
        print(f"  Episodes: {episodes}")
        print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Best Reward: {best_reward:.2f}")
        print(f"  Rewards: {rewards}")
    
    # Save test results
    results_path = os.path.join(test_dir, 'test_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Test Results for {net_type.capitalize()} Network\n")
        f.write(f"Episodes: {episodes}\n")
        f.write(f"Average Reward: {avg_reward:.2f} ± {std_reward:.2f}\n")
        f.write(f"Best Reward: {best_reward:.2f}\n")
        f.write(f"All Rewards: {rewards}\n")
    
    # Generate plots for the best episode
    if generate_plots and best_episode is not None:
        try:
            import matplotlib.pyplot as plt
            
            # Create trajectory plots
            fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
            
            # Time steps
            steps = np.arange(len(best_episode['states']))
            
            # Plot states
            for i, label in enumerate(['x_pos', 'y_pos', 'x_vel', 'y_vel', 'angle', 'angular_vel', 'leg1', 'leg2']):
                if i < 4:
                    axs[0].plot(steps[:-1], best_episode['states'][:-1, i], label=label)
                else:
                    axs[1].plot(steps[:-1], best_episode['states'][:-1, i], label=label)
            
            axs[0].set_ylabel('Position/Velocity')
            axs[0].grid(True)
            axs[0].legend()
            
            axs[1].set_ylabel('Angle/Contacts')
            axs[1].grid(True)
            axs[1].legend()
            
            # Plot actions
            axs[2].plot(steps[:-1], best_episode['actions'], 'g-', drawstyle='steps-post')
            axs[2].set_ylabel('Action')
            axs[2].set_yticks([0, 1, 2, 3])
            axs[2].set_yticklabels(['Nothing', 'Left', 'Main', 'Right'])
            axs[2].grid(True)
            
            # Plot rewards
            axs[3].plot(steps[:-1], best_episode['rewards'], 'r-')
            axs[3].plot(steps[:-1], np.cumsum(best_episode['rewards']), 'b--', label='Cumulative')
            axs[3].set_ylabel('Reward')
            axs[3].set_xlabel('Time Steps')
            axs[3].grid(True)
            axs[3].legend()
            
            plt.tight_layout()
            trajectory_path = os.path.join(test_dir, 'best_episode.png')
            plt.savefig(trajectory_path)
            if view:
                plt.show()
            else:
                plt.close()
                
            if verbose:
                print(f"Best episode trajectory saved to {trajectory_path}")
                
        except Exception as e:
            print(f"Error generating trajectory plots: {e}")
            if verbose:
                traceback.print_exc()
    
    # Create a movie of the best performance
    if create_movie:
        try:
            if verbose:
                print("\nCreating movie... (this may take a while)")
            movie_path = os.path.join(test_dir, 'simulation.mp4')
            make_movie(net, movie_path)
            if verbose:
                print(f"Movie created successfully and saved to {movie_path}")
        except Exception as e:
            print(f"Error creating movie: {str(e)}")
            if verbose:
                traceback.print_exc()
    
    if verbose:
        print(f"\nAll test results saved to {test_dir}")
    
    return avg_reward

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test a trained network on the lunar lander task')
    parser.add_argument('--genome', type=str, help='Path to the genome file')
    parser.add_argument('--view', action='store_true', help='Show plots during execution')
    parser.add_argument('--seed', type=int, default=999, help='Random seed for reproducibility')
    parser.add_argument('--net-type', type=str, default='feedforward', 
                      choices=['feedforward', 'kan'],
                      help='Type of network to test (feedforward or kan)')
    parser.add_argument('--no-movie', action='store_true', help='Skip creating simulation movie')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes to test')
    
    args = parser.parse_args()
    
    test_genome(
        genome_path=args.genome, 
        view=args.view, 
        seed=args.seed,
        net_type=args.net_type,
        create_movie=not args.no_movie,
        generate_plots=not args.no_plots,
        verbose=not args.quiet,
        episodes=args.episodes
    )