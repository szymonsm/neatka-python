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

import neat
from cart_pole import CartPole, discrete_actuator_force
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
                create_movie=True, generate_plots=True, verbose=True):
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
    
    Returns:
        float: Time the pole was balanced
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
            config, experiment_name=net_type.lower(), seed=seed)
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
    
    # Create the simulator
    sim = CartPole()
    
    # Define node names for visualization
    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    
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
    
    # Run the simulation
    if verbose:
        print("\nInitial conditions:")
        print("        x = {0:.4f}".format(sim.x))
        print("    x_dot = {0:.4f}".format(sim.dx))
        print("    theta = {0:.4f}".format(sim.theta))
        print("theta_dot = {0:.4f}".format(sim.dtheta))
        print()
    
    # Run the simulation for up to 120 seconds.
    balance_time = 0.0
    sim_data = []  # Store simulation data for plotting
    
    while sim.t < 120.0:
        inputs = sim.get_scaled_state()
        action = net.activate(inputs)
        force = discrete_actuator_force(action)
        
        # Store the current state and action
        sim_data.append((sim.t, sim.x, sim.dx, sim.theta, sim.dtheta, force))
        
        # Update the simulation
        sim.step(force)
        
        if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
            break
        
        balance_time = sim.t
    
    if verbose:
        print('Pole balanced for {0:.1f} seconds'.format(balance_time))
        
        print("\nFinal conditions:")
        print("        x = {0:.4f}".format(sim.x))
        print("    x_dot = {0:.4f}".format(sim.dx))
        print("    theta = {0:.4f}".format(sim.theta))
        print("theta_dot = {0:.4f}".format(sim.dtheta))
        print()
    
    # Save simulation results
    results_path = os.path.join(test_dir, 'simulation_results.txt')
    with open(results_path, 'w') as f:
        f.write(f"Balance time: {balance_time:.1f} seconds\n\n")
        f.write("Initial conditions:\n")
        f.write(f"        x = {sim_data[0][1]:.4f}\n")
        f.write(f"    x_dot = {sim_data[0][2]:.4f}\n")
        f.write(f"    theta = {sim_data[0][3]:.4f}\n")
        f.write(f"theta_dot = {sim_data[0][4]:.4f}\n\n")
        
        f.write("Final conditions:\n")
        f.write(f"        x = {sim.x:.4f}\n")
        f.write(f"    x_dot = {sim.dx:.4f}\n")
        f.write(f"    theta = {sim.theta:.4f}\n")
        f.write(f"theta_dot = {sim.dtheta:.4f}\n")
    
    if generate_plots:
        # Plot the simulation trajectory
        import matplotlib.pyplot as plt
        
        # Create trajectory plots
        fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
        
        # Extract data
        times = [d[0] for d in sim_data]
        x_vals = [d[1] for d in sim_data]
        dx_vals = [d[2] for d in sim_data]
        theta_vals = [d[3] for d in sim_data]
        dtheta_vals = [d[4] for d in sim_data]
        force_vals = [d[5] for d in sim_data]
        
        # Plot each variable
        axs[0].plot(times, x_vals)
        axs[0].set_ylabel('Position (x)')
        axs[0].grid(True)
        
        axs[1].plot(times, dx_vals)
        axs[1].set_ylabel('Velocity (dx)')
        axs[1].grid(True)
        
        axs[2].plot(times, theta_vals)
        axs[2].set_ylabel('Angle (theta)')
        axs[2].grid(True)
        
        axs[3].plot(times, dtheta_vals)
        axs[3].set_ylabel('Angular velocity')
        axs[3].grid(True)
        
        axs[4].plot(times, force_vals, 'g-', drawstyle='steps-post')
        axs[4].set_ylabel('Force')
        axs[4].set_xlabel('Time (s)')
        axs[4].grid(True)
        
        plt.tight_layout()
        trajectory_path = os.path.join(test_dir, 'trajectory.png')
        plt.savefig(trajectory_path)
        if view:
            plt.show()
        else:
            plt.close()
    
    # Create a movie of the simulation
    if create_movie:
        try:
            if verbose:
                print("\nCreating movie... (this may take a while)")
            movie_path = os.path.join(test_dir, 'simulation.mp4')
            fps = 50  # Higher FPS for smoother video
            make_movie(net, discrete_actuator_force, min(15.0, balance_time + 1.0), 
                    movie_path)
            if verbose:
                print(f"Movie created successfully and saved to {movie_path}")
        except Exception as e:
            print(f"Error creating movie: {str(e)}")
            if verbose:
                traceback.print_exc()
    
    if verbose:
        print(f"\nAll test results saved to {test_dir}")
    
    return balance_time

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test a trained network on the pole balancing task')
    parser.add_argument('--genome', type=str, help='Path to the genome file')
    parser.add_argument('--view', action='store_true', help='Show plots during execution')
    parser.add_argument('--seed', type=int, default=999, help='Random seed for reproducibility')
    parser.add_argument('--net-type', type=str, default='feedforward', 
                      choices=['feedforward', 'kan'],
                      help='Type of network to test (feedforward or kan)')
    parser.add_argument('--no-movie', action='store_true', help='Skip creating simulation movie')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    
    args = parser.parse_args()
    
    test_genome(
        genome_path=args.genome, 
        view=args.view, 
        seed=args.seed,
        net_type=args.net_type,
        create_movie=not args.no_movie,
        generate_plots=not args.no_plots,
        verbose=not args.quiet
    )
