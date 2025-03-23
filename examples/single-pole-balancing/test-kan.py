"""
Test the performance of the best genome produced by evolve-kan.py.
"""

import os
import pickle
import traceback
import argparse

import neat
from neat.nn.kan import KANNetwork
from neat.kan_genome import KANGenome
from cart_pole import CartPole, discrete_actuator_force
from movie import make_movie
import visualize_kan
import results_manager

def load_genome(genome_path):
    """Load a genome from the specified path."""
    try:
        with open(genome_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading genome from {genome_path}: {e}")
        return None

def test_genome(genome_path=None, view=False):
    """Test a genome and visualize its performance."""
    # Load the config file
    local_dir = os.path.dirname(__file__)
    # take one dir higher
    # local_local_dir = os.path.dirname(local_dir)
    config_path = os.path.join(local_dir, 'config-kan')
    config = neat.Config(KANGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Determine which genome to load
    if genome_path is None:
        # Check if there's a results directory with a winner
        results_dir, run_id, _ = results_manager.setup_results_directory(config)
        genome_path = os.path.join(results_dir, 'winner-kan.pkl')
        
        if not os.path.exists(genome_path):
            # Fall back to the default location
            genome_path = 'winner-kan'
    
    # Load the winner
    c = load_genome(genome_path)
    if c is None:
        return
    
    print('Loaded genome:')
    print(c)
    
    local_results_path = os.path.dirname(genome_path)
    # Set up the results directory
    # results_dir, run_id, _ = results_manager.setup_results_directory(config)
    test_dir = os.path.join(local_results_path, 'test_results')
    os.makedirs(test_dir, exist_ok=True)
    
    # Create the network and simulator
    net = KANNetwork.create(c, config)
    sim = CartPole()
    
    # Print detailed analysis of the genome
    # analysis_path = os.path.join(test_dir, 'analysis.txt')
    # with open(analysis_path, 'w') as f:
    #     # Redirect stdout to the file
    #     import sys
    #     original_stdout = sys.stdout
    #     sys.stdout = f
        
    #     visualize_kan.analyze_kan_genome(c, config.genome_config)
        
    #     # Restore stdout
    #     sys.stdout = original_stdout
    
    # Plot spline visualizations
    # try:
    #     print("\nPlotting splines...")
    #     splines_path = os.path.join(test_dir, 'splines.png')
    #     visualize_kan.plot_kan_splines(c, config.genome_config, 
    #                                   filename=splines_path, view=view)
    #     print("Splines plotted successfully.")
    # except Exception as e:
    #     print(f"Error plotting splines: {str(e)}")
    #     traceback.print_exc()
    
    # Run the simulation
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
    try:
        print("\nCreating movie... (this may take a while)")
        movie_path = os.path.join(test_dir, 'simulation.mp4')
        make_movie(net, discrete_actuator_force, min(15.0, balance_time + 1.0), 
                  movie_path)
        print(f"Movie created successfully and saved to {movie_path}")
    except Exception as e:
        print(f"Error creating movie: {str(e)}")
        traceback.print_exc()
    
    print(f"\nAll test results saved to {test_dir}")
    
    return balance_time

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test a trained KAN on the pole balancing task')
    parser.add_argument('--genome', type=str, help='Path to the genome file')
    parser.add_argument('--view', action='store_true', help='Show plots during execution')
    
    args = parser.parse_args()
    
    test_genome(genome_path=args.genome, view=args.view)