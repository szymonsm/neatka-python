"""
Single-pole balancing experiment using a Kolmogorov-Arnold Network (KAN).
"""

import multiprocessing
import os
import pickle
import time
import argparse
import random
import numpy as np

import cart_pole
import neat
from neat.nn.kan import KANNetwork
from neat.kan_genome import KANGenome
import visualize
import visualize_kan
import results_manager

runs_per_net = 5
simulation_seconds = 60.0

# Use the KAN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = KANNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        sim = cart_pole.CartPole()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.activate(inputs)

            # Apply action to the simulated cart-pole
            force = cart_pole.discrete_actuator_force(action)
            sim.step(force)

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.
            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break

            fitness = sim.t

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run(seed=None, num_generations=100):
    """Run the experiment with optional seed for reproducibility."""
    # Set seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    # Load the config file
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-kan')
    config = neat.Config(KANGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    
    # Set up results directory
    results_dir, run_id, is_new_run = results_manager.setup_results_directory(config, seed=seed)
    print(f"Results will be saved to: {results_dir}")
    
    # Create stats reporter - define it early so it can be used even for existing runs
    stats = neat.StatisticsReporter()
    
    if not is_new_run:
        print(f"Using existing run directory (config match found): {run_id}")
        
        # Check if the winner already exists
        winner_path = os.path.join(results_dir, 'winner-kan.pkl')
        if os.path.exists(winner_path):
            print(f"Winner already exists at {winner_path}")
            
            # Load the winner
            with open(winner_path, 'rb') as f:
                winner = pickle.load(f)
            
            # Check if statistics exist
            stats_path = os.path.join(results_dir, 'statistics.pkl')
            if os.path.exists(stats_path):
                # Load existing statistics
                try:
                    with open(stats_path, 'rb') as f:
                        stats = pickle.load(f)
                    print("Loaded existing statistics")
                except Exception as e:
                    print(f"Could not load statistics: {e}")
                    # Will use empty stats
            else:
                print("No statistics file found, visualizing with empty statistics")
            
            # Visualize the results
            visualize_results(winner, stats, config, results_dir)
            return winner
    
    # Save the seed used
    if seed is not None:
        with open(os.path.join(results_dir, 'seed.txt'), 'w') as f:
            f.write(f"Seed: {seed}\n")

    # Create population and add reporters
    pop = neat.Population(config)
    pop.add_reporter(stats)  # Use the already created stats reporter
    pop.add_reporter(neat.StdOutReporter(True))
    
    # Add a checkpoint reporter to save progress
    checkpoints_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoints_dir, 'neat-checkpoint-')
    pop.add_reporter(neat.Checkpointer(5, 900, checkpoint_prefix))
    
    # Record start time
    start_time = time.time()

    # Run the evolution
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate, num_generations)
    
    # Record elapsed time
    elapsed = time.time() - start_time
    print(f"Total time elapsed: {elapsed:.2f} seconds")
    
    with open(os.path.join(results_dir, 'runtime.txt'), 'w') as f:
        f.write(f"Runtime: {elapsed:.2f} seconds\n")
        f.write(f"Generations: {pop.generation}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Save the winner
    with open(os.path.join(results_dir, 'winner-kan.pkl'), 'wb') as f:
        pickle.dump(winner, f)
    
    # Save statistics for future use
    with open(os.path.join(results_dir, 'statistics.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    print(winner)
    
    # Visualize the results
    visualize_results(winner, stats, config, results_dir)
    
    return winner

def visualize_results(winner, stats, config, results_dir):
    """Generate all visualizations and save them in the results directory."""
    print("Plotting statistics...")
    if stats.generation_statistics:  # Check if we have valid statistics
        stats_path = os.path.join(results_dir, "fitness.svg")
        visualize.plot_stats(stats, ylog=True, view=False, filename=stats_path)
        
        print("Plotting species...")
        species_path = os.path.join(results_dir, "speciation.svg")
        visualize.plot_species(stats, view=False, filename=species_path)
    else:
        print("No statistics available to plot")

    # Define node names for better visualization
    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    
    # Use KAN-specific visualization functions
    print("Visualizing winner...")
    net_path = os.path.join(results_dir, "network")
    visualize.draw_kan_net(config, winner, view=False, node_names=node_names,
                        filename=net_path, fmt='png')
    
    print("Visualizing winner (pruned)...")
    pruned_path = os.path.join(results_dir, "network-pruned")
    visualize.draw_kan_net(config, winner, view=False, node_names=node_names,
                        filename=pruned_path, prune_unused=True, fmt='png')
    
    # Plot spline visualizations
    print("Plotting splines...")
    splines_path = os.path.join(results_dir, "splines.png")
    visualize_kan.plot_kan_splines(winner, config.genome_config, 
                                  filename=splines_path, view=False)
    
    # Plot KAN network with splines
    # print("Plotting KAN network with splines...")
    # kan_net_path = os.path.join(results_dir, "kan-network.svg")
    # visualize_kan.draw_kan_network_with_splines(winner, config, 
    #                                           filename=kan_net_path, view=False)
    
    # Analyze the genome and save to file
    print("Analyzing genome...")
    analysis_path = os.path.join(results_dir, "genome_analysis.txt")
    with open(analysis_path, 'w') as f:
        # Redirect stdout to the file
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        
        visualize_kan.analyze_kan_genome(winner, config.genome_config)
        
        # Restore stdout
        sys.stdout = original_stdout
    
    print(f"All results saved to {results_dir}")

if __name__ == '__main__':
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Run KAN pole balancing evolution')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--generations', type=int, default=100, 
                      help='Number of generations to run')
    
    args = parser.parse_args()
    
    run(seed=args.seed, num_generations=args.generations)