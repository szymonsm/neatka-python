"""
LunarLander-v2 experiment using either a standard feedforward neural network
or a Kolmogorov-Arnold Network (KAN).
"""
import multiprocessing
import os
import pickle
import time
import argparse
import random
import numpy as np
import sys

import lunar_lander
import neat
import visualize
import results_manager

# Import KAN-specific modules conditionally
try:
    from neat.nn.kan import KANNetwork
    from neat.kan_genome import KANGenome
    KAN_AVAILABLE = True
    print(f"KAN available: {KAN_AVAILABLE}")
except ImportError:
    KAN_AVAILABLE = False
    print(f"KAN available: {KAN_AVAILABLE}")

runs_per_net = 3

# Define separate evaluation functions for each network type
def eval_feedforward_genome(genome, config):
    """Evaluate a feedforward genome."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return _run_simulation(net)

def eval_kan_genome(genome, config):
    """Evaluate a KAN genome."""
    net = KANNetwork.create(genome, config)
    return _run_simulation(net)

def _run_simulation(net):
    """Run the lunar lander simulation with the given network."""
    fitnesses = []

    for s in range(runs_per_net):
        # Create a fresh environment for each run
        env = lunar_lander.create_env(seed=s)
        fitness = lunar_lander.run_simulation(net, env, seed=s)
        env.close()
        fitnesses.append(fitness)

    # The genome's fitness is its average performance across all runs
    return sum(fitnesses) / len(fitnesses)

# Move GenerationReporter to module level to make it picklable
class GenerationReporter(neat.reporting.BaseReporter):
    def __init__(self, generation_data):
        self.generation_data = generation_data

    def end_generation(self, config, population, species_set):
        best = None
        for genome_id, genome in population.items():
            if genome.fitness is not None:
                if best is None or genome.fitness > best.fitness:
                    best = genome
        
        avg_fitness = sum(g.fitness for g in population.values() if g.fitness is not None)
        count = sum(1 for g in population.values() if g.fitness is not None)
        avg_fitness = avg_fitness / count if count > 0 else 0
        
        self.generation_data.append({
            'generation': len(self.generation_data),
            'best_fitness': best.fitness if best is not None else 0,
            'avg_fitness': avg_fitness,
            'species_count': len(species_set.species),
            'complexity': len(best.connections) if best is not None else 0,
            'hidden_nodes': len([n for n in best.nodes 
                              if n not in config.genome_config.input_keys and 
                                 n not in config.genome_config.output_keys]) if best is not None else 0
        })

def run(net_type='feedforward', seed=None, num_generations=100, config_file=None, 
        results_file=None, skip_plots=False):
    """
    Run evolution with the specified network type.
    
    Args:
        net_type: Type of network to use ('feedforward' or 'kan')
        seed: Random seed for reproducibility (default: None)
        num_generations: Number of generations to run (default: 100)
        config_file: Optional path to custom config file
        results_file: Optional path to save generation results as CSV
        skip_plots: Skip generating plots to save time/space (default: False)
        
    Returns:
        The winning genome
    """
    # Track per-generation data if results_file is provided
    generation_data = []

    # Validate network type
    if net_type.lower() == 'kan' and not KAN_AVAILABLE:
        raise ImportError("KAN network type requested but KAN modules not available")
    
    # Set seed if provided
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
    # Load the appropriate config file
    local_dir = os.path.dirname(__file__)
    if config_file:
        config_path = config_file
    else:
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
    
    # Set up results directory - use network type as experiment name
    results_dir, run_id, is_new_run = results_manager.setup_results_directory(
        config, experiment_name=net_type.lower(), seed=seed)
    print(f"Results will be saved to: {results_dir}")
    
    # Create stats reporter - define it early so it can be used even for existing runs
    stats = neat.StatisticsReporter()
    
    if not is_new_run:
        print(f"Using existing run directory (config match found): {run_id}")
        
        # Check if the winner already exists
        winner_path = os.path.join(results_dir, f'winner-{net_type.lower()}.pkl')
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
            
            # Only generate visualizations if not skipping plots
            if not skip_plots:
                visualize_results(winner, stats, config, results_dir, net_type)
            else:
                print("Skipping plot generation (--skip-plots is set)")
                
            return winner
    
    # Save the seed used
    if seed is not None:
        with open(os.path.join(results_dir, 'seed.txt'), 'w') as f:
            f.write(f"Seed: {seed}\n")
            f.write(f"Network type: {net_type}\n")

    # Create population and add reporters
    pop = neat.Population(config)
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    
    # Add a reporter to collect per-generation stats
    if results_file:
        gen_reporter = GenerationReporter(generation_data)
        pop.add_reporter(gen_reporter)
    
    # Add a checkpoint reporter to save progress
    checkpoints_dir = os.path.join(results_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    checkpoint_prefix = os.path.join(checkpoints_dir, 'neat-checkpoint-')
    pop.add_reporter(neat.Checkpointer(5, 900, checkpoint_prefix))
    
    # Record start time
    start_time = time.time()

    # Choose the appropriate evaluation function
    if net_type.lower() == 'feedforward':
        eval_function = eval_feedforward_genome
    else:
        eval_function = eval_kan_genome

    # Run the evolution with the selected evaluation function
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_function)
    winner = pop.run(pe.evaluate, num_generations)
    
    # Record elapsed time
    elapsed = time.time() - start_time
    print(f"Total time elapsed: {elapsed:.2f} seconds")
    
    with open(os.path.join(results_dir, 'runtime.txt'), 'w') as f:
        f.write(f"Runtime: {elapsed:.2f} seconds\n")
        f.write(f"Generations: {pop.generation}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Network type: {net_type}\n")

    # Save the winner
    with open(os.path.join(results_dir, f'winner-{net_type.lower()}.pkl'), 'wb') as f:
        pickle.dump(winner, f)
    
    # Save statistics for future use
    with open(os.path.join(results_dir, 'statistics.pkl'), 'wb') as f:
        pickle.dump(stats, f)
    
    # Save generation data to CSV if requested
    if results_file:
        try:
            import pandas as pd
            results_df = pd.DataFrame(generation_data)
            results_df.to_csv(results_file, index=False)
            print(f"Generation results saved to {results_file}")
        except Exception as e:
            print(f"Error saving results to CSV: {e}")
    
    print(winner)
    
    # Only generate visualizations if not skipping plots
    if not skip_plots:
        visualize_results(winner, stats, config, results_dir, net_type)
    else:
        print("Skipping plot generation (--skip-plots is set)")
    
    return winner

def visualize_results(winner, stats, config, results_dir, net_type='feedforward'):
    """
    Generate minimal visualizations and save them in the results directory.
    For full visualization, use test.py.
    
    Args:
        winner: The winning genome
        stats: NEAT statistics reporter
        config: The experiment configuration
        results_dir: Directory to save results
        net_type: Type of network ('feedforward' or 'kan')
    """
    print("Plotting statistics...")
    if stats.generation_statistics:  # Check if we have valid statistics
        stats_path = os.path.join(results_dir, "fitness.png")
        visualize.plot_stats(stats, ylog=False, view=False, filename=stats_path)
        
        print("Plotting species...")
        species_path = os.path.join(results_dir, "speciation.png")
        visualize.plot_species(stats, view=False, filename=species_path)
    else:
        print("No statistics available to plot")

    # Define node names for better visualization
    node_names = {
        -1: 'x_pos', -2: 'y_pos', -3: 'x_vel', -4: 'y_vel', 
        -5: 'angle', -6: 'angular_vel', -7: 'leg1', -8: 'leg2', 
        0: 'nothing', 1: 'left', 2: 'main', 3: 'right'
    }
    
    # Analyze the genome and save to file (minimal but useful text output)
    print("Analyzing genome...")
    analysis_path = os.path.join(results_dir, "genome_analysis.txt")
    with open(analysis_path, 'w') as f:
        # Redirect stdout to the file
        original_stdout = sys.stdout
        sys.stdout = f
        
        visualize.analyze_genome(winner, config.genome_config, node_names, net_type=net_type)
        
        # Restore stdout
        sys.stdout = original_stdout
    
    print(f"Basic results saved to {results_dir}")
    print(f"Run 'python test.py --genome {os.path.join(results_dir, f'winner-{net_type.lower()}.pkl')} --net-type {net_type}' for full visualization and analysis")

if __name__ == '__main__':
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Run lunar lander evolution')
    parser.add_argument('--net-type', type=str, default='feedforward', 
                      choices=['feedforward', 'kan'],
                      help='Type of network to use (feedforward or kan)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--generations', type=int, default=100, 
                      help='Number of generations to run')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--results-file', type=str, help='CSV file to save generation results')
    parser.add_argument('--skip-plots', action='store_true', help='Skip generating plots to save time/space')
    
    args = parser.parse_args()
    
    run(net_type=args.net_type, seed=args.seed, num_generations=args.generations,
        config_file=args.config, results_file=args.results_file, skip_plots=args.skip_plots)