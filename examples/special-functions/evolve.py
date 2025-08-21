"""
Special Functions experiment using either a standard feedforward neural network
or a Kolmogorov-Arnold Network (KAN) with NEAT evolution.
"""
import os
import pickle
import time
import argparse
import random
import numpy as np
import sys

import special_functions
import neat
import visualize
import results_manager

try:
    from neat.nn.kan import KANNetwork
    from neat.kan_genome import KANGenome
    KAN_AVAILABLE = True
    print(f"KAN available: {KAN_AVAILABLE}")
except ImportError:
    KAN_AVAILABLE = False
    print(f"KAN available: {KAN_AVAILABLE}")

# Global variables for the current function being tested
current_function = None
test_data = None
fitness_cache = {}

def set_target_function(function_name: str, n_samples: int = 1000, seed: int = 42):
    """Set the target function for evolution."""
    global current_function, test_data, fitness_cache
    current_function = special_functions.get_function(function_name)
    
    # Generate test data for fitness evaluation
    X_train, y_train, X_test, y_test = current_function.generate_data(
        n_samples=n_samples, test_fraction=0.2, seed=seed
    )
    
    # Normalize the target values to make fitness evaluation more stable
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std > 0:
        y_train_norm = (y_train - y_mean) / y_std
        y_test_norm = (y_test - y_mean) / y_std
    else:
        y_train_norm = y_train
        y_test_norm = y_test
    
    test_data = {
        'X_train': X_train,
        'y_train': y_train_norm,
        'X_test': X_test,
        'y_test': y_test_norm,
        'y_mean': y_mean,
        'y_std': y_std,
        'function_name': function_name
    }
    
    # Clear fitness cache when changing functions
    fitness_cache.clear()
    
    print(f"Target function set to: {function_name}")
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    print(f"Target range: [{np.min(y_train):.3f}, {np.max(y_train):.3f}]")

def eval_feedforward_genome(genome, config):
    """Evaluate a feedforward genome."""
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    return _calculate_fitness(net, genome.key)

def eval_kan_genome(genome, config):
    """Evaluate a KAN genome."""
    net = KANNetwork.create(genome, config)
    return _calculate_fitness(net, genome.key)

def _calculate_fitness(net, genome_key):
    """Calculate fitness based on function approximation accuracy."""
    global test_data, fitness_cache
    
    if test_data is None:
        raise ValueError("No target function set. Call set_target_function first.")
    
    # Check cache first
    if genome_key in fitness_cache:
        return fitness_cache[genome_key]
    
    try:
        X_train = test_data['X_train']
        y_train = test_data['y_train']
        
        # Evaluate network on training data
        predictions = []
        for i in range(len(X_train)):
            inputs = X_train[i]
            try:
                output = net.activate(inputs)
                if isinstance(output, (list, tuple)):
                    output = output[0]
                
                # Check for numerical issues
                if np.isfinite(output):
                    predictions.append(float(output))
                else:
                    predictions.append(0.0)  # Replace inf/nan with 0
            except Exception as e:
                # Handle any network evaluation errors
                predictions.append(0.0)
        
        predictions = np.array(predictions)
        
        # Additional check for extreme values that could cause overflow in MSE
        predictions = np.clip(predictions, -1000, 1000)  # Clamp predictions
        
        # Calculate RMSE
        mse = np.mean((predictions - y_train) ** 2)
        rmse = np.sqrt(mse)
        
        # Use RMSE directly as fitness (lower RMSE = better fitness)
        # Since NEAT minimizes fitness when fitness_criterion = min
        fitness = rmse
        
        # Penalize for NaN or infinite predictions or extremely high values
        if not np.isfinite(fitness) or fitness > 10000:
            fitness = 10000.0  # Large but bounded penalty
        
        # Cache the result
        fitness_cache[genome_key] = fitness
        
        return fitness
        
    except Exception as e:
        print(f"Error in fitness evaluation: {e}")
        return 10000.0

class GenerationReporter(neat.reporting.BaseReporter):
    def __init__(self, generation_data):
        self.generation_data = generation_data

    def end_generation(self, config, population, species_set):
        best = None
        for genome_id, genome in population.items():
            if genome.fitness is not None:
                # Since we're minimizing RMSE, lower fitness is better
                if best is None or genome.fitness < best.fitness:
                    best = genome
        
        avg_fitness = sum(g.fitness for g in population.values() if g.fitness is not None)
        count = sum(1 for g in population.values() if g.fitness is not None)
        avg_fitness = avg_fitness / count if count > 0 else 1000.0
        
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

def run(function_name='jv', net_type='feedforward', seed=None, num_generations=100, 
        config_file=None, results_file=None, skip_plots=False, n_samples=1000):
    """
    Run evolution with the specified network type on a special function.
    
    Args:
        function_name: Name of the special function to approximate
        net_type: Type of network to use ('feedforward' or 'kan')
        seed: Random seed for reproducibility (default: None)
        num_generations: Number of generations to run (default: 100)
        config_file: Optional path to custom config file
        results_file: Optional path to save generation results as CSV
        skip_plots: Skip generating plots to save time/space (default: False)
        n_samples: Number of samples for training data (default: 1000)
        
    Returns:
        The winning genome
    """
    # Track per-generation data if results_file is provided
    generation_data = []

    # Validate network type
    if net_type.lower() == 'kan' and not KAN_AVAILABLE:
        raise ImportError("KAN network type requested but KAN modules not available")
    
    # Set the target function
    set_target_function(function_name, n_samples=n_samples, seed=seed)
    
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
    
    # Set up results directory
    experiment_name = f"{function_name}-{net_type.lower()}"
    results_dir, run_id, is_new_run = results_manager.setup_results_directory(
        config, experiment_name=experiment_name, seed=seed)
    print(f"Results will be saved to: {results_dir}")
    
    # Create stats reporter
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
                try:
                    with open(stats_path, 'rb') as f:
                        stats = pickle.load(f)
                    print("Loaded existing statistics")
                except Exception as e:
                    print(f"Could not load statistics: {e}")
            
            if not skip_plots:
                visualize_results(winner, stats, config, results_dir, net_type, function_name)
            else:
                print("Skipping plot generation (--skip-plots is set)")
                
            return winner
    
    # Save the seed and function info
    with open(os.path.join(results_dir, 'experiment_info.txt'), 'w') as f:
        f.write(f"Function: {function_name}\n")
        f.write(f"Seed: {seed}\n")
        f.write(f"Network type: {net_type}\n")
        f.write(f"Samples: {n_samples}\n")
        f.write(f"Description: {current_function.description}\n")
        f.write(f"Domain X: {current_function.domain_x}\n")
        f.write(f"Domain Y: {current_function.domain_y}\n")

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

    # Run the evolution (sequential evaluation)
    def evaluate_genomes(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = eval_function(genome, config)
    
    winner = pop.run(evaluate_genomes, num_generations)
    
    # Record elapsed time
    elapsed = time.time() - start_time
    print(f"Total time elapsed: {elapsed:.2f} seconds")
    
    with open(os.path.join(results_dir, 'runtime.txt'), 'w') as f:
        f.write(f"Runtime: {elapsed:.2f} seconds\n")
        f.write(f"Generations: {pop.generation}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Network type: {net_type}\n")
        f.write(f"Function: {function_name}\n")

    # Save the winner
    with open(os.path.join(results_dir, f'winner-{net_type.lower()}.pkl'), 'wb') as f:
        pickle.dump(winner, f)
    
    # Save statistics
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
    
    print(f"Best fitness: {winner.fitness:.6f}")
    
    # Only generate visualizations if not skipping plots
    if not skip_plots:
        visualize_results(winner, stats, config, results_dir, net_type, function_name)
    else:
        print("Skipping plot generation (--skip-plots is set)")
    
    return winner

def visualize_results(winner, stats, config, results_dir, net_type='feedforward', function_name='jv'):
    """Generate visualizations and save them in the results directory."""
    print("Plotting statistics...")
    if stats.generation_statistics:
        stats_path = os.path.join(results_dir, "fitness.png")
        visualize.plot_stats(stats, ylog=False, view=False, filename=stats_path)
        
        print("Plotting species...")
        species_path = os.path.join(results_dir, "speciation.png")
        visualize.plot_species(stats, view=False, filename=species_path)
    else:
        print("No statistics available to plot")

    # Define node names for visualization
    node_names = {-1: 'x', -2: 'y', 0: 'output'}
    
    # Analyze the genome and save to file
    print("Analyzing genome...")
    analysis_path = os.path.join(results_dir, "genome_analysis.txt")
    with open(analysis_path, 'w') as f:
        original_stdout = sys.stdout
        sys.stdout = f
        
        visualize.analyze_genome(winner, config.genome_config, node_names, net_type=net_type)
        
        sys.stdout = original_stdout
    
    # Test the winner on the function
    test_winner_on_function(winner, config, results_dir, net_type, function_name)
    
    print(f"Results saved to {results_dir}")

def test_winner_on_function(winner, config, results_dir, net_type, function_name):
    """Test the winner genome on the target function and save results."""
    global test_data
    
    if test_data is None:
        print("No test data available")
        return
    
    # Create network from winner
    if net_type.lower() == 'feedforward':
        net = neat.nn.FeedForwardNetwork.create(winner, config)
    else:
        net = KANNetwork.create(winner, config)
    
    # Test on training and test data
    X_train = test_data['X_train']
    y_train = test_data['y_train']
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    # Get predictions
    train_pred = []
    for i in range(len(X_train)):
        try:
            output = net.activate(X_train[i])
            if isinstance(output, (list, tuple)):
                output = output[0]
            train_pred.append(float(output))
        except:
            train_pred.append(0.0)
    
    test_pred = []
    for i in range(len(X_test)):
        try:
            output = net.activate(X_test[i])
            if isinstance(output, (list, tuple)):
                output = output[0]
            test_pred.append(float(output))
        except:
            test_pred.append(0.0)
    
    train_pred = np.array(train_pred)
    test_pred = np.array(test_pred)
    
    # Calculate errors
    train_rmse = np.sqrt(np.mean((train_pred - y_train) ** 2))
    test_rmse = np.sqrt(np.mean((test_pred - y_test) ** 2))
    
    # Save results
    results_path = os.path.join(results_dir, "function_approximation_results.txt")
    with open(results_path, 'w') as f:
        f.write(f"Function Approximation Results\n")
        f.write(f"==============================\n")
        f.write(f"Function: {function_name}\n")
        f.write(f"Network type: {net_type}\n")
        f.write(f"Winner fitness: {winner.fitness:.6f}\n")
        f.write(f"Training RMSE: {train_rmse:.6f}\n")
        f.write(f"Test RMSE: {test_rmse:.6f}\n")
        f.write(f"Network complexity: {len(winner.connections)} connections, ")
        f.write(f"{len([n for n in winner.nodes if n not in config.genome_config.input_keys and n not in config.genome_config.output_keys])} hidden nodes\n")
    
    print(f"Function approximation results saved to {results_path}")
    print(f"Training RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}")

if __name__ == '__main__':
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description='Run special function approximation evolution')
    parser.add_argument('--function', type=str, default='jv', 
                      choices=special_functions.list_functions(),
                      help='Special function to approximate')
    parser.add_argument('--net-type', type=str, default='feedforward', 
                      choices=['feedforward', 'kan'],
                      help='Type of network to use (feedforward or kan)')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--generations', type=int, default=100, 
                      help='Number of generations to run')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--results-file', type=str, help='CSV file to save generation results')
    parser.add_argument('--skip-plots', action='store_true', help='Skip generating plots to save time/space')
    parser.add_argument('--samples', type=int, default=1000, help='Number of training samples')
    
    args = parser.parse_args()
    
    run(function_name=args.function, net_type=args.net_type, seed=args.seed, 
        num_generations=args.generations, config_file=args.config, 
        results_file=args.results_file, skip_plots=args.skip_plots,
        n_samples=args.samples)
