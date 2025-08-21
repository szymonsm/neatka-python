"""
Test the performance of the best genome produced by evolve.py for special functions.
Supports both feedforward and KAN networks.
"""

import os
import pickle
import traceback
import argparse
import numpy as np
import json

import neat
import special_functions
import visualize

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

def calculate_test_rmse(genome, config, function_name, seed, net_type='kan'):
    """
    Calculate test RMSE for a genome on a specific function.
    
    Args:
        genome: The NEAT genome to evaluate
        config: NEAT configuration
        function_name: Name of the special function to test
        seed: Random seed used for data generation
        net_type: Type of network ('feedforward' or 'kan')
    
    Returns:
        dict: Results containing train_rmse, test_rmse, and other metrics
    """
    try:
        # Get the function and generate test data (same as training)
        func = special_functions.get_function(function_name)
        X_train, y_train, X_test, y_test = func.generate_data(
            n_samples=1000, test_fraction=0.2, seed=seed
        )
        
        # Normalize data (same as in training)
        y_mean = np.mean(y_train)
        y_std = np.std(y_train)
        if y_std > 0:
            y_train_norm = (y_train - y_mean) / y_std
            y_test_norm = (y_test - y_mean) / y_std
        else:
            y_train_norm = y_train
            y_test_norm = y_test
        
        # Create the appropriate network
        if net_type.lower() == 'feedforward':
            net = neat.nn.FeedForwardNetwork.create(genome, config)
        else:
            net = KANNetwork.create(genome, config)
        
        # Evaluate on training data
        train_predictions = []
        for x in X_train:
            output = net.activate(x)
            train_predictions.append(output[0])
        train_predictions = np.array(train_predictions)
        
        # Evaluate on test data
        test_predictions = []
        for x in X_test:
            output = net.activate(x)
            test_predictions.append(output[0])
        test_predictions = np.array(test_predictions)
        
        # Calculate RMSE on normalized data (same as fitness function)
        train_rmse = np.sqrt(np.mean((train_predictions - y_train_norm) ** 2))
        test_rmse = np.sqrt(np.mean((test_predictions - y_test_norm) ** 2))
        
        # Also calculate denormalized RMSE for interpretation
        if y_std > 0:
            train_pred_denorm = train_predictions * y_std + y_mean
            test_pred_denorm = test_predictions * y_std + y_mean
        else:
            train_pred_denorm = train_predictions
            test_pred_denorm = test_predictions
        
        train_rmse_denorm = np.sqrt(np.mean((train_pred_denorm - y_train) ** 2))
        test_rmse_denorm = np.sqrt(np.mean((test_pred_denorm - y_test) ** 2))
        
        return {
            'function_name': function_name,
            'seed': seed,
            'net_type': net_type,
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_rmse_denorm': float(train_rmse_denorm),
            'test_rmse_denorm': float(test_rmse_denorm),
            'y_mean': float(y_mean),
            'y_std': float(y_std),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        }
        
    except Exception as e:
        print(f"Error calculating RMSE: {e}")
        traceback.print_exc()
        return None

def test_genome(genome_path=None, view=False, seed=None, net_type='kan',
                function_name=None, generate_plots=True, verbose=True):
    """
    Test a genome and visualize its performance on special functions.
    
    Args:
        genome_path: Path to the genome file
        view: Whether to display plots during execution (default: False)
        seed: Random seed for reproducibility (default: from config_summary.json)
        net_type: Type of network to test ('feedforward' or 'kan')
        function_name: Name of the special function (default: from config_summary.json)
        generate_plots: Whether to generate visualization plots (default: True)
        verbose: Whether to print detailed information (default: True)
    
    Returns:
        dict: Test results including RMSE values
    """
    # Validate network type
    if net_type.lower() == 'kan' and not KAN_AVAILABLE:
        raise ImportError("KAN network type requested but KAN modules not available")
    
    if genome_path is None:
        raise ValueError("genome_path must be specified")
    
    if not os.path.exists(genome_path):
        raise FileNotFoundError(f"Genome file not found: {genome_path}")
    
    # Try to load experiment configuration from config_summary.json
    config_summary_path = os.path.join(os.path.dirname(genome_path), 'config_summary.json')
    if os.path.exists(config_summary_path):
        with open(config_summary_path, 'r') as f:
            config_summary = json.load(f)
        
        if function_name is None:
            function_name = config_summary.get('function_name')
        if seed is None:
            seed = config_summary.get('seed')
        
        if verbose:
            print(f"Loaded experiment config: function={function_name}, seed={seed}")
    
    # If function_name is still None, try to extract it from the folder path
    if function_name is None:
        # Look for pattern like "results/ellipj-kan/" or "results/jv-kan/" in the path
        path_parts = os.path.normpath(genome_path).split(os.sep)
        for part in path_parts:
            if part.endswith('-kan') or part.endswith('-feedforward'):
                # Extract function name (remove -kan or -feedforward suffix)
                if part.endswith('-kan'):
                    function_name = part[:-4]  # Remove '-kan'
                else:
                    function_name = part[:-12]  # Remove '-feedforward'
                break
        
        if verbose and function_name:
            print(f"Extracted function name from path: {function_name}")
    
    if function_name is None:
        raise ValueError("function_name must be specified, available in config_summary.json, or extractable from folder path")
    if seed is None:
        raise ValueError("seed must be specified or available in config_summary.json")
    
    # Load the appropriate config file
    config_dir = os.path.dirname(genome_path)
    config_path = None
    
    # Look for config file in the same directory
    for config_file in os.listdir(config_dir):
        if config_file.startswith('config') and config_file.endswith('.cfg'):
            config_path = os.path.join(config_dir, config_file)
            break
    
    # If not found, use default config
    if config_path is None or not os.path.exists(config_path):
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
    
    # Load the genome
    genome = load_genome(genome_path)
    if genome is None:
        return None
    
    if verbose:
        print('Loaded genome:')
        print(genome)
    
    # Set up test results directory
    test_dir = os.path.join(os.path.dirname(genome_path), 'test_results')
    os.makedirs(test_dir, exist_ok=True)
    
    # Calculate test RMSE
    results = calculate_test_rmse(genome, config, function_name, seed, net_type)
    
    if results is None:
        return None
    
    if verbose:
        print(f"\nTest Results for {function_name}:")
        print(f"Train RMSE (normalized): {results['train_rmse']:.6f}")
        print(f"Test RMSE (normalized):  {results['test_rmse']:.6f}")
        print(f"Train RMSE (original):   {results['train_rmse_denorm']:.6f}")
        print(f"Test RMSE (original):    {results['test_rmse_denorm']:.6f}")
    
    # Save test RMSE results to JSON
    rmse_json_path = os.path.join(os.path.dirname(genome_path), 'test-rmse.json')
    with open(rmse_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"Test RMSE results saved to: {rmse_json_path}")
    
    if generate_plots:
        # Create the appropriate network for visualization
        if net_type.lower() == 'feedforward':
            net = neat.nn.FeedForwardNetwork.create(genome, config)
        else:
            net = KANNetwork.create(genome, config)
        
        # Define node names for special functions (2 inputs, 1 output)
        node_names = {-1: 'x1', -2: 'x2', 0: 'output'}
        
        # Print detailed analysis of the genome
        analysis_path = os.path.join(test_dir, 'analysis.txt')
        with open(analysis_path, 'w') as f:
            # Redirect stdout to the file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            print(f"{net_type.capitalize()} Network Analysis for {function_name}:")
            print(f"Seed: {seed}")
            print(f"Test RMSE: {results['test_rmse']:.6f}")
            print("")
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
        
        # Generate function approximation plot
        try:
            if verbose:
                print("Creating function approximation plot...")
            
            import matplotlib.pyplot as plt
            
            # Get function and test data
            func = special_functions.get_function(function_name)
            X_train, y_train, X_test, y_test = func.generate_data(
                n_samples=1000, test_fraction=0.2, seed=seed
            )
            
            # Generate predictions
            train_predictions = []
            for x in X_train:
                output = net.activate(x)
                train_predictions.append(output[0])
            
            test_predictions = []
            for x in X_test:
                output = net.activate(x)
                test_predictions.append(output[0])
            
            # Denormalize predictions
            y_mean = results['y_mean']
            y_std = results['y_std']
            if y_std > 0:
                train_pred_denorm = np.array(train_predictions) * y_std + y_mean
                test_pred_denorm = np.array(test_predictions) * y_std + y_mean
            else:
                train_pred_denorm = np.array(train_predictions)
                test_pred_denorm = np.array(test_predictions)
            
            # Create approximation plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Training data plot
            ax1.scatter(y_train, train_pred_denorm, alpha=0.6, s=1)
            ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
            ax1.set_xlabel('True Values')
            ax1.set_ylabel('Predicted Values')
            ax1.set_title(f'Training Data - {function_name}\nRMSE: {results["train_rmse_denorm"]:.6f}')
            ax1.grid(True, alpha=0.3)
            
            # Test data plot
            ax2.scatter(y_test, test_pred_denorm, alpha=0.6, s=1, color='orange')
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax2.set_xlabel('True Values')
            ax2.set_ylabel('Predicted Values')
            ax2.set_title(f'Test Data - {function_name}\nRMSE: {results["test_rmse_denorm"]:.6f}')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            approx_path = os.path.join(test_dir, 'function_approximation.png')
            plt.savefig(approx_path, dpi=300, bbox_inches='tight')
            if view:
                plt.show()
            else:
                plt.close()
                
            if verbose:
                print(f"Function approximation plot saved to: {approx_path}")
                
        except Exception as e:
            print(f"Error creating function approximation plot: {str(e)}")
            if verbose:
                traceback.print_exc()
    
    if verbose:
        print(f"\nAll test results saved to {test_dir}")
    
    return results

def run_all_winners(results_dir="results", net_type="kan", verbose=True):
    """
    Run test evaluation on all winner-kan.pkl files in the results directory.
    
    Args:
        results_dir: Base directory containing experiment results
        net_type: Type of network to test
        verbose: Whether to print progress information
    """
    winner_files = []
    
    # Find all winner files
    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file == f'winner-{net_type}.pkl':
                winner_path = os.path.join(root, file)
                winner_files.append(winner_path)
    
    if verbose:
        print(f"Found {len(winner_files)} winner-{net_type}.pkl files")
    
    results_summary = []
    
    for i, winner_path in enumerate(winner_files, 1):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing {i}/{len(winner_files)}: {winner_path}")
            print(f"{'='*60}")
        
        try:
            result = test_genome(
                genome_path=winner_path,
                view=False,
                net_type=net_type,
                generate_plots=True,
                verbose=verbose
            )
            
            if result:
                results_summary.append(result)
                if verbose:
                    print(f"✓ Success: Test RMSE = {result['test_rmse']:.6f}")
            else:
                if verbose:
                    print("✗ Failed to process genome")
        
        except Exception as e:
            print(f"✗ Error processing {winner_path}: {e}")
            if verbose:
                traceback.print_exc()
    
    # Save summary of all results
    if results_summary:
        summary_path = os.path.join(results_dir, 'all_test_results_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Summary of all results saved to: {summary_path}")
            print(f"Successfully processed {len(results_summary)} out of {len(winner_files)} files")
            print(f"{'='*60}")
    
    return results_summary

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test a trained network on special function approximation')
    parser.add_argument('--genome', type=str, help='Path to the genome file')
    parser.add_argument('--view', action='store_true', help='Show plots during execution')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--net-type', type=str, default='kan', 
                      choices=['feedforward', 'kan'],
                      help='Type of network to test (feedforward or kan)')
    parser.add_argument('--function', type=str, help='Special function name')
    parser.add_argument('--no-plots', action='store_true', help='Skip generating plots')
    parser.add_argument('--quiet', action='store_true', help='Reduce verbosity')
    parser.add_argument('--all', action='store_true', help='Run on all winner files in results directory')
    parser.add_argument('--results-dir', type=str, default='results', 
                      help='Results directory to search for winner files')
    
    args = parser.parse_args()
    
    if args.all:
        # Run on all winner files
        run_all_winners(
            results_dir=args.results_dir,
            net_type=args.net_type,
            verbose=not args.quiet
        )
    else:
        # Run on single genome
        test_genome(
            genome_path=args.genome, 
            view=args.view, 
            seed=args.seed,
            net_type=args.net_type,
            function_name=args.function,
            generate_plots=not args.no_plots,
            verbose=not args.quiet
        )
