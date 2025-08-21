"""
Simplified KAN-NEAT Results Analysis Script
Focuses on core functionality without complex dependencies.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from pathlib import Path


def load_experiment_results():
    """Load the experiment results CSV."""
    results_path = Path("special_functions_results") / "experiment_results.csv"
    if results_path.exists():
        return pd.read_csv(results_path)
    else:
        print(f"Results file not found: {results_path}")
        return None


def find_kan_neat_winners():
    """Find all KAN-NEAT winner files and their associated metadata."""
    results_dir = Path("results")
    winners = {}
    
    functions = ['ellipj', 'jv', 'lpmv_1']
    
    for function_name in functions:
        winners[function_name] = {}
        function_dir = results_dir / f"{function_name}-kan"
        
        if not function_dir.exists():
            print(f"No results directory found for {function_name}")
            continue
        
        for exp_dir in function_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            # Load config summary to get seed
            config_file = exp_dir / "config_summary.json"
            if not config_file.exists():
                continue
            
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            seed = config.get('seed')
            winner_file = exp_dir / "winner-kan.pkl"
            
            if winner_file.exists():
                winners[function_name][seed] = {
                    'winner_path': winner_file,
                    'config_path': exp_dir / "reconstructed_config.ini",
                    'exp_dir': exp_dir
                }
                print(f"Found winner for {function_name}, seed {seed}")
    
    return winners


def evaluate_kan_model(winner_path, config_path, function_name, seed):
    """Evaluate a KAN model and return RMSE scores."""
    try:
        import special_functions
        import neat
        from neat.nn.kan import KANNetwork
        
        # Load winner
        with open(winner_path, 'rb') as f:
            winner = pickle.load(f)
        
        # Load config
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           str(config_path))
        
        # Get function and generate test data
        func = special_functions.get_function(function_name)
        X_train, y_train, X_test, y_test = func.generate_data(
            n_samples=1000, test_fraction=0.2, seed=seed
        )
        
        # Create network
        network = KANNetwork.create(winner, config)
        
        # Evaluate on training data
        train_predictions = []
        for x in X_train:
            pred = network.activate(x)
            train_predictions.append(pred[0])
        train_predictions = np.array(train_predictions)
        
        # Evaluate on test data
        test_predictions = []
        for x in X_test:
            pred = network.activate(x)
            test_predictions.append(pred[0])
        test_predictions = np.array(test_predictions)
        
        # Calculate RMSE
        train_rmse = np.sqrt(np.mean((train_predictions - y_train) ** 2))
        test_rmse = np.sqrt(np.mean((test_predictions - y_test) ** 2))
        
        return {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'seed': seed
        }
        
    except Exception as e:
        print(f"Error evaluating {function_name} seed {seed}: {e}")
        return None


def load_pykan_results():
    """Load PyKAN results from the experiment CSV."""
    df = load_experiment_results()
    if df is None:
        return {}
    
    pykan_results = {}
    pykan_df = df[df['method'] == 'pykan']
    
    for _, row in pykan_df.iterrows():
        function = row['function']
        seed = row['seed']
        
        if function not in pykan_results:
            pykan_results[function] = {}
        
        pykan_results[function][seed] = {
            'train_loss': row.get('train_loss', None),
            'test_loss': row.get('test_loss', None)
        }
    
    return pykan_results


def create_comparison_summary():
    """Create a summary comparison table."""
    print("Creating comparison summary...")
    
    # Load PyKAN results
    pykan_results = load_pykan_results()
    
    # Find KAN-NEAT winners
    kan_winners = find_kan_neat_winners()
    
    # Evaluate KAN-NEAT models
    kan_neat_results = {}
    
    for function_name, seeds_data in kan_winners.items():
        kan_neat_results[function_name] = {}
        
        for seed, paths in seeds_data.items():
            result = evaluate_kan_model(
                paths['winner_path'],
                paths['config_path'],
                function_name,
                seed
            )
            
            if result:
                kan_neat_results[function_name][seed] = result
    
    # Create summary table
    summary_data = []
    
    functions = ['ellipj', 'jv', 'lpmv_1']
    
    for function_name in functions:
        # Find best PyKAN result
        pykan_best_test = float('inf')
        pykan_best_train = None
        
        if function_name in pykan_results:
            for seed, results in pykan_results[function_name].items():
                test_loss = results.get('test_loss')
                if test_loss is not None and test_loss < pykan_best_test:
                    pykan_best_test = test_loss
                    pykan_best_train = results.get('train_loss')
        
        # Find best KAN-NEAT result
        kan_best_test = float('inf')
        kan_best_train = None
        
        if function_name in kan_neat_results:
            for seed, results in kan_neat_results[function_name].items():
                test_rmse = results['test_rmse']
                if test_rmse < kan_best_test:
                    kan_best_test = test_rmse
                    kan_best_train = results['train_rmse']
        
        summary_data.append({
            'Special Function': function_name,
            'Best Test RMSE PyKAN': pykan_best_test if pykan_best_test != float('inf') else 'N/A',
            'Train RMSE PyKAN': pykan_best_train if pykan_best_train is not None else 'N/A',
            'Best Test RMSE KAN-NEAT': kan_best_test if kan_best_test != float('inf') else 'N/A',
            'Train RMSE KAN-NEAT': kan_best_train if kan_best_train is not None else 'N/A'
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_path = Path("special_functions_results") / "comparison_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    
    print(f"Summary saved to: {summary_path}")
    print("\nComparison Summary:")
    print(summary_df.to_string(index=False))
    
    return summary_df, kan_neat_results, pykan_results


if __name__ == "__main__":
    print("KAN-NEAT vs PyKAN Analysis")
    print("=" * 40)
    
    try:
        summary_df, kan_neat_results, pykan_results = create_comparison_summary()
        
        print("\n✅ Analysis completed successfully!")
        print(f"Summary table saved to: special_functions_results/comparison_summary.csv")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
