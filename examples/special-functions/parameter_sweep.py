"""
Parameter sweep script for special functions experiments.
Runs comprehensive experiments comparing KAN-NEAT, MLP-NEAT, and PyKAN
on mathematical special functions.
"""

import os
import sys
import time
import argparse
import subprocess
import itertools
import pandas as pd
from pathlib import Path

import special_functions
import pykan_trainer

# Add parent directory to path for results_manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import results_manager


def run_neat_experiment(function_name, net_type, seed, num_generations=50, 
                       n_samples=1000, base_results_dir="results"):
    """Run a single NEAT experiment."""
    
    # Create results directory for this experiment
    experiment_dir = os.path.join(base_results_dir, f"{net_type}-{function_name}-seed{seed}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create log and results files
    log_file = os.path.join(experiment_dir, f'log_{seed}.txt')
    results_file = os.path.join(experiment_dir, f'results_{seed}.csv')
    
    # Create command
    script_path = os.path.join(os.path.dirname(__file__), 'evolve.py')
    cmd = [
        'python', script_path,
        '--function', function_name,
        '--net-type', net_type,
        '--seed', str(seed),
        '--generations', str(num_generations),
        '--samples', str(n_samples),
        '--results-file', results_file,
        '--skip-plots'  # Skip plots during parameter sweep to save time
    ]
    
    print(f"Running experiment: {net_type} with seed {seed}, function: {function_name}")
    
    try:
        start_time = time.time()
        
        # Run experiment exactly like single-pole-balancing: redirect all output to log file
        with open(log_file, 'w') as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
        
        elapsed_time = time.time() - start_time
        
        # Check if results file was created
        if os.path.exists(results_file):
            print(f"Results saved to {results_file}")
            status = "completed"
            
            # Try to extract best fitness from results file
            best_fitness = None
            try:
                import pandas as pd
                df = pd.read_csv(results_file)
                if not df.empty and 'best_fitness' in df.columns:
                    best_fitness = df['best_fitness'].iloc[-1]  # Last generation's best fitness
            except:
                # Try to extract from log file
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        for line in content.split('\n'):
                            if "Best fitness:" in line:
                                try:
                                    best_fitness = float(line.split("Best fitness:")[-1].strip())
                                except:
                                    pass
                except:
                    pass
            
        else:
            print(f"Warning: No results file found at {results_file}")
            status = "failed"
            best_fitness = None
        
        return {
            'status': status,
            'best_fitness': best_fitness,
            'runtime': elapsed_time,
            'log_file': log_file,
            'results_file': results_file if os.path.exists(results_file) else None
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'runtime': time.time() - start_time if 'start_time' in locals() else 0,
            'log_file': log_file,
            'results_file': None
        }


def run_pykan_experiment(function_name, seed, width=[2, 5, 1],
                        steps_per_grid=100, base_results_dir="results"):
    """Run a single PyKAN experiment."""
    
    if not pykan_trainer.PYKAN_AVAILABLE:
        return {
            'status': 'failed',
            'error': 'PyKAN not available',
            'runtime': 0
        }
    
    try:
        start_time = time.time()
        
        # Create results directory
        func_results_dir = os.path.join(base_results_dir, f"pykan-{function_name}-seed{seed}")
        
        print(f"Starting PyKAN experiment for {function_name} with seed {seed}")
        print(f"Results will be saved to: {func_results_dir}")
        
        # Train PyKAN
        model, history = pykan_trainer.train_pykan_on_function(
            function_name=function_name,
            width=width,
            steps_per_grid=steps_per_grid,
            results_dir=func_results_dir,
            seed=seed
        )
        
        elapsed_time = time.time() - start_time
        
        # Get final results
        final_train_loss = history[-1]['train_loss']
        final_test_loss = history[-1]['test_loss']
        total_parameters = history[-1]['parameters']
        
        print(f"PyKAN experiment completed successfully!")
        print(f"Final train loss: {final_train_loss:.6f}")
        print(f"Final test loss: {final_test_loss:.6f}")
        
        return {
            'status': 'completed',
            'train_loss': final_train_loss,
            'test_loss': final_test_loss,
            'parameters': total_parameters,
            'runtime': elapsed_time
        }
        
    except KeyboardInterrupt:
        elapsed_time = time.time() - start_time
        print(f"PyKAN experiment interrupted by user")
        return {
            'status': 'interrupted',
            'error': 'Interrupted by user',
            'runtime': elapsed_time
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"PyKAN experiment failed with error: {str(e)}")
        return {
            'status': 'failed',
            'error': str(e),
            'runtime': elapsed_time
        }


def run_parameter_sweep(functions=None, methods=None, seeds=None, num_generations=50,
                       n_samples=1000, results_dir="special_functions_results",
                       pykan_config=None):
    """
    Run comprehensive parameter sweep on special functions.
    
    Args:
        functions: List of function names (if None, uses a subset)
        methods: List of methods ['mlp-neat', 'kan-neat', 'pykan']
        seeds: List of random seeds
        num_generations: Number of generations for NEAT
        n_samples: Number of training samples
        results_dir: Directory to save results
        pykan_config: Dictionary with PyKAN configuration
    """
    
    # Default configurations
    if functions is None:
        # Start with a subset of functions that are known to work well
        functions = ['jv', 'ellipkinc', 'lpmv_0', 'sph_harm_01', 'iv']
    
    if methods is None:
        methods = ['mlp-neat', 'kan-neat']
        if pykan_trainer.PYKAN_AVAILABLE:
            methods.append('pykan')
    
    if seeds is None:
        seeds = [42, 123, 456, 789, 1000]  # 5 different seeds
    
    if pykan_config is None:
        pykan_config = {
            'width': [2, 5, 1],
            'steps_per_grid': 10
        }
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize results tracking
    experiment_results = []
    
    # Order methods to run PyKAN first, then NEAT methods
    ordered_methods = []
    if 'pykan' in methods:
        ordered_methods.append('pykan')
    if 'kan-neat' in methods:
        ordered_methods.append('kan-neat')
    if 'mlp-neat' in methods:
        ordered_methods.append('mlp-neat')
    
    total_experiments = len(functions) * len(ordered_methods) * len(seeds)
    completed_experiments = 0
    
    print(f"Starting parameter sweep: {len(functions)} functions × {len(ordered_methods)} methods × {len(seeds)} seeds = {total_experiments} experiments")
    print(f"Functions: {functions}")
    print(f"Methods: {ordered_methods}")
    print(f"Seeds: {seeds}")
    print(f"Execution order: PyKAN first, then NEAT methods")
    
    # Run experiments in order: PyKAN first, then NEAT
    for function_name in functions:
        for method in ordered_methods:
            for seed in seeds:
                completed_experiments += 1
                print(f"\n{'='*80}")
                print(f"Experiment {completed_experiments}/{total_experiments}")
                print(f"Function: {function_name}, Method: {method}, Seed: {seed}")
                print(f"{'='*80}")
                
                experiment_info = {
                    'function': function_name,
                    'method': method,
                    'seed': seed,
                    'num_generations': num_generations,
                    'n_samples': n_samples
                }
                
                if method == 'pykan':
                    # Create organized PyKAN results directory
                    pykan_base_dir = os.path.join(results_dir, 'pykan_results')
                    result = run_pykan_experiment(
                        function_name=function_name,
                        seed=seed,
                        width=pykan_config['width'],
                        steps_per_grid=pykan_config['steps_per_grid'],
                        base_results_dir=pykan_base_dir
                    )
                    experiment_info.update({
                        'pykan_width': pykan_config['width'],
                        'pykan_steps_per_grid': pykan_config['steps_per_grid']
                    })
                else:
                    # NEAT experiment (runs sequentially, no multiprocessing)
                    net_type = 'feedforward' if method == 'mlp-neat' else 'kan'
                    result = run_neat_experiment(
                        function_name=function_name,
                        net_type=net_type,
                        seed=seed,
                        num_generations=num_generations,
                        n_samples=n_samples,
                        base_results_dir=results_dir
                    )
                
                # Combine experiment info with results
                experiment_info.update(result)
                experiment_results.append(experiment_info)
                
                # Print result summary
                if result['status'] == 'completed':
                    if method == 'pykan':
                        print(f"✓ Completed: Train Loss = {result.get('train_loss', 'N/A'):.6f}, "
                              f"Test Loss = {result.get('test_loss', 'N/A'):.6f}")
                    else:
                        print(f"✓ Completed: Best Fitness = {result.get('best_fitness', 'N/A')}")
                else:
                    print(f"✗ {result['status'].title()}: {result.get('error', 'Unknown error')}")
                
                print(f"Runtime: {result.get('runtime', 0):.2f} seconds")
                
                # Save intermediate results after each experiment
                save_results(experiment_results, results_dir)
    
    # Save final results
    save_results(experiment_results, results_dir)
    
    # Generate summary
    generate_summary(experiment_results, results_dir)
    
    print(f"\nParameter sweep completed! Results saved to {results_dir}")
    return experiment_results


def save_results(experiment_results, results_dir):
    """Save experiment results to CSV."""
    df = pd.DataFrame(experiment_results)
    results_path = os.path.join(results_dir, 'experiment_results.csv')
    df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


def generate_summary(experiment_results, results_dir):
    """Generate summary statistics and save to file."""
    df = pd.DataFrame(experiment_results)
    
    summary_path = os.path.join(results_dir, 'experiment_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Special Functions Experiment Summary\n")
        f.write("===================================\n\n")
        
        # Overall statistics
        total_experiments = len(df)
        completed = len(df[df['status'] == 'completed'])
        failed = len(df[df['status'] == 'failed'])
        timeout = len(df[df['status'] == 'timeout'])
        
        f.write(f"Total experiments: {total_experiments}\n")
        f.write(f"Completed: {completed} ({completed/total_experiments*100:.1f}%)\n")
        f.write(f"Failed: {failed} ({failed/total_experiments*100:.1f}%)\n")
        f.write(f"Timeout: {timeout} ({timeout/total_experiments*100:.1f}%)\n\n")
        
        # Results by method
        f.write("Results by Method:\n")
        f.write("-" * 40 + "\n")
        
        completed_df = df[df['status'] == 'completed']
        
        for method in completed_df['method'].unique():
            method_df = completed_df[completed_df['method'] == method]
            f.write(f"\n{method.upper()}:\n")
            f.write(f"  Completed experiments: {len(method_df)}\n")
            f.write(f"  Average runtime: {method_df['runtime'].mean():.2f} seconds\n")
            
            if method == 'pykan':
                if 'train_loss' in method_df.columns:
                    f.write(f"  Average train loss: {method_df['train_loss'].mean():.6f}\n")
                    f.write(f"  Average test loss: {method_df['test_loss'].mean():.6f}\n")
            else:
                if 'best_fitness' in method_df.columns:
                    f.write(f"  Average best fitness: {method_df['best_fitness'].mean():.6f}\n")
        
        # Results by function
        f.write(f"\n\nResults by Function:\n")
        f.write("-" * 40 + "\n")
        
        for function in completed_df['function'].unique():
            func_df = completed_df[completed_df['function'] == function]
            f.write(f"\n{function}:\n")
            f.write(f"  Total completed: {len(func_df)}\n")
            
            for method in func_df['method'].unique():
                method_func_df = func_df[func_df['method'] == method]
                f.write(f"  {method}: {len(method_func_df)} experiments\n")
    
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run parameter sweep on special functions')
    parser.add_argument('--functions', nargs='+', 
                      choices=special_functions.list_functions(),
                      default=['ellipj', 'jv', 'lpmv_1'],  # 3 specified functions
                      help='Functions to test (default: ellipj, jv, lpmv_1)')
    parser.add_argument('--methods', nargs='+', 
                      choices=['mlp-neat', 'kan-neat', 'pykan'],
                      default=['kan-neat', 'pykan'],  # Only KAN-NEAT and PyKAN
                      help='Methods to test')
    parser.add_argument('--seeds', nargs='+', type=int,
                      default=[42, 123, 456, 789, 1000],  # 5 seeds as requested
                      help='Random seeds to use')
    parser.add_argument('--generations', type=int, default=50,
                      help='Number of generations for NEAT')
    parser.add_argument('--samples', type=int, default=1000,
                      help='Number of training samples')
    parser.add_argument('--results-dir', type=str, default='special_functions_results',
                      help='Directory to save results')
    parser.add_argument('--pykan-width', nargs='+', type=int, default=[2, 5, 1],
                      help='PyKAN network width')
    parser.add_argument('--pykan-max-grid', type=int, default=50,
                      help='PyKAN maximum grid size')
    parser.add_argument('--pykan-steps', type=int, default=10,  # Set to 10 as requested
                      help='PyKAN steps per grid')
    
    args = parser.parse_args()
    
    # Configure PyKAN
    pykan_config = {
        'width': args.pykan_width,
        'steps_per_grid': args.pykan_steps
    }
    
    # Print configuration
    print("Parameter Sweep Configuration:")
    print(f"Functions: {args.functions}")
    print(f"Methods: {args.methods}")
    print(f"Seeds: {args.seeds}")
    print(f"PyKAN steps per grid: {args.pykan_steps}")
    print(f"Results directory: {args.results_dir}")
    print(f"Execution order: PyKAN first, then NEAT methods")
    
    # Add PyKAN to methods if available
    if 'pykan' not in args.methods and pykan_trainer.PYKAN_AVAILABLE:
        print("PyKAN is available but not selected. Add --methods pykan to include it.")
    
    # Run parameter sweep
    results = run_parameter_sweep(
        functions=args.functions,
        methods=args.methods,
        seeds=args.seeds,
        num_generations=args.generations,
        n_samples=args.samples,
        results_dir=args.results_dir,
        pykan_config=pykan_config
    )
