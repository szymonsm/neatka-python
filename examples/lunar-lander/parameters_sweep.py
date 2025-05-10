"""
NEAT Parameter Sweep Script for LunarLander-v2
Systematically explores hyperparameters for both feedforward and KAN networks.
"""

import os
import json
import itertools
import subprocess
import pandas as pd
import datetime
import argparse
import multiprocessing
import time
import copy
import hashlib
from pathlib import Path

# Parameters to sweep
PARAMETERS = {
    'network_type': ['feedforward', 'kan'],
    'population_size': [100, 250, 500],
    'num_hidden': [8, 16, 32],
    'conn_add_prob': [0.2],
    'conn_delete_prob': [0.2], 
    'node_add_prob': [0.2],
    'node_delete_prob': [0.2],
}

# Seeds to use for each configuration
SEEDS = [10, 42, 123, 456, 789]

# Base directory for results
RESULTS_BASE_DIR = 'examples/lunar-lander/parameter_sweep_results'

# Number of generations for the experiment
num_generations = 50

def generate_config_file(params, base_config_path, output_path):
    """Generate a new config file with the specified parameters."""
    with open(base_config_path, 'r') as f:
        config_content = f.read()
    
    # Replace parameters in config
    replacements = {
        'pop_size': str(params['population_size']),
        'num_hidden': str(params['num_hidden']),
        'conn_add_prob': str(params['conn_add_prob']),
        'conn_delete_prob': str(params['conn_delete_prob']),
        'node_add_prob': str(params['node_add_prob']),
        'node_delete_prob': str(params['node_delete_prob']),
    }
    
    for param, value in replacements.items():
        # Find the line with the parameter and replace it
        lines = config_content.split('\n')
        for i, line in enumerate(lines):
            if line.strip().startswith(param + ' '):
                parts = line.split('=')
                if len(parts) == 2:
                    lines[i] = parts[0] + '= ' + value
        config_content = '\n'.join(lines)
    
    # Save the new config file
    with open(output_path, 'w') as f:
        f.write(config_content)
    
    return output_path

def generate_config_hash(params):
    """Generate a unique hash for the parameter configuration."""
    param_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(param_str.encode()).hexdigest()[:10]

def run_experiment(params, seed, results_dir):
    """Run a single experiment with the specified parameters and seed."""
    # Create a directory for this specific configuration
    config_hash = generate_config_hash(params)
    net_type = params['network_type']
    
    config_dir = os.path.join(results_dir, f"{net_type}_{config_hash}")
    os.makedirs(config_dir, exist_ok=True)
    
    # Generate config file
    base_config_path = os.path.join('examples', 'lunar-lander', f'config-{net_type}')
    config_path = os.path.join(config_dir, f'config-{net_type}_{seed}.cfg')
    generate_config_file(params, base_config_path, config_path)
    
    # Create a json file to store parameters
    params_with_seed = copy.deepcopy(params)
    params_with_seed['seed'] = seed
    with open(os.path.join(config_dir, f'params_{seed}.json'), 'w') as f:
        json.dump(params_with_seed, f, indent=2)
    
    # Run the experiment
    print(f"Running experiment: {net_type} with seed {seed}, params: {params}")
    log_file = os.path.join(config_dir, f'log_{seed}.txt')
    results_file = os.path.join(config_dir, f'results_{seed}.csv')
    
    # Execute evolve.py with the specified parameters
    cmd = [
        'python', 
        os.path.join('examples', 'lunar-lander', 'evolve.py'),
        '--net-type', net_type,
        '--seed', str(seed),
        '--config', config_path,
        '--results-file', results_file,
        '--generations', str(num_generations),
        '--skip-plots'
    ]
    
    with open(log_file, 'w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    
    # Check if results file was created
    if os.path.exists(results_file):
        print(f"Results saved to {results_file}")
        status = "completed"
    else:
        print(f"Warning: No results file found at {results_file}")
        status = "failed"
        
    return results_file, status

def run_parameter_sweep(parallel=True, max_parallel=None):
    """Run the full parameter sweep."""
    # Create results directory with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(RESULTS_BASE_DIR, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save the parameter ranges
    with open(os.path.join(results_dir, 'parameter_ranges.json'), 'w') as f:
        json.dump(PARAMETERS, f, indent=2)
    
    # Generate all parameter combinations
    param_names = list(PARAMETERS.keys())
    param_values = list(PARAMETERS.values())
    
    all_experiments = []
    for values in itertools.product(*param_values):
        params = dict(zip(param_names, values))
        for seed in SEEDS:
            all_experiments.append((params, seed))
    
    total_experiments = len(all_experiments)
    print(f"Running {total_experiments} experiments...")
    
    # Create a summary file to track progress
    summary_path = os.path.join(results_dir, 'experiment_summary.csv')
    summary_df = pd.DataFrame(columns=['network_type', 'population_size', 'num_hidden', 
                                      'conn_add_prob', 'conn_delete_prob', 
                                      'node_add_prob', 'node_delete_prob', 
                                      'seed', 'status', 'results_file'])
    summary_df.to_csv(summary_path, index=False)
    
    if parallel and max_parallel is None:
        max_parallel = max(1, multiprocessing.cpu_count() - 1)
    elif not parallel:
        max_parallel = 1
    
    # Run all experiments
    start_time = time.time()
    
    if parallel:
        with multiprocessing.Pool(max_parallel) as pool:
            results = []
            for i, (params, seed) in enumerate(all_experiments):
                print(f"Queuing experiment {i+1}/{total_experiments}")
                result = pool.apply_async(run_experiment, (params, seed, results_dir))
                results.append((params, seed, result))
            
            # Wait for all to complete and update summary
            for i, (params, seed, result) in enumerate(results):
                try:
                    results_file, status = result.get()
                    print(f"Experiment {i+1}/{total_experiments} {status}")
                except Exception as e:
                    results_file = None
                    status = f"failed: {str(e)}"
                    print(f"Experiment {i+1}/{total_experiments} {status}")
                
                # Update summary
                experiment_data = {**params, 'seed': seed, 'status': status, 'results_file': results_file}
                summary_df = pd.concat([summary_df, pd.DataFrame([experiment_data])], ignore_index=True)
                summary_df.to_csv(summary_path, index=False)
    else:
        # Run sequentially
        for i, (params, seed) in enumerate(all_experiments):
            print(f"Running experiment {i+1}/{total_experiments}")
            try:
                results_file, status = run_experiment(params, seed, results_dir)
                print(f"Experiment {i+1}/{total_experiments} {status}")
            except Exception as e:
                results_file = None
                status = f"failed: {str(e)}"
                print(f"Experiment {i+1}/{total_experiments} {status}")
            
            # Update summary
            experiment_data = {**params, 'seed': seed, 'status': status, 'results_file': results_file}
            summary_df = pd.concat([summary_df, pd.DataFrame([experiment_data])], ignore_index=True)
            summary_df.to_csv(summary_path, index=False)
    
    # Generate final analysis
    end_time = time.time()
    total_time = end_time - start_time
    print(f"All experiments completed in {total_time:.2f} seconds.")
    
    # Create consolidated results
    consolidate_results(results_dir, summary_path)
    
    return results_dir

def consolidate_results(results_dir, summary_path):
    """Consolidate all experiment results into comprehensive analysis files."""
    print("Generating consolidated results...")
    
    # Load summary
    summary_df = pd.read_csv(summary_path)
    completed_experiments = summary_df[summary_df['status'] == 'completed']
    
    # Collect all results
    all_data = []
    for _, row in completed_experiments.iterrows():
        if row['results_file'] and os.path.exists(row['results_file']):
            try:
                # Load the CSV file with generation data
                exp_data = pd.read_csv(row['results_file'])
                
                # Add experiment parameters to each row
                for col in row.index:
                    if col not in ['status', 'results_file']:
                        exp_data[col] = row[col]
                
                all_data.append(exp_data)
                print(f"Loaded results from {row['results_file']}")
            except Exception as e:
                print(f"Failed to load results from {row['results_file']}: {e}")
    
    if all_data:
        # Concatenate all results
        consolidated_df = pd.concat(all_data, ignore_index=True)
        
        # Save to CSV
        consolidated_path = os.path.join(results_dir, 'consolidated_results.csv')
        consolidated_df.to_csv(consolidated_path, index=False)
        print(f"Consolidated results saved to {consolidated_path}")
        
        # Also save as JSON for easier programmatic access
        consolidated_df.to_json(os.path.join(results_dir, 'consolidated_results.json'), orient='records', indent=2)
        
        # Generate analysis per network type
        for net_type in PARAMETERS['network_type']:
            net_data = consolidated_df[consolidated_df['network_type'] == net_type]
            if not net_data.empty:
                net_data.to_csv(os.path.join(results_dir, f'{net_type}_results.csv'), index=False)
    else:
        print("No valid results found to consolidate.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NEAT parameter sweep for LunarLander")
    parser.add_argument("--sequential", action="store_true", help="Run experiments sequentially instead of in parallel")
    parser.add_argument("--max-parallel", type=int, help="Maximum parallel processes to use")
    args = parser.parse_args()
    
    run_parameter_sweep(parallel=not args.sequential, max_parallel=args.max_parallel)