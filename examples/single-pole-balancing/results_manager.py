"""
Manages results from experiments, ensuring proper organization and reproducibility.
"""
import os
import hashlib
import json
import random
import time
import sys
from datetime import datetime

def get_config_hash(config, seed=None):
    """Generate a hash for the configuration and seed to identify unique experiments.
    
    Args:
        config: The experiment configuration
        seed: Random seed used for the experiment (default: None)
        
    Returns:
        str: Hash string that uniquely identifies this configuration+seed
    """
    # Extract relevant parameters from the config
    params = {
        'seed': seed,  # Include the seed in the hash
        'genome_config': {
            'num_inputs': config.genome_config.num_inputs,
            'num_outputs': config.genome_config.num_outputs,
            'num_hidden': config.genome_config.num_hidden,
            'feed_forward': config.genome_config.feed_forward,
            'initial_connection': config.genome_config.initial_connection,
        },
        'reproduction': {
            'elitism': config.reproduction_config.elitism,
            'survival_threshold': config.reproduction_config.survival_threshold,
        },
        'species_set': {
            'compatibility_threshold': config.species_set_config.compatibility_threshold,
        },
        'stagnation': {
            'species_fitness_func': config.stagnation_config.species_fitness_func,
            'max_stagnation': config.stagnation_config.max_stagnation,
            'species_elitism': config.stagnation_config.species_elitism,
        },
    }
    
    # For KAN parameters, extract them if available
    if hasattr(config.genome_config, 'spline_mutation_rate'):
        params['kan_params'] = {
            'spline_mutation_rate': config.genome_config.spline_mutation_rate,
            'spline_add_prob': config.genome_config.spline_add_prob,
            'spline_delete_prob': config.genome_config.spline_delete_prob,
            'initial_spline_segments': config.genome_config.initial_spline_segments,
            'min_spline_segments': config.genome_config.min_spline_segments,
            'max_spline_segments': config.genome_config.max_spline_segments,
        }
    
    # Convert to JSON string and hash
    params_str = json.dumps(params, sort_keys=True)
    return hashlib.md5(params_str.encode()).hexdigest()[:10]

def setup_results_directory(config, experiment_name='kan', seed=None, force_new=False):
    """Set up a results directory for the experiment.
    
    Args:
        config: The experiment configuration
        experiment_name: Name of the experiment (default: 'kan')
        seed: Random seed for reproducibility (default: None)
        force_new: Force creation of a new run directory even if this config+seed exists (default: False)
        
    Returns:
        tuple: (results_dir, run_id, is_new_run, seed)
    """
    # If no seed is provided, generate one
    # if seed is None:
    #     seed = random.randint(1, 2**31 - 1)
    
    # Create base results directory
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(base_dir, exist_ok=True)
    
    # Create experiment-specific directory
    exp_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Generate a unique hash for this configuration + seed
    config_hash = get_config_hash(config, seed)
    
    # Use timestamp for unique identification
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{config_hash}_{timestamp}"
    
    # Check if a run with this config+seed already exists
    if not force_new:
        existing_runs = [d for d in os.listdir(exp_dir) if d.startswith(config_hash)]
        
        if existing_runs:
            # Use most recent existing run
            existing_runs.sort(reverse=True)  # Sort by timestamp (descending)
            run_id = existing_runs[0]
            results_dir = os.path.join(exp_dir, run_id)
            
            # Load the seed from the existing run
            try:
                with open(os.path.join(results_dir, 'config_summary.json'), 'r') as f:
                    data = json.load(f)
                    if 'seed' in data:
                        seed = data['seed']
            except Exception as e:
                print(f"Warning: Could not load seed from existing run: {e}")
            
            return results_dir, run_id, False
    
    # Create a new run directory
    results_dir = os.path.join(exp_dir, run_id)
    os.makedirs(results_dir, exist_ok=True)
    
    # Save configuration summary
    with open(os.path.join(results_dir, 'config_summary.json'), 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'config_hash': config_hash,
            'seed': seed,
            'python_version': sys.version,
            'parameters': {
                'pop_size': config.pop_size,
                'fitness_threshold': config.fitness_threshold,
                'num_inputs': config.genome_config.num_inputs,
                'num_hidden': config.genome_config.num_hidden,
                'num_outputs': config.genome_config.num_outputs,
                'kan_params': {
                    'initial_spline_segments': getattr(config.genome_config, 'initial_spline_segments', 'n/a'),
                    'min_spline_segments': getattr(config.genome_config, 'min_spline_segments', 'n/a'),
                    'max_spline_segments': getattr(config.genome_config, 'max_spline_segments', 'n/a'),
                }
            }
        }, f, indent=4)
    
    return results_dir, run_id, True

def load_previous_results(experiment_name='kan', config_hash=None):
    """Load previous results for analysis.
    
    Args:
        experiment_name: Name of the experiment (default: 'kan')
        config_hash: Optional hash to filter results (default: None)
        
    Returns:
        list: List of result directories matching the criteria
    """
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    exp_dir = os.path.join(base_dir, experiment_name)
    
    if not os.path.exists(exp_dir):
        return []
    
    if config_hash:
        return [os.path.join(exp_dir, d) for d in os.listdir(exp_dir) 
                if d.startswith(config_hash)]
    else:
        return [os.path.join(exp_dir, d) for d in os.listdir(exp_dir)]