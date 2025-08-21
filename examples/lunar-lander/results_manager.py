"""
Manages results from experiments, ensuring proper organization and reproducibility.
"""
import os
import hashlib
import json
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
    # Extract ALL parameters from the config
    params = {
        'seed': seed,  # Include the seed in the hash
        'neat': {
            'pop_size': config.pop_size,
            'fitness_criterion': config.fitness_criterion,
            'fitness_threshold': config.fitness_threshold,
            'reset_on_extinction': config.reset_on_extinction
        },
        'genome_config': {
            # Network structure
            'num_inputs': config.genome_config.num_inputs,
            'num_outputs': config.genome_config.num_outputs,
            'num_hidden': config.genome_config.num_hidden,
            'feed_forward': config.genome_config.feed_forward,
            'initial_connection': config.genome_config.initial_connection,
            
            # Compatibility parameters
            'compatibility_disjoint_coefficient': config.genome_config.compatibility_disjoint_coefficient,
            'compatibility_weight_coefficient': config.genome_config.compatibility_weight_coefficient,
            
            # Connection addition/deletion
            'conn_add_prob': config.genome_config.conn_add_prob,
            'conn_delete_prob': config.genome_config.conn_delete_prob,
            
            # Node addition/deletion
            'node_add_prob': config.genome_config.node_add_prob,
            'node_delete_prob': config.genome_config.node_delete_prob,
            
            # Connection enable parameters
            'enabled_default': config.genome_config.enabled_default,
            'enabled_mutate_rate': config.genome_config.enabled_mutate_rate,
            'enabled_rate_to_true_add': getattr(config.genome_config, 'enabled_rate_to_true_add', 0.0),
            'enabled_rate_to_false_add': getattr(config.genome_config, 'enabled_rate_to_false_add', 0.0),
            
            # Structural mutation
            'single_structural_mutation': config.genome_config.single_structural_mutation,
            'structural_mutation_surer': str(config.genome_config.structural_mutation_surer),
            
            # Activation and aggregation
            'activation_default': config.genome_config.activation_default,
            'activation_options': list(config.genome_config.activation_options),
            'activation_mutate_rate': config.genome_config.activation_mutate_rate,
            'aggregation_default': config.genome_config.aggregation_default,
            'aggregation_options': list(config.genome_config.aggregation_options),
            'aggregation_mutate_rate': config.genome_config.aggregation_mutate_rate,
            
            # Node bias parameters
            'bias_init_mean': config.genome_config.bias_init_mean,
            'bias_init_stdev': config.genome_config.bias_init_stdev,
            'bias_max_value': config.genome_config.bias_max_value,
            'bias_min_value': config.genome_config.bias_min_value,
            'bias_mutate_power': config.genome_config.bias_mutate_power,
            'bias_mutate_rate': config.genome_config.bias_mutate_rate,
            'bias_replace_rate': config.genome_config.bias_replace_rate,
            
            # Node response parameters
            'response_init_mean': config.genome_config.response_init_mean,
            'response_init_stdev': config.genome_config.response_init_stdev,
            'response_max_value': config.genome_config.response_max_value,
            'response_min_value': config.genome_config.response_min_value,
            'response_mutate_power': config.genome_config.response_mutate_power,
            'response_mutate_rate': config.genome_config.response_mutate_rate,
            'response_replace_rate': config.genome_config.response_replace_rate,
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
    if hasattr(config.genome_config, 'spline_mutate_rate'):
        params['kan_params'] = {
            # KAN-specific initialization parameters
            'weight_s_init_mean': config.genome_config.weight_s_init_mean,
            'weight_s_init_stdev': config.genome_config.weight_s_init_stdev,
            'weight_s_replace_rate': config.genome_config.weight_s_replace_rate,
            'weight_b_init_mean': config.genome_config.weight_b_init_mean,
            'weight_b_init_stdev': config.genome_config.weight_b_init_stdev,
            'weight_b_replace_rate': config.genome_config.weight_b_replace_rate,
            'spline_init_mean': config.genome_config.spline_init_mean,
            'spline_init_stdev': config.genome_config.spline_init_stdev,
            'spline_replace_rate': config.genome_config.spline_replace_rate,
            
            # KAN-specific coefficient ranges
            'weight_s_coefficient_range': config.genome_config.weight_s_coefficient_range,
            'weight_b_coefficient_range': config.genome_config.weight_b_coefficient_range,
            'spline_coefficient_range': config.genome_config.spline_coefficient_range,
            
            # KAN-specific mutation parameters
            'weight_s_mutate_rate': config.genome_config.weight_s_mutate_rate,
            'weight_s_mutate_power': config.genome_config.weight_s_mutate_power,
            'weight_s_min_value': config.genome_config.weight_s_min_value,
            'weight_s_max_value': config.genome_config.weight_s_max_value,
            'weight_b_mutate_rate': config.genome_config.weight_b_mutate_rate,
            'weight_b_mutate_power': config.genome_config.weight_b_mutate_power,
            'weight_b_min_value': config.genome_config.weight_b_min_value,
            'weight_b_max_value': config.genome_config.weight_b_max_value,
            'spline_mutate_rate': config.genome_config.spline_mutate_rate,
            'spline_mutate_power': config.genome_config.spline_mutate_power,
            'spline_min_value': config.genome_config.spline_min_value,
            'spline_max_value': config.genome_config.spline_max_value,
            
            # Spline segment parameters
            'spline_add_prob': config.genome_config.spline_add_prob,
            'spline_delete_prob': config.genome_config.spline_delete_prob,
            'initial_spline_segments': config.genome_config.initial_spline_segments,
            'min_spline_segments': config.genome_config.min_spline_segments,
            'max_spline_segments': config.genome_config.max_spline_segments,
            'spline_range_min': config.genome_config.spline_range_min,
            'spline_range_max': config.genome_config.spline_range_max,
            
            # KAN-specific crossover parameters
            'kan_segments_distance_treshold': config.genome_config.kan_segments_distance_treshold,
            'kan_connection_crossover_add_segment_rate': config.genome_config.kan_connection_crossover_add_segment_rate,
            'kan_segment_crossover_better_fitness_rate': config.genome_config.kan_segment_crossover_better_fitness_rate
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
        tuple: (results_dir, run_id, is_new_run)
    """
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
    
    # Create detailed configuration dump
    # This will be a complete copy of all parameters
    config_data = {
        'timestamp': timestamp,
        'config_hash': config_hash,
        'seed': seed,
        'python_version': sys.version,
        'neat': {
            'pop_size': config.pop_size,
            'fitness_criterion': config.fitness_criterion,
            'fitness_threshold': config.fitness_threshold,
            'reset_on_extinction': config.reset_on_extinction
        },
        'genome_config': {
            # Network structure
            'num_inputs': config.genome_config.num_inputs,
            'num_outputs': config.genome_config.num_outputs,
            'num_hidden': config.genome_config.num_hidden,
            'feed_forward': config.genome_config.feed_forward,
            'initial_connection': str(config.genome_config.initial_connection),
            
            # Compatibility parameters
            'compatibility_disjoint_coefficient': config.genome_config.compatibility_disjoint_coefficient,
            'compatibility_weight_coefficient': config.genome_config.compatibility_weight_coefficient,
            
            # Connection addition/deletion
            'conn_add_prob': config.genome_config.conn_add_prob,
            'conn_delete_prob': config.genome_config.conn_delete_prob,
            
            # Node addition/deletion
            'node_add_prob': config.genome_config.node_add_prob,
            'node_delete_prob': config.genome_config.node_delete_prob,
            
            # Connection enable parameters
            'enabled_default': config.genome_config.enabled_default,
            'enabled_mutate_rate': config.genome_config.enabled_mutate_rate,
            'enabled_rate_to_true_add': getattr(config.genome_config, 'enabled_rate_to_true_add', 0.0),
            'enabled_rate_to_false_add': getattr(config.genome_config, 'enabled_rate_to_false_add', 0.0),
            
            # Structural mutation
            'single_structural_mutation': config.genome_config.single_structural_mutation,
            'structural_mutation_surer': str(config.genome_config.structural_mutation_surer),
            
            # Activation and aggregation
            'activation_default': config.genome_config.activation_default,
            'activation_options': list(config.genome_config.activation_options),
            'activation_mutate_rate': config.genome_config.activation_mutate_rate,
            'aggregation_default': config.genome_config.aggregation_default,
            'aggregation_options': list(config.genome_config.aggregation_options),
            'aggregation_mutate_rate': config.genome_config.aggregation_mutate_rate,
            
            # Node bias parameters
            'bias_init_mean': config.genome_config.bias_init_mean,
            'bias_init_stdev': config.genome_config.bias_init_stdev,
            'bias_max_value': config.genome_config.bias_max_value,
            'bias_min_value': config.genome_config.bias_min_value,
            'bias_mutate_power': config.genome_config.bias_mutate_power,
            'bias_mutate_rate': config.genome_config.bias_mutate_rate,
            'bias_replace_rate': config.genome_config.bias_replace_rate,
            
            # Node response parameters
            'response_init_mean': config.genome_config.response_init_mean,
            'response_init_stdev': config.genome_config.response_init_stdev,
            'response_max_value': config.genome_config.response_max_value,
            'response_min_value': config.genome_config.response_min_value,
            'response_mutate_power': config.genome_config.response_mutate_power,
            'response_mutate_rate': config.genome_config.response_mutate_rate,
            'response_replace_rate': config.genome_config.response_replace_rate,
        },
        'reproduction_config': {
            'elitism': config.reproduction_config.elitism,
            'survival_threshold': config.reproduction_config.survival_threshold,
        },
        'species_set_config': {
            'compatibility_threshold': config.species_set_config.compatibility_threshold,
        },
        'stagnation_config': {
            'species_fitness_func': config.stagnation_config.species_fitness_func,
            'max_stagnation': config.stagnation_config.max_stagnation,
            'species_elitism': config.stagnation_config.species_elitism,
        }
    }
    
    # Add KAN parameters if they exist
    if hasattr(config.genome_config, 'spline_mutate_rate'):
        config_data['kan_params'] = {
            # KAN-specific initialization parameters
            'weight_s_init_mean': config.genome_config.weight_s_init_mean,
            'weight_s_init_stdev': config.genome_config.weight_s_init_stdev,
            'weight_s_replace_rate': config.genome_config.weight_s_replace_rate,
            'weight_b_init_mean': config.genome_config.weight_b_init_mean,
            'weight_b_init_stdev': config.genome_config.weight_b_init_stdev,
            'weight_b_replace_rate': config.genome_config.weight_b_replace_rate,
            'spline_init_mean': config.genome_config.spline_init_mean,
            'spline_init_stdev': config.genome_config.spline_init_stdev,
            'spline_replace_rate': config.genome_config.spline_replace_rate,
            
            # KAN-specific coefficient ranges
            'weight_s_coefficient_range': config.genome_config.weight_s_coefficient_range,
            'weight_b_coefficient_range': config.genome_config.weight_b_coefficient_range,
            'spline_coefficient_range': config.genome_config.spline_coefficient_range,
            
            # KAN-specific mutation parameters
            'weight_s_mutate_rate': config.genome_config.weight_s_mutate_rate,
            'weight_s_mutate_power': config.genome_config.weight_s_mutate_power,
            'weight_s_min_value': config.genome_config.weight_s_min_value,
            'weight_s_max_value': config.genome_config.weight_s_max_value,
            'weight_b_mutate_rate': config.genome_config.weight_b_mutate_rate,
            'weight_b_mutate_power': config.genome_config.weight_b_mutate_power,
            'weight_b_min_value': config.genome_config.weight_b_min_value,
            'weight_b_max_value': config.genome_config.weight_b_max_value,
            'spline_mutate_rate': config.genome_config.spline_mutate_rate,
            'spline_mutate_power': config.genome_config.spline_mutate_power,
            'spline_min_value': config.genome_config.spline_min_value,
            'spline_max_value': config.genome_config.spline_max_value,
            
            # Spline segment parameters
            'spline_add_prob': config.genome_config.spline_add_prob,
            'spline_delete_prob': config.genome_config.spline_delete_prob,
            'initial_spline_segments': config.genome_config.initial_spline_segments,
            'min_spline_segments': config.genome_config.min_spline_segments,
            'max_spline_segments': config.genome_config.max_spline_segments,
            'spline_range_min': config.genome_config.spline_range_min,
            'spline_range_max': config.genome_config.spline_range_max,
            
            # KAN-specific crossover parameters
            'kan_segments_distance_treshold': config.genome_config.kan_segments_distance_treshold,
            'kan_connection_crossover_add_segment_rate': config.genome_config.kan_connection_crossover_add_segment_rate,
            'kan_segment_crossover_better_fitness_rate': config.genome_config.kan_segment_crossover_better_fitness_rate
        }
    
    # Save configuration summary
    with open(os.path.join(results_dir, 'config_summary.json'), 'w') as f:
        json.dump(config_data, f, indent=4)
    
    # Also save the original config file
    try:
        config_filename = config.genome_config.config_filename
        if config_filename:
            import shutil
            shutil.copy2(config_filename, os.path.join(results_dir, 'original_config.ini'))
    except (AttributeError, FileNotFoundError):
        # If we can't get or copy the config file, create one from parsed parameters
        try:
            from configparser import ConfigParser
            cfg = ConfigParser()
            
            # NEAT section
            cfg.add_section('NEAT')
            cfg.set('NEAT', 'fitness_criterion', str(config.fitness_criterion))
            cfg.set('NEAT', 'fitness_threshold', str(config.fitness_threshold))
            cfg.set('NEAT', 'pop_size', str(config.pop_size))
            cfg.set('NEAT', 'reset_on_extinction', str(int(config.reset_on_extinction)))
            
            # Genome section (appropriate class name)
            genome_section = 'KANGenome' if hasattr(config.genome_config, 'spline_mutate_rate') else 'DefaultGenome'
            cfg.add_section(genome_section)
            
            # Add all parameters from config_data
            for key, value in config_data['genome_config'].items():
                cfg.set(genome_section, key, str(value))
            
            # Add KAN parameters if applicable
            if 'kan_params' in config_data:
                for key, value in config_data['kan_params'].items():
                    cfg.set(genome_section, key, str(value))
            
            # Add other sections
            cfg.add_section('DefaultSpeciesSet')
            cfg.set('DefaultSpeciesSet', 'compatibility_threshold', 
                   str(config.species_set_config.compatibility_threshold))
            
            cfg.add_section('DefaultStagnation')
            cfg.set('DefaultStagnation', 'species_fitness_func', 
                   str(config.stagnation_config.species_fitness_func))
            cfg.set('DefaultStagnation', 'max_stagnation', 
                   str(config.stagnation_config.max_stagnation))
            
            cfg.add_section('DefaultReproduction')
            cfg.set('DefaultReproduction', 'elitism', 
                   str(config.reproduction_config.elitism))
            cfg.set('DefaultReproduction', 'survival_threshold', 
                   str(config.reproduction_config.survival_threshold))
            
            # Write the config file
            with open(os.path.join(results_dir, 'reconstructed_config.ini'), 'w') as f:
                cfg.write(f)
                
        except Exception as e:
            print(f"Warning: Could not reconstruct config file: {e}")
    
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