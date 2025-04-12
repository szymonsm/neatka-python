"""
Analysis script for NEAT parameter sweep results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import json

def load_results(results_dir):
    """Load consolidated results from the given directory."""
    consolidated_path = os.path.join(results_dir, 'consolidated_results.csv')
    if os.path.exists(consolidated_path):
        return pd.read_csv(consolidated_path)
    else:
        print(f"No consolidated results found at {consolidated_path}")
        return None

def plot_performance_by_parameter(df, param_name, output_dir):
    """Plot performance across different values of a parameter."""
    if df is None or df.empty:
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by parameter and network type
    param_values = sorted(df[param_name].unique())
    network_types = sorted(df['network_type'].unique())
    
    # Create a figure with two subplots: one for best fitness, one for convergence speed
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot best achieved fitness
    for net_type in network_types:
        best_fitness = []
        error_bars = []
        
        for val in param_values:
            subset = df[(df[param_name] == val) & (df['network_type'] == net_type)]
            if not subset.empty:
                # Group by seed and get the best fitness for each run
                seed_groups = subset.groupby('seed')
                max_fitness_per_seed = seed_groups['best_fitness'].max()
                
                best_fitness.append(max_fitness_per_seed.mean())
                error_bars.append(max_fitness_per_seed.std())
        
        axes[0].errorbar(param_values, best_fitness, yerr=error_bars, marker='o', 
                       label=f"{net_type}")
    
    axes[0].set_xlabel(param_name)
    axes[0].set_ylabel('Best Fitness')
    axes[0].set_title(f'Best Fitness by {param_name}')
    axes[0].grid(True)
    axes[0].legend()
    
    # Plot convergence speed (generations to 90% of max)
    for net_type in network_types:
        convergence_gens = []
        error_bars = []
        
        for val in param_values:
            subset = df[(df[param_name] == val) & (df['network_type'] == net_type)]
            if not subset.empty:
                # Group by seed
                conv_per_seed = []
                for seed, seed_data in subset.groupby('seed'):
                    seed_data = seed_data.sort_values('generation')
                    max_fitness = seed_data['best_fitness'].max()
                    threshold = 0.9 * max_fitness
                    
                    # Find first generation where fitness exceeds 90% of maximum
                    exceeded = seed_data[seed_data['best_fitness'] >= threshold]
                    if not exceeded.empty:
                        conv_per_seed.append(exceeded.iloc[0]['generation'])
                    else:
                        # If never exceeds threshold, use max generations
                        conv_per_seed.append(seed_data['generation'].max())
                
                if conv_per_seed:
                    convergence_gens.append(np.mean(conv_per_seed))
                    error_bars.append(np.std(conv_per_seed))
                else:
                    convergence_gens.append(np.nan)
                    error_bars.append(np.nan)
        
        axes[1].errorbar(param_values, convergence_gens, yerr=error_bars, marker='o', 
                       label=f"{net_type}")
    
    axes[1].set_xlabel(param_name)
    axes[1].set_ylabel('Generations to 90% Max Fitness')
    axes[1].set_title(f'Convergence Speed by {param_name}')
    axes[1].grid(True)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'performance_by_{param_name}.png'), dpi=300)
    plt.close()

def plot_fitness_over_generations(df, output_dir):
    """Plot fitness over generations for different parameter configurations."""
    if df is None or df.empty:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare network types
    network_types = sorted(df['network_type'].unique())
    
    plt.figure(figsize=(10, 6))
    
    for net_type in network_types:
        net_data = df[df['network_type'] == net_type].groupby('generation')['best_fitness'].mean()
        gen_std = df[df['network_type'] == net_type].groupby('generation')['best_fitness'].std()
        
        generations = net_data.index
        plt.plot(generations, net_data, label=f"{net_type}")
        plt.fill_between(generations, net_data - gen_std, net_data + gen_std, alpha=0.2)
    
    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')
    plt.title('Fitness over Generations by Network Type')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'fitness_by_network_type.png'), dpi=300)
    plt.close()
    
    # Compare population sizes
    population_sizes = sorted(df['population_size'].unique())
    
    plt.figure(figsize=(12, 6))
    
    for net_type in network_types:
        plt.subplot(1, len(network_types), network_types.index(net_type) + 1)
        
        for pop_size in population_sizes:
            pop_data = df[(df['network_type'] == net_type) & (df['population_size'] == pop_size)]
            pop_mean = pop_data.groupby('generation')['best_fitness'].mean()
            
            generations = pop_mean.index
            plt.plot(generations, pop_mean, label=f"Pop {pop_size}")
        
        plt.xlabel('Generation')
        plt.ylabel('Best Fitness')
        plt.title(f'{net_type} Network')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fitness_by_population_size.png'), dpi=300)
    plt.close()

def generate_summary_table(df, output_dir):
    """Generate summary statistics table."""
    if df is None or df.empty:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Group by network type and parameter combinations
    grouped = df.groupby([
        'network_type', 'population_size', 'num_hidden', 
        'conn_add_prob', 'conn_delete_prob', 'node_add_prob', 'node_delete_prob'
    ])
    
    # Calculate statistics
    stats = grouped.agg({
        'best_fitness': ['mean', 'std', 'max'],
        'complexity': ['mean', 'max'],
        'hidden_nodes': ['mean', 'max']
    }).reset_index()
    
    # Sort by average best fitness
    stats = stats.sort_values([('best_fitness', 'mean')], ascending=False)
    
    # Save to CSV
    stats.to_csv(os.path.join(output_dir, 'parameter_stats_summary.csv'))
    
    # Top 10 configurations per network type
    for net_type in df['network_type'].unique():
        net_stats = stats[stats['network_type'] == net_type].head(10)
        net_stats.to_csv(os.path.join(output_dir, f'top_configs_{net_type}.csv'))

def run_analysis(results_dir):
    """Run comprehensive analysis on parameter sweep results."""
    output_dir = os.path.join(results_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    results = load_results(results_dir)
    if results is None:
        print("No results to analyze.")
        return
    
    # Plot performance by each parameter
    for param in ['population_size', 'num_hidden', 'conn_add_prob', 
                 'conn_delete_prob', 'node_add_prob', 'node_delete_prob']:
        plot_performance_by_parameter(results, param, output_dir)
    
    # Plot fitness over generations
    plot_fitness_over_generations(results, output_dir)
    
    # Generate summary tables
    generate_summary_table(results, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze NEAT parameter sweep results")
    parser.add_argument("results_dir", help="Directory containing parameter sweep results")
    args = parser.parse_args()
    
    run_analysis(args.results_dir)
