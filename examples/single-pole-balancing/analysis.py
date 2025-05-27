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

def plot_convergence_distribution(df, output_dir, target_fitness=60.0):
    """
    Plot the distribution of convergence lengths (generations needed to reach target fitness).
    Collects all convergence points across all runs for each network type.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save output files
        target_fitness: Target fitness threshold (default: 60.0)
    """
    if df is None or df.empty:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare network types
    network_types = sorted(df['network_type'].unique())
    
    plt.figure(figsize=(14, 8))
    
    # Collect all convergence generations for each network type
    convergence_data = {}
    max_gen = 0  # Track maximum generation for x-axis limit
    
    for net_type in network_types:
        net_data = df[df['network_type'] == net_type]
        convergence_gens = []
        
        # Get unique experiment combinations (identified by seed, population_size, etc.)
        experiments = net_data.drop_duplicates(subset=['seed', 'population_size', 'num_hidden', 
                                                     'conn_add_prob', 'conn_delete_prob', 
                                                     'node_add_prob', 'node_delete_prob'])
        
        # For each experiment, find when it first reached target fitness
        for _, exp in experiments.iterrows():
            # Get all data for this experiment
            exp_data = net_data[
                (net_data['seed'] == exp['seed']) &
                (net_data['population_size'] == exp['population_size']) &
                (net_data['num_hidden'] == exp['num_hidden']) &
                (net_data['conn_add_prob'] == exp['conn_add_prob']) &
                (net_data['conn_delete_prob'] == exp['conn_delete_prob']) &
                (net_data['node_add_prob'] == exp['node_add_prob']) &
                (net_data['node_delete_prob'] == exp['node_delete_prob'])
            ].sort_values('generation')
            
            # Find when target fitness was reached
            exceeded = exp_data[exp_data['best_fitness'] >= target_fitness]
            if not exceeded.empty:
                conv_gen = exceeded.iloc[0]['generation']
                convergence_gens.append(conv_gen)
                max_gen = max(max_gen, conv_gen)
        
        convergence_data[net_type] = convergence_gens
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(network_types)))
    # Reverse order for better visibility
    colors = colors[::-1]

    # Plot distribution for each network type
    for i, net_type in enumerate(network_types):
        valid_gens = convergence_data[net_type]
        if valid_gens:
            # Use kernel density estimation with bounded support (min=0)
            sns.kdeplot(valid_gens, label=f"{net_type} (n={len(valid_gens)})",
                       fill=True, alpha=0.3, linewidth=2, clip=(0, None), color=colors[i])
            
            # Mark the median with a vertical line
            median_gen = np.median(valid_gens)
            plt.axvline(median_gen, 
                      color=colors[i], 
                      linestyle='--', 
                      alpha=0.8, 
                      linewidth=2)
            
            # Add summary statistics as text
            plt.text(median_gen + (-7.5) if i == 0 else median_gen + 1, 0.01, 
                   f"Median: {median_gen:.1f}\nMean: {np.mean(valid_gens):.1f}\nStd: {np.std(valid_gens):.1f}", 
                   color=colors[i], 
                   fontweight='bold',
                   fontsize=9,
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.3'))
    
    # Set x-axis limits to avoid negative values
    plt.xlim(0, max_gen * 1.1)  # Add 10% padding to the right
    
    plt.xlabel('Generations to Reach Fitness ≥ 60', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title(f'Distribution of Convergence Speed (Target Fitness: {target_fitness})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Network Type")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'convergence_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_convergence_failure_analysis(df, summary_df, output_dir, target_fitness=60.0):
    """
    Plot analysis of convergence failures - how many experiments failed to reach the target fitness
    or failed entirely (status="failed").
    
    Args:
        df: DataFrame with results from completed experiments
        summary_df: DataFrame with experiment summary including failed runs
        output_dir: Directory to save output files
        target_fitness: Target fitness threshold (default: 60.0)
    """
    if summary_df is None or summary_df.empty:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare network types
    network_types = sorted(summary_df['network_type'].unique())
    
    # Collect data by network type
    param_names = ['population_size', 'num_hidden', 'conn_add_prob', 'conn_delete_prob', 
                  'node_add_prob', 'node_delete_prob']
    
    # Create overall success/failure stats first
    plt.figure(figsize=(12, 7))
    
    # Prepare data for the plot
    success_counts = []
    convergence_failure_counts = []  # Ran but didn't reach target fitness
    runtime_failure_counts = []      # Failed to complete (status="failed")
    total_counts = []
    
    for net_type in network_types:
        # Get all experiments for this network type from summary
        net_exps = summary_df[summary_df['network_type'] == net_type]
        total_count = len(net_exps)
        
        # Count runtime failures (status="failed")
        runtime_failures = sum(net_exps['status'] == 'failed')
        
        # For convergence failures, we need to look at the completed experiments
        completed_exps = net_exps[net_exps['status'] == 'completed']
        
        # Find completed experiments in the results dataframe
        if df is not None and not df.empty:
            net_data = df[df['network_type'] == net_type]
            
            # Group by seed to find max fitness per experiment
            convergence_failures = 0
            success_count = 0
            
            # Check each completed experiment
            for _, row in completed_exps.iterrows():
                seed = row['seed']
                pop_size = row['population_size']
                num_hidden = row['num_hidden']
                _data = net_data[(net_data['seed'] == seed) &
                                   (net_data['population_size'] == pop_size) &
                                      (net_data['num_hidden'] == num_hidden)]
                
                if not _data.empty:
                    max_fitness = _data['best_fitness'].max()
                    if max_fitness < target_fitness:
                        convergence_failures += 1
                    else:
                        success_count += 1
        else:
            # If no results data, assume all completed runs were successful
            convergence_failures = 0
            success_count = len(completed_exps)
        
        success_counts.append(success_count)
        convergence_failure_counts.append(convergence_failures)
        runtime_failure_counts.append(runtime_failures)
        total_counts.append(total_count)
    
    # Create stacked bar chart
    bar_width = 0.6
    x = np.arange(len(network_types))
    
    # Calculate percentages
    success_pcts = [s/t * 100 for s, t in zip(success_counts, total_counts)]
    conv_fail_pcts = [c/t * 100 for c, t in zip(convergence_failure_counts, total_counts)]
    runtime_fail_pcts = [r/t * 100 for r, t in zip(runtime_failure_counts, total_counts)]
    
    # Create stacked bars
    plt.bar(x, success_pcts, bar_width, color='#2ecc71', label='Success')
    plt.bar(x, conv_fail_pcts, bar_width, bottom=success_pcts, color='#f39c12', 
           label='Failed to reach target')
    plt.bar(x, runtime_fail_pcts, bar_width, 
           bottom=[s+c for s, c in zip(success_pcts, conv_fail_pcts)], 
           color='#e74c3c', label='Failed to complete')
    
    # Add percentage labels on bars
    for i, (s, c, r, t) in enumerate(zip(success_counts, convergence_failure_counts, 
                                        runtime_failure_counts, total_counts)):
        # Success percentage
        if s > 0:
            plt.text(x[i], success_pcts[i]/2, 
                   f"{success_pcts[i]:.1f}%\n({s}/{t})", 
                   ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
        
        # Convergence failure percentage
        if c > 0:
            plt.text(x[i], success_pcts[i] + conv_fail_pcts[i]/2, 
                   f"{conv_fail_pcts[i]:.1f}%\n({c}/{t})", 
                   ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
        
        # Runtime failure percentage
        if r > 0:
            plt.text(x[i], success_pcts[i] + conv_fail_pcts[i] + runtime_fail_pcts[i]/2, 
                   f"{runtime_fail_pcts[i]:.1f}%\n({r}/{t})", 
                   ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='white')
    
    plt.ylabel('Percentage of Experiments', fontsize=12)
    plt.title(f'Experiment Results (Target Fitness ≥ {target_fitness})', fontsize=14)
    plt.xticks(x, network_types)
    plt.yticks(np.arange(0, 101, 10))
    plt.legend(loc='upper right')
    plt.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'experiment_results_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Now analyze failure by parameter for each network type
    for param in param_names:
        plt.figure(figsize=(12, 7))
        
        param_values = sorted(summary_df[param].dropna().unique())
        x = np.arange(len(param_values))
        width = 0.8 / len(network_types)

        for i, net_type in enumerate(network_types):
            success_by_param = []
            conv_fail_by_param = []
            runtime_fail_by_param = []
            totals_by_param = []

            for val in param_values:
                param_exps = summary_df[(summary_df['network_type'] == net_type) & 
                                        (summary_df[param] == val)]
                total_count = len(param_exps)
                runtime_failures = sum(param_exps['status'] == 'failed')
                completed_exps = param_exps[param_exps['status'] == 'completed']
                
                convergence_failures = 0
                success_count = 0

                if df is not None and not df.empty:
                    for _, row in completed_exps.iterrows():
                        seed = row['seed']
                        subset = df[(df['network_type'] == net_type) & 
                                    (df[param] == val) & 
                                    (df['seed'] == seed)]
                        if not subset.empty:
                            max_fitness = subset['best_fitness'].max()
                            if max_fitness < target_fitness:
                                convergence_failures += 1
                            else:
                                success_count += 1
                        else:
                            convergence_failures += 1
                else:
                    success_count = len(completed_exps)

                success_by_param.append(success_count)
                conv_fail_by_param.append(convergence_failures)
                runtime_fail_by_param.append(runtime_failures)
                totals_by_param.append(total_count)

            # Percentages
            success_pcts = [s/t * 100 if t > 0 else 0 for s, t in zip(success_by_param, totals_by_param)]
            conv_fail_pcts = [c/t * 100 if t > 0 else 0 for c, t in zip(conv_fail_by_param, totals_by_param)]
            runtime_fail_pcts = [r/t * 100 if t > 0 else 0 for r, t in zip(runtime_fail_by_param, totals_by_param)]

            # Align bars centered on x
            pos = x - (width * len(network_types) / 2) + (i + 0.5) * width

            plt.bar(pos, success_pcts, width, color='#2ecc71', alpha=0.8, label=None if i > 0 else 'Success')
            plt.bar(pos, conv_fail_pcts, width, bottom=success_pcts, color='#f39c12', alpha=0.8,
                    label=None if i > 0 else 'Failed to reach target')
            plt.bar(pos, runtime_fail_pcts, width, bottom=[s + c for s, c in zip(success_pcts, conv_fail_pcts)],
                    color='#e74c3c', alpha=0.8, label=None if i > 0 else 'Failed to complete')

        plt.xlabel(param, fontsize=12)
        plt.ylabel('Percentage of Experiments', fontsize=12)
        plt.title(f'Results by {param} (Target Fitness ≥ {target_fitness})', fontsize=14)
        plt.xticks(x, param_values, rotation=45 if len(param_values) > 10 else 0)
        plt.ylim(0, 105)
        plt.grid(axis='y', alpha=0.3)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'results_by_{param}.png'), dpi=300, bbox_inches='tight')
        plt.close()


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

def plot_fitness_by_hidden_nodes(df, output_dir):
    """
    Plot fitness evolution over generations for different numbers of hidden nodes.
    Shows one subplot per network type for easier comparison.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save output files
    """
    if df is None or df.empty:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare network types
    network_types = sorted(df['network_type'].unique())
    
    # Get unique hidden node counts
    hidden_node_counts = sorted(df['num_hidden'].unique())
    
    # Create subplots - one for each network type
    fig, axes = plt.subplots(1, len(network_types), figsize=(16, 6), sharey=True)
    if len(network_types) == 1:
        axes = [axes]  # Make axes iterable even with just one subplot
    
    for i, net_type in enumerate(network_types):
        ax = axes[i]
        
        # Define colors for different hidden node counts
        colors = plt.cm.viridis(np.linspace(0, 1, len(hidden_node_counts)))
        
        for j, num_hidden in enumerate(hidden_node_counts):
            # Get data for this network type and hidden node count
            subset = df[(df['network_type'] == net_type) & (df['num_hidden'] == num_hidden)]
            
            if not subset.empty:
                # Plot individual runs with transparency
                for seed in subset['seed'].unique():
                    seed_data = subset[subset['seed'] == seed].sort_values('generation')
                    ax.plot(seed_data['generation'], seed_data['best_fitness'], 
                          color=colors[j], alpha=0.2, linewidth=0.6)
                
                # Calculate mean fitness per generation
                mean_fitness = subset.groupby('generation')['best_fitness'].mean()
                generations = mean_fitness.index.tolist()
                
                # Plot mean fitness with full opacity
                ax.plot(generations, mean_fitness.values, 
                       color=colors[j], linewidth=2.5, 
                       label=f"{num_hidden} hidden nodes")
                
                # Find and mark the maximum fitness point
                if len(mean_fitness.values) > 0:
                    max_gen_idx = mean_fitness.values.argmax()
                    max_gen = generations[max_gen_idx]
                    max_fitness = mean_fitness.values[max_gen_idx]
                    
                    ax.scatter([max_gen], [max_fitness], 
                             s=100, color=colors[j], marker='o', 
                             edgecolor='black', linewidth=1, zorder=10)
                    
                    # Add text annotation for max fitness
                    ax.annotate(f"Max: {max_fitness:.1f}", 
                              xy=(max_gen, max_fitness), 
                              xytext=(5, 5), textcoords='offset points', 
                              fontsize=8, color=colors[j], fontweight='bold')
        
        ax.set_xlabel('Generation', fontsize=11)
        if i == 0:  # Only add y-label to leftmost subplot
            ax.set_ylabel('Best Fitness', fontsize=11)
        ax.set_title(f'{net_type} Network', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Hidden Nodes')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fitness_by_hidden_nodes.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Keep the comparative bar chart as a separate visualization
    plt.figure(figsize=(12, 8))
    
    # Track the best performance for each configuration
    max_fitness_by_config = {}
    
    # Group by network type and hidden node count
    for i, net_type in enumerate(network_types):
        # Get best fitness for each hidden node count
        best_fitness = []
        std_devs = []
        
        for num_hidden in hidden_node_counts:
            subset = df[(df['network_type'] == net_type) & (df['num_hidden'] == num_hidden)]
            if not subset.empty:
                # Group by experiment and get max fitness
                grouped = subset.groupby(['seed', 'population_size', 'conn_add_prob', 
                                         'conn_delete_prob', 'node_add_prob', 'node_delete_prob'])
                max_fitness_per_exp = grouped['best_fitness'].max()
                
                mean_max = max_fitness_per_exp.mean()
                std_dev = max_fitness_per_exp.std()
                
                best_fitness.append(mean_max)
                std_devs.append(std_dev)
                
                # Store for tracking best config
                max_fitness_by_config[(net_type, num_hidden)] = mean_max
            else:
                best_fitness.append(0)
                std_devs.append(0)
        
        # Plot with offset for better visibility
        offset = i * 0.15 - 0.15 * (len(network_types) - 1) / 2
        x_pos = np.array(range(len(hidden_node_counts))) + offset
        
        plt.bar(x_pos, best_fitness, width=0.15, 
               label=f"{net_type}", 
               color=plt.cm.Set1(i/max(len(network_types), 1)), 
               alpha=0.8)
        
        # Add error bars
        plt.errorbar(x_pos, best_fitness, yerr=std_devs, fmt='none', color='black', 
                    capsize=4, elinewidth=1, alpha=0.7)
        
        # Add value labels
        for j, (x, y) in enumerate(zip(x_pos, best_fitness)):
            plt.text(x, y + 0.5, f"{y:.1f}", ha='center', va='bottom', 
                   fontsize=8, rotation=0)
    
    # Find the best overall configuration
    if max_fitness_by_config:
        best_config = max(max_fitness_by_config.items(), key=lambda x: x[1])
        best_net_type, best_hidden = best_config[0]
        best_fitness = best_config[1]
        
        # Mark the best configuration
        best_idx = hidden_node_counts.index(best_hidden)
        best_net_idx = network_types.index(best_net_type)
        x_offset = best_net_idx * 0.15 - 0.15 * (len(network_types) - 1) / 2
        best_x = best_idx + x_offset
        
        plt.scatter([best_x], [best_fitness], s=150, marker='o', 
                   edgecolor='black', color='gold', zorder=10,
                   label="Best Config")
        
        plt.annotate(f"Best: {best_fitness:.1f}\n{best_net_type}, {best_hidden} nodes", 
                   xy=(best_x, best_fitness), 
                   xytext=(0, 20), textcoords='offset points', 
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", fc='white', ec='gold', alpha=0.9),
                   ha='center',
                   arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2", 
                                 color='black'))
    
    plt.xlabel('Number of Hidden Nodes', fontsize=12)
    plt.ylabel('Best Fitness (Mean across runs)', fontsize=12)
    plt.title('Performance Comparison by Hidden Node Count', fontsize=14)
    plt.xticks(range(len(hidden_node_counts)), hidden_node_counts)
    plt.grid(axis='y', alpha=0.3)
    plt.legend(title="Network Type")
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'hidden_nodes_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_fitness_over_generations(df, output_dir):
    """
    Plot fitness over generations for different parameter configurations.
    Shows raw data with transparency (alpha=0.4) and highlights when
    maximum fitness is reached with full opacity (alpha=1.0).
    """
    if df is None or df.empty:
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare network types
    network_types = sorted(df['network_type'].unique())
    
    plt.figure(figsize=(12, 7))
    
    # Define colors for consistency
    colors = plt.cm.Set1(np.linspace(0, 1, len(network_types)))
    # Reverse order for better visibility
    colors = colors[::-1]
    
    for i, net_type in enumerate(network_types):
        net_type_data = df[df['network_type'] == net_type]
        
        # Plot individual runs with transparency
        for seed in net_type_data['seed'].unique():
            seed_data = net_type_data[net_type_data['seed'] == seed].sort_values('generation')
            plt.plot(seed_data['generation'], seed_data['best_fitness'], 
                    color=colors[i], alpha=0.4, linewidth=0.8)
        
        # Calculate mean fitness per generation
        mean_fitness = net_type_data.groupby('generation')['best_fitness'].mean()
        generations = mean_fitness.index
        
        # Plot all mean data with semi-transparency
        plt.plot(generations, mean_fitness, color=colors[i], alpha=0.6, linewidth=1.5)
        
        # Find the generation where maximum fitness is reached
        max_gen_idx = mean_fitness.argmax()
        max_gen = generations[max_gen_idx]
        max_fitness = mean_fitness.iloc[max_gen_idx]
        
        # Plot mean fitness up to max with full opacity
        plt.plot(generations[:max_gen_idx+1], mean_fitness[:max_gen_idx+1], 
                color=colors[i], linewidth=2.5, alpha=1.0, label=f"{net_type}")
        
        # Add marker at max point
        plt.scatter([max_gen], [max_fitness], 
                   s=120, color=colors[i], marker='o', edgecolor='black',
                   label=f"{net_type} max at gen {max_gen}", zorder=10)
        
        # Add text annotation for max fitness
        plt.annotate(f"Max: {max_fitness:.2f} at gen {max_gen}", 
                    xy=(max_gen, max_fitness), 
                    xytext=(10, 10), textcoords='offset points', 
                    fontsize=9, color='black', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", fc=colors[i], alpha=0.7))
    
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Best Fitness', fontsize=12)
    plt.title('Fitness over Generations by Network Type', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Create custom legend without duplicating entries
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='lower right')
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'fitness_by_network_type.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Compare population sizes
    population_sizes = sorted(df['population_size'].unique())
    
    # Create subplots - one for each network type
    fig, axes = plt.subplots(1, len(network_types), figsize=(16, 6), sharey=True)
    if len(network_types) == 1:
        axes = [axes]  # Make axes iterable even with just one subplot
    
    for i, net_type in enumerate(network_types):
        ax = axes[i]
        
        for j, pop_size in enumerate(population_sizes):
            pop_color = plt.cm.viridis(j/max(len(population_sizes)-1, 1))
            
            # Get data for this network type and population size
            pop_data = df[(df['network_type'] == net_type) & (df['population_size'] == pop_size)]
            
            if not pop_data.empty:
                # Plot individual runs with transparency
                for seed in pop_data['seed'].unique():
                    seed_data = pop_data[pop_data['seed'] == seed].sort_values('generation')
                    ax.plot(seed_data['generation'], seed_data['best_fitness'], 
                          color=pop_color, alpha=0.4, linewidth=0.8)
                
                # Calculate mean fitness per generation
                pop_mean = pop_data.groupby('generation')['best_fitness'].mean()
                generations = pop_mean.index
                
                # Find the generation where maximum fitness is reached
                max_gen_idx = pop_mean.argmax()
                max_gen = generations[max_gen_idx]
                max_fitness = pop_mean.iloc[max_gen_idx]
                
                # Plot mean fitness up to max with full opacity
                ax.plot(generations[:max_gen_idx+1], pop_mean[:max_gen_idx+1], 
                      color=pop_color, linewidth=2.5, alpha=1.0, label=f"Pop {pop_size}")
                
                # Add marker at max point
                ax.scatter([max_gen], [max_fitness], 
                         s=80, color=pop_color, marker='o', zorder=10)
                
                # Add text annotation for max generation only
                ax.annotate(f"Gen {max_gen}", 
                          xy=(max_gen, max_fitness), 
                          xytext=(5, 0), textcoords='offset points', 
                          fontsize=8, color=pop_color, fontweight='bold')
        
        ax.set_xlabel('Generation', fontsize=11)
        if i == 0:  # Only add y-label to leftmost subplot
            ax.set_ylabel('Best Fitness', fontsize=11)
        ax.set_title(f'{net_type} Network', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(title='Population Size')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fitness_by_population_size.png'), dpi=300, bbox_inches='tight')
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

def load_experiment_summary(results_dir):
    """Load experiment summary from the given directory."""
    summary_path = os.path.join(results_dir, 'experiment_summary.csv')
    if os.path.exists(summary_path):
        return pd.read_csv(summary_path)
    else:
        print(f"No experiment summary found at {summary_path}")
        return None

def run_analysis(results_dir):
    """Run comprehensive analysis on parameter sweep results."""
    output_dir = os.path.join(results_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results data
    results = load_results(results_dir)
    
    # Load experiment summary (includes failed experiments)
    summary = load_experiment_summary(results_dir)
    
    if results is None and summary is None:
        print("No results to analyze.")
        return
    
    # Plot performance by each parameter (if results data is available)
    if results is not None:
        for param in ['population_size', 'num_hidden', 'conn_add_prob', 
                     'conn_delete_prob', 'node_add_prob', 'node_delete_prob']:
            plot_performance_by_parameter(results, param, output_dir)
        
        # Plot fitness over generations
        plot_fitness_over_generations(results, output_dir)
        
        # Plot fitness by number of hidden nodes
        plot_fitness_by_hidden_nodes(results, output_dir)
        
        # Plot convergence distribution (generations to reach fitness 60)
        plot_convergence_distribution(results, output_dir, target_fitness=60.0)
    
    # Plot failure analysis (using both results and summary data)
    plot_convergence_failure_analysis(results, summary, output_dir, target_fitness=60.0)
    
    # Generate summary tables (if results data is available)
    if results is not None:
        generate_summary_table(results, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze NEAT parameter sweep results")
    parser.add_argument("results_dir", help="Directory containing parameter sweep results")
    args = parser.parse_args()
    
    run_analysis(args.results_dir)
