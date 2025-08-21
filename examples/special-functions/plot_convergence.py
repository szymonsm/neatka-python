#!/usr/bin/env python3
"""
Convergence plots comparing KAN-NEAT and PyKAN training across all functions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from typing import Dict, List, Tuple

def load_best_results(filepath: str) -> pd.DataFrame:
    """Load the best results per function and method."""
    return pd.read_csv(filepath)

def load_kan_neat_results(results_dir: str) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Load KAN-NEAT results from all seed directories.
    
    Returns:
        Dict with structure: {function: {seed: dataframe}}
    """
    kan_data = {}
    
    # Find all KAN-NEAT result directories
    kan_dirs = glob.glob(os.path.join(results_dir, "kan-*-seed*"))
    
    for kan_dir in kan_dirs:
        # Parse directory name: kan-{function}-seed{seed}
        dir_name = os.path.basename(kan_dir)
        parts = dir_name.split('-')
        if len(parts) >= 3 and parts[0] == 'kan':
            function = '-'.join(parts[1:-1])  # Handle functions with hyphens
            seed_part = parts[-1]
            if seed_part.startswith('seed'):
                seed = int(seed_part[4:])
                
                # Look for results file
                results_files = glob.glob(os.path.join(kan_dir, f"results_{seed}.csv"))
                if results_files:
                    try:
                        df = pd.read_csv(results_files[0])
                        if function not in kan_data:
                            kan_data[function] = {}
                        kan_data[function][seed] = df
                        print(f"Loaded KAN-NEAT data: {function}, seed {seed}")
                    except Exception as e:
                        print(f"Error loading {results_files[0]}: {e}")
    
    return kan_data

def load_pykan_results(results_dir: str) -> Dict[str, Dict[int, pd.DataFrame]]:
    """
    Load PyKAN results from all seed directories.
    
    Returns:
        Dict with structure: {function: {seed: dataframe}}
    """
    pykan_data = {}
    
    # Find all PyKAN result directories
    pykan_dirs = glob.glob(os.path.join(results_dir, "pykan_results", "pykan-*-seed*"))
    
    for pykan_dir in pykan_dirs:
        # Parse directory name: pykan-{function}-seed{seed}
        dir_name = os.path.basename(pykan_dir)
        parts = dir_name.split('-')
        if len(parts) >= 3 and parts[0] == 'pykan':
            function = '-'.join(parts[1:-1])  # Handle functions with hyphens
            seed_part = parts[-1]
            if seed_part.startswith('seed'):
                seed = int(seed_part[4:])
                
                # Look for losses file
                losses_file = os.path.join(pykan_dir, "losses.txt")
                if os.path.exists(losses_file):
                    try:
                        df = pd.read_csv(losses_file, sep='\t')
                        if function not in pykan_data:
                            pykan_data[function] = {}
                        pykan_data[function][seed] = df
                        print(f"Loaded PyKAN data: {function}, seed {seed}")
                    except Exception as e:
                        print(f"Error loading {losses_file}: {e}")
    
    return pykan_data

def plot_convergence_comparison(kan_data: Dict[str, Dict[int, pd.DataFrame]], 
                              pykan_data: Dict[str, Dict[int, pd.DataFrame]], 
                              best_results: pd.DataFrame,
                              output_dir: str):
    """
    Create convergence plots for each function comparing KAN-NEAT and PyKAN on the same plot.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    functions = set(kan_data.keys()) | set(pykan_data.keys())
    
    for function in functions:
        print(f"\nPlotting convergence for function: {function}")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        fig.suptitle(f'Training Convergence Comparison: {function}', fontsize=16, fontweight='bold')
        
        # Get best results for highlighting
        best_kan = best_results[(best_results['function'] == function) & 
                               (best_results['method'] == 'kan-neat')]
        best_pykan = best_results[(best_results['function'] == function) & 
                                 (best_results['method'] == 'pykan')]
        
        best_kan_seed = best_kan['seed'].iloc[0] if not best_kan.empty else None
        best_pykan_seed = best_pykan['seed'].iloc[0] if not best_pykan.empty else None
        
        # Plot KAN-NEAT results (blue color family)
        if function in kan_data:
            for seed, df in kan_data[function].items():
                if 'generation' in df.columns and 'best_fitness' in df.columns:
                    alpha = 1.0 if seed == best_kan_seed else 0.4
                    linewidth = 3 if seed == best_kan_seed else 1.5
                    color = 'darkblue' if seed == best_kan_seed else 'blue'
                    linestyle = '-' if seed == best_kan_seed else '-'
                    
                    label = f'KAN-NEAT seed {seed}' + (' (best)' if seed == best_kan_seed else '')
                    
                    ax.plot(df['generation'], df['best_fitness'], 
                           alpha=alpha, linewidth=linewidth, color=color, 
                           linestyle=linestyle, label=label)
        
        # Plot PyKAN results (red color family)
        if function in pykan_data:
            for seed, df in pykan_data[function].items():
                if 'Step' in df.columns and 'Train_Loss' in df.columns:
                    alpha = 1.0 if seed == best_pykan_seed else 0.4
                    linewidth = 3 if seed == best_pykan_seed else 1.5
                    color = 'darkred' if seed == best_pykan_seed else 'red'
                    linestyle = '--' if seed == best_pykan_seed else '--'
                    
                    label = f'PyKAN seed {seed}' + (' (best)' if seed == best_pykan_seed else '')
                    
                    ax.plot(df['Step'], df['Train_Loss'], 
                           alpha=alpha, linewidth=linewidth, color=color,
                           linestyle=linestyle, label=label)
        
        ax.set_xlabel('Training Steps/Generations')
        ax.set_ylabel('Training RMSE/Loss')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(output_dir, f'convergence_{function}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()

def plot_combined_convergence(kan_data: Dict[str, Dict[int, pd.DataFrame]], 
                            pykan_data: Dict[str, Dict[int, pd.DataFrame]], 
                            best_results: pd.DataFrame,
                            output_dir: str):
    """
    Create a combined plot showing all functions in one figure with consistent colors.
    """
    functions = sorted(set(kan_data.keys()) | set(pykan_data.keys()))
    
    fig, axes = plt.subplots(len(functions), 1, figsize=(12, 6 * len(functions)))
    if len(functions) == 1:
        axes = [axes]
    
    fig.suptitle('Training Convergence Comparison: KAN-NEAT vs PyKAN', fontsize=16, fontweight='bold')
    
    for i, function in enumerate(functions):
        print(f"\nProcessing function: {function}")
        ax = axes[i]
        
        # Get best results for highlighting
        best_kan = best_results[(best_results['function'] == function) & 
                               (best_results['method'] == 'kan-neat')]
        best_pykan = best_results[(best_results['function'] == function) & 
                                 (best_results['method'] == 'pykan')]
        
        best_kan_seed = best_kan['seed'].iloc[0] if not best_kan.empty else None
        best_pykan_seed = best_pykan['seed'].iloc[0] if not best_pykan.empty else None
        
        # Plot KAN-NEAT results (blue color family)
        if function in kan_data:
            for seed, df in kan_data[function].items():
                if 'generation' in df.columns and 'best_fitness' in df.columns:
                    alpha = 1.0 if seed == best_kan_seed else 0.4
                    linewidth = 3 if seed == best_kan_seed else 1.5
                    color = 'darkblue' if seed == best_kan_seed else 'blue'
                    
                    label = f'KAN-NEAT seed {seed}' + (' (best)' if seed == best_kan_seed else '') if i == 0 else None
                    
                    ax.plot(df['generation'], df['best_fitness'], 
                           alpha=alpha, linewidth=linewidth, color=color, label=label)
        
        # Plot PyKAN results (red color family)
        if function in pykan_data:
            for seed, df in pykan_data[function].items():
                if 'Step' in df.columns and 'Train_Loss' in df.columns:
                    alpha = 1.0 if seed == best_pykan_seed else 0.4
                    linewidth = 3 if seed == best_pykan_seed else 1.5
                    color = 'darkred' if seed == best_pykan_seed else 'red'
                    linestyle = '--'
                    
                    label = f'PyKAN seed {seed}' + (' (best)' if seed == best_pykan_seed else '') if i == 0 else None
                    
                    ax.plot(df['Step'], df['Train_Loss'], 
                           alpha=alpha, linewidth=linewidth, color=color,
                           linestyle=linestyle, label=label)
        
        ax.set_xlabel('Training Steps/Generations')
        ax.set_ylabel('Training RMSE/Loss')
        ax.set_title(f'{function}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        if i == 0:  # Only show legend on first subplot
            ax.legend()
    
    plt.tight_layout()
    
    # Save combined plot
    output_file = os.path.join(output_dir, 'convergence_combined.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved combined plot: {output_file}")
    plt.close()

def plot_method_comparison(kan_data: Dict[str, Dict[int, pd.DataFrame]], 
                          pykan_data: Dict[str, Dict[int, pd.DataFrame]], 
                          best_results: pd.DataFrame,
                          output_dir: str):
    """
    Create a clean comparison plot with consistent colors per method.
    """
    functions = sorted(set(kan_data.keys()) | set(pykan_data.keys()))
    
    fig, axes = plt.subplots(1, len(functions), figsize=(6 * len(functions), 6))
    if len(functions) == 1:
        axes = [axes]
    
    fig.suptitle('Training Convergence: KAN-NEAT vs PyKAN', fontsize=16, fontweight='bold')
    
    for i, function in enumerate(functions):
        ax = axes[i]
        
        # Get best results for highlighting
        best_kan = best_results[(best_results['function'] == function) & 
                               (best_results['method'] == 'kan-neat')]
        best_pykan = best_results[(best_results['function'] == function) & 
                                 (best_results['method'] == 'pykan')]
        
        best_kan_seed = best_kan['seed'].iloc[0] if not best_kan.empty else None
        best_pykan_seed = best_pykan['seed'].iloc[0] if not best_pykan.empty else None
        
        # Track if we've added legend labels
        kan_legend_added = False
        pykan_legend_added = False
        
        # Plot KAN-NEAT results (all blue)
        if function in kan_data:
            for seed, df in kan_data[function].items():
                if 'generation' in df.columns and 'best_fitness' in df.columns:
                    alpha = 1.0 if seed == best_kan_seed else 0.3
                    linewidth = 3 if seed == best_kan_seed else 1
                    color = 'blue'
                    
                    label = 'KAN-NEAT' if not kan_legend_added else None
                    if not kan_legend_added:
                        kan_legend_added = True
                    
                    ax.plot(df['generation'], df['best_fitness'], 
                           alpha=alpha, linewidth=linewidth, color=color, 
                           label=label, linestyle='-')
        
        # Plot PyKAN results (all red)
        if function in pykan_data:
            for seed, df in pykan_data[function].items():
                if 'Step' in df.columns and 'Train_Loss' in df.columns:
                    alpha = 1.0 if seed == best_pykan_seed else 0.3
                    linewidth = 3 if seed == best_pykan_seed else 1
                    color = 'red'
                    
                    label = 'PyKAN' if not pykan_legend_added else None
                    if not pykan_legend_added:
                        pykan_legend_added = True
                    
                    ax.plot(df['Step'], df['Train_Loss'], 
                           alpha=alpha, linewidth=linewidth, color=color,
                           label=label, linestyle='--')
        
        ax.set_xlabel('Training Steps/Generations')
        ax.set_ylabel('Training RMSE/Loss')
        ax.set_title(f'{function}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    plt.tight_layout()
    
    # Save plot
    output_file = os.path.join(output_dir, 'method_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nSaved method comparison plot: {output_file}")
    plt.close()

def create_convergence_summary(kan_data: Dict[str, Dict[int, pd.DataFrame]], 
                             pykan_data: Dict[str, Dict[int, pd.DataFrame]], 
                             best_results: pd.DataFrame,
                             output_dir: str):
    """
    Create a summary table of convergence characteristics.
    """
    summary_data = []
    
    functions = sorted(set(kan_data.keys()) | set(pykan_data.keys()))
    
    for function in functions:
        # KAN-NEAT summary
        if function in kan_data:
            best_kan = best_results[(best_results['function'] == function) & 
                                   (best_results['method'] == 'kan-neat')]
            best_kan_seed = best_kan['seed'].iloc[0] if not best_kan.empty else None
            
            if best_kan_seed and best_kan_seed in kan_data[function]:
                df = kan_data[function][best_kan_seed]
                final_rmse = df['best_fitness'].iloc[-1]
                total_generations = len(df)
                
                # Find generation where 95% of final performance was reached
                target_rmse = final_rmse * 1.05
                convergence_gen = len(df) - 1
                for gen, rmse in enumerate(df['best_fitness']):
                    if rmse <= target_rmse:
                        convergence_gen = gen
                        break
                
                summary_data.append({
                    'function': function,
                    'method': 'KAN-NEAT',
                    'seed': best_kan_seed,
                    'final_rmse': final_rmse,
                    'total_steps': total_generations,
                    'convergence_step': convergence_gen,
                    'convergence_ratio': convergence_gen / total_generations if total_generations > 0 else 0
                })
        
        # PyKAN summary
        if function in pykan_data:
            best_pykan = best_results[(best_results['function'] == function) & 
                                     (best_results['method'] == 'pykan')]
            best_pykan_seed = best_pykan['seed'].iloc[0] if not best_pykan.empty else None
            
            if best_pykan_seed and best_pykan_seed in pykan_data[function]:
                df = pykan_data[function][best_pykan_seed]
                final_loss = df['Train_Loss'].iloc[-1]
                total_steps = len(df)
                
                # Find step where 95% of final performance was reached
                target_loss = final_loss * 1.05
                convergence_step = len(df) - 1
                for step_idx, loss in enumerate(df['Train_Loss']):
                    if loss <= target_loss:
                        convergence_step = step_idx
                        break
                
                summary_data.append({
                    'function': function,
                    'method': 'PyKAN',
                    'seed': best_pykan_seed,
                    'final_rmse': final_loss,
                    'total_steps': total_steps,
                    'convergence_step': convergence_step,
                    'convergence_ratio': convergence_step / total_steps if total_steps > 0 else 0
                })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_file = os.path.join(output_dir, 'convergence_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"\nSaved convergence summary: {summary_file}")
    
    return summary_df

def main():
    # Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "special_functions_results")
    best_results_file = os.path.join(base_dir, "best_results_per_function_method.csv")
    output_dir = os.path.join(base_dir, "convergence_plots")
    
    print("Loading data...")
    
    # Load best results table
    best_results = load_best_results(best_results_file)
    print(f"Best results table:\n{best_results}")
    
    # Load training data
    kan_data = load_kan_neat_results(results_dir)
    pykan_data = load_pykan_results(results_dir)
    
    print(f"\nLoaded KAN-NEAT data for functions: {list(kan_data.keys())}")
    print(f"Loaded PyKAN data for functions: {list(pykan_data.keys())}")
    
    # Create plots
    print("\nCreating convergence plots...")
    plot_convergence_comparison(kan_data, pykan_data, best_results, output_dir)
    plot_combined_convergence(kan_data, pykan_data, best_results, output_dir)
    plot_method_comparison(kan_data, pykan_data, best_results, output_dir)
    
    # Create summary
    summary_df = create_convergence_summary(kan_data, pykan_data, best_results, output_dir)
    print(f"\nConvergence summary:\n{summary_df}")
    
    print(f"\nAll plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
