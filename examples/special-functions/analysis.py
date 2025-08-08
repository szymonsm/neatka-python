"""
Analysis script for special functions experiments.
Compares performance of KAN-NEAT, MLP-NEAT, and PyKAN on mathematical functions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_results(results_dir):
    """Load experiment results from CSV file."""
    results_path = os.path.join(results_dir, 'experiment_results.csv')
    if os.path.exists(results_path):
        return pd.read_csv(results_path)
    else:
        print(f"No results found at {results_path}")
        return None


def plot_method_comparison(df, output_dir):
    """Plot comparison between different methods."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to completed experiments
    completed_df = df[df['status'] == 'completed'].copy()
    
    if len(completed_df) == 0:
        print("No completed experiments to analyze")
        return
    
    # Normalize metrics for comparison
    # For NEAT methods, use fitness (higher is better)
    # For PyKAN, use negative test loss (higher is better)
    completed_df['performance'] = np.nan
    
    for idx, row in completed_df.iterrows():
        if row['method'] == 'pykan':
            # Use negative test loss as performance metric
            if not pd.isna(row.get('test_loss')):
                completed_df.loc[idx, 'performance'] = -row['test_loss']
        else:
            # Use fitness for NEAT methods
            if not pd.isna(row.get('best_fitness')):
                completed_df.loc[idx, 'performance'] = row['best_fitness']
    
    # Remove rows where performance couldn't be calculated
    completed_df = completed_df[~pd.isna(completed_df['performance'])]
    
    if len(completed_df) == 0:
        print("No valid performance data to analyze")
        return
    
    # 1. Performance by method (box plot)
    plt.figure(figsize=(12, 8))
    
    # Rename methods for better display
    method_names = {
        'mlp-neat': 'MLP-NEAT',
        'kan-neat': 'KAN-NEAT',
        'pykan': 'PyKAN'
    }
    completed_df['method_display'] = completed_df['method'].map(method_names)
    
    sns.boxplot(data=completed_df, x='method_display', y='performance')
    plt.title('Performance Comparison Across Methods', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Performance (higher is better)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Performance by function
    functions = completed_df['function'].unique()
    if len(functions) > 1:
        fig, axes = plt.subplots(1, 1, figsize=(14, 8))
        
        # Create a grouped bar plot
        method_performance = completed_df.groupby(['function', 'method_display'])['performance'].mean().unstack()
        method_performance.plot(kind='bar', ax=axes, width=0.8)
        
        axes.set_title('Average Performance by Function and Method', fontsize=16)
        axes.set_xlabel('Function', fontsize=14)
        axes.set_ylabel('Average Performance', fontsize=14)
        axes.grid(True, alpha=0.3)
        axes.legend(title='Method', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_by_function.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Runtime comparison
    plt.figure(figsize=(12, 8))
    sns.boxplot(data=completed_df, x='method_display', y='runtime')
    plt.title('Runtime Comparison Across Methods', fontsize=16)
    plt.xlabel('Method', fontsize=14)
    plt.ylabel('Runtime (seconds)', fontsize=14)
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'runtime_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pareto_frontier(df, output_dir):
    """Plot Pareto frontier of performance vs. complexity."""
    os.makedirs(output_dir, exist_ok=True)
    
    completed_df = df[df['status'] == 'completed'].copy()
    
    # Estimate complexity for different methods
    completed_df['complexity'] = np.nan
    
    for idx, row in completed_df.iterrows():
        if row['method'] == 'pykan':
            # Use number of parameters
            if not pd.isna(row.get('parameters')):
                completed_df.loc[idx, 'complexity'] = row['parameters']
        else:
            # For NEAT, we would need to parse the network structure
            # For now, use a placeholder based on typical NEAT network sizes
            completed_df.loc[idx, 'complexity'] = 50  # Approximate
    
    # Calculate performance metric
    completed_df['performance'] = np.nan
    for idx, row in completed_df.iterrows():
        if row['method'] == 'pykan':
            if not pd.isna(row.get('test_loss')):
                completed_df.loc[idx, 'performance'] = -row['test_loss']
        else:
            if not pd.isna(row.get('best_fitness')):
                completed_df.loc[idx, 'performance'] = row['best_fitness']
    
    # Remove invalid data
    plot_df = completed_df[~pd.isna(completed_df['complexity']) & ~pd.isna(completed_df['performance'])]
    
    if len(plot_df) == 0:
        print("No valid data for Pareto frontier plot")
        return
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    method_names = {
        'mlp-neat': 'MLP-NEAT',
        'kan-neat': 'KAN-NEAT',
        'pykan': 'PyKAN'
    }
    
    colors = {'MLP-NEAT': 'blue', 'KAN-NEAT': 'orange', 'PyKAN': 'green'}
    
    for method in plot_df['method'].unique():
        method_df = plot_df[plot_df['method'] == method]
        display_name = method_names.get(method, method)
        plt.scatter(method_df['complexity'], method_df['performance'], 
                   label=display_name, color=colors.get(display_name, 'gray'),
                   alpha=0.7, s=60)
    
    plt.xlabel('Model Complexity (parameters)', fontsize=14)
    plt.ylabel('Performance (higher is better)', fontsize=14)
    plt.title('Performance vs. Complexity', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pareto_frontier.png'), dpi=300, bbox_inches='tight')
    plt.close()


def generate_statistical_summary(df, output_dir):
    """Generate statistical summary of results."""
    os.makedirs(output_dir, exist_ok=True)
    
    completed_df = df[df['status'] == 'completed']
    
    summary_path = os.path.join(output_dir, 'statistical_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("Statistical Summary of Special Functions Experiments\n")
        f.write("=" * 60 + "\n\n")
        
        # Overall statistics
        total_experiments = len(df)
        completed = len(completed_df)
        success_rate = completed / total_experiments * 100
        
        f.write(f"Total experiments: {total_experiments}\n")
        f.write(f"Completed successfully: {completed} ({success_rate:.1f}%)\n")
        f.write(f"Failed: {total_experiments - completed}\n\n")
        
        # Results by method
        f.write("Results by Method:\n")
        f.write("-" * 30 + "\n")
        
        for method in completed_df['method'].unique():
            method_df = completed_df[completed_df['method'] == method]
            f.write(f"\n{method.upper()}:\n")
            f.write(f"  Number of experiments: {len(method_df)}\n")
            f.write(f"  Average runtime: {method_df['runtime'].mean():.2f} ± {method_df['runtime'].std():.2f} seconds\n")
            
            if method == 'pykan':
                if 'test_loss' in method_df.columns:
                    test_losses = method_df['test_loss'].dropna()
                    if len(test_losses) > 0:
                        f.write(f"  Average test loss: {test_losses.mean():.6f} ± {test_losses.std():.6f}\n")
                        f.write(f"  Best test loss: {test_losses.min():.6f}\n")
                if 'parameters' in method_df.columns:
                    params = method_df['parameters'].dropna()
                    if len(params) > 0:
                        f.write(f"  Average parameters: {params.mean():.0f} ± {params.std():.0f}\n")
            else:
                if 'best_fitness' in method_df.columns:
                    fitnesses = method_df['best_fitness'].dropna()
                    if len(fitnesses) > 0:
                        f.write(f"  Average fitness: {fitnesses.mean():.6f} ± {fitnesses.std():.6f}\n")
                        f.write(f"  Best fitness: {fitnesses.max():.6f}\n")
        
        # Results by function
        f.write(f"\n\nResults by Function:\n")
        f.write("-" * 30 + "\n")
        
        for function in completed_df['function'].unique():
            func_df = completed_df[completed_df['function'] == function]
            f.write(f"\n{function}:\n")
            f.write(f"  Total experiments: {len(func_df)}\n")
            
            # Performance by method for this function
            for method in func_df['method'].unique():
                method_func_df = func_df[func_df['method'] == method]
                f.write(f"  {method}: {len(method_func_df)} experiments\n")
                
                # Calculate best performance for this method on this function
                if method == 'pykan':
                    if 'test_loss' in method_func_df.columns:
                        best_loss = method_func_df['test_loss'].min()
                        if not pd.isna(best_loss):
                            f.write(f"    Best test loss: {best_loss:.6f}\n")
                else:
                    if 'best_fitness' in method_func_df.columns:
                        best_fitness = method_func_df['best_fitness'].max()
                        if not pd.isna(best_fitness):
                            f.write(f"    Best fitness: {best_fitness:.6f}\n")
        
        # Statistical tests (if scipy is available)
        try:
            from scipy import stats
            
            f.write(f"\n\nStatistical Tests:\n")
            f.write("-" * 20 + "\n")
            
            # Compare methods pairwise
            methods = completed_df['method'].unique()
            if len(methods) >= 2:
                f.write("Pairwise comparisons (Wilcoxon rank-sum test):\n")
                
                # Prepare performance data
                performance_data = {}
                for method in methods:
                    method_df = completed_df[completed_df['method'] == method]
                    if method == 'pykan':
                        perf = -method_df['test_loss'].dropna()  # Negative loss
                    else:
                        perf = method_df['best_fitness'].dropna()
                    performance_data[method] = perf
                
                for i, method1 in enumerate(methods):
                    for method2 in methods[i+1:]:
                        if len(performance_data[method1]) > 0 and len(performance_data[method2]) > 0:
                            statistic, p_value = stats.ranksums(
                                performance_data[method1], 
                                performance_data[method2]
                            )
                            f.write(f"  {method1} vs {method2}: p = {p_value:.4f}\n")
            
        except ImportError:
            f.write("\nSciPy not available for statistical tests\n")
    
    print(f"Statistical summary saved to {summary_path}")


def run_analysis(results_dir):
    """Run comprehensive analysis on experiment results."""
    output_dir = os.path.join(results_dir, 'analysis')
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    df = load_results(results_dir)
    if df is None:
        print("No results to analyze")
        return
    
    print(f"Loaded {len(df)} experiment results")
    print(f"Status distribution:")
    print(df['status'].value_counts())
    
    # Generate plots
    print("Generating comparison plots...")
    plot_method_comparison(df, output_dir)
    
    print("Generating Pareto frontier plot...")
    plot_pareto_frontier(df, output_dir)
    
    # Generate statistical summary
    print("Generating statistical summary...")
    generate_statistical_summary(df, output_dir)
    
    print(f"Analysis complete. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze special functions experiment results")
    parser.add_argument("results_dir", help="Directory containing experiment results")
    args = parser.parse_args()
    
    run_analysis(args.results_dir)
