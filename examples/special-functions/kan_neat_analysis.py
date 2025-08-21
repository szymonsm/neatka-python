"""
KAN-NEAT Results Analysis Script

This script analyzes the KAN-NEAT experiment results, evaluates saved models,
and creates comparison plots and summary tables with PyKAN results.
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import glob

import special_functions
import neat
from neat.nn.kan import KANNetwork


class KANNEATAnalyzer:
    """Analyzer for KAN-NEAT experimental results."""
    
    def __init__(self, results_dir="special_functions_results", results_kan_dir="results"):
        self.results_dir = Path(results_dir)
        self.results_kan_dir = Path(results_kan_dir)
        self.functions = ['ellipj', 'jv', 'lpmv_1']
        self.seeds = [42, 123, 456, 789, 1000]
        
        # Initialize data structures
        self.kan_neat_results = {}
        self.pykan_results = {}
        self.summary_data = []
        
    def load_pykan_results(self):
        """Load PyKAN experiment results."""
        print("Loading PyKAN results...")
        
        pykan_dir = self.results_dir / "pykan_results"
        
        for function_name in self.functions:
            self.pykan_results[function_name] = {}
            
            for seed in self.seeds:
                result_dir = pykan_dir / f"pykan-{function_name}-seed{seed}"
                
                if not result_dir.exists():
                    print(f"Warning: PyKAN result not found for {function_name}, seed {seed}")
                    continue
                
                # Load training history - try both .json and .txt formats
                history_file_json = result_dir / "training_history.json"
                history_file_txt = result_dir / "training_history.txt"
                losses_file = result_dir / "losses.txt"
                
                history_data = None
                
                # Try JSON format first
                if history_file_json.exists():
                    try:
                        with open(history_file_json, 'r') as f:
                            history_data = json.load(f)
                    except Exception as e:
                        print(f"Error reading JSON history for {function_name}, seed {seed}: {e}")
                
                # Try TXT format
                elif history_file_txt.exists():
                    try:
                        with open(history_file_txt, 'r') as f:
                            content = f.read().strip()
                            if content:
                                try:
                                    history_data = json.loads(content)
                                except json.JSONDecodeError:
                                    # Try to parse as simple text format
                                    lines = content.split('\n')
                                    if len(lines) >= 2:
                                        try:
                                            train_loss = float(lines[-2].strip())
                                            test_loss = float(lines[-1].strip())
                                            history_data = [{'train_loss': train_loss, 'test_loss': test_loss}]
                                        except ValueError:
                                            pass
                    except Exception as e:
                        print(f"Error reading TXT history for {function_name}, seed {seed}: {e}")
                
                # Try losses.txt as fallback
                elif losses_file.exists():
                    try:
                        with open(losses_file, 'r') as f:
                            lines = f.readlines()
                            if len(lines) >= 2:
                                train_loss = float(lines[0].strip())
                                test_loss = float(lines[1].strip())
                                history_data = [{'train_loss': train_loss, 'test_loss': test_loss}]
                    except Exception as e:
                        print(f"Error reading losses file for {function_name}, seed {seed}: {e}")
                
                if history_data and len(history_data) > 0:
                    self.pykan_results[function_name][seed] = {
                        'train_losses': [h['train_loss'] for h in history_data],
                        'test_losses': [h['test_loss'] for h in history_data],
                        'final_train_loss': history_data[-1]['train_loss'],
                        'final_test_loss': history_data[-1]['test_loss']
                    }
                    
                    print(f"Loaded PyKAN results for {function_name}, seed {seed}")
                else:
                    print(f"Warning: Could not load PyKAN results for {function_name}, seed {seed}")
    
    def evaluate_kan_neat_model(self, winner_path, function_name, seed):
        """Evaluate a KAN-NEAT winner model on test data."""
        try:
            # Load the winner genome
            with open(winner_path, 'rb') as f:
                winner = pickle.load(f)
            
            # Get the function and generate test data
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
            
            # Try to find config file or use default
            config_path = None
            
            # Look for config in the same directory as winner
            winner_dir = winner_path.parent
            for config_file in winner_dir.glob('config*.cfg'):
                config_path = str(config_file)
                break
            
            # If not found, look in parent directory
            if not config_path:
                parent_dir = winner_dir.parent
                for config_file in parent_dir.glob('config*.cfg'):
                    config_path = str(config_file)
                    break
            
            # Use default config if still not found
            if not config_path:
                config_path = os.path.join(os.path.dirname(__file__), 'config-kan')
            
            # Create config with KANGenome instead of DefaultGenome for KAN networks
            from neat.kan_genome import KANGenome
            config = neat.Config(KANGenome, neat.DefaultReproduction,
                               neat.DefaultSpeciesSet, neat.DefaultStagnation,
                               config_path)
            
            # Create KAN network
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
            
            # Calculate RMSE on normalized data (same as fitness function)
            train_rmse = np.sqrt(np.mean((train_predictions - y_train_norm) ** 2))
            test_rmse = np.sqrt(np.mean((test_predictions - y_test_norm) ** 2))
            
            return {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_predictions': train_predictions,
                'test_predictions': test_predictions,
                'y_train': y_train_norm,
                'y_test': y_test_norm,
                'X_train': X_train,
                'X_test': X_test,
                'winner': winner
            }
            
        except Exception as e:
            print(f"Error evaluating model {winner_path}: {e}")
            return None
    
    def load_kan_neat_results(self):
        """Load and evaluate KAN-NEAT experiment results."""
        print("Loading and evaluating KAN-NEAT results...")
        
        for function_name in self.functions:
            self.kan_neat_results[function_name] = {}
            
            # Check the results directory structure - use function-kan format
            function_result_dir = self.results_dir / f"{function_name}-kan"
            
            if function_result_dir.exists():
                print(f"Found results directory: {function_result_dir}")
                
                # Iterate through all experiment directories
                seed_count = 0
                for exp_dir in function_result_dir.iterdir():
                    if not exp_dir.is_dir():
                        continue
                    
                    print(f"Checking experiment directory: {exp_dir}")
                    
                    # Load config summary to get seed
                    config_summary_path = exp_dir / "config_summary.json"
                    if not config_summary_path.exists():
                        print(f"No config_summary.json found in {exp_dir}")
                        continue
                    
                    try:
                        with open(config_summary_path, 'r') as f:
                            config_summary = json.load(f)
                    except Exception as e:
                        print(f"Error reading config_summary.json from {exp_dir}: {e}")
                        continue
                    
                    exp_seed = config_summary.get('seed', f'run_{seed_count}')
                    seed_count += 1
                    
                    # Check if winner exists
                    winner_path = exp_dir / "winner-kan.pkl"
                    if not winner_path.exists():
                        print(f"Warning: No winner found for {function_name}, seed {exp_seed}")
                        continue
                    
                    # Evaluate the model
                    print(f"Evaluating model for {function_name}, seed {exp_seed}")
                    evaluation = self.evaluate_kan_neat_model(winner_path, function_name, exp_seed)
                    
                    if evaluation is not None:
                        evaluation['seed'] = exp_seed
                        self.kan_neat_results[function_name][str(exp_seed)] = evaluation
                        print(f"Successfully evaluated KAN-NEAT model for {function_name}, seed {exp_seed}")
                    else:
                        print(f"Failed to evaluate model for {function_name}, seed {exp_seed}")
            else:
                print(f"No results directory found for function: {function_name} (looking for {function_result_dir})")
    
    def create_summary_table(self):
        """Create summary comparison table."""
        print("Creating summary table...")
        
        summary_data = []
        
        for function_name in self.functions:
            # Get best PyKAN result (lowest test loss)
            pykan_best_test = float('inf')
            pykan_best_train = None
            
            if function_name in self.pykan_results:
                for seed, results in self.pykan_results[function_name].items():
                    if results['final_test_loss'] < pykan_best_test:
                        pykan_best_test = results['final_test_loss']
                        pykan_best_train = results['final_train_loss']
            
            # Get best KAN-NEAT result (lowest test RMSE)
            kan_neat_best_test = float('inf')
            kan_neat_best_train = None
            
            if function_name in self.kan_neat_results:
                for seed, results in self.kan_neat_results[function_name].items():
                    if results['test_rmse'] < kan_neat_best_test:
                        kan_neat_best_test = results['test_rmse']
                        kan_neat_best_train = results['train_rmse']
            
            summary_data.append({
                'Special Function': function_name,
                'Best Test RMSE PyKAN': pykan_best_test if pykan_best_test != float('inf') else 'N/A',
                'Train RMSE PyKAN': pykan_best_train if pykan_best_train is not None else 'N/A',
                'Best Test RMSE KAN-NEAT': kan_neat_best_test if kan_neat_best_test != float('inf') else 'N/A',
                'Train RMSE KAN-NEAT': kan_neat_best_train if kan_neat_best_train is not None else 'N/A'
            })
        
        self.summary_df = pd.DataFrame(summary_data)
        
        # Save summary table
        summary_path = self.results_dir / "comparison_summary.csv"
        self.summary_df.to_csv(summary_path, index=False)
        print(f"Summary table saved to {summary_path}")
        
        return self.summary_df
    
    def plot_function_comparison(self, function_name, save_path=None):
        """Create comparison plot for a specific function."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot PyKAN results
        if function_name in self.pykan_results:
            best_test_loss = float('inf')
            best_seed = None
            
            for seed, results in self.pykan_results[function_name].items():
                # Check if this is the best run
                if results['final_test_loss'] < best_test_loss:
                    best_test_loss = results['final_test_loss']
                    best_seed = seed
            
            # Plot all runs
            for seed, results in self.pykan_results[function_name].items():
                if seed == best_seed:
                    alpha = 1.0
                    linewidth = 3
                    label = f'PyKAN (seed {seed}) - Best'
                    color = 'darkblue'
                else:
                    alpha = 0.4
                    linewidth = 1
                    label = None
                    color = 'lightblue'
                
                ax1.plot(results['train_losses'], alpha=alpha, linewidth=linewidth, 
                        label=label, color=color)
        
        ax1.set_title(f'PyKAN Training Loss Convergence - {function_name}')
        ax1.set_xlabel('Grid Training Steps')
        ax1.set_ylabel('Training Loss')
        ax1.grid(True, alpha=0.3)
        if best_seed is not None:
            ax1.legend()
        
        # Plot KAN-NEAT training convergence
        if function_name in self.kan_neat_results:
            # Find best test RMSE run
            best_test_rmse = float('inf')
            best_kan_neat_seed = None
            
            for seed, results in self.kan_neat_results[function_name].items():
                if results['test_rmse'] < best_test_rmse:
                    best_test_rmse = results['test_rmse']
                    best_kan_neat_seed = seed
            
            # Plot convergence curves if available
            for seed, results in self.kan_neat_results[function_name].items():
                if 'convergence_data' in results and results['convergence_data']:
                    convergence = results['convergence_data']
                    generations = [row['generation'] for row in convergence]
                    best_fitness = [row['best_fitness'] for row in convergence]
                    
                    if seed == best_kan_neat_seed:
                        alpha = 1.0
                        linewidth = 3
                        label = f'KAN-NEAT (seed {seed}) - Best'
                        color = 'darkred'
                    else:
                        alpha = 0.4
                        linewidth = 1
                        label = None
                        color = 'lightcoral'
                    
                    ax2.plot(generations, best_fitness, alpha=alpha, linewidth=linewidth,
                            label=label, color=color)
        
        ax2.set_title(f'KAN-NEAT Training Fitness Convergence - {function_name}')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Best Fitness (RMSE)')
        ax2.grid(True, alpha=0.3)
        if best_kan_neat_seed is not None:
            ax2.legend()
        
        # Plot KAN-NEAT final results scatter
        if function_name in self.kan_neat_results and self.kan_neat_results[function_name]:
            seeds = list(self.kan_neat_results[function_name].keys())
            test_rmses = [self.kan_neat_results[function_name][seed]['test_rmse'] for seed in seeds]
            train_rmses = [self.kan_neat_results[function_name][seed]['train_rmse'] for seed in seeds]
            
            if len(test_rmses) > 0:  # Check if we have any results
                # Highlight best result
                best_idx = np.argmin(test_rmses)
                colors = ['darkred' if i == best_idx else 'lightcoral' for i in range(len(seeds))]
                alphas = [1.0 if i == best_idx else 0.6 for i in range(len(seeds))]
                sizes = [120 if i == best_idx else 60 for i in range(len(seeds))]
                
                ax3.scatter(train_rmses, test_rmses, c=colors, alpha=alphas, s=sizes)
                
                # Add seed labels
                for i, seed in enumerate(seeds):
                    ax3.annotate(f'Seed {seed}', (train_rmses[i], test_rmses[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            else:
                ax3.text(0.5, 0.5, 'No KAN-NEAT results available', 
                        transform=ax3.transAxes, ha='center', va='center', fontsize=12)
        else:
            ax3.text(0.5, 0.5, 'No KAN-NEAT results available', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
        
        ax3.set_title(f'KAN-NEAT Final Results - {function_name}')
        ax3.set_xlabel('Train RMSE')
        ax3.set_ylabel('Test RMSE')
        ax3.grid(True, alpha=0.3)
        
        # Plot comparison of best results
        if (function_name in self.pykan_results and 
            function_name in self.kan_neat_results and 
            self.kan_neat_results[function_name]):
            
            methods = ['PyKAN', 'KAN-NEAT']
            pykan_best_test = min(r['final_test_loss'] for r in self.pykan_results[function_name].values())
            kan_neat_best_test = min(r['test_rmse'] for r in self.kan_neat_results[function_name].values())
            
            test_values = [pykan_best_test, kan_neat_best_test]
            colors = ['darkblue', 'darkred']
            
            bars = ax4.bar(methods, test_values, color=colors, alpha=0.7)
            ax4.set_title(f'Best Test RMSE Comparison - {function_name}')
            ax4.set_ylabel('Test RMSE')
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, test_values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data for comparison', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def visualize_best_kan_neat_network(self, function_name, save_path=None):
        """Visualize the best KAN-NEAT network for a function."""
        if function_name not in self.kan_neat_results:
            print(f"No KAN-NEAT results found for {function_name}")
            return
        
        # Find best model (lowest test RMSE)
        best_test_rmse = float('inf')
        best_seed = None
        best_result_dir = None
        
        function_result_dir = self.results_kan_dir / f"{function_name}-kan"
        
        for exp_dir in function_result_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            config_summary_path = exp_dir / "config_summary.json"
            if not config_summary_path.exists():
                continue
            
            with open(config_summary_path, 'r') as f:
                config_summary = json.load(f)
            
            seed = config_summary['seed']
            
            if seed in self.kan_neat_results[function_name]:
                test_rmse = self.kan_neat_results[function_name][seed]['test_rmse']
                if test_rmse < best_test_rmse:
                    best_test_rmse = test_rmse
                    best_seed = seed
                    best_result_dir = exp_dir
        
        if best_result_dir is None:
            print(f"No valid results found for {function_name}")
            return
        
        # Load the best winner
        winner_path = best_result_dir / "winner-kan.pkl"
        with open(winner_path, 'rb') as f:
            winner = pickle.load(f)
        
        # Load config
        config_path = best_result_dir / "reconstructed_config.ini"
        config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                           neat.DefaultSpeciesSet, neat.DefaultStagnation,
                           str(config_path))
        
        # Create network visualization (similar to other examples)
        self._draw_kan_network(winner, config, function_name, best_seed, save_path)
    
    def _draw_kan_network(self, genome, config, function_name, seed, save_path=None):
        """Draw KAN network architecture."""
        try:
            import graphviz
        except ImportError:
            print("Graphviz not available for network visualization")
            return
        
        dot = graphviz.Digraph(format='png')
        dot.attr(rankdir='LR')
        
        # Input nodes
        inputs = config.genome_config.input_keys
        for i, input_key in enumerate(inputs):
            dot.node(str(input_key), f'Input {i}', shape='box', style='filled', fillcolor='lightblue')
        
        # Output nodes
        outputs = config.genome_config.output_keys
        for i, output_key in enumerate(outputs):
            dot.node(str(output_key), f'Output {i}', shape='box', style='filled', fillcolor='lightcoral')
        
        # Hidden nodes
        hidden_nodes = [n for n in genome.nodes.keys() 
                       if n not in inputs and n not in outputs]
        for node_key in hidden_nodes:
            node = genome.nodes[node_key]
            dot.node(str(node_key), f'KAN\n{node_key}', shape='circle', style='filled', fillcolor='lightgreen')
        
        # Connections
        for conn_key, conn in genome.connections.items():
            if conn.enabled:
                input_key, output_key = conn_key
                weight = conn.weight
                color = 'red' if weight < 0 else 'blue'
                width = str(min(abs(weight) * 2, 3))
                dot.edge(str(input_key), str(output_key), 
                        label=f'{weight:.2f}', color=color, penwidth=width)
        
        dot.attr(label=f'Best KAN-NEAT Network\\n{function_name} (seed {seed})')
        
        if save_path:
            output_path = str(save_path).replace('.png', '')
            dot.render(output_path, cleanup=True)
            print(f"Network visualization saved to {output_path}.png")
        else:
            dot.view()
    
    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("Starting full KAN-NEAT analysis...")
        
        # Load all results
        self.load_pykan_results()
        self.load_kan_neat_results()
        
        # Create summary table
        summary_df = self.create_summary_table()
        print("\nSummary Table:")
        print(summary_df.to_string(index=False))
        
        # Create plots for each function
        plots_dir = self.results_dir / "comparison_plots"
        plots_dir.mkdir(exist_ok=True)
        
        for function_name in self.functions:
            print(f"\nCreating plots for {function_name}...")
            
            # Comparison plot
            plot_path = plots_dir / f"{function_name}_comparison.png"
            self.plot_function_comparison(function_name, save_path=plot_path)
            
            # Network visualization
            network_path = plots_dir / f"{function_name}_best_network"
            self.visualize_best_kan_neat_network(function_name, save_path=network_path)
        
        print(f"\nAnalysis complete! Results saved in {self.results_dir}")
        
        return summary_df


if __name__ == "__main__":
    # Run the analysis
    analyzer = KANNEATAnalyzer()
    summary = analyzer.run_full_analysis()
