"""
PyKAN implementation for special functions approximation.
This module provides a wrapper for training PyKAN models on special functions
to compare with NEAT-evolved networks.
"""

import os
import time
import numpy as np
import torch
import pickle
from pathlib import Path

try:
    from kan import KAN, create_dataset
    PYKAN_AVAILABLE = True
except ImportError:
    PYKAN_AVAILABLE = False
    print("PyKAN not available. Install with: pip install pykan")

import special_functions


class PyKANTrainer:
    """Trainer for PyKAN models on special functions."""
    
    def __init__(self, function_name: str, device=None, seed: int = 42):
        if not PYKAN_AVAILABLE:
            raise ImportError("PyKAN not available")
        
        self.function_name = function_name
        self.function = special_functions.get_function(function_name)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        
        # Set random seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        print(f"PyKAN Trainer initialized for function: {function_name}")
        print(f"Device: {self.device}")
    
    def create_dataset(self, n_samples: int = 1000, test_fraction: float = 0.2):
        """Create PyKAN dataset from the special function."""
        X_train, y_train, X_test, y_test = self.function.generate_data(
            n_samples=n_samples, test_fraction=test_fraction, seed=self.seed
        )
        
        # Convert to torch tensors
        X_train_torch = torch.tensor(X_train, dtype=torch.float32, device=self.device)
        y_train_torch = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32, device=self.device)
        X_test_torch = torch.tensor(X_test, dtype=torch.float32, device=self.device)
        y_test_torch = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32, device=self.device)
        
        # Create PyKAN dataset format
        dataset = {
            'train_input': X_train_torch,
            'train_label': y_train_torch,
            'test_input': X_test_torch,
            'test_label': y_test_torch
        }
        
        return dataset
    
    def train_model(self, width=[2, 5, 1], steps_per_grid=200, 
                   grids=None, results_dir=None):
        """
        Train a PyKAN model with grid refinement.
        
        Args:
            width: Network width specification [input, hidden, output]
            steps_per_grid: Training steps per grid size
            grids: Custom grid sizes (if None, uses exponential progression)
            results_dir: Directory to save results
        """
        if grids is None:
            grids = np.array([3, 5, 10, 20, 50])
        
        # Create dataset
        dataset = self.create_dataset()
        
        # Initialize results tracking
        train_losses = []
        test_losses = []
        training_history = []
        
        # Start timing
        start_time = time.time()
        
        model = None
        for i, grid in enumerate(grids):
            print(f"\nTraining with grid size: {grid}")
            
            if i == 0:
                # Initialize model
                model = KAN(width=width, grid=grid, k=3, seed=self.seed, device=self.device)
            else:
                # Refine existing model
                model = model.refine(grid)
            
            # Train the model
            results = model.fit(dataset, opt="LBFGS", steps=steps_per_grid, lamb=0.01)
            
            # Record losses
            train_losses.extend(results['train_loss'])
            test_losses.extend(results['test_loss'])
            
            # Record grid-level statistics
            final_train_loss = results['train_loss'][-1]
            final_test_loss = results['test_loss'][-1]
            
            training_history.append({
                'grid': grid,
                'train_loss': final_train_loss,
                'test_loss': final_test_loss,
                'parameters': self._count_parameters(model),
                'steps': steps_per_grid
            })
            
            print(f"Grid {grid}: Train Loss = {final_train_loss:.6f}, Test Loss = {final_test_loss:.6f}")
        
        # Record total training time
        total_time = time.time() - start_time
        
        # Save results if directory provided
        if results_dir:
            self._save_results(model, training_history, train_losses, test_losses, 
                             total_time, results_dir)
        
        return model, training_history, train_losses, test_losses
    
    def _count_parameters(self, model):
        """Count the number of parameters in the model."""
        try:
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        except:
            # Fallback if parameter counting fails
            return len(model.grid) * len(model.width)
    
    def _save_results(self, model, training_history, train_losses, test_losses, 
                     total_time, results_dir):
        """Save training results and model."""
        os.makedirs(results_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(results_dir, 'pykan_model.pkl')
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Could not save model: {e}")
        
        # Save training history
        history_path = os.path.join(results_dir, 'training_history.txt')
        with open(history_path, 'w') as f:
            f.write(f"PyKAN Training Results\n")
            f.write(f"=====================\n")
            f.write(f"Function: {self.function_name}\n")
            f.write(f"Description: {self.function.description}\n")
            f.write(f"Total training time: {total_time:.2f} seconds\n")
            f.write(f"Device: {self.device}\n")
            f.write(f"Seed: {self.seed}\n\n")
            
            f.write("Grid-by-grid results:\n")
            f.write("Grid\tParameters\tTrain Loss\tTest Loss\n")
            f.write("-" * 50 + "\n")
            
            for entry in training_history:
                f.write(f"{entry['grid']}\t{entry['parameters']}\t")
                f.write(f"{entry['train_loss']:.6f}\t{entry['test_loss']:.6f}\n")
            
            # Final results
            final_entry = training_history[-1]
            f.write(f"\nFinal Results:\n")
            f.write(f"Best train loss: {final_entry['train_loss']:.6f}\n")
            f.write(f"Best test loss: {final_entry['test_loss']:.6f}\n")
            f.write(f"Total parameters: {final_entry['parameters']}\n")
        
        # Save losses for plotting
        losses_path = os.path.join(results_dir, 'losses.txt')
        with open(losses_path, 'w') as f:
            f.write("Step\tTrain_Loss\tTest_Loss\n")
            for i, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
                f.write(f"{i+1}\t{train_loss:.6f}\t{test_loss:.6f}\n")
        
        print(f"Training results saved to {results_dir}")


def train_pykan_on_function(function_name: str, width=[2, 5, 1], 
                           steps_per_grid=200, results_dir=None, seed=42):
    """
    Convenience function to train PyKAN on a special function.
    
    Args:
        function_name: Name of the special function
        width: Network architecture
        steps_per_grid: Training steps per grid size
        results_dir: Directory to save results
        seed: Random seed
        
    Returns:
        Trained model and training history
    """
    if not PYKAN_AVAILABLE:
        raise ImportError("PyKAN not available")
    
    trainer = PyKANTrainer(function_name, seed=seed)
    model, history, train_losses, test_losses = trainer.train_model(
        width=width, steps_per_grid=steps_per_grid, 
        results_dir=results_dir
    )
    
    return model, history


def train_all_functions(functions=None, width=[2, 5, 1], 
                       steps_per_grid=100, base_results_dir="pykan_results", seed=42):
    """
    Train PyKAN on all special functions.
    
    Args:
        functions: List of function names (if None, uses all)
        width: Network architecture
        steps_per_grid: Training steps per grid size
        base_results_dir: Base directory for results
        seed: Random seed
    """
    if not PYKAN_AVAILABLE:
        raise ImportError("PyKAN not available")
    
    if functions is None:
        functions = special_functions.list_functions()
    
    results = {}
    
    for func_name in functions:
        print(f"\n{'='*60}")
        print(f"Training PyKAN on function: {func_name}")
        print(f"{'='*60}")
        
        # Create results directory for this function
        func_results_dir = os.path.join(base_results_dir, func_name)
        
        try:
            model, history = train_pykan_on_function(
                function_name=func_name,
                width=width,
                steps_per_grid=steps_per_grid,
                results_dir=func_results_dir,
                seed=seed
            )
            
            results[func_name] = {
                'model': model,
                'history': history,
                'final_train_loss': history[-1]['train_loss'],
                'final_test_loss': history[-1]['test_loss'],
                'parameters': history[-1]['parameters']
            }
            
            print(f"✓ {func_name}: Train Loss = {history[-1]['train_loss']:.6f}, "
                  f"Test Loss = {history[-1]['test_loss']:.6f}")
            
        except Exception as e:
            print(f"✗ Failed to train on {func_name}: {e}")
            results[func_name] = {'error': str(e)}
    
    # Save summary
    summary_path = os.path.join(base_results_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("PyKAN Training Summary\n")
        f.write("====================\n\n")
        f.write("Function\t\tTrain Loss\tTest Loss\tParameters\n")
        f.write("-" * 60 + "\n")
        
        for func_name, result in results.items():
            if 'error' not in result:
                f.write(f"{func_name:<15}\t{result['final_train_loss']:.6f}\t")
                f.write(f"{result['final_test_loss']:.6f}\t{result['parameters']}\n")
            else:
                f.write(f"{func_name:<15}\tERROR: {result['error']}\n")
    
    print(f"\nTraining summary saved to {summary_path}")
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train PyKAN on special functions')
    parser.add_argument('--function', type=str, 
                      choices=special_functions.list_functions() + ['all'],
                      default='jv', help='Function to train on (or "all")')
    parser.add_argument('--width', nargs='+', type=list, default=[2, 5, 1],
                      help='Network width specification')
    parser.add_argument('--steps-per-grid', type=int, default=100,
                      help='Training steps per grid size')
    parser.add_argument('--results-dir', type=str, default='pykan_results',
                      help='Directory to save results')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed')
    
    args = parser.parse_args()
    
    if args.function == 'all':
        train_all_functions(
            width=args.width,
            steps_per_grid=args.steps_per_grid,
            base_results_dir=args.results_dir,
            seed=args.seed
        )
    else:
        train_pykan_on_function(
            function_name=args.function,
            width=args.width,
            steps_per_grid=args.steps_per_grid,
            results_dir=os.path.join(args.results_dir, args.function),
            seed=args.seed
        )
