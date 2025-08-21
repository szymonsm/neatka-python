"""
Special functions module for testing KAN-NEAT, MLP-NEAT, and PyKAN on mathematical functions.
Based on the special functions from the original KAN paper.
"""

import numpy as np
from scipy import special
from typing import Tuple, Callable


class SpecialFunction:
    """Base class for special functions."""
    
    def __init__(self, name: str, func: Callable, domain_x: Tuple[float, float], 
                 domain_y: Tuple[float, float], description: str = ""):
        self.name = name
        self.func = func
        self.domain_x = domain_x
        self.domain_y = domain_y
        self.description = description
    
    def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate the function at given points."""
        return self.func(x, y)
    
    def generate_data(self, n_samples: int = 1000, test_fraction: float = 0.2, 
                     seed: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate training and test data for the function."""
        if seed is not None:
            np.random.seed(seed)
        
        # Generate random points in the domain
        x_vals = np.random.uniform(self.domain_x[0], self.domain_x[1], n_samples)
        y_vals = np.random.uniform(self.domain_y[0], self.domain_y[1], n_samples)
        
        # Evaluate function
        z_vals = self.func(x_vals, y_vals)
        
        # Handle potential NaN or infinite values
        valid_mask = np.isfinite(z_vals)
        x_vals = x_vals[valid_mask]
        y_vals = y_vals[valid_mask]
        z_vals = z_vals[valid_mask]
        
        # Split into train/test
        n_train = int(len(x_vals) * (1 - test_fraction))
        indices = np.random.permutation(len(x_vals))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        X_train = np.column_stack([x_vals[train_idx], y_vals[train_idx]])
        y_train = z_vals[train_idx]
        X_test = np.column_stack([x_vals[test_idx], y_vals[test_idx]])
        y_test = z_vals[test_idx]
        
        return X_train, y_train, X_test, y_test
    
    def for_neat_evaluation(self, inputs: np.ndarray) -> float:
        """
        Evaluate function for NEAT fitness calculation.
        inputs: [x, y] array
        Returns: function value (scalar)
        """
        if len(inputs) != 2:
            raise ValueError(f"Expected 2 inputs, got {len(inputs)}")
        
        x, y = inputs[0], inputs[1]
        
        # Check if inputs are in valid domain
        if not (self.domain_x[0] <= x <= self.domain_x[1] and 
                self.domain_y[0] <= y <= self.domain_y[1]):
            return 0.0
        
        try:
            result = self.func(x, y)
            return float(result) if np.isfinite(result) else 0.0
        except:
            return 0.0


# Define all 15 special functions from the KAN paper
SPECIAL_FUNCTIONS = {
    'ellipj': SpecialFunction(
        name='ellipj',
        func=lambda x, y: special.ellipj(x, y)[0],  # sn component
        domain_x=(0.1, 2.0),
        domain_y=(0.1, 0.9),
        description="Jacobian elliptic functions (sn component)"
    ),
    
    'ellipkinc': SpecialFunction(
        name='ellipkinc',
        func=lambda x, y: special.ellipkinc(x, y),
        domain_x=(0.1, np.pi/2 - 0.1),
        domain_y=(0.1, 0.9),
        description="Incomplete elliptic integral of the first kind"
    ),
    
    'ellipeinc': SpecialFunction(
        name='ellipeinc',
        func=lambda x, y: special.ellipeinc(x, y),
        domain_x=(0.1, np.pi/2 - 0.1),
        domain_y=(0.1, 0.9),
        description="Incomplete elliptic integral of the second kind"
    ),
    
    'jv': SpecialFunction(
        name='jv',
        func=lambda x, y: special.jv(x, y),
        domain_x=(0.1, 5.0),
        domain_y=(0.1, 10.0),
        description="Bessel function of the first kind"
    ),
    
    'yv': SpecialFunction(
        name='yv',
        func=lambda x, y: special.yv(x, y),
        domain_x=(0.1, 5.0),
        domain_y=(0.1, 10.0),
        description="Bessel function of the second kind"
    ),
    
    'kv': SpecialFunction(
        name='kv',
        func=lambda x, y: special.kv(x, y),
        domain_x=(0.1, 5.0),
        domain_y=(0.1, 10.0),
        description="Modified Bessel function of the second kind"
    ),
    
    'iv': SpecialFunction(
        name='iv',
        func=lambda x, y: special.iv(x, y),
        domain_x=(0.1, 5.0),
        domain_y=(0.1, 5.0),  # Smaller domain to avoid overflow
        description="Modified Bessel function of the first kind"
    ),
    
    'lpmv_0': SpecialFunction(
        name='lpmv_0',
        func=lambda x, y: special.lpmv(0, x, y),
        domain_x=(0, 10),
        domain_y=(-0.99, 0.99),
        description="Associated Legendre function (m=0)"
    ),
    
    'lpmv_1': SpecialFunction(
        name='lpmv_1',
        func=lambda x, y: special.lpmv(1, x, y),
        domain_x=(0, 10),
        domain_y=(-0.99, 0.99),
        description="Associated Legendre function (m=1)"
    ),
    
    'lpmv_2': SpecialFunction(
        name='lpmv_2',
        func=lambda x, y: special.lpmv(2, x, y),
        domain_x=(0, 10),
        domain_y=(-0.99, 0.99),
        description="Associated Legendre function (m=2)"
    ),
    
    'sph_harm_01': SpecialFunction(
        name='sph_harm_01',
        func=lambda x, y: np.real(special.sph_harm(0, 1, x, y)),
        domain_x=(0, 2*np.pi),
        domain_y=(0, np.pi),
        description="Spherical harmonics (m=0, n=1) - real part"
    ),
    
    'sph_harm_11': SpecialFunction(
        name='sph_harm_11',
        func=lambda x, y: np.real(special.sph_harm(1, 1, x, y)),
        domain_x=(0, 2*np.pi),
        domain_y=(0, np.pi),
        description="Spherical harmonics (m=1, n=1) - real part"
    ),
    
    'sph_harm_02': SpecialFunction(
        name='sph_harm_02',
        func=lambda x, y: np.real(special.sph_harm(0, 2, x, y)),
        domain_x=(0, 2*np.pi),
        domain_y=(0, np.pi),
        description="Spherical harmonics (m=0, n=2) - real part"
    ),
    
    'sph_harm_12': SpecialFunction(
        name='sph_harm_12',
        func=lambda x, y: np.real(special.sph_harm(1, 2, x, y)),
        domain_x=(0, 2*np.pi),
        domain_y=(0, np.pi),
        description="Spherical harmonics (m=1, n=2) - real part"
    ),
    
    'sph_harm_22': SpecialFunction(
        name='sph_harm_22',
        func=lambda x, y: np.real(special.sph_harm(2, 2, x, y)),
        domain_x=(0, 2*np.pi),
        domain_y=(0, np.pi),
        description="Spherical harmonics (m=2, n=2) - real part"
    ),
}


def get_function(name: str) -> SpecialFunction:
    """Get a special function by name."""
    if name not in SPECIAL_FUNCTIONS:
        raise ValueError(f"Unknown function: {name}. Available: {list(SPECIAL_FUNCTIONS.keys())}")
    return SPECIAL_FUNCTIONS[name]


def list_functions() -> list:
    """List all available special functions."""
    return list(SPECIAL_FUNCTIONS.keys())


def test_all_functions():
    """Test all functions to ensure they work correctly."""
    print("Testing all special functions...")
    for name, func in SPECIAL_FUNCTIONS.items():
        try:
            X_train, y_train, X_test, y_test = func.generate_data(n_samples=100, seed=42)
            print(f"✓ {name}: Train shape {X_train.shape}, Test shape {X_test.shape}")
            print(f"  Train y range: [{y_train.min():.3f}, {y_train.max():.3f}]")
        except Exception as e:
            print(f"✗ {name}: Error - {e}")


if __name__ == "__main__":
    test_all_functions()
