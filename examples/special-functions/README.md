# Special Functions Experiments

This directory contains experiments comparing KAN-NEAT, MLP-NEAT, and PyKAN on mathematical special functions from the original KAN paper.

## Purpose

These experiments address the feedback that the thesis should show KAN-NEAT as an alternative method for discovering KAN architectures automatically, rather than just comparing it to MLP-NEAT. The special functions provide cases where KANs are known to outperform MLPs, making this a fair comparison ground.

## Files

- `special_functions.py`: Defines 15 mathematical special functions from scipy.special
- `evolve.py`: NEAT evolution script for function approximation
- `pykan_trainer.py`: PyKAN training wrapper 
- `parameter_sweep.py`: Comprehensive experiment runner
- `analysis.py`: Results analysis and visualization
- `config-feedforward`: NEAT config for MLP networks
- `config-kan`: NEAT config for KAN networks

## Available Functions

1. **ellipj**: Jacobian elliptic functions
2. **ellipkinc**: Incomplete elliptic integral of the first kind
3. **ellipeinc**: Incomplete elliptic integral of the second kind
4. **jv**: Bessel function of the first kind
5. **yv**: Bessel function of the second kind
6. **kv**: Modified Bessel function of the second kind
7. **iv**: Modified Bessel function of the first kind
8. **lpmv_0/1/2**: Associated Legendre functions
9. **sph_harm_01/11/02/12/22**: Spherical harmonics

## Running Experiments

### Quick Start - Single Function
```bash
# Test MLP-NEAT on Bessel function
python evolve.py --function jv --net-type feedforward --seed 42

# Test KAN-NEAT on Bessel function  
python evolve.py --function jv --net-type kan --seed 42

# Test PyKAN on Bessel function
python pykan_trainer.py --function jv --seed 42
```

### Parameter Sweep - Multiple Functions and Seeds
```bash
# Run comprehensive comparison (requires PyKAN: pip install pykan)
python parameter_sweep.py --methods mlp-neat kan-neat pykan --seeds 42 123 456 --generations 50

# Run only NEAT methods
python parameter_sweep.py --methods mlp-neat kan-neat --seeds 42 123 456 789 1000

# Test specific functions
python parameter_sweep.py --functions jv ellipkinc lpmv_0 --methods mlp-neat kan-neat
```

### Analysis
```bash
# Analyze results and generate plots
python analysis.py special_functions_results/
```

## Expected Results

Based on the original KAN paper, we expect:

1. **PyKAN** should outperform traditional MLPs on these functions
2. **KAN-NEAT** should discover compact KAN architectures automatically
3. **MLP-NEAT** should serve as a baseline for evolutionary approaches

## Key Metrics

- **Performance**: Function approximation accuracy (fitness/RMSE)
- **Complexity**: Number of parameters/connections
- **Efficiency**: Training time and convergence speed
- **Robustness**: Consistency across different random seeds

## Thesis Contribution

This experiment demonstrates that:

1. **KAN-NEAT provides automatic architecture discovery** for KAN networks
2. **No manual architecture design** required (unlike standard PyKAN)
3. **Evolutionary approach** can find compact, efficient KAN representations
4. **Alternative training paradigm** for KANs beyond gradient-based methods

## Dependencies

- numpy
- scipy
- pandas  
- matplotlib
- seaborn
- torch (for PyKAN)
- pykan (optional, for PyKAN comparison)

Install PyKAN for full comparison:
```bash
pip install pykan
```

## Results Directory Structure

```
special_functions_results/
├── experiment_results.csv       # Raw experiment data
├── experiment_summary.txt       # Success/failure summary  
├── analysis/
│   ├── method_comparison.png    # Performance comparison plots
│   ├── runtime_comparison.png   # Training time analysis
│   ├── pareto_frontier.png      # Performance vs complexity
│   └── statistical_summary.txt  # Detailed statistics
├── mlp-neat-{function}-seed{N}/ # Individual NEAT results
├── kan-neat-{function}-seed{N}/
└── pykan-{function}-seed{N}/    # PyKAN results
```

## Configuration

### NEAT Parameters
- Population: 150
- Generations: 50-100 (configurable)
- Activation functions: tanh, relu, sigmoid, sin, cos (feedforward) / spline (KAN)
- Mutation rates optimized for function approximation

### PyKAN Parameters  
- Width: [2, 5, 1] (input, hidden, output)
- Grid progression: 3 → 5 → 10 → 20 → 50
- Spline order: 3
- Optimizer: LBFGS

## Citation

This work builds upon:
- KAN: Kolmogorov-Arnold Networks (Liu et al., 2024)
- NEAT: Evolving Neural Networks through Augmenting Topologies (Stanley & Miikkulainen, 2002)
