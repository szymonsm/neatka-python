# NEAT Core Library with KAN Extensions

This directory contains the core NEAT implementation with **Kolmogorov-Arnold Network (KAN)** extensions.

## Key Innovation: `kan_genome.py` 🚀

The primary contribution of this work is the **`kan_genome.py`** module, which implements KAN-based genomes for evolutionary optimization:

### Core Components

- **`KANNodeGene`**: Standard NEAT nodes adapted for KAN networks with summation aggregation
- **`KANConnectionGene`**: Revolutionary connection genes that replace scalar weights with learnable B-spline functions
- **`SplineSegmentGene`**: Individual control points that define spline curves on connections

### Innovation Highlights

- **Spline-Based Connections**: Each connection contains multiple spline segments with evolutionary control points
- **Dynamic Spline Structure**: Ability to add/remove control points during evolution through mutation
- **Specialized Operators**: Custom crossover and mutation operators for spline parameters
- **Genetic Distance Calculation**: Novel distance metrics for spline-based genomes

## Other Key Files

- **`kan_utils.py`**: Visualization and utility functions for KAN networks
- **`nn/kan.py`**: Neural network implementation for executing KAN genomes
- All other files are inherited from the original neat-python library

## Usage

```python
from neat.kan_genome import KANGenome
from neat.nn.kan import KANNetwork

# Create and evolve KAN networks through standard NEAT interface
# Networks automatically use spline-based connections instead of weights
```

The KAN implementation seamlessly integrates with the existing NEAT framework, allowing easy comparison between traditional feedforward and KAN architectures.
