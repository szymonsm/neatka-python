# NEATKA-Python: NEAT with Kolmogorov-Arnold Networks

## About ##

This repository presents **NEATKA** (NEAT + KAN), an innovative combination of **NEAT** (NeuroEvolution of Augmenting Topologies) and **Kolmogorov-Arnold Networks (KANs)**. This work was developed as part of the Master's thesis "Neuroevolutionary Techniques for Enhancing the Efficiency of Kolmogorov-Arnold Networks" by **Szymon Matuszewski** at **Warsaw University of Technology**.

### Background

NEAT is a method developed by Kenneth O. Stanley for evolving arbitrary neural networks. This project extends the excellent `neat-python` implementation by integrating Kolmogorov-Arnold Networks, which replace traditional Multi-Layer Perceptrons' linear weights with learnable spline functions.

**Key Innovation**: Instead of fixed activation functions and linear weights, KANs use learnable univariate functions (implemented as B-splines) on the edges, making them more interpretable and potentially more efficient for certain tasks.

### Original NEAT-Python

This work is based on the `neat-python` library, a pure-Python implementation of NEAT. The original project was created by @MattKallada and is maintained by CodeReclaimers.

For information on the original NEAT algorithm and theory, see:
- [Selected Publications](http://www.cs.ucf.edu/~kstanley/#publications) by Kenneth O. Stanley
- [Current publications](https://www.kenstanley.net/papers) on Stanley's website

## License ##

This project extends `neat-python` which is licensed under the [3-clause BSD license](https://opensource.org/licenses/BSD-3-Clause). 

**Copyright Notice:**
- Original neat-python: Copyright (c) 2007-2019, CodeReclaimers, LLC and contributors
- KAN extensions: Copyright (c) 2025, Szymon Matuszewski, Warsaw University of Technology

This extended work maintains compatibility with the original BSD-3-Clause license while adding KAN-specific functionality for research purposes.

## Getting Started ##

### Installation

1. Clone this repository:
```bash
git clone https://github.com/szymonsm/neatka-python.git
cd neatka-python
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Usage

The library supports both traditional feedforward networks and the new KAN networks. Choose your network type using configuration files:

- **Feedforward networks**: Use `config-feedforward` files
- **KAN networks**: Use `config-kan` files

### Examples

Three comprehensive examples are provided:

1. **`examples/lunar-lander/`** - OpenAI Gym LunarLander-v2 control task
2. **`examples/single-pole-balancing/`** - Classic cart-pole balancing problem  
3. **`examples/special-functions/`** - Mathematical function approximation

Each example includes:
- `evolve.py` - Run evolution with either network type
- `test.py` - Test and visualize best evolved solutions
- `parameters_sweep.py` - Systematic hyperparameter exploration

### Key Features

- **Dual Network Support**: Seamlessly switch between feedforward and KAN architectures
- **B-spline Connections**: Learnable spline functions replace traditional linear connections in KANs
- **Evolutionary Spline Optimization**: Spline control points evolve through mutation and crossover
- **Comprehensive Examples**: Real-world applications demonstrating both network types
- **Visualization Tools**: Built-in plotting and network visualization capabilities

## Technical Innovation ##

The core innovation lies in the `neat/kan_genome.py` module, which implements:

- **KANConnectionGene**: Connections with learnable B-spline functions instead of scalar weights
- **SplineSegmentGene**: Individual control points that define the spline curves
- **Evolutionary Operators**: Specialized mutation and crossover for spline-based connections
- **Dynamic Spline Structure**: Ability to add/remove spline control points during evolution

## Citing ##

If you use this work, please cite both the original neat-python library and this KAN extension:

### This Work (NEATKA)
```
@mastersthesis{Matuszewski2025NEATKA,
    author = {Matuszewski, Szymon},
    title = {Neuroevolutionary Techniques for Enhancing the Efficiency of Kolmogorov-Arnold Networks},
    school = {Warsaw University of Technology},
    year = {2025}
}
```

### Original NEAT-Python
**APA:**
```
McIntyre, A., Kallada, M., Miguel, C. G., Feher de Silva, C., & Netto, M. L. neat-python [Computer software]
```

**Bibtex:**
```
@software{McIntyre_neat-python,
    author = {McIntyre, Alan and Kallada, Matt and Miguel, Cesar G. and Feher de Silva, Carolina and Netto, Marcio Lobo},
    title = {{neat-python}}
}
```

## Acknowledgments ##

This work builds upon the excellent `neat-python` library by CodeReclaimers, LLC. Special thanks to the original authors and maintainers for providing a solid foundation for this research.

**Master's Thesis Supervision**: Warsaw University of Technology  
**Author**: Szymon Matuszewski 
