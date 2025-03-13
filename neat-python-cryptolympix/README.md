# 🧠 NEAT Algorithm Implementation

This project implements the NEAT (NeuroEvolution of Augmenting Topologies) algorithm for evolving neural networks. NEAT is a genetic algorithm specifically designed for evolving neural networks with varying topologies.

## Overview

- **NEAT Algorithm**: The NEAT algorithm is a method for evolving artificial neural networks. It allows the evolution of both the structure and weights of neural networks, starting from minimal structures and incrementally adding complexity over generations.

## Quick Start

```python
from neat.__types__ import NeatConfig
from neat.default_config import default_config
from neat.population import Population
from neat.genome import Genome

# Initialize NEAT configuration
config: NeatConfig = default_config

# Initialize population
population = Population(config)

def evaluate_genome(genome: Genome, generation: int):
    # Your evaluation logic here
    # Evaluate the fitness of the genome and set genome.fitness

# Run NEAT algorithm
generations = 100
population.run(evaluate_genome, generations)
```

## Customize NEAT Configuration

Adjust the NEAT configuration parameters in the default_config.py file to suit your specific problem and preferences.

Now you're all set to evolve neural networks using the NEAT algorithm! Feel free to explore and experiment with different configurations.

The default_config.py file contains configuration parameters for the NEAT algorithm. Adjust these parameters to customize the algorithm's behavior.

- **Population Settings**

  - `population_size`: Number of individuals in each generation.
  - `fitness_threshold`: Termination threshold for the fitness criterion.
  - `no_fitness_termination`: If True, the evolution process won't terminate based on fitness.
  - `reset_on_extinction`: If True, a new random population is created when all species go extinct.

- **Genome Settings**

  - `activation_default`: Default activation function for neurons.
  - `activation_mutate_rate`: Mutation rate for activation functions.
  - `num_inputs`: Number of input nodes in the neural network.
  - `num_outputs`: Number of output nodes in the neural network.

- **Connection Settings**

  - `compatibility_disjoint_coefficient`: Coefficient for compatibility calculation related to disjoint genes.
  - `compatibility_weight_coefficient`: Coefficient for compatibility calculation related to weight differences.
  - `conn_add_prob`: Probability of adding a new connection.
  - `conn_delete_prob`: Probability of deleting an existing connection.
  - `enabled_default`: Default state (enabled or disabled) for new connections.
  - `enabled_mutate_rate`: Mutation rate for connection enable/disable state.
  - `initial_connections`: Type of initial connections ("none" or "full").
  - `weight_init_mean`: Mean for the initialization of weight values.
  - `weight_init_stdev`: Standard deviation for the initialization of weight values.
  - `weight_init_type`: Initialization type for weight values ("normal" or "uniform").
  - `weight_max_value`: Maximum value for weight values.
  - `weight_min_value`: Minimum value for weight values.
  - `weight_mutate_rate`: Mutation rate for weight values.
  - `weight_replace_rate`: Replacement rate for weight values.

- **Node Settings**

  - `node_add_prob`: Probability of adding a new node.
  - `node_delete_prob`: Probability of deleting an existing node.
  - `bias_init_mean`: Mean for the initialization of bias values.
  - `bias_init_type`: Initialization type for bias values ("normal" or "uniform").
  - `bias_init_stdev`: Standard deviation for the initialization of bias values.
  - `bias_max_value`: Maximum value for bias values.
  - `bias_min_value`: Minimum value for bias values.
  - `bias_mutate_rate`: Mutation rate for bias values.
  - `bias_replace_rate`: Replacement rate for bias values.

- **Stagnation Settings**

  - `max_stagnation`: Maximum number of generations without improvement before species are removed.
  - `species_elitism`: Number of species protected from stagnation.

- **Reproduction Settings**

  - `elitism`: Number of most-fit individuals preserved from one generation to the next.
  - `survival_threshold`: Fraction of each species allowed to reproduce.
  - `min_species_size`: Minimum number of genomes per species after reproduction.

- **Species Settings**
  - `compatibility_threshold`: Genomic distance threshold for considering individuals in the same species.
  - `bad_species_threshold`: Threshold for average fitness, below which a species is considered bad.

## Contributing

Contributions are welcome! If you find issues or have improvements, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Original NEAT Paper: Stanley, K. O., & Miikkulainen, R. (2009). Evolving Neural Networks through Augmenting Topologies.

Inspired by the NEAT algorithm developed by Kenneth O. Stanley and Risto Miikkulainen.

Happy evolving! 🚀
