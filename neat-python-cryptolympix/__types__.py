from typing import TypedDict, Literal

InitialConnection = Literal["none", "full"]
ActivationFunctions = Literal["step", "sigmoid", "tanh", "relu",
                              "leaky_relu", "prelu", "elu", "softmax", "linear", "swish"]
DistributionType = Literal["normal", "uniform"]


class NeatConfig(TypedDict):
    """
    NEAT Configuration

    This TypedDict defines the configuration parameters for the NEAT (NeuroEvolution of Augmenting Topologies) algorithm.

    Attributes:
    - population_size (int): The number of individuals in each generation.
    - fitness_threshold (float): Termination threshold for the fitness criterion.
    - no_fitness_termination (bool): If True, the evolution process won't terminate based on fitness.
    - reset_on_extinction (bool): If True, a new random population is created when all species go extinct.

    - activation_default (ActivationFunctions): Default activation function for neurons.
    - activation_mutate_rate (float): Mutation rate for activation functions.
    - num_inputs (int): Number of input nodes in the neural network.
    - num_outputs (int): Number of output nodes in the neural network.
    - bias_init_mean (float): Mean for the initialization of bias values.
    - bias_init_type (DistributionType): Initialization type for bias values.
    - bias_init_stdev (float): Standard deviation for the initialization of bias values.
    - bias_max_value (float): Maximum value for bias values.
    - bias_min_value (float): Minimum value for bias values.
    - bias_mutate_rate (float): Mutation rate for bias values.
    - bias_replace_rate (float): Replacement rate for bias values.
    - compatibility_disjoint_coefficient (float): Coefficient for compatibility calculation related to disjoint genes.
    - compatibility_weight_coefficient (float): Coefficient for compatibility calculation related to weight differences.
    - conn_add_prob (float): Probability of adding a new connection.
    - conn_delete_prob (float): Probability of deleting an existing connection.
    - enabled_default (bool): Default state (enabled or disabled) for new connections.
    - enabled_mutate_rate (float): Mutation rate for connection enable/disable state.
    - initial_connections (InitialConnection): Type of initial connections ("none" or "full").
    - node_add_prob (float): Probability of adding a new node.
    - node_delete_prob (float): Probability of deleting an existing node.
    - weight_init_mean (float): Mean for the initialization of weight values.
    - weight_init_stdev (float): Standard deviation for the initialization of weight values.
    - weight_init_type (DistributionType): Initialization type for weight values.
    - weight_max_value (float): Maximum value for weight values.
    - weight_min_value (float): Minimum value for weight values.
    - weight_mutate_rate (float): Mutation rate for weight values.
    - weight_replace_rate (float): Replacement rate for weight values.

    - max_stagnation (int): Maximum number of generations without improvement before species are removed.
    - species_elitism (int): Number of species protected from stagnation.

    - elitism (int): Number of most-fit individuals preserved from one generation to the next.
    - survival_threshold (float): Fraction of each species allowed to reproduce.
    - min_species_size (int): Minimum number of genomes per species after reproduction.

    - compatibility_threshold (float): Genomic distance threshold for considering individuals in the same species.
    - bad_species_threshold (float): Threshold for average fitness, below which a species is considered bad.

    """
    # ======== NEAT =========== #
    population_size: int
    fitness_threshold: float
    no_fitness_termination: bool
    reset_on_extinction: bool

    # ======== GENOME =========== #
    activation_default: ActivationFunctions
    activation_mutate_rate: float
    num_inputs: int
    num_outputs: int
    bias_init_mean: float
    bias_init_type: DistributionType
    bias_init_stdev: float
    bias_max_value: float
    bias_min_value: float
    bias_mutate_rate: float
    bias_replace_rate: float
    compatibility_disjoint_coefficient: float
    compatibility_weight_coefficient: float
    conn_add_prob: float
    conn_delete_prob: float
    enabled_default: bool
    enabled_mutate_rate: float
    initial_connections: InitialConnection
    node_add_prob: float
    node_delete_prob: float
    weight_init_mean: float
    weight_init_stdev: float
    weight_init_type: DistributionType
    weight_max_value: float
    weight_min_value: float
    weight_mutate_rate: float
    weight_replace_rate: float

    # ======== STAGNATION =========== #
    max_stagnation: int
    species_elitism: int

    # ======== REPRODUCTION =========== #
    elitism: int
    survival_threshold: float
    min_species_size: int

    # ======== SPECIES =========== #
    compatibility_threshold: float
    bad_species_threshold: float
