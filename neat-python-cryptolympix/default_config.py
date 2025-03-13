from __types__ import NeatConfig

default_config: NeatConfig = {
    # ======== NEAT =========== #
    "population_size": 100,
    "fitness_threshold": 100,
    "no_fitness_termination": True,
    "reset_on_extinction": True,

    # ======== GENOME =========== #
    "activation_default": "sigmoid",
    "activation_mutate_rate": 0.1,
    "num_inputs": 0,
    "num_outputs": 5,  # enter long | enter short | close long | close short | wait
    "bias_init_mean": 0.0,
    "bias_init_type": "normal",
    "bias_init_stdev": 1.0,
    "bias_max_value": 1.0,
    "bias_min_value": -1.0,
    "bias_mutate_rate": 0.5,
    "bias_replace_rate": 0.1,
    "compatibility_disjoint_coefficient": 1.0,
    "compatibility_weight_coefficient": 0.5,
    "conn_add_prob": 0.1,
    "conn_delete_prob": 0.0,
    "enabled_default": True,
    "enabled_mutate_rate": 0.1,
    "initial_connections": "full",
    "node_add_prob": 0.1,
    "node_delete_prob": 0.0,
    "weight_init_mean": 0.0,
    "weight_init_stdev": 1.0,
    "weight_init_type": "normal",
    "weight_max_value": 1.0,
    "weight_min_value": -1.0,
    "weight_mutate_rate": 0.9,
    "weight_replace_rate": 0.1,

    # ======== STAGNATION =========== #
    "max_stagnation": 15,
    "species_elitism": 2,

    # ======== REPRODUCTION =========== #
    "elitism": 2,
    "survival_threshold": 0.2,
    "min_species_size": 2,

    # ======== SPECIES =========== #
    "compatibility_threshold": 3.0,
    "bad_species_threshold": 0.25
}
