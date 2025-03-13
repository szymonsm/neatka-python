from __future__ import annotations
from random import random
from numpy.random import normal, uniform
from __init__ import NeatConfig
from node import Node


class ConnectionGene():
    """
    Connection Gene

    Represents a connection gene in the NEAT algorithm, connecting two nodes in a neural network.

    Attributes:
    - from_node (Node): The source node of the connection.
    - to_node (Node): The target node of the connection.
    - weight (float): The weight associated with the connection.
    - enabled (bool): Indicates if the connection is enabled or disabled.
    - innovation_nb (int): The innovation number uniquely identifying this connection.

    Methods:
    - mutate(config: NeatConfig) -> None: Randomly mutates the connection's weight or enables/disables it based on NEAT configuration.
    - is_equal(other: ConnectionGene) -> bool: Compare two connection genes.
    - clone(from_node: Node, to_node: Node) -> ConnectionGene: Creates a copy of the connection gene with new source and target nodes.

    """

    def __init__(self, from_node: Node, to_node: Node, weight: float, innovation_nb: int, enabled: bool) -> None:
        """
        Initialize a ConnectionGene instance.

        Args:
        - from_node (Node): The source node of the connection.
        - to_node (Node): The target node of the connection.
        - weight (float): The weight associated with the connection.
        - innovation_nb (int): The innovation number uniquely identifying this connection.
        - enabled (bool): Indicates if the connection is enabled or disabled.

        """
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        self.enabled = enabled
        self.innovation_nb = innovation_nb

    def mutate(self, config: NeatConfig):
        """
        Mutate the connection's weight or enable/disable it based on the NEAT configuration.

        Args:
        - config (NeatConfig): NEAT configuration used for mutation probabilities.

        """
        if random() < config["weight_replace_rate"]:
            self.weight = uniform(
                config["weight_min_value"], config["weight_max_value"])

        elif random() < config["weight_mutate_rate"]:
            # otherwise slightly change it
            self.weight += normal(config["weight_init_mean"],
                                  config["weight_init_stdev"]) / 50
            # keep weight between bounds
            if self.weight > config["weight_max_value"]:
                self.weight = config["weight_max_value"]
            if self.weight < config["weight_min_value"]:
                self.weight = config["weight_min_value"]

        if random() < config["enabled_mutate_rate"]:
            self.enabled = False if self.enabled else True

    def is_equal(self, other: ConnectionGene):
        """
        Compare two connection genes.

        Args:
            other (ConnectionGene): An other connection gene to compare with it.

        Returns:
            bool: True if the connection genes are equals, otherwise false.

        """
        return self.from_node.id == other.from_node.id and self.to_node.id == other.to_node.id and self.weight == other.weight and self.innovation_nb == other.innovation_nb and self.enabled == other.enabled

    def clone(self, from_node: Node, to_node: Node):
        """
        Create a copy of the connection gene with new source and target nodes.

        Args:
        - from_node (Node): The new source node for the copied connection.
        - to_node (Node): The new target node for the copied connection.

        Returns:
        - ConnectionGene: A copy of the connection gene with new source and target nodes.

        """
        return ConnectionGene(from_node, to_node, self.weight, self.innovation_nb, self.enabled)
