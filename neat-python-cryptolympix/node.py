from __future__ import annotations
from random import random, choice
from numpy.random import uniform, normal
from __init__ import NeatConfig, ActivationFunctions
import activation_functions as activation_functions


class Node():
    """
    Node

    Represents a node in the NEAT algorithm, used within the context of a neural network.

    Attributes:
    - id (int): Unique identifier for the node.
    - input_sum (float): Current sum before activation.
    - output_value (float): After the activation function is applied.
    - output_connections (list[ConnectionGene]): List of connection genes representing connections from this node to others.
    - layer (int): Layer in the neural network to which the node belongs.
    - activation_function (ActivationFunctions): Activation function for the node.

    Methods:
    - get_activation_function(activation_function: ActivationFunctions) -> callable: Get the activation function based on the specified string.
    - activate() -> None: Activate the node using its activation function.
    - propagate_output() -> None: Propagate the output to connected nodes.
    - mutate(config: NeatConfig, is_bias_node: bool = False) -> None: Mutate the node's properties based on the NEAT configuration.
    - is_connected_to(node: Node) -> bool: Check if this node is connected to the specified node.
    - is_equal(other: Node) -> bool: Compare two nodes. 
    - clone() -> Node: Return a copy of this node.

    """

    def __init__(self, id: int, activation_function: ActivationFunctions, layer=0) -> None:
        """
        Initialize a Node instance.

        Args:
        - id (int): Unique identifier for the node.
        - activation_function (ActivationFunctions): Activation function for the node.
        - layer (int): Layer in the neural network to which the node belongs (default is 0).

        """
        from connection_gene import ConnectionGene
        self.id = id
        self.input_sum = 0  # current sum before activation
        self.output_value = 0  # after activation function is applied
        self.output_connections: list[ConnectionGene] = []
        self.layer = layer
        self.activation_function = activation_function

    def get_activation_function(self, activation_function: ActivationFunctions):
        """
        Get the activation function based on the specified string.

        Args:
        - activation_function (ActivationFunctions): String representing the activation function.

        Returns:
        - callable: The corresponding activation function.

        """
        function_mapping = {
            "elu": activation_functions.elu,
            "leaky_relu": activation_functions.leaky_relu,
            "linear": activation_functions.linear,
            "prelu": activation_functions.prelu,
            "relu": activation_functions.relu,
            "sigmoid": activation_functions.sigmoid,
            "softmax": activation_functions.softmax,
            "step": activation_functions.step,
            "swish": activation_functions.swish,
            "tanh": activation_functions.tanh,
        }
        return function_mapping.get(activation_function, activation_functions.sigmoid)

    def activate(self) -> None:
        """
        Activate the node using its activation function.

        """
        if self.layer != 0:
            activation = self.get_activation_function(self.activation_function)
            self.output_value = activation(self.input_sum)

    def propagate_output(self) -> None:
        """
        Propagate the output to connected nodes.

        """
        for c in self.output_connections:
            if c.enabled:
                c.to_node.input_sum += c.weight * self.output_value

    def mutate(self, config: NeatConfig, is_bias_node=False) -> None:
        """
        Mutate the node's properties based on the NEAT configuration.

        Args:
        - config (NeatConfig): NEAT configuration settings.
        - is_bias_node (bool): Flag indicating whether the node is a bias node (default is False).

        """
        if is_bias_node:
            if random() < config["bias_replace_rate"]:
                self.output_value = uniform(
                    config["bias_min_value"], config["bias_max_value"])

            elif random() < config["bias_mutate_rate"]:
                # otherwise slightly change it
                self.output_value += normal(config["bias_init_mean"],
                                            config["bias_init_stdev"]) / 50
                # keep weight between bounds
                if self.output_value > config["bias_max_value"]:
                    self.output_value = config["bias_max_value"]
                if self.output_value < config["bias_min_value"]:
                    self.output_value = config["bias_min_value"]

        if random() < config["activation_mutate_rate"]:
            activations_functions = ["step", "sigmoid", "tanh", "relu",
                                     "leaky_relu", "prelu", "elu", "softmax", "linear", "swish"]
            random_function = choice(activations_functions)
            self.activation_function = self.get_activation_function(
                random_function)

    def is_connected_to(self, node: Node) -> bool:
        """
        Check if this node is connected to the specified node.

        Args:
        - node (Node): The node to check for a connection.

        Returns:
        - bool: True if connected, False otherwise.

        """
        if node.layer == self.layer:
            return False

        if node.layer < self.layer:
            for c in node.output_connections:
                if c.to_node == self:
                    return True
        else:
            for c in self.output_connections:
                if c.to_node == node:
                    return True

        return False
    
    def is_equal(self, other: Node):
        """
        Compare two nodes.
        
        Args:
            other (Node): An other node to compare with it.

        Returns:
            bool: True if the nodes are equals, otherwise false.
        """
        if self.id != other.id or self.activation_function != other.activation_function or self.layer != other.layer:
            return False
        
        for c1 in self.output_connections:
            found = False
            for c2 in other.output_connections:
                if c1.is_equal(c2):
                    found = True
            if not found:
                return False
            
        return True

    def clone(self) -> Node:
        """
        Return a copy of this node.

        Returns:
        - Node: A copy of this node.

        """
        return Node(self.id, self.activation_function, self.layer)
