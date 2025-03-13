from __future__ import annotations
import os
import math
import pickle
import string
import secrets
from termcolor import colored
from numpy.random import normal, uniform
from random import random, randrange, choice
from __init__ import NeatConfig
from node import Node
from connection_gene import ConnectionGene
from connection_history import ConnectionHistory


def generate_uid(size: int):
    characters = string.ascii_letters + string.digits
    uid = ''.join(secrets.choice(characters) for _ in range(size))
    return uid


next_innovation_nb = 1


class Genome():
    """
    Genome

    Represents an individual genome in the NEAT algorithm, containing nodes and connection genes.

    Attributes:
    - id (str): Unique identifier for the genome.
    - config (NeatConfig): NEAT configuration settings.
    - genes (list[ConnectionGene]): List of connection genes representing the structure of the genome.
    - nodes (list[Node]): List of nodes in the genome.
    - inputs (int): Number of input nodes.
    - outputs (int): Number of output nodes.
    - layers (int): Number of layers in the neural network.
    - next_node (int): Next available node ID for mutation.
    - network (list[Node]): List of nodes in order for the neural network.
    - fitness (float): Fitness value of the genome.

    Methods:
    - fully_connect(innovation_history: list[ConnectionHistory]) -> None: Fully connect the neural network.
    - get_node(id: int) -> Node: Get the node with the specified ID.
    - connect_nodes() -> None: Add connections going out of a node to that node.
    - feed_forward(input_values: list[int]) -> list[float]: Perform feedforward computation.
    - generate_network() -> None: Set up the neural network as a list of nodes in order to be engaged.
    - add_node(innovation_history: list[ConnectionHistory]) -> None: Mutate the genome by adding a new node.
    - remove_node() -> None: Remove a random node from the genome.
    - add_connection(innovation_history: list[ConnectionHistory]) -> None: Add a connection between two nodes.
    - remove_connection() -> None: Remove a random connection from the genome.
    - new_connection_weight() -> float: Generate a new connection weight.
    - get_innovation_number(innovation_history: list[ConnectionHistory], from_node: Node, to_node: Node) -> int: Get the innovation number for a new mutation.
    - fully_connected() -> bool: Check if the network is fully connected.
    - mutate(innovation_history: list[ConnectionHistory]) -> None: Mutate the genome.
    - crossover(parent: Genome) -> Genome: Perform crossover with another parent genome.
    - matching_gene(parent: Genome, innovation: int) -> int: Check if there is a gene matching the input innovation number in the parent genome.
    - print_genome() -> None: Print information about the genome to the console.
    - is_equal(other: Genome) -> bool: Compare two genomes.
    - clone() -> Genome: Return a copy of this genome.
    - save(file_path: str) -> None: Save the genome to a file.
    - load(file_path: str) -> Genome: Load a genome from a file.

    """

    def __init__(self, config: NeatConfig, crossover=False) -> None:
        """
        Initialize a Genome instance.

        Args:
        - config (NeatConfig): NEAT configuration settings.
        - crossover (bool): Flag indicating whether the genome is created through crossover.

        """
        self.id = generate_uid(8)
        self.config = config
        self.genes: list[ConnectionGene] = []
        self.nodes: list[Node] = []
        self.inputs = config["num_inputs"]
        self.outputs = config["num_outputs"]
        self.layers = 2
        self.next_node = 0
        # a list of the nodes in the order that they need to be considered in the NN
        self.network: list[Node] = []
        self.fitness = 0

        if crossover == True:
            return

        # create input nodes
        for i in range(0, self.inputs):
            self.nodes.append(Node(i, self.config["activation_default"], 0))
            self.next_node += 1

        # create output nodes
        for i in range(0, self.outputs):
            self.nodes.append(
                Node(i + self.inputs, self.config["activation_default"], 1))
            self.next_node += 1

        # create bias nodes
        self.nodes.append(
            Node(self.next_node, self.config["activation_default"], 0))
        self.bias_node = self.next_node
        self.next_node += 1

    def fully_connect(self, innovation_history: "list[ConnectionHistory]") -> None:
        """
        Fully connect the neural network.

        Args:
        - innovation_history (list[ConnectionHistory]): List of connection histories for innovation tracking.

        """
        # connect the inputs nodes and outputs nodes
        for i in range(0, self.inputs):
            for j in range(0, self.outputs):
                connection_innovation_nb = self.get_innovation_number(
                    innovation_history,
                    self.nodes[i],
                    self.nodes[self.inputs + j]
                )

                self.genes.append(
                    ConnectionGene(
                        self.nodes[i],
                        self.nodes[self.inputs + j],
                        self.new_connection_weight(),
                        connection_innovation_nb,
                        self.config["enabled_default"]
                    )
                )

        # connect the bias nodes to outputs nodes
        for i in range(0, self.outputs):
            connection_innovation_nb = self.get_innovation_number(
                innovation_history,
                self.nodes[self.bias_node],
                self.nodes[self.inputs + i]
            )

            self.genes.append(
                ConnectionGene(
                    self.nodes[self.bias_node],
                    self.nodes[self.inputs + i],
                    self.new_connection_weight(),
                    connection_innovation_nb,
                    self.config["enabled_default"]
                )
            )

        # changed this so if error here
        self.connect_nodes()

    def get_node(self, id: int) -> Node:
        """
        Get the node with the specified ID.

        Args:
        - id (int): ID of the node to retrieve.

        Returns:
        - Node: Node with the specified ID, or None if not found.

        """
        for n in self.nodes:
            if n.id == id:
                return n
        return None

    def connect_nodes(self) -> None:
        """
        Add connections going out of a node to that node so that it can access the next node during feeding forward.

        """
        for n in self.nodes:
            # clear the connections
            n.output_connections = []

        for g in self.genes:
            g.from_node.output_connections.append(g)  # add it to node

    def feed_forward(self, input_values: "list[int]") -> "list[float]":
        """
        Perform feedforward computation.

        Args:
        - input_values (list[int]): List of input values.

        Returns:
        - list[float]: List of output values.

        """
        try:
            # set the outputs of the input nodes
            for i in range(0, self.inputs):
                self.nodes[i].output_value = input_values[i]

            # output of bias is 1
            self.nodes[self.bias_node].output_value = 1

            for n in self.network:
                # for each node in the network activate it (see node class for what this does)
                n.activate()
                n.propagate_output()

            # the outputs are nodes[inputs] to nodes[inputs+outputs-1]
            outs: list[float] = [0] * self.outputs
            for i in range(0, self.outputs):
                outs[i] = self.nodes[self.inputs + i].output_value

            for i in range(0, len(self.nodes)):
                # reset all the nodes for the next feed forward
                self.nodes[i].input_sum = 0

            return outs
        except Exception as e:
            # Handle the exception here or re-raise it if needed
            print(f"An error occurred during feed_forward: {e}")
            # Optionally, re-raise the exception to propagate it further
            # raise e

    def generate_network(self) -> None:
        """
        Set up the neural network as a list of nodes in order to be engaged.

        """
        self.connect_nodes()
        self.network: list[Node] = []

        # for each layer add the node in that layer, since layers cannot connect to themselves there is no need to order the nodes within a layer
        for l in range(0, self.layers):
            for n in self.nodes:
                if n.layer == l:
                    # if that node is in that layer
                    self.network.append(n)

    def add_node(self, innovation_history: "list[ConnectionHistory]") -> None:
        """
        Mutate the genome by adding a new node. It does this by picking a random connection and disabling
        it then 2 new connections are added: one between the input node of the disabled connection and the new node,
        and the other between the new node and the output of the disabled connection.

        Args:
        - innovation_history (list[ConnectionHistory]): List of connection histories for innovation tracking.

        """
        # pick a random connection to create a node between
        if len(self.genes) == 0:
            self.add_connection(innovation_history)
            return

        random_connection = math.floor(randrange(0, len(self.genes)))

        while self.genes[random_connection].from_node == self.nodes[self.bias_node] and len(self.genes) != 1:
            # dont disconnect bias
            random_connection = math.floor(randrange(0, len(self.genes)))

        self.genes[random_connection].enabled = False  # disable it

        new_node_nb = self.next_node
        self.nodes.append(Node(new_node_nb, self.config["activation_default"]))
        self.next_node += 1

        # add a new connection to the new node with a weight of 1
        connection_innovation_nb = self.get_innovation_number(
            innovation_history,
            self.genes[random_connection].from_node,
            self.get_node(new_node_nb)
        )

        self.genes.append(
            ConnectionGene(
                self.genes[random_connection].from_node,
                self.get_node(new_node_nb),
                1,
                connection_innovation_nb,
                self.config["enabled_default"]
            )
        )

        connection_innovation_nb = self.get_innovation_number(
            innovation_history,
            self.get_node(new_node_nb),
            self.genes[random_connection].to_node
        )

        # add a new connection from the new node with a weight the same as the disabled connection
        self.genes.append(
            ConnectionGene(
                self.get_node(new_node_nb),
                self.genes[random_connection].to_node,
                self.genes[random_connection].weight,
                connection_innovation_nb,
                self.config["enabled_default"]
            )
        )

        self.get_node(
            new_node_nb).layer = self.genes[random_connection].from_node.layer + 1

        connection_innovation_nb = self.get_innovation_number(
            innovation_history,
            self.nodes[self.bias_node],
            self.get_node(new_node_nb)
        )

        # get the value for the bias node
        bias_value = 0.0
        if self.config["bias_init_type"] == "normal":
            bias_value = normal(
                self.config["bias_init_mean"], self.config["bias_init_stdev"])
            # keep value between bounds
            if bias_value > self.config["bias_max_value"]:
                bias_value = self.config["bias_max_value"]
            if bias_value < self.config["bias_min_value"]:
                bias_value = self.config["bias_min_value"]
        elif self.config["bias_init_type"] == "uniform":
            bias_value = uniform(
                self.config["bias_min_value"], self.config["bias_max_value"])

        # connect the bias to the new node
        self.genes.append(
            ConnectionGene(
                self.nodes[self.bias_node],
                self.get_node(new_node_nb),
                bias_value,
                connection_innovation_nb,
                self.config["enabled_default"]
            )
        )

        # If the layer of the new node is equal to the layer of the output node of the old connection then a new layer needs to be created
        # more accurately the layer numbers of all layers equal to or greater than this new node need to be incremented
        if self.get_node(new_node_nb).layer == self.genes[random_connection].to_node.layer:
            for n in self.nodes[:-1]:
                # dont include this newest node
                if (n.layer >= self.get_node(new_node_nb).layer):
                    n.layer += 1
            self.layers += 1

        self.connect_nodes()

    def remove_node(self):
        """
        Remove a random node from the genome.

        This method randomly selects a node (excluding input, output, and bias nodes) and removes it
        from the genome along with any connections associated with that node.

        Returns:
        - None

        """
        # select a random node by excluding inputs, outputs and bias nodes
        random_node = choice(self.nodes[self.bias_node:])
        self.nodes.remove(random_node)

        # remove the connection that are connected to the random node selected
        for g in self.genes:
            if g.from_node == random_node or g.to_node == random_node:
                self.genes.remove(g)

    def add_connection(self, innovation_history: "list[ConnectionHistory]") -> None:
        """
        Adds a connection between two nodes that aren't currently connected.

        This method randomly selects two nodes in the genome and creates a connection between them,
        provided that the selected nodes are not in the same layer and are not already connected.

        Args:
        - innovation_history (list[ConnectionHistory]): List of connection histories for innovation tracking.

        Returns:
        - None

        """
        # cannot add a connection to a fully connected network
        if self.fully_connected():
            return

        def random_connection_nodes_are_valid(rand1: int, rand2: int) -> bool:
            if (self.nodes[rand1].layer == self.nodes[rand2].layer):
                return False  # if the nodes are in the same layer
            if (self.nodes[rand1].is_connected_to(self.nodes[rand2])):
                return False  # if the nodes are already connected
            return True

        # get random nodes
        random_node_1 = math.floor(randrange(0, len(self.nodes)))
        random_node_2 = math.floor(randrange(0, len(self.nodes)))
        while not random_connection_nodes_are_valid(random_node_1, random_node_2):
            # while the random this.nodes are no good get new ones
            random_node_1 = math.floor(randrange(0, len(self.nodes)))
            random_node_2 = math.floor(randrange(0, len(self.nodes)))

        temp: int
        if self.nodes[random_node_1].layer > self.nodes[random_node_2].layer:
            # if the first random node is after the second then switch
            temp = random_node_2
            random_node_2 = random_node_1
            random_node_1 = temp

        # get the innovation number of the connection
        # this will be a new number if no identical genome has mutated in the same way
        connection_innovation_nb = self.get_innovation_number(
            innovation_history,
            self.nodes[random_node_1],
            self.nodes[random_node_2]
        )

        # add the connection with a random array
        self.genes.append(
            ConnectionGene(
                self.nodes[random_node_1],
                self.nodes[random_node_2],
                self.new_connection_weight(),
                connection_innovation_nb,
                self.config["enabled_default"]
            )
        )

        # changed this so if error here
        self.connect_nodes()

    def remove_connection(self):
        """
        Removes a random connection from the genome.

        This method randomly selects a connection in the genome and removes it.

        Returns:
        - None

        """
        random_gene = choice(self.genes)
        self.genes.remove(random_gene)

    def new_connection_weight(self) -> float:
        """
        Generates a new random connection weight based on the specified initialization type.

        Returns:
        - float: A new random connection weight.

        """
        weight = 0.0
        if self.config["weight_init_type"] == "normal":
            weight = normal(
                self.config["weight_init_mean"], self.config["weight_init_stdev"])
            # keep value between bounds
            if weight > self.config["weight_max_value"]:
                weight = self.config["weight_max_value"]
            if weight < self.config["weight_min_value"]:
                weight = self.config["weight_min_value"]
        elif self.config["weight_init_type"] == "uniform":
            weight = uniform(
                self.config["weight_min_value"], self.config["weight_max_value"])

        return weight

    def get_innovation_number(self, innovation_history: "list[ConnectionHistory]", from_node: Node, to_node: Node) -> int:
        """
        Returns the innovation number for a new mutation. If this mutation has never been seen before, it will be given
        a new unique innovation number. If this mutation matches a previous mutation, it will be given the same innovation
        number as the previous one.

        Args:
        - innovation_history (list[ConnectionHistory]): List of connection histories for innovation tracking.
        - from_node (Node): The node from which the connection originates.
        - to_node (Node): The node to which the connection leads.

        Returns:
        - int: The innovation number for the new mutation.

        """
        is_new = True
        global next_innovation_nb
        connection_innovation_nb = next_innovation_nb

        for i in innovation_history:
            # for each previous mutation
            if i.matches(from_node, to_node):
                # if match found
                is_new = False  # its not a new mutation
                # set the innovation number as the innovation number of the match
                connection_innovation_nb = i.innovation_nb
                break

        if is_new:
            # then add this mutation to the innovationHistory
            innovation_history.append(
                ConnectionHistory(
                    from_node,
                    to_node,
                    connection_innovation_nb
                )
            )
            next_innovation_nb += 1

        return connection_innovation_nb

    def fully_connected(self) -> bool:
        """
        Returns whether the network is fully connected or not.

        Returns:
        - bool: True if the network is fully connected, False otherwise.

        """
        max_connections = 0
        # array which stored the amount of this.nodes in each layer
        nodes_in_layers: list[int] = [0] * self.layers

        # populate array
        for n in self.nodes:
            nodes_in_layers[n.layer] += 1

        # for each layer the maximum amount of connections is the number in this layer * the number of this.nodes in front of it
        # so lets add the max for each layer together and then we will get the maximum amount of connections in the network
        for i in range(0, self.layers - 1):
            nodes_in_front = 0
            for j in range(i + 1, self.layers):
                # for each layer in front of this layer
                nodes_in_front += nodes_in_layers[j]  # add up nodes

            max_connections += nodes_in_layers[i] * nodes_in_front

        if max_connections <= len(self.genes):
            # if the number of connections is equal to the max number of connections possible then it is full
            return True

        return False

    def mutate(self, innovation_history: "list[ConnectionHistory]") -> None:
        """
        Mutates the genome by applying various mutations, including node and connection mutations.

        Args:
        - innovation_history (list[ConnectionHistory]): List of connection histories for innovation tracking.

        """
        if len(self.genes) == 0:
            self.add_connection(innovation_history)

        for i in range(len(self.nodes)):
            is_bias_node = i == self.bias_node
            self.nodes[i].mutate(self.config, is_bias_node)

        for i in range(len(self.genes)):
            self.genes[i].mutate(self.config)

        if random() < self.config["conn_add_prob"]:
            self.add_connection(innovation_history)

        if random() < self.config["conn_delete_prob"]:
            self.remove_connection()

        if random() < self.config["node_add_prob"]:
            self.add_node(innovation_history)

        if random() < self.config["node_delete_prob"]:
            self.remove_node()

    def crossover(self, parent: Genome) -> Genome:
        """
        Performs crossover with another parent genome to create a new child genome.

        Args:
        - parent (Genome): The other parent genome for crossover.

        Returns:
        - Genome: The child genome resulting from crossover.

        """
        child = Genome(self.config, True)
        child.genes = []
        child.nodes = []
        child.layers = self.layers
        child.next_node = self.next_node
        child.bias_node = self.bias_node
        # list of genes to be inherited form the parents
        child_genes: list[ConnectionGene] = []
        is_enabled: list[bool] = []

        # all inherited genes
        for g in self.genes:
            set_enabled = True  # is this node in the child going to be enabled
            parent_gene = self.matching_gene(
                parent, g.innovation_nb)
            if parent_gene != -1:
                # if the genes match
                if not g.enabled or not parent.genes[parent_gene].enabled:
                    # if either of the matching genes are disabled
                    if random() < 0.75:
                        # 75% of the time disable the child gene
                        set_enabled = False

                if random() < 0.5:
                    child_genes.append(g)
                else:
                    # get gene from parent
                    child_genes.append(parent.genes[parent_gene])
            else:
                # disjoint or excess gene
                child_genes.append(g)
                set_enabled = g.enabled
            is_enabled.append(set_enabled)

        # since all excess and disjoint genes are inherited from the more fit parent (this Genome) the child structure is no different from this parent | with exception of dormant connections being enabled but this wont effect this.nodes
        # so all the this.nodes can be inherited from this parent
        for n in self.nodes:
            child.nodes.append(n.clone())

        # clone all the connections so that they connect the child nodes
        for i in range(0, len(child_genes)):
            child.genes.append(
                child_genes[i].clone(
                    child.get_node(child_genes[i].from_node.id),
                    child.get_node(child_genes[i].to_node.id)
                )
            )
            child.genes[i].enabled = is_enabled[i]

        child.connect_nodes()
        return child

    def matching_gene(self, parent: Genome, innovation: int):
        """
        Checks whether there is a gene matching the input innovation number in the input genome.

        Args:
        - parent (Genome): The parent genome to search for a matching gene.
        - innovation (int): The innovation number to match.

        Returns:
        - int: Index of the matching gene in the parent genome, or -1 if no matching gene is found.

        """
        for i in range(len(parent.genes)):
            if (parent.genes[i].innovation_nb == innovation):
                return i
        return -1  # no matching gene found

    def print_genome(self):  # pragma: no cover
        """
        Prints information about the genome to the console.

        This method provides a detailed summary of the genome, including the number of layers, bias nodes, and the list
        of connection genes with their respective details.

        Note:
        The method uses colored console output for better readability.

        """
        str_genes = '\n\t'.join([
            str('{' +
                colored('innovation_nb', attrs=["bold"]) + ': ' + str(g.innovation_nb) + ', ' +
                colored('from_node', attrs=["bold"]) + ': ' + str(g.from_node.id) + ', ' +
                colored('to_node', attrs=["bold"]) + ': ' + str(g.to_node.id) + ', ' +
                colored("enabled", attrs=["bold"]) + ': ' + str(g.enabled) + ', ' +
                colored('from_layer', attrs=["bold"]) + ': ' + str(g.from_node.layer) + ', ' +
                colored('to_layer', attrs=["bold"]) + ': ' + str(g.to_node.layer) + ', ' +
                colored('weight', attrs=["bold"]) + ': ' + str(g.weight) +
                '}') for g in self.genes])

        print(
            f'''
        ------------------------------ {colored("GENOME", attrs=["bold"])} ----------------------------
        {colored("⚪️ Resume:", attrs=["bold"])}
        {colored('{layers', attrs=["bold"]) + ": " + str(self.layers) + ", " + colored('bias nodes', attrs=["bold"]) + ": " +
                str(self.bias_node) + ", " + colored('nodes', attrs=["bold"]) + ": " + str(len(self.nodes)) + "}"}
        {colored("⚪️ Connection genes:", attrs=["bold"])}
        {str_genes}
        ''')

    def is_equal(self, other: Genome):
        """
        Compare two genomes.

        Args:
            other (Genome): The other genome to compare with it

        Returns:
            bool: True if the genome are equals, otherwise false.
        """

        # Compare the number of nodes
        if len(self.nodes) != len(other.nodes):
            return False

        # Compare the number of genes
        if len(self.genes) != len(other.genes):
            return False

        # Compare each node

        def get_node_id(node: Node):
            return node.id

        self.nodes.sort(key=get_node_id)
        other.nodes.sort(key=get_node_id)

        for i in range(len(self.nodes)):
            if not self.nodes[i].is_equal(other.nodes[i]):
                return False

        # Compare each gene

        def get_gene_innovation_nb(gene: ConnectionGene):
            return gene.innovation_nb

        self.genes.sort(key=get_gene_innovation_nb)
        other.genes.sort(key=get_gene_innovation_nb)

        for i in range(len(self.genes)):
            if not self.genes[i].is_equal(other.genes[i]):
                return False

        return True

    def clone(self):
        """
        Returns a copy of this genome.

        Returns:
        - Genome: A copy of the current genome.

        """
        clone = Genome(self.config, True)
        for n in self.nodes:
            # copy nodes
            clone.nodes.append(n.clone())

        # copy all the connections so that they connect the clone new nodes
        for g in self.genes:
            # copy genes
            clone.genes.append(
                g.clone(
                    clone.get_node(g.from_node.id),
                    clone.get_node(g.to_node.id)
                )
            )

        clone.layers = self.layers
        clone.next_node = self.next_node
        clone.bias_node = self.bias_node
        clone.connect_nodes()

        return clone

    def save(self, file_path: str):
        """
        Save the genome to a file.

        Args:
        - file_path (str): The path to the file where the genome will be saved.

        """
        # Check if the directory exist
        dir = os.path.dirname(file_path)
        if not os.path.exists(dir):
            os.makedirs(dir)

        base, ext = os.path.splitext(file_path)
        if not ext:
            file_path = f"{file_path}.pkl"

        # Serialize and save the model to a file
        with open(file_path, 'wb') as file:
            pickle.dump(self, file)

        print(f"Genome saved to '{file_path}'")

    @staticmethod
    def load(file_path: str) -> Genome:
        """
        Load a genome from a file.

        Args:
        - file_path (str): The path to the file from which the genome will be loaded.

        Returns:
        - Genome: The loaded genome.

        """
        with open(file_path, 'rb') as file:
            print(f"Genome {file_path} loaded")
            return pickle.load(file)
