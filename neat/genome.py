"""Handles genomes (individuals in the population)."""
import copy
import sys
from itertools import count
from random import choice, random, shuffle

from neat.activations import ActivationFunctionSet
from neat.aggregations import AggregationFunctionSet
from neat.config import ConfigParameter, write_pretty_params
from neat.genes import DefaultConnectionGene, DefaultNodeGene
from neat.graphs import creates_cycle
from neat.graphs import required_for_output


class DefaultGenomeConfig(object):
    """Sets up and holds configuration information for the DefaultGenome class."""
    allowed_connectivity = ['unconnected', 'fs_neat_nohidden', 'fs_neat', 'fs_neat_hidden',
                            'full_nodirect', 'full', 'full_direct',
                            'partial_nodirect', 'partial', 'partial_direct']

    def __init__(self, params):
        # Create full set of available activation functions.
        self.activation_defs = ActivationFunctionSet()
        # ditto for aggregation functions - name difference for backward compatibility
        self.aggregation_function_defs = AggregationFunctionSet()
        self.aggregation_defs = self.aggregation_function_defs

        self._params = [ConfigParameter('num_inputs', int),
                        ConfigParameter('num_outputs', int),
                        ConfigParameter('num_hidden', int),
                        ConfigParameter('feed_forward', bool),
                        ConfigParameter('compatibility_disjoint_coefficient', float),
                        ConfigParameter('compatibility_weight_coefficient', float),
                        ConfigParameter('conn_add_prob', float),
                        ConfigParameter('conn_delete_prob', float),
                        ConfigParameter('node_add_prob', float),
                        ConfigParameter('node_delete_prob', float),
                        ConfigParameter('single_structural_mutation', bool, 'false'),
                        ConfigParameter('structural_mutation_surer', str, 'default'),
                        ConfigParameter('initial_connection', str, 'unconnected')]

        # Gather configuration data from the gene classes.
        self.node_gene_type = params['node_gene_type']
        self._params += self.node_gene_type.get_config_params()
        self.connection_gene_type = params['connection_gene_type']
        self._params += self.connection_gene_type.get_config_params()

        # Use the configuration data to interpret the supplied parameters.
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        self.node_gene_type.validate_attributes(self)
        self.connection_gene_type.validate_attributes(self)

        # By convention, input pins have negative keys, and the output
        # pins have keys 0,1,...
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]

        self.connection_fraction = None

        # Verify that initial connection type is valid.
        # pylint: disable=access-member-before-definition
        if 'partial' in self.initial_connection:
            c, p = self.initial_connection.split()
            self.initial_connection = c
            self.connection_fraction = float(p)
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")

        assert self.initial_connection in self.allowed_connectivity

        # Verify structural_mutation_surer is valid.
        # pylint: disable=access-member-before-definition
        if self.structural_mutation_surer.lower() in ['1', 'yes', 'true', 'on']:
            self.structural_mutation_surer = 'true'
        elif self.structural_mutation_surer.lower() in ['0', 'no', 'false', 'off']:
            self.structural_mutation_surer = 'false'
        elif self.structural_mutation_surer.lower() == 'default':
            self.structural_mutation_surer = 'default'
        else:
            error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            raise RuntimeError(error_string)

        self.node_indexer = None

    def add_activation(self, name, func):
        self.activation_defs.add(name, func)

    def add_aggregation(self, name, func):
        self.aggregation_function_defs.add(name, func)

    def save(self, f):
        if 'partial' in self.initial_connection:
            if not (0 <= self.connection_fraction <= 1):
                raise RuntimeError(
                    "'partial' connection value must be between 0.0 and 1.0, inclusive.")
            f.write(f'initial_connection      = {self.initial_connection} {self.connection_fraction}\n')
        else:
            f.write(f'initial_connection      = {self.initial_connection}\n')

        assert self.initial_connection in self.allowed_connectivity

        write_pretty_params(f, self, [p for p in self._params
                                      if 'initial_connection' not in p.name])

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            if node_dict:
                self.node_indexer = count(max(list(node_dict)) + 1)
            else:
                self.node_indexer = count(max(list(node_dict)) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id

    def check_structural_mutation_surer(self):
        if self.structural_mutation_surer == 'true':
            return True
        elif self.structural_mutation_surer == 'false':
            return False
        elif self.structural_mutation_surer == 'default':
            return self.single_structural_mutation
        else:
            error_string = f"Invalid structural_mutation_surer {self.structural_mutation_surer!r}"
            raise RuntimeError(error_string)


class DefaultGenome(object):
    """
    A genome for generalized neural networks.

    Terminology
        pin: Point at which the network is conceptually connected to the external world;
             pins are either input or output.
        node: Analog of a physical neuron.
        connection: Connection between a pin/node output and a node's input, or between a node's
             output and a pin/node input.
        key: Identifier for an object, unique within the set of similar objects.

    Design assumptions and conventions.
        1. Each output pin is connected only to the output of its own unique
           neuron by an implicit connection with weight one. This connection
           is permanently enabled.
        2. The output pin's key is always the same as the key for its
           associated neuron.
        3. Output neurons can be modified but not deleted.
        4. The input values are applied to the input pins unmodified.
    """

    @classmethod
    def parse_config(cls, param_dict):
        param_dict['node_gene_type'] = DefaultNodeGene
        param_dict['connection_gene_type'] = DefaultConnectionGene
        return DefaultGenomeConfig(param_dict)

    @classmethod
    def write_config(cls, f, config):
        config.save(f)

    def __init__(self, key):
        # Unique identifier for a genome instance.
        self.key = key

        # (gene_key, gene) pairs for gene sets.
        self.connections = {}
        self.nodes = {}

        # Fitness results.
        self.fitness = None

    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""

        # Create node genes for the output pins.
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(config, node_key)

        # Add hidden nodes if requested.
        if config.num_hidden > 0:
            for i in range(config.num_hidden):
                node_key = config.get_new_node_key(self.nodes)
                assert node_key not in self.nodes
                node = self.create_node(config, node_key)
                self.nodes[node_key] = node

        # Add connections based on initial connectivity type.

        if 'fs_neat' in config.initial_connection:
            if config.initial_connection == 'fs_neat_nohidden':
                self.connect_fs_neat_nohidden(config)
            elif config.initial_connection == 'fs_neat_hidden':
                self.connect_fs_neat_hidden(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = fs_neat will not connect to hidden nodes;",
                        "\tif this is desired, set initial_connection = fs_neat_nohidden;",
                        "\tif not, set initial_connection = fs_neat_hidden",
                        sep='\n', file=sys.stderr)
                self.connect_fs_neat_nohidden(config)
        elif 'full' in config.initial_connection:
            if config.initial_connection == 'full_nodirect':
                self.connect_full_nodirect(config)
            elif config.initial_connection == 'full_direct':
                self.connect_full_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = full with hidden nodes will not do direct input-output connections;",
                        "\tif this is desired, set initial_connection = full_nodirect;",
                        "\tif not, set initial_connection = full_direct",
                        sep='\n', file=sys.stderr)
                self.connect_full_nodirect(config)
        elif 'partial' in config.initial_connection:
            if config.initial_connection == 'partial_nodirect':
                self.connect_partial_nodirect(config)
            elif config.initial_connection == 'partial_direct':
                self.connect_partial_direct(config)
            else:
                if config.num_hidden > 0:
                    print(
                        "Warning: initial_connection = partial with hidden nodes will not do direct input-output connections;",
                        f"\tif this is desired, set initial_connection = partial_nodirect {config.connection_fraction};",
                        f"\tif not, set initial_connection = partial_direct {config.connection_fraction}",
                        sep='\n', file=sys.stderr)
                self.connect_partial_nodirect(config)

    def configure_crossover(self, genome1, genome2, config):
        """ Configure a new genome by crossover from two parent genomes. """
        if genome1.fitness > genome2.fitness:
            parent1, parent2 = genome1, genome2
        else:
            parent1, parent2 = genome2, genome1

        # Inherit connection genes
        for key, cg1 in parent1.connections.items():
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2)

        # Inherit node genes
        parent1_set = parent1.nodes
        parent2_set = parent2.nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)

    def mutate(self, config):
        """ Mutates this genome. """

        if config.single_structural_mutation:
            div = max(1, (config.node_add_prob + config.node_delete_prob +
                          config.conn_add_prob + config.conn_delete_prob))
            r = random()
            if r < (config.node_add_prob / div):
                self.mutate_add_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob) / div):
                self.mutate_delete_node(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob) / div):
                self.mutate_add_connection(config)
            elif r < ((config.node_add_prob + config.node_delete_prob +
                       config.conn_add_prob + config.conn_delete_prob) / div):
                self.mutate_delete_connection()
        else:
            if random() < config.node_add_prob:
                self.mutate_add_node(config)

            if random() < config.node_delete_prob:
                self.mutate_delete_node(config)

            if random() < config.conn_add_prob:
                self.mutate_add_connection(config)

            if random() < config.conn_delete_prob:
                self.mutate_delete_connection()

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    def mutate_add_node(self, config):
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return

        # Choose a random connection to split
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.  The new node+connections have roughly the same behavior as
        # the original connection (depending on the activation function of the new node).
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        self.add_connection(config, i, new_node_id, 1.0, True)
        self.add_connection(config, new_node_id, o, conn_to_split.weight, True)

    def add_connection(self, config, input_key, output_key, weight, enabled):
        # TODO: Add further validation of this connection addition?
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)
        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        connection.weight = weight
        connection.enabled = enabled
        self.connections[key] = connection

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, the only restriction being that the output
        node cannot be one of the network input pins.
        """
        possible_outputs = list(self.nodes)
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            # TODO: Should this be using mutation to/from rates? Hairy to configure...
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return

        # Don't allow connections between two output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return

        # No need to check for connections between input nodes:
        # they cannot be the output end of a connection (see above).

        # For feed-forward networks, avoid creating cycles.
        if config.feed_forward and creates_cycle(list(self.connections), key):
            return

        cg = self.create_connection(config, in_node, out_node)
        self.connections[cg.key] = cg

    def mutate_delete_node(self, config):
        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in self.nodes if k not in config.output_keys]
        if not available_nodes:
            return -1

        del_key = choice(available_nodes)

        connections_to_delete = set()
        for k, v in self.connections.items():
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        return del_key

    def mutate_delete_connection(self):
        if self.connections:
            key = choice(list(self.connections.keys()))
            del self.connections[key]

    def distance(self, other, config):
        """
        Returns the genetic distance between this genome and the other. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in other.nodes:
                if k2 not in self.nodes:
                    disjoint_nodes += 1

            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance +
                             (config.compatibility_disjoint_coefficient *
                              disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in other.connections:
                if k2 not in self.connections:
                    disjoint_connections += 1

            for k1, c1 in self.connections.items():
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += c1.distance(c2, config)

            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance

    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes), num_enabled_connections

    def __str__(self):
        s = f"Key: {self.key}\nFitness: {self.fitness}\nNodes:"
        for k, ng in self.nodes.items():
            s += f"\n\t{k} {ng!s}"
        s += "\nConnections:"
        connections = list(self.connections.values())
        connections.sort()
        for c in connections:
            s += "\n\t" + str(c)
        return s

    @staticmethod
    def create_node(config, node_id):
        node = config.node_gene_type(node_id)
        node.init_attributes(config)
        return node

    @staticmethod
    def create_connection(config, input_id, output_id):
        connection = config.connection_gene_type((input_id, output_id))
        connection.init_attributes(config)
        return connection

    def connect_fs_neat_nohidden(self, config):
        """
        Randomly connect one input to all output nodes
        (FS-NEAT without connections to hidden, if any).
        Originally connect_fs_neat.
        """
        input_id = choice(config.input_keys)
        for output_id in config.output_keys:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_fs_neat_hidden(self, config):
        """
        Randomly connect one input to all hidden and output nodes
        (FS-NEAT with connections to hidden, if any).
        """
        input_id = choice(config.input_keys)
        others = [i for i in self.nodes if i not in config.input_keys]
        for output_id in others:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def compute_full_connections(self, config, direct):
        """
        Compute connections for a fully-connected feed-forward genome--each
        input connected to all hidden nodes
        (and output nodes if ``direct`` is set or there are no hidden nodes),
        each hidden node connected to all output nodes.
        (Recurrent genomes will also include node self-connections.)
        """
        hidden = [i for i in self.nodes if i not in config.output_keys]
        output = [i for i in self.nodes if i in config.output_keys]
        connections = []
        if hidden:
            for input_id in config.input_keys:
                for h in hidden:
                    connections.append((input_id, h))
            for h in hidden:
                for output_id in output:
                    connections.append((h, output_id))
        if direct or (not hidden):
            for input_id in config.input_keys:
                for output_id in output:
                    connections.append((input_id, output_id))

        # For recurrent genomes, include node self-connections.
        if not config.feed_forward:
            for i in self.nodes:
                connections.append((i, i))

        return connections

    def connect_full_nodirect(self, config):
        """
        Create a fully-connected genome
        (except without direct input-output unless no hidden nodes).
        """
        for input_id, output_id in self.compute_full_connections(config, False):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_full_direct(self, config):
        """ Create a fully-connected genome, including direct input-output connections. """
        for input_id, output_id in self.compute_full_connections(config, True):
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_nodirect(self, config):
        """
        Create a partially-connected genome,
        with (unless no hidden nodes) no direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, False)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def connect_partial_direct(self, config):
        """
        Create a partially-connected genome,
        including (possibly) direct input-output connections.
        """
        assert 0 <= config.connection_fraction <= 1
        all_connections = self.compute_full_connections(config, True)
        shuffle(all_connections)
        num_to_add = int(round(len(all_connections) * config.connection_fraction))
        for input_id, output_id in all_connections[:num_to_add]:
            connection = self.create_connection(config, input_id, output_id)
            self.connections[connection.key] = connection

    def get_pruned_copy(self, genome_config):
        used_node_genes, used_connection_genes = get_pruned_genes(self.nodes, self.connections,
                                                                  genome_config.input_keys, genome_config.output_keys)
        new_genome = DefaultGenome(None)
        new_genome.nodes = used_node_genes
        new_genome.connections = used_connection_genes
        return new_genome


def get_pruned_genes(node_genes, connection_genes, input_keys, output_keys):
    used_nodes = required_for_output(input_keys, output_keys, connection_genes)
    used_pins = used_nodes.union(input_keys)

    # Copy used nodes into a new genome.
    used_node_genes = {}
    for n in used_nodes:
        used_node_genes[n] = copy.deepcopy(node_genes[n])

    # Copy enabled and used connections into the new genome.
    used_connection_genes = {}
    for key, cg in connection_genes.items():
        in_node_id, out_node_id = key
        if cg.enabled and in_node_id in used_pins and out_node_id in used_pins:
            used_connection_genes[key] = copy.deepcopy(cg)

    return used_node_genes, used_connection_genes

# from random import random, choice, gauss
# import numpy as np
# from itertools import count
# from neat.genome import DefaultGenome, DefaultGenomeConfig
# from neat.genes import BaseGene, DefaultNodeGene, DefaultConnectionGene
# # from neat.kan_genes import KANNodeGene, KANConnectionGene, SplineSegmentGene
# from neat.config import ConfigParameter, DefaultClassConfig

# class KANGenomeConfig(DefaultGenomeConfig):
#     """Configuration for KANGenome class."""
    
#     def __init__(self, params):
#         super().__init__(params)
        
#         # Additional KAN-specific parameters
#         self._params += [
#             ConfigParameter('scale_coefficient_range', float, 5.0),
#             ConfigParameter('bias_coefficient_range', float, 5.0),
#             ConfigParameter('spline_coefficient_range', float, 5.0),
#             ConfigParameter('scale_mutation_rate', float, 0.1),
#             ConfigParameter('scale_mutation_power', float, 0.5),
#             ConfigParameter('scale_min_value', float, -5.0),
#             ConfigParameter('scale_max_value', float, 5.0),
#             ConfigParameter('bias_mutation_rate', float, 0.1),
#             ConfigParameter('bias_mutation_power', float, 0.5),
#             ConfigParameter('bias_min_value', float, -5.0),
#             ConfigParameter('bias_max_value', float, 5.0),
#             ConfigParameter('spline_mutation_rate', float, 0.1),
#             ConfigParameter('spline_mutation_power', float, 0.5),
#             ConfigParameter('spline_min_value', float, -5.0),
#             ConfigParameter('spline_max_value', float, 5.0),
#             ConfigParameter('spline_add_prob', float, 0.1),
#             ConfigParameter('spline_delete_prob', float, 0.05),
#             ConfigParameter('initial_spline_segments', int, 3),
#             ConfigParameter('max_spline_segments', int, 10),
#             ConfigParameter('spline_range_min', float, -1.0),
#             ConfigParameter('spline_range_max', float, 1.0),
#         ]

# class KANGenome(DefaultGenome):
#     """Genome class for Kolmogorov-Arnold Networks."""
    
#     @classmethod
#     def parse_config(cls, param_dict):
#         return KANGenomeConfig(param_dict)
    
#     def __init__(self, key):
#         super().__init__(key)
#         self.nodes = {}        # Dictionary of node genes (key: node_id, value: KANNodeGene)
#         self.connections = {}  # Dictionary of connection genes (key: (in_node_id, out_node_id), value: KANConnectionGene)
#         self.spline_segments = {}  # Dictionary mapping connection key to dictionary of spline segments
        
#     def configure_new(self, config):
#         """Configure a new genome based on the given configuration."""
#         super().configure_new(config)
        
#         # Configure initial splines for all connections
#         for conn_key in list(self.connections.keys()):
#             conn = self.connections[conn_key]
#             self.spline_segments[conn_key] = {}
            
#             # Create initial spline segments
#             for i in range(config.genome_config.initial_spline_segments):
#                 grid_pos = config.genome_config.spline_range_min + i * (
#                     (config.genome_config.spline_range_max - config.genome_config.spline_range_min) / 
#                     (config.genome_config.initial_spline_segments - 1)
#                 )
#                 seg_key = (conn_key[0], conn_key[1], grid_pos)
#                 value = gauss(0.0, 0.1)
#                 conn.add_spline_segment(seg_key, grid_pos, value)
                
#     def configure_crossover(self, parent1, parent2, config):
#         """Configure this genome as a crossover of the two parent genomes."""
#         super().configure_crossover(parent1, parent2, config)
        
#         # Handle spline segments
#         for conn_key, conn in self.connections.items():
#             self.spline_segments[conn_key] = {}
            
#             # Check if this connection exists in either parent
#             p1_conn = parent1.connections.get(conn_key)
#             p2_conn = parent2.connections.get(conn_key)
            
#             if p1_conn and p2_conn:  # Connection in both parents
#                 # Inherit spline segments from either parent
#                 for grid_pos in set(list(p1_conn.spline_segments.keys()) + list(p2_conn.spline_segments.keys())):
#                     if grid_pos in p1_conn.spline_segments and grid_pos in p2_conn.spline_segments:
#                         # Get from either parent
#                         seg = choice([p1_conn.spline_segments[grid_pos], p2_conn.spline_segments[grid_pos]])
#                         conn.spline_segments[grid_pos] = SplineSegmentGene(
#                             (conn_key[0], conn_key[1], grid_pos), 
#                             seg.value, 
#                             grid_pos
#                         )
#                     elif grid_pos in p1_conn.spline_segments:
#                         seg = p1_conn.spline_segments[grid_pos]
#                         conn.spline_segments[grid_pos] = SplineSegmentGene(
#                             (conn_key[0], conn_key[1], grid_pos), 
#                             seg.value, 
#                             grid_pos
#                         )
#                     else:
#                         seg = p2_conn.spline_segments[grid_pos]
#                         conn.spline_segments[grid_pos] = SplineSegmentGene(
#                             (conn_key[0], conn_key[1], grid_pos), 
#                             seg.value, 
#                             grid_pos
#                         )
#             elif p1_conn:  # Connection only in parent1
#                 for grid_pos, seg in p1_conn.spline_segments.items():
#                     conn.spline_segments[grid_pos] = SplineSegmentGene(
#                         (conn_key[0], conn_key[1], grid_pos), 
#                         seg.value, 
#                         grid_pos
#                     )
#             elif p2_conn:  # Connection only in parent2
#                 for grid_pos, seg in p2_conn.spline_segments.items():
#                     conn.spline_segments[grid_pos] = SplineSegmentGene(
#                         (conn_key[0], conn_key[1], grid_pos), 
#                         seg.value, 
#                         grid_pos
#                     )
#             else:  # New connection
#                 # Create initial spline segments
#                 for i in range(config.genome_config.initial_spline_segments):
#                     grid_pos = config.genome_config.spline_range_min + i * (
#                         (config.genome_config.spline_range_max - config.genome_config.spline_range_min) / 
#                         (config.genome_config.initial_spline_segments - 1)
#                     )
#                     seg_key = (conn_key[0], conn_key[1], grid_pos)
#                     value = gauss(0.0, 0.1)
#                     conn.add_spline_segment(seg_key, grid_pos, value)
    
#     def mutate(self, config):
#         """Mutate this genome."""
#         super().mutate(config)
        
#         # Additional mutations for KAN-specific elements
#         for conn_key, conn in self.connections.items():
#             # Mutate scale and bias
#             if random() < config.genome_config.scale_mutation_rate:
#                 conn.scale += gauss(0, config.genome_config.scale_mutation_power)
#                 conn.scale = max(min(conn.scale, config.genome_config.scale_max_value), config.genome_config.scale_min_value)
                
#             if random() < config.genome_config.bias_mutation_rate:
#                 conn.bias += gauss(0, config.genome_config.bias_mutation_power)
#                 conn.bias = max(min(conn.bias, config.genome_config.bias_max_value), config.genome_config.bias_min_value)
            
#             # Mutate existing spline segments
#             for grid_pos, segment in list(conn.spline_segments.items()):
#                 if random() < config.genome_config.spline_mutation_rate:
#                     segment.value += gauss(0, config.genome_config.spline_mutation_power)
#                     segment.value = max(min(segment.value, config.genome_config.spline_max_value), 
#                                         config.genome_config.spline_min_value)
            
#             # Add a new spline segment with some probability
#             if random() < config.genome_config.spline_add_prob and len(conn.spline_segments) < config.genome_config.max_spline_segments:
#                 # Find a position between existing grid points
#                 existing_positions = sorted(seg.grid_position for seg in conn.spline_segments.values())
                
#                 if len(existing_positions) <= 1:
#                     # If there are 0 or 1 positions, create a random new one
#                     new_pos = random() * (config.genome_config.spline_range_max - config.genome_config.spline_range_min) + config.genome_config.spline_range_min
#                 else:
#                     # Choose a random gap between existing positions
#                     idx = choice(range(len(existing_positions) - 1))
#                     new_pos = (existing_positions[idx] + existing_positions[idx + 1]) / 2
                
#                 # Add new segment
#                 new_key = (conn_key[0], conn_key[1], new_pos)
#                 value = gauss(0.0, 0.1)
#                 conn.add_spline_segment(new_key, new_pos, value)
            
#             # Delete a spline segment with some probability
#             if random() < config.genome_config.spline_delete_prob and len(conn.spline_segments) > 2:
#                 # Don't delete all segments - keep at least 2
#                 to_delete = choice(list(conn.spline_segments.keys()))
#                 del conn.spline_segments[to_delete]
    
#     def distance(self, other, config):
#         """Return the genetic distance between this genome and the other."""
#         distance = super().distance(other, config)
        
#         # Add distance component for spline segments
#         if not self.connections or not other.connections:
#             return distance
            
#         # Get connections that exist in both genomes
#         common_connections = set(self.connections.keys()) & set(other.connections.keys())
#         if not common_connections:
#             return distance
            
#         spline_distance = 0.0
#         for key in common_connections:
#             conn1 = self.connections[key]
#             conn2 = other.connections[key]
            
#             # Compare scale and bias
#             spline_distance += abs(conn1.scale - conn2.scale) / config.genome_config.scale_coefficient_range
#             spline_distance += abs(conn1.bias - conn2.bias) / config.genome_config.bias_coefficient_range
            
#             # Compare spline segments at matching grid positions
#             common_positions = set(seg.grid_position for seg in conn1.spline_segments.values()) & \
#                                set(seg.grid_position for seg in conn2.spline_segments.values())
                               
#             if common_positions:
#                 seg_dist = 0.0
#                 for pos in common_positions:
#                     seg1 = next(seg for seg in conn1.spline_segments.values() if seg.grid_position == pos)
#                     seg2 = next(seg for seg in conn2.spline_segments.values() if seg.grid_position == pos)
#                     seg_dist += abs(seg1.value - seg2.value) / config.genome_config.spline_coefficient_range
                
#                 spline_distance += seg_dist / len(common_positions)
        
#         # Normalize by number of common connections
#         spline_distance /= len(common_connections)
        
#         # Add this component to total distance
#         distance += spline_distance
        
#         return distance