import unittest
import os
import tempfile
from unittest.mock import MagicMock
from __init__ import default_config, Genome, Node, ConnectionGene, ConnectionHistory, NeatConfig


def assert_instance_properties_equal(test_case: unittest.TestCase, instance1, instance2, properties):
    for prop in properties:
        with test_case.subTest(property=prop):
            value1 = getattr(instance1, prop)
            value2 = getattr(instance2, prop)
            test_case.assertEqual(value1, value2)


class TestGenome(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        # Mock NeatConfig
        self.config = default_config
        self.config["num_inputs"] = 2
        self.config["num_outputs"] = 2

        # Mock ConnectionHistory
        self.connection_history = [
            MagicMock(spec=ConnectionHistory, innovation_nb=1),
            MagicMock(spec=ConnectionHistory, innovation_nb=2),
            MagicMock(spec=ConnectionHistory, innovation_nb=3),
            MagicMock(spec=ConnectionHistory, innovation_nb=4)
        ]

    def runTest(self):
        self.setUpClass()
        self.test_init()
        self.test_fully_connect()
        self.test_get_node()
        self.test_connect_nodes()
        self.test_feed_forward()
        self.test_generate_network()
        self.test_add_node()
        self.test_remove_node()
        self.test_add_connection()
        self.setUp()
        self.test_remove_connection()
        self.test_new_connection_weight()
        self.test_get_innovation_number()
        self.test_fully_connected()
        self.test_mutate()
        self.test_crossover()
        self.test_matching_gene()
        self.test_clone()
        self.test_save_load()

    def test_init(self):
        genome = Genome(self.config)

        self.assertIsNotNone(genome.id)
        self.assertEqual(genome.config, self.config)
        self.assertEqual(genome.genes, [])
        self.assertEqual(len(genome.nodes), 5)
        self.assertEqual(genome.inputs, 2)
        self.assertEqual(genome.outputs, 2)
        self.assertEqual(genome.layers, 2)
        self.assertEqual(genome.next_node, 5)
        self.assertEqual(genome.network, [])
        self.assertEqual(genome.fitness, 0)

    def test_fully_connect(self):
        genome = Genome(self.config)
        genome.fully_connect(self.connection_history)

        # Check if the genes are generated properly
        self.assertEqual(len(genome.genes), 6)
        # First input node
        self.assertEqual(genome.genes[0].from_node.id, 0)
        self.assertEqual(genome.genes[0].to_node.id, 2)
        self.assertEqual(genome.genes[1].from_node.id, 0)
        self.assertEqual(genome.genes[1].to_node.id, 3)
        # Second input node
        self.assertEqual(genome.genes[2].from_node.id, 1)
        self.assertEqual(genome.genes[2].to_node.id, 2)
        self.assertEqual(genome.genes[3].from_node.id, 1)
        self.assertEqual(genome.genes[3].to_node.id, 3)
        # Bias node
        self.assertEqual(genome.genes[4].from_node.id, 4)
        self.assertEqual(genome.genes[4].to_node.id, 2)
        self.assertEqual(genome.genes[5].from_node.id, 4)
        self.assertEqual(genome.genes[5].to_node.id, 3)

    def test_get_node(self):
        genome = Genome(self.config)
        node = genome.get_node(0)

        self.assertIsNotNone(node)
        self.assertEqual(node.id, 0)

    def test_connect_nodes(self):
        genome = Genome(self.config)
        genome.fully_connect(self.connection_history)
        genome.connect_nodes()

        # Check if the connections are set properly
        self.assertEqual(len(genome.nodes[0].output_connections), 2)
        self.assertEqual(len(genome.nodes[1].output_connections), 2)
        self.assertEqual(len(genome.nodes[2].output_connections), 0)

    def test_feed_forward(self):
        genome = Genome(self.config)
        genome.fully_connect(self.connection_history)

        input_values = [1, 0]
        output_values = genome.feed_forward(input_values)

        # Check if the output values are calculated properly
        self.assertEqual(len(output_values), 2)
        self.assertEqual(output_values[0], genome.nodes[2].output_value)
        self.assertEqual(output_values[1], genome.nodes[3].output_value)

    def test_generate_network(self):
        genome = Genome(self.config)
        genome.fully_connect(self.connection_history)
        genome.generate_network()

        # Check if the network is generated properly
        # 2 input nodes + 2 output nodes
        self.assertEqual(len(genome.network), 5)
        self.assertEqual(genome.network[0].id, 0)   # First input node
        self.assertEqual(genome.network[1].id, 1)   # Second input node
        self.assertEqual(genome.network[2].id, 4)   # Bias node
        self.assertEqual(genome.network[3].id, 2)   # First output node
        self.assertEqual(genome.network[4].id, 3)   # Second output node

    def test_add_node(self):
        genome = Genome(self.config)
        genome.fully_connect(self.connection_history)
        initial_num_genes = len(genome.genes)
        initial_num_nodes = len(genome.nodes)

        genome.add_node(self.connection_history)

        # Check if a new node and connections are added properly
        # 1 new connection + 2 existing connections
        self.assertEqual(len(genome.genes), initial_num_genes + 3)
        self.assertEqual(len(genome.nodes),
                         initial_num_nodes + 1)  # 1 new node

    def test_remove_node(self):
        genome = Genome(self.config)
        genome.fully_connect(self.connection_history)
        initial_num_genes = len(genome.genes)
        initial_num_nodes = len(genome.nodes)

        genome.remove_node()

        # Check if a node and its connections are removed properly
        # At least one connection removed
        self.assertLess(len(genome.genes), initial_num_genes)
        # 1 node removed
        self.assertEqual(len(genome.nodes), initial_num_nodes - 1)

    def test_add_connection(self):
        genome = Genome(self.config)
        initial_num_genes = len(genome.genes)

        genome.add_connection(self.connection_history)

        # Check if a new connection is added properly
        self.assertEqual(len(genome.genes), initial_num_genes + 1)

    def test_remove_connection(self):
        genome = Genome(self.config)
        genome.fully_connect(self.connection_history)
        initial_num_genes = len(genome.genes)

        genome.remove_connection()

        # Check if a connection is removed properly
        self.assertEqual(len(genome.genes), initial_num_genes - 1)

    def test_new_connection_weight(self):
        genome = Genome(self.config)
        weight = genome.new_connection_weight()

        # Check if the new connection weight is within the specified range
        self.assertTrue(self.config["weight_min_value"]
                        <= weight <= self.config["weight_max_value"])

    def test_get_innovation_number(self):
        genome = Genome(self.config)
        from_node = Node(0, "sigmoid", 0)
        to_node = Node(2, "sigmoid", 1)
        innovation_number = genome.get_innovation_number(
            self.connection_history, from_node, to_node)

        # Check if the innovation number is obtained properly
        self.assertEqual(innovation_number, 1)

    def test_fully_connected(self):
        genome = Genome(self.config)
        genome.fully_connect(self.connection_history)

        # Check if the genome is considered fully connected
        self.assertTrue(genome.fully_connected())

    def test_mutate(self):
        genome = Genome(self.config)
        initial_num_genes = len(genome.genes)
        initial_num_nodes = len(genome.nodes)

        genome.mutate(self.connection_history)

        # Check if mutations are applied properly
        # Mutations may add new genes
        self.assertGreaterEqual(len(genome.genes), initial_num_genes)
        # Mutations may add new nodes
        self.assertGreaterEqual(len(genome.nodes), initial_num_nodes)

    def test_crossover(self):
        parent1 = Genome(self.config)
        parent2 = Genome(self.config)
        parent1.fully_connect(self.connection_history)
        parent2.fully_connect(self.connection_history)
        parent1.fitness = 1
        parent2.fitness = 0.5

        child = parent1.crossover(parent2)

        # Check if crossover produces a child genome
        self.assertIsInstance(child, Genome)
        self.assertIsNot(child, parent1)
        self.assertIsNot(child, parent2)

    def test_matching_gene(self):
        genome = Genome(self.config)
        genome.fully_connect(self.connection_history)
        matching_gene_index = genome.matching_gene(genome, 1)

        # Check if matching gene is found properly
        self.assertEqual(matching_gene_index, 0)

    def test_clone(self):
        genome = Genome(self.config)
        genome.fully_connect(self.connection_history)
        clone = genome.clone()

        # Check if cloning the genome produces a valid clone
        self.assertTrue(clone.is_equal(genome))

    def test_save_load(self):
        genome = Genome(self.config)
        genome.fully_connect(self.connection_history)

        # Create a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "test_genome")

            # Save the genome
            genome.save(file_path)

            # Load the saved genome
            loaded_genome = Genome.load(file_path + ".pkl")

            # Check if the loaded genome is valid
            self.assertIsInstance(loaded_genome, Genome)
            self.assertIsNot(loaded_genome, genome)

            # Check if the connection genes are valid
            for i in range(len(genome.genes)):
                assert_instance_properties_equal(self, loaded_genome.genes[i], genome.genes[i], [
                                                 "weight", "enabled", "innovation_nb"])
                assert_instance_properties_equal(
                    self, loaded_genome.genes[i].from_node, genome.genes[i].from_node, ["id", "activation_function", "layer"])
                assert_instance_properties_equal(
                    self, loaded_genome.genes[i].to_node, genome.genes[i].to_node, ["id", "activation_function", "layer"])

            # Check if the nodes are valid
            for i in range(len(genome.nodes)):
                assert_instance_properties_equal(self, loaded_genome.nodes[i], genome.nodes[i], [
                                                 "id", "activation_function", "layer"])
