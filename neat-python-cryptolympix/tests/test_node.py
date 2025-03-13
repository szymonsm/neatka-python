import unittest
from __init__ import default_config, Node, ConnectionGene


class TestNode(unittest.TestCase):
    def setUp(self):
        self.config = default_config
        self.node = Node(1, "relu", 1)

    def runTest(self):
        self.setUp()
        self.test_initialization()
        self.setUp()
        self.test_activation()
        self.setUp()
        self.test_mutate()
        self.setUp()
        self.test_is_connected_to()
        self.setUp()
        self.test_clone()

    def test_initialization(self):
        self.assertEqual(self.node.id, 1)
        self.assertEqual(self.node.input_sum, 0)
        self.assertEqual(self.node.output_value, 0)
        self.assertEqual(self.node.output_connections, [])
        self.assertEqual(self.node.layer, 1)
        self.assertEqual(self.node.activation_function, "relu")

    def test_activation(self):
        # Test activate method when layer is not 0
        self.node.input_sum = 0.5
        self.node.activate()
        self.node.propagate_output()
        # Output value should be updated
        self.assertNotEqual(self.node.output_value, 0)

    def test_mutate(self):
        config = default_config
        # Test mutation of a bias node
        config["bias_mutate_rate"] = 1.0
        config["bias_replace_rate"] = 0.0
        self.node.output_value = 0.0
        self.node.mutate(config, is_bias_node=True)
        self.assertNotEqual(self.node.output_value, 0.0)

        # Test mutation of a bias node
        config["bias_mutate_rate"] = 0.0
        config["bias_replace_rate"] = 1.0
        self.node.output_value = 0.0
        self.node.mutate(config, is_bias_node=True)
        self.assertNotEqual(self.node.output_value, 0.0)

        # Test activation mutation
        config["activation_mutate_rate"] = 1.0
        self.node.mutate(config)
        self.assertNotEqual(self.node.activation_function, "relu")

    def test_is_connected_to(self):
        node1 = Node(2, "step", 2)
        node2 = Node(3, "sigmoid", 3)
        self.node.output_connections = [
            ConnectionGene(self.node, node1, 1.0, 1, True),
            ConnectionGene(self.node, node2, 1.0, 1, True)
        ]
        # Validate if the node is connected to the provided node
        self.assertTrue(self.node.is_connected_to(node1))
        self.assertTrue(self.node.is_connected_to(node2))

        node3 = Node(4, "tanh", 1)
        # Can't be connected if is on the same layer
        self.assertFalse(self.node.is_connected_to(node3))

        node4 = Node(4, "relu", 0)
        node4.output_connections = [
            ConnectionGene(node4, self.node, 1.0, 1, True),
        ]
        # Connected because of the connection gene
        self.assertTrue(self.node.is_connected_to(node4))

        node5 = Node(5, "softmax", 0)
        node5.output_connections = []
        # Can't be connected if is on a previous layer and no connection gene
        self.assertFalse(self.node.is_connected_to(node5))

    def test_clone(self):
        cloned_node = self.node.clone()
        # Validate that the cloned node is a separate instance
        self.assertTrue(cloned_node.is_equal(self.node))
