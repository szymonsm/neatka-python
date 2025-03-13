import unittest
from __init__ import default_config, Node, ConnectionGene, activation_functions


class TestConnectionGene(unittest.TestCase):
    def setUp(self):
        self.config = default_config
        self.from_node = Node(1, activation_functions.sigmoid, 1)
        self.to_node = Node(2, activation_functions.sigmoid, 2)
        self.gene = ConnectionGene(self.from_node, self.to_node, 0.5, 1, True)

    def runTest(self):
        self.setUp()
        self.test_init()
        self.setUp()
        self.test_mutate_weight_replace()
        self.setUp()
        self.test_mutate_weight_mutate()
        self.setUp()
        self.test_clone()

    def test_init(self):
        self.assertEqual(self.gene.from_node, self.from_node)
        self.assertEqual(self.gene.to_node, self.to_node)
        self.assertEqual(self.gene.weight, 0.5)
        self.assertEqual(self.gene.innovation_nb, 1)
        self.assertTrue(self.gene.enabled)

    def test_mutate_weight_replace(self):
        self.config["weight_replace_rate"] = 1.0
        self.gene.mutate(self.config)
        self.assertTrue(-1 <= self.gene.weight <= 1)

    def test_mutate_weight_mutate(self):
        self.config["weight_replace_rate"] = 0.0
        self.config["weight_mutate_rate"] = 1.0
        self.gene.mutate(self.config)
        self.assertNotEqual(self.gene.weight, 0.5)

    def test_mutate_enabled_mutate(self):
        self.config["enabled_mutate_rate"] = 1.0
        self.gene.mutate(self.config)
        self.assertFalse(self.gene.enabled)

    def test_clone(self):
        cloned_gene = self.gene.clone(self.from_node, self.to_node)
        self.assertTrue(cloned_gene.is_equal(self.gene))
