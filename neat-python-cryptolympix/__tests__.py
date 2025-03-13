import unittest
from tests.test_activation_functions import TestActivationFunctions
from tests.test_connection_gene import TestConnectionGene
from tests.test_genome import TestGenome
from tests.test_connection_history import TestConnectionHistory
from tests.test_population import TestPopulation
from tests.test_node import TestNode
from tests.test_species import TestSpecies
from tests.test_visualize import TestVisualization


def test_suite():
    suite = unittest.TestSuite()
    suite.addTests([
        TestActivationFunctions(),
        TestConnectionGene(),
        TestGenome(),
        TestConnectionHistory(),
        TestPopulation(),
        TestNode(),
        TestSpecies(),
        TestVisualization()
    ])
    return suite

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(test_suite())

