import unittest
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
from visualize import plot_population_history, plot_genome, plot_species
from __init__ import Population, Species, Genome, Node, ConnectionGene, default_config


class TestVisualization(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.test_dir = tempfile.mkdtemp()  # Create a temporary directory for testing

    @classmethod
    def tearDownClass(self):
        # Check if the directory exists before attempting to remove it
        if os.path.exists(self.test_dir):
            # Remove the temporary directory and its contents
            shutil.rmtree(self.test_dir)

    def runTest(self):
        self.setUpClass()
        self.test_plot_population_history()
        self.test_plot_species()
        self.test_plot_genome()
        self.tearDownClass()

    @patch('matplotlib.pyplot')
    def test_plot_population_history(self, mock_pyplot):
        # Set up population history
        population_history = {
            0: Population(default_config),
            1: Population(default_config),
            2: Population(default_config),
        }
        filename = os.path.join(self.test_dir, "test_plot.png")

        # Call the plot_population_history function
        plot_population_history(
            population_history, filename, ylog=True, view=False)

        # Assertions
        self.assertTrue(os.path.exists(filename))

    @patch('graphviz.Digraph')
    def test_plot_species(self, mock_pyplot):
        # Set up population history
        population_history = {
            0: MagicMock(spec=Population, species=[MagicMock(spec=Species, genomes=[MagicMock(spec=Genome)]), MagicMock(spec=Species, genomes=[MagicMock(spec=Genome)])]),
            1: MagicMock(spec=Population, species=[MagicMock(spec=Species, genomes=[MagicMock(spec=Genome), MagicMock(spec=Genome)]), MagicMock(spec=Species, genomes=[MagicMock(spec=Genome)])]),
            2: MagicMock(spec=Population, species=[MagicMock(spec=Species, genomes=[MagicMock(spec=Genome)]), MagicMock(spec=Species, genomes=[MagicMock(spec=Genome)])]),

        }
        filename = os.path.join(self.test_dir, "test_plot.png")

        # Call the plot_species function
        plot_species(population_history, filename)

        # Assertions
        self.assertTrue(os.path.exists(filename))

    @patch('graphviz.Digraph')
    def test_plot_genome(self, mock_digraph):
        # Mock Genome class
        genome_mock = MagicMock(spec=Genome, nodes=[MagicMock(spec=Node, id=0, layer=0)], genes=[
                                MagicMock(spec=ConnectionGene, from_node=MagicMock(spec=Node, id=0), to_node=MagicMock(spec=Node, id=1), weight=0.5, enabled=True)])
        filename = os.path.join(self.test_dir, "test_genome.png")

        # Call the plot_genome function
        plot_genome(genome_mock, filename)

        # Assertions
        self.assertTrue(os.path.exists(
            os.path.join(self.test_dir, filename)))
