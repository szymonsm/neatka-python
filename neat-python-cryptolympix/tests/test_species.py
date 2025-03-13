import unittest
from unittest.mock import MagicMock
from __init__ import default_config, Genome, Species, ConnectionHistory, Node


def init_connection_history(num_inputs: int, num_outputs: int):
    connections: list[ConnectionHistory] = []
    innovation_nb = 0
    for i in range(num_inputs):
        node_input = MagicMock(spec=Node, id=i)
        for j in range(num_outputs):
            node_output = MagicMock(spec=Node, id=j)
            connections.append(ConnectionHistory(
                from_node=node_input, to_node=node_output, innovation_nb=innovation_nb))
            innovation_nb += 1
    return connections


class TestSpecies(unittest.TestCase):
    def setUp(self):
        self.config = {**default_config, **
                       {"num_inputs": 10, "num_outputs": 2}}
        self.genome = Genome(self.config)
        self.connection_history = init_connection_history(10, 2)
        self.genome.fully_connect(self.connection_history)
        self.species = Species(self.genome)

    def runTest(self):
        self.setUp()
        self.test_init()
        self.setUp()
        self.test_same_specie_true()
        self.setUp()
        self.test_same_specie_false()
        self.setUp()
        self.test_add_to_specie()
        self.setUp()
        self.test_excess_disjoint_genes()
        self.setUp()
        self.test_average_weight_diff()
        self.setUp()
        self.test_sort_genomes_stagnation_increment()
        self.setUp()
        self.test_sort_genomes_with_new_best_genome()
        self.setUp()
        self.test_set_average_fitness()
        self.setUp()
        self.test_give_me_baby()
        self.setUp()
        self.test_select_genome()
        self.setUp()
        self.test_kill_genomes()
        self.setUp()
        self.test_fitness_sharing()

    def test_init(self):
        self.assertEqual(self.species.genomes, [self.genome])
        self.assertEqual(self.species.champion, self.genome)
        self.assertEqual(self.species.best_fitness, 0)
        self.assertEqual(self.species.average_fitness, 0)
        self.assertEqual(self.species.stagnation, 0)

    def test_same_specie_true(self):
        other_genome = self.genome.clone()
        self.assertTrue(self.species.same_species(other_genome, self.config))

    def test_same_specie_false(self):
        other_config = {**default_config, **
                        {"num_inputs": 5, "num_outputs": 1}}
        other_genome = Genome(other_config)
        connection_history = init_connection_history(5, 1)
        other_genome.fully_connect(connection_history)
        self.assertFalse(self.species.same_species(other_genome, self.config))

    def test_add_to_specie(self):
        new_genome = Genome(self.config)
        self.species.add_to_species(new_genome)
        self.assertEqual(self.species.genomes, [self.genome, new_genome])

    def test_excess_disjoint_genes(self):
        # Test with two same genomes
        other_genome = self.genome.clone()
        result = self.species.get_excess_disjoint_genes(
            self.genome, other_genome)
        self.assertEqual(result, 0.0)

        # Test with two totally different genomes
        other_genome = Genome(self.config)
        result = self.species.get_excess_disjoint_genes(
            self.genome, other_genome)
        self.assertGreater(result, 0.0)

        # Test with two genomes a little different
        other_genome = self.genome.clone()
        other_genome.remove_connection()
        result = self.species.get_excess_disjoint_genes(
            self.genome, other_genome)
        self.assertEqual(result, 1.0)

    def test_average_weight_diff(self):
        # Test average weight diff of two same genomes
        other_genome = self.genome.clone()
        result = self.species.average_weight_diff(self.genome, other_genome)
        self.assertEqual(result, 0.0)

        # Test average weight diff of two totally different genomes
        other_genome = Genome(self.config)
        connection_history = init_connection_history(10, 2)
        other_genome.fully_connect(connection_history)
        result = self.species.average_weight_diff(self.genome, other_genome)
        self.assertEqual(result, 100.0)

        # Test average weight diff without connection genes
        other_genome = Genome(self.config)
        result = self.species.average_weight_diff(self.genome, other_genome)
        self.assertEqual(result, 0.0)

        # Test average weight diff with two genomes a little bit different
        other_genome = self.genome.clone()
        other_genome.mutate([MagicMock(innovation_nb=10)])
        result = self.species.average_weight_diff(self.genome, other_genome)
        self.assertGreater(result, 0.0)

    def test_sort_genomes_stagnation_increment(self):
        self.species.stagnation = 0
        self.species.sort_genomes()
        self.assertEqual(self.species.stagnation, 1)

    def test_sort_genomes_with_new_best_genome(self):
        self.species.stagnation = 100
        new_best = Genome(self.config)
        new_best.fitness = 10
        self.species.add_to_species(new_best)
        self.species.sort_genomes()
        self.assertEqual(self.species.stagnation, 0)

    def test_set_average_fitness(self):
        self.species.genomes = [MagicMock(fitness=5), MagicMock(
            fitness=10), MagicMock(fitness=15)]
        self.species.set_average_fitness()
        self.assertEqual(self.species.average_fitness, (5 + 10 + 15) / 3)

    def test_give_me_baby(self):
        baby = self.species.give_me_baby([MagicMock(innovation_nb=10)])
        self.assertIsInstance(baby, Genome)

    def test_select_genome(self):
        selected_genome = self.species.select_genome()
        self.assertIsInstance(selected_genome, Genome)

    def test_kill_genomes(self):
        self.species.genomes = [MagicMock()] * 10
        self.species.kill_genomes(self.config)
        self.assertEqual(len(self.species.genomes),
                         self.config["min_species_size"])

    def test_fitness_sharing(self):
        fitness = [5, 10, 15]
        self.species.genomes = [MagicMock(fitness=f) for f in fitness]
        self.species.fitness_sharing()
        for i in range(len(self.species.genomes)):
            self.assertEqual(
                self.species.genomes[i].fitness, fitness[i] / len(fitness))
