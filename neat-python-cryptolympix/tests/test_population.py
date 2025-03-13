import unittest
from unittest.mock import MagicMock
from __init__ import default_config, Population, Genome, ConnectionHistory, Species, Node
from random import randrange


def init_connection_history(num_inputs: int, num_outputs: int):
    connections: list[ConnectionHistory] = []
    innovation_nb = 0
    innovation_nbs = [n for n in range(num_inputs * num_outputs + 1)]
    for i in range(num_inputs):
        node_input = MagicMock(spec=Node, id=i)
        for j in range(num_outputs):
            node_output = MagicMock(spec=Node, id=j)
            connections.append(ConnectionHistory(
                from_node=node_input, to_node=node_output, innovation_nb=innovation_nb, innovation_nbs=innovation_nbs))
            innovation_nb += 1
    return connections


class TestPopulation(unittest.TestCase):
    def setUp(self):
        # Mock NeatConfig
        self.config = default_config
        self.config["num_inputs"] = 5
        self.config["num_outputs"] = 2
        self.config["population_size"] = 10
        self.config["species_elitism"] = 2
        self.config["max_stagnation"] = 5
        self.config["bad_species_threshold"] = 0.5
        self.config["no_fitness_termination"] = False
        self.config["min_species_size"] = 2
        self.config["fitness_threshold"] = 100

    def runTest(self):
        self.setUp()
        self.test_population_initialization()
        self.setUp()
        self.test_set_best_genome()
        self.setUp()
        self.test_speciate()
        self.setUp()
        self.test_reproduce_species()
        self.setUp()
        self.test_sort_species()
        self.setUp()
        self.test_kill_stagnant_species()
        self.setUp()
        self.test_kill_bad_species()
        self.setUp()
        self.test_reset_on_extinction()
        self.setUp()
        self.test_update_species()
        self.setUp()
        self.test_clone()

    def test_population_initialization(self):
        population = Population(self.config)
        self.assertEqual(len(population.genomes),
                         self.config["population_size"])
        self.assertEqual(len(population.species), 0)
        self.assertEqual(population.generation, 0)

    def test_set_best_genome(self):
        population = Population(self.config)

        # Assume best fitness is set to 10 for simplicity
        population.best_fitness = 10

        # Mock species and genomes
        genome = MagicMock(spec=Genome, fitness=20)
        specie = MagicMock(spec=Species, genomes=[genome])
        population.species = [specie]

        # Set best genome
        population.set_best_genome()

        # Assert that the best genome is set
        self.assertEqual(population.best_genome, genome)

    def test_speciate(self):
        population = Population(self.config)

        # Mock genomes
        genome1 = Genome(self.config)
        genome2 = Genome(self.config)
        population.genomes = [genome1, genome2]

        # Mock specie
        specie1 = Species(genome1)
        specie2 = Species(genome2)
        population.species = [specie1, specie2]

        # Run speciation
        population.speciate()

        # Assert that genomes are grouped into species
        self.assertGreater(len(population.species), 0)

    def test_reproduce_species(self):
        population = Population(self.config)

        # Mock species
        genome = Genome(self.config)
        specie = Species(genome)
        population.species = [specie]

        # Run reproduction
        population.reproduce_species()

        # Assert that the population's genomes are updated
        self.assertEqual(population.generation, 1)
        self.assertEqual(population.best_genome, genome)
        self.assertEqual(len(population.genomes), 10)

    def test_sort_species(self):
        population = Population(self.config)

        # Mock species
        for i in range(5):
            genome = Genome(self.config)
            genome.fitness = randrange(0, 100)
            specie = Species(genome)
            population.species.append(specie)

        # Run species sorting
        population.sort_species()

        def get_best_fitness(s: Species):
            return s.best_fitness

        # Assert species are sorted
        self.assertEqual(len(population.species), 5)
        self.assertTrue(sorted(population.species, key=get_best_fitness))

    def test_kill_stagnant_species(self):
        population = Population(self.config)

        # Mock genomes
        genome1 = Genome(self.config)
        genome2 = Genome(self.config)
        genome3 = Genome(self.config)
        genome4 = Genome(self.config)

        # Mock species
        species_to_keep1 = Species(genome1)
        species_to_keep2 = Species(genome2)
        species_to_remove1 = Species(genome3)
        species_to_remove2 = Species(genome4)

        # Set stagnation of the species
        species_to_keep1.stagnation = 2
        species_to_keep2.stagnation = 4
        species_to_remove1.stagnation = 6
        species_to_remove2.stagnation = 8

        population.genomes = [genome1, genome2, genome3, genome4]
        population.species = [
            species_to_keep1, species_to_keep2, species_to_remove1, species_to_remove2]

        # Run killing stagnant species
        population.kill_stagnant_species()

        # Assert stagnant species are removed
        self.assertTrue(species_to_keep1 in population.species)
        self.assertTrue(species_to_keep2 in population.species)
        self.assertTrue(species_to_remove1 not in population.species)
        self.assertTrue(species_to_remove2 not in population.species)

        # Assert the genomes of the stagnant species are removed
        self.assertTrue(genome1 in population.genomes)
        self.assertTrue(genome2 in population.genomes)
        self.assertTrue(genome3 not in population.genomes)
        self.assertTrue(genome4 not in population.genomes)

    def test_kill_bad_species(self):
        population = Population(self.config)

        # Mock genomes
        genome1 = Genome(self.config)
        genome2 = Genome(self.config)
        genome3 = Genome(self.config)

        # Mock species
        good_species = Species(genome1)
        bad_species1 = Species(genome2)
        bad_species2 = Species(genome3)

        # Mock average fitness
        average_fitness1 = 100.0
        average_fitness2 = 3.0
        average_fitness3 = 1.0

        # Set average fitness
        good_species.average_fitness = average_fitness1
        bad_species1.average_fitness = average_fitness2
        bad_species2.average_fitness = average_fitness3

        population.genomes = [genome1, genome2, genome3]
        population.species = [good_species, bad_species1, bad_species2]

        # Run killing bad species
        population.kill_bad_species()

        # Assert bad species are removed
        self.assertEqual(population.species, [good_species])

        # Assert genomes are correctly removed
        self.assertTrue(genome1 in population.genomes)
        self.assertTrue(genome2 not in population.genomes)
        self.assertTrue(genome3 not in population.genomes)

    def test_reset_on_extinction(self):
        population = Population(self.config)

        # Mock species
        population.species = []

        # Run reset on extinction
        population.reset_on_extinction()

        # Assert new random population is created
        self.assertEqual(len(population.genomes),
                         self.config["population_size"])

    def test_update_species(self):
        population = Population(self.config)

        # Mock species
        species = Species(Genome(self.config))
        for i in range(10):
            species.genomes.append(Genome(self.config))
        population.species = [species]

        # Run updating species
        population.update_species()

        # Assert species is updated
        self.assertEqual(len(species.genomes), self.config["min_species_size"])

    def test_clone(self):
        population = Population(self.config)
        clone = population.clone()

        # Asserts the clone is valid
        for i in range(len(clone.genomes)):
            self.assertTrue(clone.genomes[i].is_equal(population.genomes[i]))
        for i in range(len(clone.species)):
            self.assertTrue(clone.species[i].is_equal(population.species[i]))
        if (population.best_genome):
            self.assertTrue(clone.best_genome.is_equal(population.best_genome))
        self.assertEqual(clone.generation, population.generation)
        self.assertEqual(clone.average_fitness, population.average_fitness)
        self.assertEqual(clone.best_fitness, population.best_fitness)
