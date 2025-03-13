from __future__ import annotations
import math
from random import random
from __init__ import NeatConfig
from genome import Genome
from connection_history import ConnectionHistory


class Species():
    """
    Species

    Represents a species of genomes in the NEAT algorithm for evolving neural networks.

    Attributes:
    - genomes (list[Genome]): List of genomes in the species.
    - champion (Genome): The best genome in the species.
    - best_fitness (float): Best fitness in the species.
    - average_fitness (float): Average fitness of the genomes in the species.
    - stagnation (int): Number of generations the species has gone without an improvement.

    Methods:
    - __init__(self, genome: Genome) -> None: Initialize a Species instance.
    - add_to_specie(self, genome: Genome) -> None: Add a genome to the species.
    - same_species(self, genome: Genome, config: NeatConfig) -> bool: Check if the genome belongs to this species.
    - get_excess_disjoint_genes(self, genome1: Genome, genome2: Genome) -> int: Return the number of excess and disjoint genes between two genomes.
    - average_weight_diff(self, genome1: Genome, genome2: Genome) -> float: Return the average weight difference between matching genes in two genomes.
    - sort_genomes(self) -> None: Sort genomes in the species by fitness.
    - set_average_fitness(self) -> None: Set the average fitness of the species.
    - give_me_baby(self, innovation_history: "list[ConnectionHistory]") -> Genome: Get a baby genome from the species.
    - select_genome(self) -> Genome: Select a genome from the species based on its fitness.
    - kill_genomes(self, config: NeatConfig) -> None: Kill a part of the species based on a survival threshold.
    - fitness_sharing(self) -> None: Apply fitness sharing to protect unique genomes.
    - is_equal(other: Species) -> bool: Compare two species.
    - clone() -> Species: Return a copy of this node. 

    """

    def __init__(self, genome: Genome = None) -> None:
        """
        Initialize a Species instance.

        Args:
        - genome (Genome): The initial genome for the species.

        """
        self.genomes: list[Genome] = []
        self.champion: Genome = None
        self.best_fitness = 0
        self.average_fitness = 0
        self.stagnation = 0  # how many generations the species has gone without an improvement

        if genome is not None:
            self.genomes.append(genome)
            # since it is the only one in the species it is by default the best
            self.best_fitness = genome.fitness
            self.champion = genome

    def add_to_species(self, genome: Genome) -> None:
        """
        Add a genome to the species.

        Args:
        - genome (Genome): The genome to add to the species.

        """
        self.genomes.append(genome)

    def same_species(self, genome: Genome, config: NeatConfig) -> bool:
        """
        Check if the genome belongs to this species.

        Args:
        - genome (Genome): The genome to check.
        - config (NeatConfig): NEAT configuration settings.

        Returns:
        - bool: True if the genome is in this species, False otherwise.

        """
        compatibility_threshold = config["compatibility_threshold"]
        compatibility_disjoint_coefficient = config["compatibility_disjoint_coefficient"]
        compatibility_weight_coefficient = config["compatibility_weight_coefficient"]

        excess_and_disjoint = self.get_excess_disjoint_genes(
            genome, self.champion)
        average_weight_diff = self.average_weight_diff(genome, self.champion)

        large_genome_normalizer = len(genome.genes) - 20
        if large_genome_normalizer < 1:
            large_genome_normalizer = 1

        # compatibility formula
        compatibility = (compatibility_disjoint_coefficient * excess_and_disjoint) / large_genome_normalizer + \
            compatibility_weight_coefficient * average_weight_diff

        return compatibility_threshold > compatibility

    def get_excess_disjoint_genes(self, genome1: Genome, genome2: Genome):
        """
        Return the number of excess and disjoint genes between two genomes.
        i.e., returns the number of genes which don't match.

        Args:
        - genome1 (Genome): The first genome.
        - genome2 (Genome): The second genome.

        Returns:
        - int: The number of excess and disjoint genes.

        """
        matching = 0
        for g1 in genome1.genes:
            for g2 in genome2.genes:
                if g1.innovation_nb == g2.innovation_nb:
                    matching += 1
                    break

        # return the number of excess and disjoint genes
        return len(genome1.genes) + len(genome2.genes) - 2 * matching

    def average_weight_diff(self, genome1: Genome, genome2: Genome):
        """
        Return the average weight difference between matching genes in the input genomes.

        Args:
        - genome1 (Genome): The first genome.
        - genome2 (Genome): The second genome.

        Returns:
        - float: The average weight difference.

        """
        if len(genome1.genes) == 0 or len(genome2.genes) == 0:
            return 0

        matching = 0
        total_diff = 0
        for g1 in genome1.genes:
            for g2 in genome2.genes:
                if g1.innovation_nb == g2.innovation_nb:
                    matching += 1
                    total_diff += abs(g1.weight - g2.weight)
                    break

        if matching == 0:
            # divide by 0 error
            return 100

        return total_diff / matching

    def sort_genomes(self):
        """
        Sort genomes in the species by fitness.

        """
        def get_fitness(g: Genome):
            return g.fitness

        # Sort the genomes by their fitness
        self.genomes.sort(key=get_fitness, reverse=True)

        if self.genomes[0].fitness > self.best_fitness:
            self.stagnation = 0
            self.best_fitness = self.genomes[0].fitness
            self.champion = self.genomes[0].clone()
        else:
            self.stagnation += 1

    def set_average_fitness(self):
        """
        Set the average fitness of the species.

        """
        sum = 0
        for g in self.genomes:
            sum += g.fitness
        self.average_fitness = sum / len(self.genomes)

    def give_me_baby(self, innovation_history: "list[ConnectionHistory]"):
        """
        Get a baby genome from the genomes in this species.

        Args:
        - innovation_history ("list[ConnectionHistory]"): List of connection history to track innovations.

        Returns:
        - Genome: The baby genome.

        """
        baby: Genome = None
        if random() < 0.25:
            baby = self.select_genome().clone()
        else:
            # 75% of the time do crossover
            parent1: Genome = self.select_genome()
            parent2: Genome = self.select_genome()

            # the crossover function expects the highest fitness parent to be the object and the lowest as the argument
            if parent1.fitness < parent2.fitness:
                baby = parent2.crossover(parent1)
            else:
                baby = parent1.crossover(parent2)

        baby.mutate(innovation_history)  # mutate that baby genome
        return baby

    def select_genome(self) -> Genome:
        """
        Select a genome from the species based on its fitness.

        Returns:
        - Genome: The selected genome.

        """
        fitness_sum = 0
        for i in range(0, len(self.genomes)):
            fitness_sum += self.genomes[i].fitness

        running_sum = 0
        for i in range(0, len(self.genomes)):
            running_sum += self.genomes[i].fitness
            if running_sum > random() * fitness_sum:
                return self.genomes[i]

        return self.genomes[0]

    def kill_genomes(self, config: NeatConfig):
        """
        Kill a part of the species.

        Args:
        - config (NeatConfig): NEAT configuration settings.

        """
        survivals_nb = math.floor(
            len(self.genomes) * config["survival_threshold"])
        if survivals_nb > config["min_species_size"]:
            self.genomes = self.genomes[0:survivals_nb]
        else:
            self.genomes = self.genomes[0:config["min_species_size"]]

    def fitness_sharing(self):
        """
        Apply fitness sharing to protect unique genomes.

        """
        for g in self.genomes:
            g.fitness /= len(self.genomes)

    def is_equal(self, other: Species):
        """
        Compare two species.

        Args:
            other (Species): The other species to compare with it

        Returns:
            bool: True if the species are equals, otherwise false.
        """

        def get_genome_id(genome: Genome):
            return genome.id

        self.genomes.sort(key=get_genome_id)
        other.genomes.sort(key=get_genome_id)

        # Compare the genomes
        for i in range(len(self.genomes)):
            if self.genomes[i].is_equal(other.genomes[i]):
                return False

        return True

    def clone(self):
        """
        Return a copy of this species.

        Returns:
        - Node: A copy of this species.

        """
        clone = Species()
        clone.champion = self.champion.clone()
        clone.average_fitness = self.average_fitness
        clone.best_fitness = self.best_fitness
        clone.stagnation = self.stagnation
        for g in self.genomes:
            clone.add_to_species(g)
        return clone
