from __future__ import annotations
import math
from typing import Callable
from multiprocessing.pool import ThreadPool
from __init__ import NeatConfig
from genome import Genome
from connection_history import ConnectionHistory
from species import Species


class Population():
    """
    Population

    Represents a population of genomes in the NEAT algorithm for evolving neural networks.

    Attributes:
    - config (NeatConfig): NEAT configuration settings.
    - genomes (list[Genome]): List of genomes in the population.
    - best_genome (Genome): Best genome in the population.
    - generation (int): Current generation number.
    - best_fitness (float): Fitness of the best genome.
    - average_fitness (float): Average fitness of the population.
    - innovation_history (list[ConnectionHistory]): List of connection history to track innovations.
    - species (list[Species]): List of species within the population.

    Methods:
    - set_best_genome() -> None: Set the best genome globally and for this generation.
    - run(evaluate_genome: Callable[[Genome, int], None], nb_generations: int, callback_generation: Callable[[int], None] = None) -> None:
        Run the training on the population.
    - speciate() -> None: Separate genomes into species based on similarity to leaders of each species in the previous generation.
    - reproduce_species() -> None: Reproduction of the species.
    - sort_species() -> None: Sort genomes within a species and the species by their fitness.
    - kill_stagnant_species() -> None: Kill all species that haven't improved in N generations.
    - kill_bad_species() -> None: Kill species with fitness below a threshold.
    - reset_on_extinction() -> None: Reset the population if all species become extinct due to stagnation.
    - get_average_fitness_sum() -> float: Returns the sum of each species' average fitness.
    - set_average_fitness() -> None: Update the average fitness for all species.
    - update_species() -> None: Update the species for the next generation.
    - clone() -> Population: Return a copy of this population.

    """

    def __init__(self, config: NeatConfig) -> None:
        """
        Initialize a Population instance.

        Args:
        - config (NeatConfig): NEAT configuration settings.

        """
        self.config = config
        self.genomes: list[Genome] = []
        self.best_genome = None
        self.generation = 0
        self.best_fitness = 0
        self.average_fitness = 0
        self.innovation_history: list[ConnectionHistory] = []
        self.species: list[Species] = []

        for i in range(0, config["population_size"]):
            genome = Genome(config)
            genome.mutate(self.innovation_history)
            genome.generate_network()

            if config["initial_connections"] == "full":
                genome.fully_connect(self.innovation_history)
            self.genomes.append(genome)

    def set_best_genome(self) -> None:
        """
        Set the best genome globally and for this generation.

        """
        temp_best = self.species[0].genomes[0]
        if temp_best.fitness >= self.best_fitness:
            self.best_genome = temp_best
            self.best_fitness = temp_best.fitness

    def run(self, evaluate_genome: Callable[[Genome, int], None], nb_generations: int, callback_generation: Callable[[int], None] = None):
        """
        Run the training on the population.

        Args:
        - evaluate_genome (Callable[[Genome, int], None]): Function to evaluate the fitness of a genome.
        - nb_generations (int): Number of generations to run.
        - callback_generation (Callable[[int], None]): Callback function after each generation (default is None).

        """
        for i in range(nb_generations):

            # Calculate the fitness of each genomes
            pool = ThreadPool(processes=len(self.genomes))
            args = [[genome, i] for genome in self.genomes]
            results = pool.starmap_async(evaluate_genome, args)
            results.wait()

            self.speciate()
            self.sort_species()
            self.update_species()
            self.set_best_genome()
            self.set_average_fitness()
            self.kill_stagnant_species()
            self.kill_bad_species()
            self.reproduce_species()
            self.reset_on_extinction()

            if callback_generation:
                callback_generation(i)

            if not self.config["no_fitness_termination"] and self.best_genome.fitness > self.config["fitness_threshold"]:
                break

    def speciate(self) -> None:
        """
        Separate genomes into species based on how similar they are to the leaders of each species in the previous generation.

        """
        # Reset the genomes in each species
        for s in self.species:
            s.genomes = []

        # Group the genomes by species
        for g in self.genomes:
            species_found = False
            for s in self.species:
                if (s.same_species(g, self.config)):
                    s.add_to_species(g)
                    species_found = True
                    break
            if species_found == False:
                new_species = Species(g)
                self.species.append(new_species)

        # Remove the empty species
        for s in self.species:
            if len(s.genomes) == 0:
                for g in s.genomes:
                    self.genomes.remove(g)

    def reproduce_species(self):
        """
        Reproduction of the species.

        """
        average_fitness_sum = self.get_average_fitness_sum()
        population_size = self.config["population_size"]

        children: list[Genome] = []
        for s in self.species:
            children.append(s.champion.clone())
            # get the calculated amount of children from this species
            nb_of_children = 0 if average_fitness_sum == 0 else math.floor(
                (s.average_fitness / average_fitness_sum) * population_size) - 1
            for i in range(nb_of_children):
                children.append(s.give_me_baby(self.innovation_history))

        previous_best = self.genomes[0]
        if len(children) < population_size:
            children.append(previous_best.clone())

        # if not enough babies (due to flooring the number of children to get a whole var) get babies from the best species
        while len(children) < population_size:
            children.append(self.species[0].give_me_baby(
                self.innovation_history))

        self.genomes = children.copy()
        self.generation += 1
        for g in self.genomes:
            g.generate_network()
        self.set_best_genome()

    def sort_species(self) -> None:
        """
        Sorts the genomes within a species and the species by their fitness.

        """
        for s in self.species:
            s.sort_genomes()

        def get_best_fitness(e: Species):
            return e.best_fitness

        # sort the species by the fitness of its best genomes using selection sort
        self.species.sort(key=get_best_fitness, reverse=True)

    def kill_stagnant_species(self) -> None:
        """
        Kills all species which haven't improved in N generations.

        """
        for s in self.species[self.config["species_elitism"]:]:
            if s.stagnation >= self.config["max_stagnation"]:
                for g in s.genomes:
                    self.genomes.remove(g)
                self.species.remove(s)

    def kill_bad_species(self) -> None:
        """
        If a species has a fitness below a threshold, kill it.

        """
        species_average_fitness = self.get_average_fitness_sum() / len(self.species)
        for s in self.species[1:]:
            if s.average_fitness < species_average_fitness * self.config["bad_species_threshold"]:
                for g in s.genomes:
                    self.genomes.remove(g)
                self.species.remove(s)

    def reset_on_extinction(self):
        """
        When all species simultaneously become extinct due to stagnation, a new random population will be created.

        """
        if len(self.species) == 0:
            self.genomes = []
            for i in range(self.config["population_size"]):
                self.genomes.append(Genome(self.config))

    def get_average_fitness_sum(self) -> float:
        """
        Returns the sum of each species' average fitness.

        Returns:
        - float: The sum of average fitness for all species.

        """
        average_sum = 0
        for s in self.species:
            average_sum += s.average_fitness
        return average_sum

    def set_average_fitness(self) -> None:
        """
        Update the average fitness of the population of genomes.

        """
        self.average_fitness = self.get_average_fitness_sum() / len(self.species)

    def update_species(self) -> None:
        """
        Update the species for the next generation.

        """
        for s in self.species:
            s.kill_genomes(self.config)
            s.fitness_sharing()
            s.set_average_fitness()

    def clone(self) -> Population:
        """
        Return a copy of this population.

        Returns:
            Population: A copy of the population
        """
        clone = Population(self.config)
        clone.genomes = []
        clone.species = []

        # Clone the species
        for s in self.species:
            clone.species.append(s.clone())

        # Add the genomes of cloned species to the population
        for s in self.species:
            for g in s.genomes:
                clone.genomes.append(g)

        clone.generation = self.generation
        clone.average_fitness = self.average_fitness
        clone.best_fitness = self.best_fitness
        clone.innovation_history = self.innovation_history

        if self.best_genome:
            clone.best_genome = self.best_genome.clone()

        return clone
