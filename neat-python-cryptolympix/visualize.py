import warnings
import numpy
from graphviz import Digraph
from statistics import stdev
import matplotlib.pyplot as plt
from population import Population
from genome import Genome


def plot_population_history(population_history: "dict[int, Population]", filename: str, ylog=False, view=False):
    """
    Plots the average and best fitness of a population across generations.

    Parameters:
    - population_history (dict[int, Population]): A dictionary containing the historical records of Population for each generation.
    - filename (str): The filename (including path and extension) to save the plot.
    - ylog (bool): Flag to indicate whether to use a logarithmic scale on the y-axis. Default is False.
    - view (bool): Flag to indicate whether to display the plot interactively. Default is False.

    Returns:
    - None
    """

    if plt is None:  # pragma: no cover
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)")
        return

    nb_generations = len(population_history.keys())
    generations = range(nb_generations)
    best_fitness = numpy.array(
        [population_history[g].best_fitness for g in range(nb_generations)])
    avg_fitness = numpy.array(
        [population_history[g].average_fitness for g in range(nb_generations)])
    stdev_fitness = numpy.array([stdev(
        [genome.fitness for genome in population_history[g].genomes]) for g in range(nb_generations)])

    plt.plot(generations, avg_fitness, 'b-', label="average")
    plt.plot(generations, avg_fitness - stdev_fitness, 'g-.', label="-1 std")
    plt.plot(generations, avg_fitness + stdev_fitness, 'g-.', label="+1 std")
    plt.plot(generations, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:  # pragma: no cover
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:  # pragma: no cover
        plt.show()

    plt.close()


def plot_species(population_history: "dict[int, Population]", filename: str, ylog=False, view=False):
    """ 
    Visualizes speciation throughout evolution. 

    Parameters:
    - population_history (dict[int, Population]): A dictionary containing the historical records of Population for each generation.
    - filename (str): The filename (including path and extension) to save the plot.
    - ylog (bool): Flag to indicate whether to use a logarithmic scale on the y-axis. Default is False.
    - view (bool): Flag to indicate whether to display the plot interactively. Default is False.

    Returns:
    - None
    """
    if plt is None:  # pragma: no cover
        warnings.warn(
            "This display is not available due to a missing optional dependency (matplotlib)")
        return

    nb_generations = len(population_history.keys())
    num_species_per_generation = []

    for i in range(nb_generations):
        num_species_per_generation.append(len(population_history[i].species))

    plt.plot(range(1, nb_generations + 1), [[n]
             for n in num_species_per_generation])

    # Set integer ticks on the x and y axes
    plt.yticks(range(0, max(num_species_per_generation) + 1))
    plt.xticks(range(1, nb_generations + 1))

    plt.xlabel("Generation")
    plt.ylabel("Number of species")
    plt.title('Evolution of species')
    if ylog:  # pragma: no cover
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:  # pragma: no cover
        plt.show()

    plt.close()


def plot_genome(genome: Genome, filename: str):
    """
    Visualizes the neural network structure of a given genome and saves it as an image file.

    Parameters:
    - genome (Genome): The genome object representing the neural network.
    - filename (str): The filename (including path and extension) to save the visualization.

    Returns:
    - None
    """

    dot = Digraph(format='png')
    dot.attr(dpi='300', concentrate='true', rankdir='LR')

    # Add nodes to the graph
    layers = set(node.layer for node in genome.nodes)
    for layer in sorted(layers):
        with dot.subgraph() as subgraph:
            subgraph.attr(rank='same')
            color = 'lightblue' if layer == 0 else 'lightpink'
            for node in filter(lambda n: n.layer == layer, genome.nodes):
                subgraph.node(str(node.id), shape='circle',
                              style='filled', fillcolor=color)

    # Add edges to the graph
    for gene in genome.genes:
        color = 'green' if gene.enabled else 'red'
        dot.edge(str(gene.from_node.id), str(gene.to_node.id),
                 label=str(gene.weight), color=color)

    # Save the graph visualization to a file
    dot.render(filename, view=False)
