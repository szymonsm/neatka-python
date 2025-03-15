"""
Single-pole balancing experiment using a Kolmogorov-Arnold Network (KAN).
"""

import multiprocessing
import os
import pickle

import cart_pole
import neat
from neat.nn.kan import KANNetwork
from neat.kan_genome import KANGenome
import visualize
import visualize_kan

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

runs_per_net = 5
simulation_seconds = 60.0

# Use the KAN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = KANNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        sim = cart_pole.CartPole()

        # Run the given simulation for up to num_steps time steps.
        fitness = 0.0
        while sim.t < simulation_seconds:
            inputs = sim.get_scaled_state()
            action = net.activate(inputs)

            # Apply action to the simulated cart-pole
            force = cart_pole.discrete_actuator_force(action)
            sim.step(force)

            # Stop if the network fails to keep the cart within the position or angle limits.
            # The per-run fitness is the number of time steps the network can balance the pole
            # without exceeding these limits.
            if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
                break

            fitness = sim.t

        fitnesses.append(fitness)

    # The genome's fitness is its worst performance across all runs.
    return min(fitnesses)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-kan')
    config = neat.Config(KANGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner-kan', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    print("Plotting statistics...")
    visualize.plot_stats(stats, ylog=True, view=True, filename="kan-fitness.svg")
    print("Plotting species...")
    visualize.plot_species(stats, view=True, filename="kan-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    
    # Use KAN-specific visualization functions
    print("Visualizing winner...")
    visualize.draw_kan_net(config, winner, view=True, node_names=node_names,
                        filename="winner-kan.gv")
    print("Visualizing winner (pruned)...")
    visualize.draw_kan_net(config, winner, view=True, node_names=node_names,
                        filename="winner-kan-pruned.gv", prune_unused=True)
    
    # Plot spline visualizations
    print("Plotting splines...")
    visualize_kan.plot_kan_splines(winner, config.genome_config, 
                                  filename="winner-kan-splines.png", view=True)
    
    # Print detailed analysis of the genome
    print("Analyzing genome...")
    visualize_kan.analyze_kan_genome(winner, config.genome_config)

if __name__ == '__main__':
    run()