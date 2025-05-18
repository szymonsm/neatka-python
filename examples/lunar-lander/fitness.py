"""
Fitness evaluation functions for the LunarLander-v2 environment.
"""
import lunar_lander

# Default number of runs per network
runs_per_net = 10

def evaluate_genome(genome, config, net_creator, env=None):
    """
    Evaluate a single genome.
    
    Args:
        genome: Genome to evaluate
        config: Configuration object
        net_creator: Function to create a network from genome (either FeedForwardNetwork.create or KANNetwork.create)
        env: Optional environment instance (will create if None)
        
    Returns:
        float: Fitness score
    """
    net = net_creator(genome, config)
    
    # Create or reuse environment
    local_env = env is None
    if local_env:
        env = lunar_lander.create_env()
    
    fitnesses = []
    
    # Run multiple times to get a more reliable fitness measure
    for _ in range(runs_per_net):
        fitness = lunar_lander.run_simulation(net, env)
        fitnesses.append(fitness)
    
    # Clean up if we created the environment
    if local_env:
        env.close()
    
    # Return the average fitness
    return sum(fitnesses) / len(fitnesses)

def evaluate_population(genomes, config, net_creator):
    """
    Evaluate all genomes in a population.
    
    Args:
        genomes: List of (id, genome) tuples
        config: Configuration object
        net_creator: Function to create a network from genome (either FeedForwardNetwork.create or KANNetwork.create)
    """
    env = lunar_lander.create_env()
    
    for genome_id, genome in genomes:
        genome.fitness = evaluate_genome(genome, config, net_creator, env)
    
    env.close()