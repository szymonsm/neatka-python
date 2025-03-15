"""
Test the performance of the best genome produced by evolve-kan.py.
"""

import os
import pickle
import traceback

import neat
from neat.nn.kan import KANNetwork
from neat.kan_genome import KANGenome
from cart_pole import CartPole, discrete_actuator_force
from movie import make_movie
import visualize_kan

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# load the winner
with open('winner-kan', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-kan')
config = neat.Config(KANGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = KANNetwork.create(c, config)
sim = CartPole()

# Print detailed analysis of the genome
visualize_kan.analyze_kan_genome(c, config.genome_config)

# Try to plot spline visualizations with error handling
try:
    print("\nPlotting splines...")
    visualize_kan.plot_kan_splines(c, config.genome_config, 
                                  filename="winner-kan-splines.png", view=True)
    print("Splines plotted successfully.")
except Exception as e:
    print(f"Error plotting splines: {str(e)}")
    traceback.print_exc()

print("\nInitial conditions:")
print("        x = {0:.4f}".format(sim.x))
print("    x_dot = {0:.4f}".format(sim.dx))
print("    theta = {0:.4f}".format(sim.theta))
print("theta_dot = {0:.4f}".format(sim.dtheta))
print()

# Run the simulation for up to 120 seconds.
balance_time = 0.0
while sim.t < 120.0:
    inputs = sim.get_scaled_state()
    action = net.activate(inputs)

    force = discrete_actuator_force(action)
    sim.step(force)

    if abs(sim.x) >= sim.position_limit or abs(sim.theta) >= sim.angle_limit_radians:
        break

    balance_time = sim.t

print('Pole balanced for {0:.1f} of 120.0 seconds'.format(balance_time))

print("\nFinal conditions:")
print("        x = {0:.4f}".format(sim.x))
print("    x_dot = {0:.4f}".format(sim.dx))
print("    theta = {0:.4f}".format(sim.theta))
print("theta_dot = {0:.4f}".format(sim.dtheta))
print()

try:
    print("\nCreating movie... (this may take a while)")
    make_movie(net, discrete_actuator_force, 15.0, "kan-movie.mp4")
    print("Movie created successfully.")
except Exception as e:
    print(f"Error creating movie: {str(e)}")
    traceback.print_exc()