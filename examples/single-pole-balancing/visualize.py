import copy
import warnings
import os

import graphviz
import matplotlib.pyplot as plt
import numpy as np


def plot_stats(statistics, ylog=False, view=False, filename='avg_fitness.svg'):
    """ Plots the population's average and best fitness. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    generation = range(len(statistics.most_fit_genomes))
    best_fitness = [c.fitness for c in statistics.most_fit_genomes]
    avg_fitness = np.array(statistics.get_fitness_mean())
    stdev_fitness = np.array(statistics.get_fitness_stdev())

    plt.plot(generation, avg_fitness, 'b-', label="average")
    # plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
    plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
    plt.plot(generation, best_fitness, 'r-', label="best")

    plt.title("Population's average and best fitness")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.grid()
    plt.legend(loc="best")
    if ylog:
        plt.gca().set_yscale('symlog')

    plt.savefig(filename)
    if view:
        plt.show()

    plt.close()


def plot_spikes(spikes, view=False, filename=None, title=None):
    """ Plots the trains for a single spiking neuron. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    t_values = [t for t, I, v, u in spikes]
    v_values = [v for t, I, v, u in spikes]
    u_values = [u for t, I, v, u in spikes]
    I_values = [I for t, I, v, u in spikes]

    fig = plt.figure()
    plt.subplot(3, 1, 1)
    plt.ylabel("Potential (mv)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, v_values, "g-")

    if title is None:
        plt.title("Izhikevich's spiking neuron model")
    else:
        plt.title("Izhikevich's spiking neuron model ({0!s})".format(title))

    plt.subplot(3, 1, 2)
    plt.ylabel("Recovery (u)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, u_values, "r-")

    plt.subplot(3, 1, 3)
    plt.ylabel("Current (I)")
    plt.xlabel("Time (in ms)")
    plt.grid()
    plt.plot(t_values, I_values, "r-o")

    if filename is not None:
        plt.savefig(filename)

    if view:
        plt.show()
        plt.close()
        fig = None

    return fig


def plot_species(statistics, view=False, filename='speciation.svg'):
    """ Visualizes speciation throughout evolution. """
    if plt is None:
        warnings.warn("This display is not available due to a missing optional dependency (matplotlib)")
        return

    species_sizes = statistics.get_species_sizes()
    num_generations = len(species_sizes)
    curves = np.array(species_sizes).T

    fig, ax = plt.subplots()
    ax.stackplot(range(num_generations), *curves)

    plt.title("Speciation")
    plt.ylabel("Size per Species")
    plt.xlabel("Generations")

    plt.savefig(filename)

    if view:
        plt.show()

    plt.close()


def draw_kan_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
                 node_colors=None, fmt='svg'):
    """Draws a neural network with KAN-specific spline visualizations."""
    if graphviz is None:
        warnings.warn("Graphviz is not available.")
        return
    
    if prune_unused:
        if show_disabled:
            warnings.warn("show_disabled has no effect when prune_unused is True")
        genome = genome.get_pruned_copy(config.genome_config)
    
    node_names = node_names or {}
    node_colors = node_colors or {}
    
    node_attrs = {
        'shape': 'circle', 'fontsize': '9', 'height': '0.2', 'width': '0.2'
    }
    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)
    
    # Generate mini spline plots and store their paths
    spline_plots = {}
    # Drop the last part of the filename to get the directory
    spline_img_path_dir = filename.split("\\")[:-1]
    # Create a new directory for the spline plots
    spline_img_path_dir = "\\".join(spline_img_path_dir)
    spline_img_path_dir = os.path.join(spline_img_path_dir, 'spline_plots')
    os.makedirs(spline_img_path_dir, exist_ok=True)
    for cg in genome.connections.values():
        if (cg.enabled or show_disabled) and hasattr(cg, 'spline_segments') and len(cg.spline_segments) > 0:
            input_key, output_key = cg.key
            spline_img_path = _generate_mini_spline_plot(cg, os.path.join(spline_img_path_dir, f"spline_{input_key}_{output_key}.png"))
            if spline_img_path:
                spline_plots[cg.key] = spline_img_path
    # Draw input and output nodes
    for k in config.genome_config.input_keys:
        dot.node(node_names.get(k, str(k)), style='filled', shape='box', fillcolor=node_colors.get(k, 'lightgray'))
    for k in config.genome_config.output_keys:
        dot.node(node_names.get(k, str(k)), style='filled', fillcolor=node_colors.get(k, 'lightblue'))
    
    # Draw hidden nodes
    for n in genome.nodes.keys():
        if n not in config.genome_config.input_keys and n not in config.genome_config.output_keys:
            dot.node(str(n), label=f"ID:{str(n)}\n∑", style='filled', fillcolor=node_colors.get(n, 'white'))
    
    # Draw connections, including KAN-specific splines
    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input_key, output_key = cg.key
            a = node_names.get(input_key, str(input_key))
            b = node_names.get(output_key, str(output_key))
            
            if hasattr(cg, 'spline_segments') and len(cg.spline_segments) > 0:
                # Add an intermediate node for the spline
                spline_node = f"{input_key}_{output_key}_spline"

                print(spline_plots[cg.key])
                
                # If we have a spline plot image, use it in the node
                if cg.key in spline_plots:
                    dot.node(spline_node, 
                            #  label=f"SPLINE\n{len(cg.spline_segments)} segments",
                             label='',
                             style='filled', 
                             shape='box', 
                             fillcolor='white', 
                             fontsize='10',
                             width='1.0', 
                             height='1.0',
                             image=spline_plots[cg.key],
                             imagescale='true',
                             imagepos='tc')
                else:
                    dot.node(spline_node, 
                             label=f"SPLINE\n{len(cg.spline_segments)} segments", 
                             style='filled', 
                             shape='box', 
                             fillcolor='yellow', 
                             fontsize='10',
                             width='1.0', 
                             height='1.0')
                
                dot.edge(a, spline_node, style='solid')
                dot.edge(spline_node, b, 
                         style='solid' if cg.enabled else 'dotted', 
                         color='green' if cg.weight > 0 else 'red', 
                         penwidth=str(0.1 + abs(cg.weight)),
                         label=f"w={cg.weight:.2f}\ns={cg.scale:.2f}\nb={cg.bias:.2f}")
            else:
                # Regular connection
                dot.edge(a, b, 
                         style='solid' if cg.enabled else 'dotted', 
                         color='green' if cg.weight > 0 else 'red', 
                         penwidth=str(0.1 + abs(cg.weight / 5.0)))
    
    if filename:
        dot.render(filename, view=view, format=fmt, cleanup=True)
    
    return dot

def _generate_mini_spline_plot(connection, filename):
    """Generate a small image of the spline for a connection."""
    try:
        # Extract spline points
        points = [(seg.grid_position, seg.value) 
                 for seg in connection.spline_segments.values()]
        
        if not points:
            return None
        
        # Sort points by x coordinate
        points.sort(key=lambda x: x[0])
        
        # Generate mini plot
        fig = plt.figure(figsize=(2, 1.5), dpi=100)
        ax = fig.add_subplot(111)
        
        # Create x values for the spline
        x_min = min(p[0] for p in points) - 0.1
        x_max = max(p[0] for p in points) + 0.1
        x = np.linspace(x_min, x_max, 100)
        
        # Get the spline function
        spline_func = connection.get_spline_function()
        
        # Evaluate and plot the spline
        y = [spline_func(xi) for xi in x]
        ax.plot(x, y, 'b-')
        
        # Plot control points
        x_points, y_points = zip(*points)
        ax.scatter(x_points, y_points, c='r', marker='o', s=20)
        
        # Clean up the plot
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_title(f"w={connection.weight:.2f}", fontsize=8)
        
        # Remove axes and save with transparent background
        # ax.axis('off')
        fig.tight_layout(pad=0)
        fig.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
        # view the plot
        plt.close(fig)
        
        return filename
    except Exception as e:
        print(f"Error generating mini spline plot: {e}")
        return None
