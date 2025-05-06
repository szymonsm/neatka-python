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
        fig.tight_layout(pad=0)
        fig.savefig(filename, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        return filename
    except Exception as e:
        print(f"Error generating mini spline plot: {e}")
        return None


def draw_net(config, genome, view=False, filename=None, node_names=None, show_disabled=True, prune_unused=False,
             node_colors=None, fmt='svg', net_type='feedforward'):
    """
    Draws a neural network with support for both standard feedforward and KAN networks.
    
    Args:
        config: The NEAT configuration object
        genome: The genome to visualize
        view: Whether to open the rendered image (default: False)
        filename: Where to save the rendered image (default: None)
        node_names: Dictionary of node names keyed by node IDs (default: None)
        show_disabled: Show disabled connections (default: True)
        prune_unused: Remove unused nodes and connections (default: False)
        node_colors: Dictionary of node colors keyed by node IDs (default: None)
        fmt: Format of the output file, e.g., 'svg', 'png' (default: 'svg')
        net_type: Type of network, either 'feedforward' or 'kan' (default: 'feedforward')
    
    Returns:
        The graphviz Digraph object
    """
    if graphviz is None:
        warnings.warn("Graphviz is not available.")
        return
    
    # Handle network pruning
    if prune_unused:
        if show_disabled:
            warnings.warn("show_disabled has no effect when prune_unused is True")
        if hasattr(genome, 'get_pruned_copy'):
            genome = genome.get_pruned_copy(config.genome_config)
        else:
            warnings.warn("This genome doesn't support pruning, using original")
    
    # Set up node names and colors
    node_names = node_names or {}
    node_colors = node_colors or {}
    
    # Create graphviz graph
    node_attrs = {
        'shape': 'circle', 'fontsize': '9', 'height': '0.2', 'width': '0.2'
    }
    dot = graphviz.Digraph(format=fmt, node_attr=node_attrs)
    
    # Generate spline plots for KAN networks
    spline_plots = {}
    if net_type.lower() == 'kan' and filename:
        # Create directory for spline plots based on output filename
        spline_img_path_dir = filename.split("\\")[:-1]
        # Create a new directory for the spline plots
        spline_img_path_dir = "\\".join(spline_img_path_dir)
        spline_img_path_dir = os.path.join(spline_img_path_dir, 'spline_plots')
        os.makedirs(spline_img_path_dir, exist_ok=True)
        for cg in genome.connections.values():
            if (cg.enabled or show_disabled) and hasattr(cg, 'spline_segments') and len(cg.spline_segments) > 0:
                input_key, output_key = cg.key
                # Generate plot name
                plot_file = os.path.join(spline_img_path_dir, f"spline_{input_key}_{output_key}.png")
                # Generate the plot
                spline_img_path = _generate_mini_spline_plot(cg, plot_file)
                if spline_img_path:
                    spline_plots[cg.key] = spline_img_path
    
    # Draw input nodes
    for k in config.genome_config.input_keys:
        name = node_names.get(k, str(k))
        dot.node(str(k), label=f"{name}", style='filled', 
                shape='box', fillcolor=node_colors.get(k, 'lightgray'))
    
    # Draw output nodes
    for k in config.genome_config.output_keys:
        node = genome.nodes.get(k)
        name = node_names.get(k, str(k))
        if node is not None:
            bias = node.bias
        else:
            bias = 0.0
        dot.node(str(k), label=f"{name}\nb={bias:.2f}", 
                style='filled', shape='box', fillcolor=node_colors.get(k, 'lightblue'))
    
    # Draw hidden nodes
    for k in genome.nodes.keys():
        if k not in config.genome_config.input_keys and k not in config.genome_config.output_keys:
            node = genome.nodes[k]
            if net_type.lower() == 'feedforward':
                # For feedforward, show activation function
                dot.node(str(k), label=f"{k}\n{node.activation}\nb={node.bias:.2f}",
                        style='filled', fillcolor=node_colors.get(k, 'white'))
            else:
                # For KAN, simpler node display
                dot.node(str(k), label=f"ID:{str(k)}\n∑\nb={node.bias:.2f}", 
                        style='filled', fillcolor=node_colors.get(k, 'white'))
    
    # Draw connections
    for cg in genome.connections.values():
        if cg.enabled or show_disabled:
            input_key, output_key = cg.key
            a = str(input_key)
            b = str(output_key)
            
            # For KAN networks with splines
            if net_type.lower() == 'kan' and hasattr(cg, 'spline_segments') and len(cg.spline_segments) > 0:
                # Add an intermediate node for the spline
                spline_node = f"{input_key}_{output_key}_spline"
                print(spline_plots[cg.key])
                # If we have a spline plot image, use it in the node
                if cg.key in spline_plots:
                    # For formats supporting embedded images
                    if fmt.lower() == 'png':
                        dot.node(spline_node, 
                                 label='',
                                 style='filled', 
                                 shape='box', 
                                 fillcolor='white', 
                                 fontsize='10',
                                 width='1.0', 
                                 height='1.0',
                                 image="C:/Users/szymo/WUT/Masters/Thesis/neatka-python/" + spline_plots[cg.key],
                                 imagescale='true',
                                 imagepos='tc')
                    else:
                        # For SVG and other formats, use HTML-like label with image
                        label = f'<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">' + \
                                f'<TR><TD><IMG SRC="{spline_plots[cg.key]}"/></TD></TR>' + \
                                f'<TR><TD>SPLINE<BR/>{len(cg.spline_segments)} segments</TD></TR>' + \
                                f'</TABLE>>'
                        
                        dot.node(spline_node,
                                 label=label,
                                 shape='none')
                else:
                    # No image version
                    dot.node(spline_node, 
                             label=f"SPLINE\n{len(cg.spline_segments)} segments", 
                             style='filled', 
                             shape='box', 
                             fillcolor='yellow', 
                             fontsize='10',
                             width='1.0', 
                             height='1.0')
                
                # Connect the spline node
                dot.edge(a, spline_node, style='solid')
                
                # KAN connections use weight_s and weight_b
                if hasattr(cg, 'weight_s') and hasattr(cg, 'weight_b'):
                    dot.edge(spline_node, b, 
                             style='solid' if cg.enabled else 'dotted', 
                             color='green' if cg.weight_s > 0 else 'red', 
                             penwidth=str(0.1 + abs(cg.weight_s)),
                             label=f"ws={cg.weight_s:.2f}\nwb={cg.weight_b:.2f}")
                else:
                    # Fallback to weight if KAN attributes not found
                    dot.edge(spline_node, b, 
                             style='solid' if cg.enabled else 'dotted', 
                             color='green' if cg.weight > 0 else 'red', 
                             penwidth=str(0.1 + abs(cg.weight / 5.0)),
                             label=f"w={cg.weight:.2f}")
                             
            else:
                # Regular connection (either feedforward or KAN without splines)
                if net_type.lower() == 'kan' and hasattr(cg, 'weight_s'):
                    # Use KAN-specific weights if available
                    style = 'solid' if cg.enabled else 'dotted'
                    color = 'green' if cg.weight_s > 0 else 'red'
                    width = str(0.1 + abs(cg.weight_s / 5.0))
                    label = f"ws={cg.weight_s:.2f}, wb={cg.weight_b:.2f}" if show_disabled else ''
                    
                    dot.edge(a, b, style=style, color=color, penwidth=width, label=label)
                else:
                    # Use standard NEAT weights
                    style = 'solid' if cg.enabled else 'dotted'
                    color = 'green' if cg.weight > 0 else 'red' if cg.enabled else 'gray'
                    width = str(0.1 + abs(cg.weight / 5.0)) if cg.enabled else str(0.1 + abs(cg.weight / 10.0))
                    label = f"{cg.weight:.2f}" if show_disabled else ''
                    
                    dot.edge(a, b, style=style, color=color, penwidth=width, label=label)
    
    # Render the graph
    if filename:
        dot.render(filename, view=view, cleanup=True, format=fmt)
    
    return dot

def plot_kan_splines(genome, config, filename="kan_splines.png", view=False):
    """
    Plot all splines in the KAN genome.
    
    Args:
        genome: A KANGenome object
        config: Configuration object
        filename: Output file name
        view: Whether to display the visualization
    """
    fig = plt.figure(figsize=(12, 8))
    
    # Get all connections
    connections = [conn for conn in genome.connections.values() if conn.enabled]
    print(f"Plotting {len(connections)} splines")
    
    if not connections:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No enabled connections", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes)
        fig.savefig(filename)
        if view:
            plt.show()
        return
        
    # Calculate grid layout
    n_conn = len(connections)
    cols = min(3, n_conn)
    rows = (n_conn + cols - 1) // cols
    
    # Plot each spline
    for i, conn in enumerate(connections):
        ax = fig.add_subplot(rows, cols, i+1)
        
        # Get spline points
        if hasattr(conn, 'spline_segments'):
            points = [(seg.grid_position, seg.value) 
                    for seg in conn.spline_segments.values()]
            
            # Handle connections with no spline points
            if not points:
                ax.text(0.5, 0.5, "No spline points", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
                ax.set_title(f"Connection {conn.key}")
                continue
                
            points.sort(key=lambda x: x[0])
            
            # Create x values for plotting
            try:
                x_min = config.spline_range_min
                x_max = config.spline_range_max
            except AttributeError:
                x_min = -1.0
                x_max = 1.0
                
            x = np.linspace(x_min, x_max, 100)
            
            try:
                # Get the spline function
                if hasattr(conn, 'get_spline_function'):
                    spline_func = conn.get_spline_function()
                    print(f"Using get_spline_function for connection {conn.key}")
                else:
                    # Create a simple function if method doesn't exist
                    from neat.nn.kan import SplineFunctionImpl
                    spline_func = SplineFunctionImpl(points)
                    print(f"Creating SplineFunctionImpl for connection {conn.key}")
                    
                # Plot the spline if we have points
                y = []
                for xi in x:
                    try:
                        y.append(spline_func(xi))
                    except Exception as e:
                        print(f"Error evaluating spline at x={xi}: {str(e)}")
                        y.append(0.0)
                        
                # Always use ax for plotting in subplots
                ax.plot(x, y)
                
                # Plot control points
                x_points, y_points = zip(*points)
                ax.scatter(x_points, y_points, c='r', marker='o')
                
            except Exception as e:
                print(f"Error plotting spline for connection {conn.key}: {str(e)}")
                ax.text(0.5, 0.5, f"Error: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "Not a KAN connection", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes)
        
        # Add connection info - use ax for all labels
        ax.set_title(f"Connection {conn.key}")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.grid(True)
    
    fig.tight_layout()
    try:
        fig.savefig(filename)
        print(f"Saved spline plots to {filename}")
    except Exception as e:
        print(f"Error saving figure: {e}")
    
    if view:
        plt.show()
    else:
        plt.close()

def analyze_genome(genome, config, node_names=None, net_type='feedforward'):
    """
    Print detailed information about a genome.
    
    Args:
        genome: The genome to analyze
        config: The configuration object
        node_names: Dictionary mapping node IDs to names (default: None)
        net_type: Type of network ('feedforward' or 'kan')
    """
    print(f"Genome ID: {genome.key}")
    print(f"Fitness: {genome.fitness}")
    print(f"Nodes: {len(genome.nodes)}")
    print(f"Connections: {len(genome.connections)}")
    
    # Count enabled connections
    enabled_connections = [c for c in genome.connections.values() if c.enabled]
    print(f"Enabled connections: {len(enabled_connections)}")

    # Print node information
    for k in config.input_keys:
        print(f"  Input Node: {node_names.get(k, str(k))} ({k})")
    
    # Print hidden node information
    for k in genome.nodes.keys():
        if k not in config.input_keys and k not in config.output_keys:
            node = genome.nodes[k]
            if net_type.lower() == 'feedforward':
                print(f"  Hidden Node {k}: bias={node.bias:.3f}, response={node.response:.3f}")
                print(f"    activation: {node.activation}")
                print(f"    aggregation: {node.aggregation}")
            else:
                print(f"  Hidden Node {k}: bias={node.bias:.3f}")

    # Print output node information
    for k in config.output_keys:
        if k in genome.nodes:
            node = genome.nodes[k]
            if net_type.lower() == 'feedforward':
                print(f"  Output Node {node_names.get(k, str(k))} ({k}): bias={node.bias:.3f}, response={node.response:.3f}")
                print(f"    activation: {node.activation}")
                print(f"    aggregation: {node.aggregation}")
            else:
                print(f"  Output Node {node_names.get(k, str(k))} ({k}): bias={node.bias:.3f}")
        else:
            print(f"  Output Node {node_names.get(k, str(k))} ({k}): bias=0.0")
    
    # Print connection information
    print("\nConnections:")
    connections = list(genome.connections.values())
    connections.sort(key=lambda x: (x.key[0], x.key[1]))
    
    for conn in connections:
        if conn.enabled:
            source, target = conn.key
            if net_type.lower() == 'kan' and hasattr(conn, 'weight_s'):
                # Print KAN-specific connection info
                n_segments = len(conn.spline_segments) if hasattr(conn, 'spline_segments') else 0
                print(f"  {source} -> {target}: ws={conn.weight_s:.3f}, wb={conn.weight_b:.3f}, segments={n_segments}")
            else:
                # Print standard connection info
                print(f"  {source} -> {target}: weight={conn.weight:.3f}")
    
    # Print additional KAN-specific information
    if net_type.lower() == 'kan':
        total_segments = sum(len(conn.spline_segments) if hasattr(conn, 'spline_segments') else 0 
                            for conn in genome.connections.values() if conn.enabled)
        print(f"\nTotal spline segments: {total_segments}")
        
    # Calculate network complexity metrics
    num_hidden = len([k for k in genome.nodes.keys() 
                     if k not in config.input_keys and k not in config.output_keys])
    
    print("\nNetwork Complexity:")
    print(f"  Hidden nodes: {num_hidden}")
    print(f"  Total connections: {len(genome.connections)}")
    print(f"  Enabled connections: {len(enabled_connections)}")


def plot_lunar_lander_trajectory(episode_data, filename='trajectory.png', view=False):
    """
    Plot the trajectory of the lunar lander during an episode.
    
    Args:
        episode_data: Dictionary with 'states', 'actions', 'rewards' arrays
        filename: Where to save the plot
        view: Whether to display the plot
    """
    states = episode_data['states']
    actions = episode_data['actions']
    rewards = episode_data['rewards']
    
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    # Time steps
    steps = np.arange(len(states))
    
    # Plot positions and velocities
    axs[0].plot(steps[:-1], states[:-1, 0], label='x_pos')
    axs[0].plot(steps[:-1], states[:-1, 1], label='y_pos')
    axs[0].set_ylabel('Position')
    axs[0].grid(True)
    axs[0].legend()
    
    # Plot velocities
    axs[1].plot(steps[:-1], states[:-1, 2], label='x_vel')
    axs[1].plot(steps[:-1], states[:-1, 3], label='y_vel')
    axs[1].set_ylabel('Velocity')
    axs[1].grid(True)
    axs[1].legend()
    
    # Plot angle, angular velocity, and leg contact points
    axs[2].plot(steps[:-1], states[:-1, 4], label='angle')
    axs[2].plot(steps[:-1], states[:-1, 5], label='angular_vel')
    axs[2].plot(steps[:-1], states[:-1, 6], label='leg1')
    axs[2].plot(steps[:-1], states[:-1, 7], label='leg2')
    axs[2].set_ylabel('Angle/Legs')
    axs[2].grid(True)
    axs[2].legend()
    
    # Plot actions and rewards
    ax3 = axs[3]
    ax3.plot(steps[:-1], rewards, 'r-', label='Reward')
    ax3.set_ylabel('Reward')
    ax3.set_xlabel('Time Steps')
    ax3.grid(True)
    
    # Add cumulative reward
    ax4 = ax3.twinx()
    ax4.plot(steps[:-1], np.cumsum(rewards), 'b--', label='Cumulative')
    ax4.set_ylabel('Cumulative Reward')
    
    # Add action markers
    action_labels = ['Noop', 'Left', 'Main', 'Right']
    markers = ['o', 's', '^', 'd']
    colors = ['blue', 'green', 'red', 'purple']
    
    for a in range(4):
        mask = actions == a
        if np.any(mask):
            ax3.scatter(steps[:-1][mask], np.zeros_like(steps[:-1])[mask] - 2, 
                      marker=markers[a], color=colors[a], label=action_labels[a])
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax4.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    plt.savefig(filename)
    
    if view:
        plt.show()
    else:
        plt.close()
    
    return fig