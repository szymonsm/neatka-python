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
                    print(f"Creating simple spline function for connection {conn.key}")
                    # Simple linear interpolation between points
                    def spline_func(x_val):
                        if not points:
                            return 0.0
                        x_points, y_points = zip(*points)
                        return np.interp(x_val, x_points, y_points)
                    
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


def plot_function_approximation(winner, config, test_data, results_dir, net_type='feedforward', function_name=''):
    """
    Plot the function approximation results.
    
    Args:
        winner: The winning genome
        config: Configuration object
        test_data: Dictionary with test data
        results_dir: Directory to save plots
        net_type: Type of network
        function_name: Name of the function being approximated
    """
    try:
        # Create network from winner
        if net_type.lower() == 'feedforward':
            from neat.nn import FeedForwardNetwork
            net = FeedForwardNetwork.create(winner, config)
        else:
            from neat.nn.kan import KANNetwork
            net = KANNetwork.create(winner, config)
        
        X_test = test_data['X_test']
        y_test = test_data['y_test']
        
        # Get predictions
        predictions = []
        for i in range(len(X_test)):
            try:
                output = net.activate(X_test[i])
                if isinstance(output, (list, tuple)):
                    output = output[0]
                predictions.append(float(output))
            except:
                predictions.append(0.0)
        
        predictions = np.array(predictions)
        
        # Create 2D visualization if possible
        if X_test.shape[1] == 2:
            fig = plt.figure(figsize=(15, 5))
            
            # Plot 1: True function
            ax1 = fig.add_subplot(131, projection='3d')
            ax1.scatter(X_test[:, 0], X_test[:, 1], y_test, c='blue', alpha=0.6, s=1)
            ax1.set_title(f'True Function: {function_name}')
            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Output')
            
            # Plot 2: Network prediction
            ax2 = fig.add_subplot(132, projection='3d')
            ax2.scatter(X_test[:, 0], X_test[:, 1], predictions, c='red', alpha=0.6, s=1)
            ax2.set_title(f'{net_type.upper()} Prediction')
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Output')
            
            # Plot 3: Error
            ax3 = fig.add_subplot(133)
            errors = np.abs(predictions - y_test)
            ax3.scatter(X_test[:, 0], X_test[:, 1], c=errors, cmap='viridis', alpha=0.6, s=1)
            ax3.set_title('Absolute Error')
            ax3.set_xlabel('X')
            ax3.set_ylabel('Y')
            plt.colorbar(ax3.collections[0], ax=ax3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'function_approximation_3d.png'), dpi=300)
            plt.close()
        
        # Always create scatter plot comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Scatter plot: predicted vs true
        ax1.scatter(y_test, predictions, alpha=0.6, s=1)
        min_val = min(np.min(y_test), np.min(predictions))
        max_val = max(np.max(y_test), np.max(predictions))
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
        ax1.set_xlabel('True values')
        ax1.set_ylabel('Predicted values')
        ax1.set_title(f'{net_type.upper()} vs True Values')
        ax1.legend()
        ax1.grid(True)
        
        # Error distribution
        errors = predictions - y_test
        ax2.hist(errors, bins=50, alpha=0.7)
        ax2.set_xlabel('Prediction Error')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Error Distribution')
        ax2.axvline(0, color='red', linestyle='--', label='Perfect prediction')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'function_approximation_2d.png'), dpi=300)
        plt.close()
        
        print(f"Function approximation plots saved to {results_dir}")
        
    except Exception as e:
        print(f"Error creating function approximation plots: {e}")
