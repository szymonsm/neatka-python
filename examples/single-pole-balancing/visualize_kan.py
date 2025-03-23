"""
Visualization utilities for Kolmogorov-Arnold Networks (KAN).
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import tempfile
import hashlib
import graphviz
from neat.kan_utils import plot_spline, visualize_kan_network

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

            # print(f"x={x}")
            # print(f"y={y}")
            # print(f"points={points}")
            
        except Exception as e:
            print(f"Error plotting spline for connection {conn.key}: {str(e)}")
            ax.text(0.5, 0.5, f"Error: {str(e)}", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes)
        
        # Add connection info - use ax for all labels
        ax.set_title(f"Connection {conn.key}")
        ax.set_xlabel("Input")
        ax.set_ylabel("Output")
        ax.grid(True)
        print(f"Plotted spline for connection {conn.key}")
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

def analyze_kan_genome(genome, config, node_names=None):
    """Print detailed information about a KAN genome."""
    print(f"Genome ID: {genome.key}")
    print(f"Fitness: {genome.fitness}")
    print(f"Nodes: {len(genome.nodes)}")
    print(f"Connections: {len(genome.connections)}")
    
    # Count enabled connections
    enabled_connections = [c for c in genome.connections.values() if c.enabled]
    print(f"Enabled connections: {len(enabled_connections)}")

    # Print node information
    for k in config.input_keys:
        print(f"  Input Node: {node_names.get(k, str(k))} ({k}): {genome.nodes[k].bias:.3f}")
    
    # Print hidden Node information
    for k in genome.nodes.keys():
        if k not in config.input_keys and k not in config.output_keys:
            print(f"  Hidden Node {genome.nodes[k]}: {genome.nodes[k].bias:.3f}")

    # Print output node information
    for k in config.output_keys:
        print(f"  Output Node {node_names.get(k, str(k))} ({k}): {genome.nodes[k].bias:.3f}")
    
    # Print spline information
    total_segments = 0
    for key, conn in genome.connections.items():
        if conn.enabled:
            n_segments = len(conn.spline_segments)
            total_segments += n_segments
            print(f"  Connection {key}: {n_segments} segments, weight_s={conn.weight_s:.3f}, weight_b={conn.weight_b:.3f}")
    
    print(f"Total spline segments: {total_segments}")
    
def plot_kan_connection_responses(genome, config, filename="kan_responses.png", view=False):
    """
    Plot response curves for each connection in the KAN genome.
    Shows how different input values are transformed by each connection's spline.
    
    Args:
        genome: A KANGenome object
        config: Configuration object
        filename: Output file name
        view: Whether to display the visualization
    """
    # Get all enabled connections
    connections = [conn for conn in genome.connections.values() if conn.enabled]
    if not connections:
        print("No enabled connections to plot")
        return
    
    # Determine subplot layout
    n_conns = len(connections)
    cols = min(3, n_conns)
    rows = (n_conns + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    fig.suptitle("KAN Connection Responses", fontsize=16)
    
    # Make axes iterable even for a single subplot
    if n_conns == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Generate common input range
    x_range = np.linspace(-1, 1, 100)
    
    for i, conn in enumerate(connections):
        if i < len(axes):
            ax = axes[i]
            
            # Get spline function
            spline_func = conn.get_spline_function()
            
            # Calculate raw spline output
            raw_output = np.array([spline_func(x) for x in x_range])
            
            # Calculate scaled and biased output (final connection output)
            scaled_output = conn.weight * (conn.scale * raw_output + conn.bias)
            
            # Plot both curves
            ax.plot(x_range, raw_output, 'b-', label='Spline')
            ax.plot(x_range, scaled_output, 'r-', label='Final output')
            
            # Plot control points
            points = [(seg.grid_position, seg.value) for seg in conn.spline_segments.values()]
            if points:
                x_points, y_points = zip(*sorted(points))
                ax.scatter(x_points, y_points, c='b', marker='o', s=30)
            
            # Add title and labels
            ax.set_title(f"Connection {conn.key}")
            ax.set_xlabel("Input")
            ax.set_ylabel("Output")
            ax.legend()
            ax.grid(True)
    
    # Hide unused subplots
    for i in range(n_conns, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    if filename:
        plt.savefig(filename)
    
    if view:
        plt.show()
    else:
        plt.close()

def input_response_analysis(genome, config, input_idx=0, view=False, filename="input_response.png"):
    """
    Analyze how a specific input affects the network output.
    Varies one input while keeping others at zero to see the response.
    
    Args:
        genome: A KANGenome object
        config: Configuration object
        input_idx: Index of the input to analyze (0-3 for cart pole)
        view: Whether to display the plot
        filename: Output file name
    """
    try:
        from neat.nn.kan import KANNetwork
    except ImportError:
        print("KANNetwork class not found!")
        return
        
    # Create the network
    net = KANNetwork.create(genome, config)
    
    # Generate a range of input values
    input_range = np.linspace(-1.0, 1.0, 100)
    outputs = []
    
    # For each input value, activate the network and record the output
    for val in input_range:
        # Create inputs (all zeros except for the one we're varying)
        inputs = [0.0] * len(config.genome_config.input_keys)
        inputs[input_idx] = val
        
        # Activate network and get output
        output = net.activate(inputs)
        outputs.append(output[0])  # Assuming one output
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(input_range, outputs)
    
    # Add labels
    input_names = {0: 'x', 1: 'dx', 2: 'theta', 3: 'dtheta'}
    input_name = input_names.get(input_idx, f"Input {input_idx}")
    
    plt.title(f"Network Response to {input_name}")
    plt.xlabel(f"{input_name} value")
    plt.ylabel("Network output")
    plt.grid(True)
    
    if filename:
        plt.savefig(filename)
        
    if view:
        plt.show()
    else:
        plt.close()

def draw_kan_network_with_splines(genome, config, filename="kan_network.svg", view=False):
    """
    Draws the KAN network with embedded spline visualizations.
    
    Args:
        genome: A KANGenome object
        config: Configuration object
        filename: Output filename
        view: Whether to display the visualization
    """
    if not graphviz:
        print("Graphviz not available. Cannot create network visualization.")
        return None
        
    # Create a temporary directory for spline images
    with tempfile.TemporaryDirectory() as temp_dir:
        # Generate spline images for each connection
        spline_images = {}
        for conn_key, conn in genome.connections.items():
            if not conn.enabled or not hasattr(conn, 'spline_segments'):
                continue
                
            # Generate a mini spline plot
            points = [(seg.grid_position, seg.value) for seg in conn.spline_segments.values()]
            if not points:
                continue
                
            # Create the plot
            fig, ax = plt.subplots(figsize=(2, 1.5), dpi=100)
            
            # Sort points and get x-range
            points.sort(key=lambda p: p[0])
            x_min = min(p[0] for p in points) - 0.1
            x_max = max(p[0] for p in points) + 0.1
            
            # Plot spline
            x = np.linspace(x_min, x_max, 100)
            spline_func = conn.get_spline_function()
            y = [spline_func(xi) for xi in x]
            
            ax.plot(x, y, 'b-')
            
            # Plot control points
            x_points, y_points = zip(*points)
            ax.scatter(x_points, y_points, c='r', marker='o', s=20)
            
            # Clean up plot
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            
            # Save the plot
            img_path = os.path.join(temp_dir, f"spline_{conn_key[0]}_{conn_key[1]}.png")
            fig.savefig(img_path, transparent=True, bbox_inches='tight', pad_inches=0)
            plt.close(fig)
            
            spline_images[conn_key] = img_path
        
        # Create the network visualization
        node_attrs = {'shape': 'circle', 'fontsize': '9', 'height': '0.2', 'width': '0.2'}
        dot = graphviz.Digraph(format='svg', node_attr=node_attrs)
        
        # Add input nodes
        for k in config.genome_config.input_keys:
            dot.node(str(k), style='filled', shape='box', fillcolor='lightgray')
        
        # Add output nodes
        for k in config.genome_config.output_keys:
            dot.node(str(k), style='filled', shape='box', fillcolor='lightblue')
        
        # Add hidden nodes
        for k in genome.nodes.keys():
            if k not in config.genome_config.input_keys and k not in config.genome_config.output_keys:
                dot.node(str(k), label=f"{k}\n∑", style='filled', fillcolor='white')
        
        # Add connections
        for conn_key, conn in genome.connections.items():
            if not conn.enabled:
                continue
                
            input_key, output_key = conn_key
            
            if conn_key in spline_images:
                # Add a spline node
                spline_node = f"{input_key}_{output_key}_spline"
                dot.node(spline_node,
                        label=f"SPLINE\n{len(conn.spline_segments)} segments",
                        style='filled',
                        shape='box',
                        fillcolor='yellow',
                        fontsize='10',
                        width='1.5',
                        height='1.5',
                        image=spline_images[conn_key],
                        imagescale='true')
                
                # Connect input -> spline -> output
                dot.edge(str(input_key), spline_node, style='solid')
                dot.edge(spline_node, str(output_key), 
                        style='solid',
                        color='green' if conn.weight > 0 else 'red',
                        penwidth=str(0.1 + abs(conn.weight / 5.0)),
                        label=f"w={conn.weight:.2f}\ns={conn.scale:.2f}\nb={conn.bias:.2f}")
            else:
                # Direct connection
                dot.edge(str(input_key), str(output_key),
                        style='solid',
                        color='green' if conn.weight > 0 else 'red',
                        penwidth=str(0.1 + abs(conn.weight / 5.0)))
        
        # Render the graph
        dot.render(filename, view=view)
        
        return dot