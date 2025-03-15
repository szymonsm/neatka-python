import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_spline(connection, x_range=(-1, 1), num_points=100):
    """Plot a spline function for visualization.
    
    Args:
        connection: A KANConnectionGene object
        x_range: Range of x values to plot
        num_points: Number of points to plot
    """
    spline_func = connection.get_spline_function()
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = [spline_func(xi) for xi in x]
    
    plt.figure(figsize=(8, 4))
    plt.plot(x, y, label=f"Connection {connection.key}")
    
    # Plot the control points
    control_points = [(seg.grid_position, seg.value) for seg in connection.spline_segments.values()]
    if control_points:
        control_x, control_y = zip(*sorted(control_points))
        plt.scatter(control_x, control_y, c='r', marker='o', label="Control points")
    
    plt.title(f"Spline Function for Connection {connection.key}")
    plt.xlabel("Input")
    plt.ylabel("Output")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def visualize_kan_network(genome, config, view=False, filename='kan_network.png'):
    """Visualize a KAN network.
    
    Args:
        genome: A KANGenome object
        config: Configuration object
        view: Whether to display the visualization
        filename: Output file name
    """
    try:
        import graphviz
    except ImportError:
        raise ImportError("This function requires graphviz. Please install it first.")
        
    dot = graphviz.Digraph(format='png')
    dot.attr(rankdir='LR')
    
    # Add nodes
    for key, node in genome.nodes.items():
        if key in config.genome_config.input_keys:
            node_color = 'lightblue'
            node_shape = 'box'
            label = f"Input {key}"
        elif key in config.genome_config.output_keys:
            node_color = 'lightgreen'
            node_shape = 'box'
            label = f"Output {key}"
        else:
            node_color = 'white'
            node_shape = 'ellipse'
            label = f"Node {key}"
            
        dot.node(str(key), label=label, shape=node_shape, style='filled', fillcolor=node_color)
    
    # Add connections
    for key, conn in genome.connections.items():
        in_node, out_node = key
        
        # Skip disabled connections
        if not conn.enabled:
            continue
            
        # Get number of spline segments
        n_segments = len(conn.spline_segments)
        
        # Set edge attributes based on connection properties
        edge_attrs = {
            'label': f"w={conn.weight:.2f}\ns={conn.scale:.2f}\nb={conn.bias:.2f}\nn={n_segments}",
            'color': 'green' if conn.weight > 0 else 'red',
            'penwidth': str(0.5 + abs(conn.weight)/2),
            'arrowhead': 'normal',
        }
        
        # Draw the edge
        dot.edge(str(in_node), str(out_node), **edge_attrs)
    
    # Save the visualization
    dot.render(filename, view=view)
    return dot

def convert_splines_to_tensor(genome):
    """Convert spline segments to tensor format compatible with KANLayer."""
    connections = {}
    
    # For each connection in the genome
    for key, conn in genome.connections.items():
        if not conn.enabled:
            continue
            
        in_node, out_node = key
        
        # Get spline points sorted by grid position
        points = [(seg.grid_position, seg.value) 
                 for seg in conn.spline_segments.values()]
        points.sort(key=lambda x: x[0])
        
        # Convert to numpy arrays
        if points:
            x_coords = np.array([p[0] for p in points])
            y_coords = np.array([p[1] for p in points])
            
            # Store with connection info
            connections[key] = {
                'grid': x_coords,
                'values': y_coords,
                'weight': conn.weight,
                'scale': conn.scale,
                'bias': conn.bias
            }
    
    return connections

def plot_all_splines(genome, config, view=False, filename='kan_splines.png'):
    """Plot all spline functions in a KAN genome.
    
    Args:
        genome: A KANGenome object
        config: Configuration object
        view: Whether to display the visualization
        filename: Output file name
    """
    # Get enabled connections
    enabled_connections = [conn for key, conn in genome.connections.items() if conn.enabled]
    if not enabled_connections:
        print("No enabled connections to plot")
        return
    
    # Determine subplot layout
    n_conns = len(enabled_connections)
    cols = min(3, n_conns)
    rows = (n_conns + cols - 1) // cols  # Ceiling division
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    fig.suptitle("Spline Functions for KAN Genome", fontsize=16)
    
    # Make axes iterable even for a single subplot
    if n_conns == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each connection's spline
    for i, conn in enumerate(enabled_connections):
        if i < len(axes):
            ax = axes[i]
            
            # Get the spline function
            spline_func = conn.get_spline_function()
            
            # Generate x values for plotting
            x_range = (config.spline_range_min, config.spline_range_max)
            x = np.linspace(x_range[0], x_range[1], 100)
            
            # Evaluate spline for each x value
            y = [spline_func(xi) for xi in x]
            
            # Plot the spline curve
            ax.plot(x, y)
            
            # Plot control points
            control_points = [(seg.grid_position, seg.value) 
                             for seg in conn.spline_segments.values()]
            if control_points:
                control_x, control_y = zip(*sorted(control_points))
                ax.scatter(control_x, control_y, c='r', marker='o')
            
            # Set labels and title
            ax.set_title(f"Connection {conn.key}")
            ax.set_xlabel("Input")
            ax.set_ylabel("Output")
            ax.grid(True)
            
            # Add weight, scale, bias info
            ax.text(0.02, 0.95, f"w={conn.weight:.2f}, s={conn.scale:.2f}, b={conn.bias:.2f}", 
                   transform=ax.transAxes, fontsize=8, 
                   verticalalignment='top', bbox={'boxstyle':'round', 'alpha':0.2})
    
    # Hide unused subplots
    for i in range(n_conns, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save the figure
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if view:
        plt.show()
    else:
        plt.close()

def plot_3d_kan_surface(genome, config, input_indices=(0, 1), output_index=None, 
                       view=False, filename='kan_surface.png'):
    """Plot a 3D surface for a KAN network with 2 inputs.
    
    Args:
        genome: A KANGenome object
        config: Configuration object
        input_indices: Tuple of two input indices to vary
        output_index: Output index to visualize (default: first output)
        view: Whether to display the visualization
        filename: Output file name
    """
    try:
        from neat.nn.kan import KANNetwork
    except ImportError:
        print("KANNetwork class not found. Make sure it's properly implemented.")
        return
    
    # Create the network
    net = KANNetwork.create(genome, config)
    
    # Default to first output if not specified
    if output_index is None:
        output_index = 0
        
    # Get input and output keys    
    input_keys = config.genome_config.input_keys
    output_keys = config.genome_config.output_keys
    
    if len(input_keys) < 2:
        print("Network needs at least 2 inputs for 3D visualization")
        return
        
    if output_index >= len(output_keys):
        print(f"Invalid output index: {output_index}. Network has {len(output_keys)} outputs")
        return
    
    # Get the selected input and output keys
    in1_key = input_keys[input_indices[0]]
    in2_key = input_keys[input_indices[1]]
    out_key = output_keys[output_index]
    
    # Create a grid of input values
    n_points = 50
    x1 = np.linspace(config.spline_range_min, config.spline_range_max, n_points)
    x2 = np.linspace(config.spline_range_min, config.spline_range_max, n_points)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Evaluate the network at each point
    Z = np.zeros_like(X1)
    
    for i in range(n_points):
        for j in range(n_points):
            # Prepare inputs (all zeros except the two we're varying)
            inputs = [0.0] * len(input_keys)
            inputs[input_indices[0]] = X1[i, j]
            inputs[input_indices[1]] = X2[i, j]
            
            # Activate the network
            outputs = net.activate(inputs)
            
            # Store the output
            Z[i, j] = outputs[output_index]
    
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(X1, X2, Z, cmap='viridis', edgecolor='none', alpha=0.8)
    
    # Add labels
    ax.set_xlabel(f'Input {in1_key}')
    ax.set_ylabel(f'Input {in2_key}')
    ax.set_zlabel(f'Output {out_key}')
    ax.set_title(f'KAN Network Response Surface')
    
    # Add a color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Save the figure
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    if view:
        plt.show()
    else:
        plt.close()

def analyze_kan_genome(genome, config):
    """Print detailed analysis of a KAN genome.
    
    Args:
        genome: A KANGenome object
        config: Configuration object
    """
    print(f"=== KAN Genome Analysis ===")
    print(f"Genome ID: {genome.key}")
    print(f"Fitness: {genome.fitness}")
    print(f"Nodes: {len(genome.nodes)} total")
    
    # Count node types
    input_nodes = [node for key, node in genome.nodes.items() 
                  if key in config.genome_config.input_keys]
    output_nodes = [node for key, node in genome.nodes.items() 
                   if key in config.genome_config.output_keys]
    hidden_nodes = [node for key, node in genome.nodes.items() 
                   if key not in config.genome_config.input_keys 
                   and key not in config.genome_config.output_keys]
                   
    print(f"  - Input nodes: {len(input_nodes)}")
    print(f"  - Hidden nodes: {len(hidden_nodes)}")
    print(f"  - Output nodes: {len(output_nodes)}")
    
    # Connection analysis
    all_connections = list(genome.connections.items())
    enabled_connections = [(key, conn) for key, conn in all_connections if conn.enabled]
    
    print(f"Connections: {len(all_connections)} total, {len(enabled_connections)} enabled")
    
    if enabled_connections:
        # Calculate connection stats
        weights = [conn.weight for _, conn in enabled_connections]
        scales = [conn.scale for _, conn in enabled_connections]
        biases = [conn.bias for _, conn in enabled_connections]
        spline_counts = [len(conn.spline_segments) for _, conn in enabled_connections]
        
        print(f"Connection stats:")
        print(f"  - Weight: min={min(weights):.4f}, max={max(weights):.4f}, avg={np.mean(weights):.4f}")
        print(f"  - Scale: min={min(scales):.4f}, max={max(scales):.4f}, avg={np.mean(scales):.4f}")
        print(f"  - Bias: min={min(biases):.4f}, max={max(biases):.4f}, avg={np.mean(biases):.4f}")
        print(f"  - Spline points: min={min(spline_counts)}, max={max(spline_counts)}, avg={np.mean(spline_counts):.1f}")
        
        # Print details of each connection
        print("\nConnection details:")
        for (in_node, out_node), conn in enabled_connections:
            print(f"  {in_node} → {out_node}: weight={conn.weight:.4f}, scale={conn.scale:.4f}, bias={conn.bias:.4f}, segments={len(conn.spline_segments)}")