import numpy as np
import matplotlib.pyplot as plt

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