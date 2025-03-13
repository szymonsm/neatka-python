import matplotlib.pyplot as plt
import numpy as np
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
    plt.figure(figsize=(12, 8))
    
    # Get all connections
    connections = [conn for conn in genome.connections.values() if conn.enabled]
    
    # Calculate grid layout
    n_conn = len(connections)
    cols = min(3, n_conn)
    rows = (n_conn + cols - 1) // cols
    
    # Plot each spline
    for i, conn in enumerate(connections):
        plt.subplot(rows, cols, i+1)
        
        # Get spline points
        points = [(seg.grid_position, seg.value) 
                 for seg in conn.spline_segments.values()]
        points.sort(key=lambda x: x[0])
        
        # Create x values for plotting
        x = np.linspace(config.spline_range_min, config.spline_range_max, 100)
        
        # Create spline function
        spline_func = conn.get_spline_function()
        y = [spline_func(xi) for xi in x]
        
        # Plot the spline
        plt.plot(x, y, label=f"c{conn.key}")
        
        # Plot control points
        if points:
            x_points, y_points = zip(*points)
            plt.scatter(x_points, y_points, c='r', marker='o')
        
        # Add connection info
        plt.title(f"Connection {conn.key}")
        plt.xlabel("Input")
        plt.ylabel("Output")
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(filename)
    
    if view:
        plt.show()

def analyze_kan_genome(genome, config):
    """Print detailed information about a KAN genome."""
    print(f"Genome ID: {genome.key}")
    print(f"Fitness: {genome.fitness}")
    print(f"Nodes: {len(genome.nodes)}")
    print(f"Connections: {len(genome.connections)}")
    
    # Count enabled connections
    enabled_connections = [c for c in genome.connections.values() if c.enabled]
    print(f"Enabled connections: {len(enabled_connections)}")
    
    # Print spline information
    total_segments = 0
    for key, conn in genome.connections.items():
        if conn.enabled:
            n_segments = len(conn.spline_segments)
            total_segments += n_segments
            print(f"  Connection {key}: {n_segments} segments, weight={conn.weight:.3f}, scale={conn.scale:.3f}, bias={conn.bias:.3f}")
    
    print(f"Total spline segments: {total_segments}")