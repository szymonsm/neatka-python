# import torch
import numpy as np
from neat.graphs import feed_forward_layers

import numpy as np
from scipy.interpolate import make_interp_spline

class SplineFunctionImpl:
    """Implements a smooth B-spline function."""
    
    def __init__(self, control_points, degree=3):
        """Initialize with control points [(x1, y1), (x2, y2), ...] and spline degree."""
        sorted_points = sorted(control_points, key=lambda p: p[0]) if control_points else []
        self.x_points = np.array([p[0] for p in sorted_points])
        self.y_points = np.array([p[1] for p in sorted_points])
        self.degree = min(degree, max(1, len(self.x_points) - 1))  # Adjust degree based on number of points
        self.spline = None
        
        # Create the B-spline if we have enough points
        if len(self.x_points) > 1:
            try:
                # Use scipy's B-spline implementation
                self.spline = make_interp_spline(self.x_points, self.y_points, k=self.degree)
            except Exception:
                # Fall back to linear interpolation if B-spline creation fails
                pass
        
    def __call__(self, x):
        """Evaluate the spline at point x."""
        # Handle empty case
        if not len(self.x_points):
            return 0.0
            
        # Handle single point case
        if len(self.x_points) == 1:
            return self.y_points[0]
            
        # Handle boundary cases
        if x <= self.x_points[0]:
            return self.y_points[0]
        if x >= self.x_points[-1]:
            return self.y_points[-1]
            
        # If we have a spline, use it
        if self.spline is not None:
            try:
                return float(self.spline(x))
            except Exception:
                pass  # Fall back to linear interpolation
                
        # Fall back to linear interpolation
        for i in range(len(self.x_points) - 1):
            if self.x_points[i] <= x <= self.x_points[i+1]:
                # Linear interpolation
                t = (x - self.x_points[i]) / (self.x_points[i+1] - self.x_points[i])
                return (1-t) * self.y_points[i] + t * self.y_points[i+1]
        
        # Fallback (should never reach here)
        return 0.0

class KANNetwork:
    """
    A neural network implementation using Kolmogorov-Arnold Networks.
    Similar to FeedForwardNetwork but using spline functions.
    """
    
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)
        
    def activate(self, inputs):
        """
        Run the network using the provided inputs.
        
        Args:
            inputs: List of input values, must match the number of input nodes
        
        Returns:
            List of output values
        """
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError(f"Expected {len(self.input_nodes)} inputs, got {len(inputs)}")
            
        # Set input values
        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v
            
        # Process nodes in topological order
        for node, node_inputs in self.node_evals:
            node_sum = 0.0
            
            # Apply spline functions to inputs and sum the results
            for input_node, spline_func, weight_s, weight_b in node_inputs:
                x = self.values[input_node]
                # Apply the spline transformation
                # y = spline_func(x)
                # Apply scaling, bias, and weight
                transformed_value = weight_b * (x / (1 + np.exp(-x))) + weight_s * spline_func(x)
                # transformed_value = weight * (scale * y + bias)
                node_sum += transformed_value
                
            # Store the summed value
            self.values[node] = node_sum
            
        # Return output values
        return [self.values[i] for i in self.output_nodes]
        
    @staticmethod
    def create(genome, config):
        """Create a KANNetwork from a genome and configuration."""
        # Get enabled connections from the genome
        connections = [(key, conn) for key, conn in genome.connections.items() if conn.enabled]
        
        # Create layers based on feed forward structure
        connection_keys = [conn[0] for conn in connections]
        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connection_keys)
        
        # Process node evaluations
        node_evals = []
        
        for layer in layers:
            for node in layer:
                # Find all connections that connect to this node
                node_inputs = []
                
                for (in_node, out_node), conn in connections:
                    if out_node != node:
                        continue
                        
                    # Create a spline function for this connection
                    control_points = []
                    if hasattr(conn, 'spline_segments') and conn.spline_segments:
                        control_points = [(seg.grid_position, seg.value) 
                                         for seg in conn.spline_segments.values()]
                        
                    # Ensure at least 2 control points for interpolation
                    if len(control_points) < 2:
                        # Create default control points (identity function) if needed
                        control_points = [(-1.0, -1.0), (1.0, 1.0)]
                                        
                    spline_func = SplineFunctionImpl(control_points)
                    
                    # Add connection info to node inputs
                    node_inputs.append((in_node, spline_func, conn.weight_s, conn.weight_b))
                
                # Add node evaluation info
                node_evals.append((node, node_inputs))
                
        return KANNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)