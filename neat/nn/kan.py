# import torch
# import numpy as np
from neat.graphs import feed_forward_layers
# from kan.spline import coef2curve

class KANNetwork(object):
    """A neural network based on the Kolmogorov-Arnold representation theorem."""
    
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)
        
    def activate(self, inputs):
        """Activate the network with the given inputs."""
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError("Expected {0:n} inputs, got {1:n}".format(len(self.input_nodes), len(inputs)))
            
        # Set input values
        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v
            
        # Process each node
        for node, agg_func, bias, response, connections in self.node_evals:
            node_inputs = []
            for input_node, spline_func, weight, scale, bias in connections:
                # Apply spline transformation to input value
                x = self.values[input_node]
                transformed_value = spline_func(x) * scale + bias
                node_inputs.append(transformed_value * weight)
                
            # Apply aggregation function (sum for KAN)
            s = agg_func(node_inputs)
            self.values[node] = response * s + bias
            
        # Return output values
        return [self.values[i] for i in self.output_nodes]
        
    @staticmethod
    def create(genome, config):
        """Create a KANNetwork from a genome and configuration."""
        # Get connections from genome
        connections = [(key, conn) for key, conn in genome.connections.items() if conn.enabled]
        
        # Create layers based on feed forward structure
        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys,
                                   [key for key, _ in connections])
        
        # Process node evaluations
        node_evals = []
        
        for layer in layers:
            for node_key in layer:
                # Find connections to this node
                node_connections = []
                for (in_node, out_node), conn in connections:
                    if out_node == node_key:
                        # Create spline function for this connection
                        spline_segments = sorted(
                            [(seg.grid_position, seg.value) for seg in conn.spline_segments.values()],
                            key=lambda x: x[0]
                        )
                        
                        # Simple linear interpolation function
                        def spline_func(x, segments=spline_segments):
                            # Special cases
                            if not segments:
                                return 0.0
                            if len(segments) == 1:
                                return segments[0][1]
                                
                            # Find position in the grid
                            if x <= segments[0][0]:
                                return segments[0][1]
                            if x >= segments[-1][0]:
                                return segments[-1][1]
                                
                            # Linear interpolation between points
                            for i in range(len(segments)-1):
                                x1, y1 = segments[i]
                                x2, y2 = segments[i+1]
                                
                                if x1 <= x <= x2:
                                    # Linear interpolation: y = y1 + (y2-y1)*(x-x1)/(x2-x1)
                                    t = (x - x1) / (x2 - x1) if x2 != x1 else 0
                                    return y1 + t * (y2 - y1)
                            
                            # Should never reach here
                            return 0.0
                            
                        node_connections.append((in_node, spline_func, conn.weight, conn.scale, conn.bias))
                
                # Get node info
                ng = genome.nodes[node_key]
                agg_func = sum  # KAN uses summation as aggregation
                
                # Add node evaluation
                node_evals.append((node_key, agg_func, ng.bias, ng.response, node_connections))
                
        return KANNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)