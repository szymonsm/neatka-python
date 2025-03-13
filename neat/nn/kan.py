import torch
import numpy as np
from neat.graphs import feed_forward_layers

class KANNetwork:
    """A neural network based on the Kolmogorov-Arnold representation theorem.
    
    This implementation is inspired by the KANLayer and MultKAN implementations,
    providing enhanced functionality and performance.
    """
    
    def __init__(self, inputs, outputs, node_evals, use_torch=False):
        """Initialize the KAN network.
        
        Args:
            inputs: List of input keys
            outputs: List of output keys
            node_evals: List of (node_key, activation_function, aggregation_function,
                               bias, response, connections)
            use_torch: Whether to use torch for computation instead of numpy
        """
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = {}  # Maps node keys to their output values
        self.use_torch = use_torch
        
        # Create zeroed arrays for the node activations
        for k in inputs + outputs:
            self.values[k] = 0.0
            
        # Store intermediate activations for inspection if needed
        self.activations = {}
        self.spline_outputs = {}
        
    def activate(self, inputs):
        """Activate the network with the given inputs."""
        if len(self.input_nodes) != len(inputs):
            raise RuntimeError(f"Expected {len(self.input_nodes)} inputs, got {len(inputs)}")
            
        # Convert inputs to tensor if using torch
        if self.use_torch:
            inputs = torch.tensor(inputs, dtype=torch.float32)
            
        # Set input values
        for k, v in zip(self.input_nodes, inputs):
            self.values[k] = v
            
        # Process each node in the network
        for node, act_func, agg_func, bias, response, connections in self.node_evals:
            node_inputs = []
            for input_node, spline_func, weight, scale, conn_bias in connections:
                # Get the input value
                x = self.values[input_node]
                
                # Apply spline transformation
                spline_output = spline_func(x)
                
                # Apply scale and bias to spline output
                transformed_value = spline_output * scale + conn_bias
                
                # Apply connection weight
                weighted_value = transformed_value * weight
                
                # Store for potential later analysis
                key = (input_node, node)
                self.spline_outputs[key] = spline_output
                
                # Add to node inputs
                node_inputs.append(weighted_value)
                
            # Apply aggregation function (sum for KAN)
            s = agg_func(node_inputs)
            
            # Apply bias and response
            self.values[node] = act_func(bias + response * s)
            
            # Store node activation
            self.activations[node] = self.values[node]
            
        # Return output values
        return [self.values[i] for i in self.output_nodes]
        
    def update_grid_from_samples(self, x, grid_eps=0.02):
        """Update spline grids based on samples to better fit the data distribution.
        
        Args:
            x: Input samples
            grid_eps: Controls interpolation between uniform and adaptive grid
        """
        if self.use_torch and not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            
        # Process each node and update its connections' grids
        for node, _, _, _, _, connections in self.node_evals:
            for i, (input_node, spline_func, _, _, _) in enumerate(connections):
                # Get input values for this connection
                if input_node in self.input_nodes:
                    # For input nodes, use the corresponding column from x
                    idx = self.input_nodes.index(input_node)
                    if self.use_torch:
                        input_values = x[:, idx]
                    else:
                        input_values = x[:, idx]
                else:
                    # For hidden nodes, we need to compute their values
                    # This requires activating the network for each sample
                    input_values = []
                    for sample in x:
                        self.activate(sample)
                        input_values.append(self.values[input_node])
                    
                    if self.use_torch:
                        input_values = torch.tensor(input_values, dtype=torch.float32)
                    else:
                        input_values = np.array(input_values)
                
                # Sort input values
                if self.use_torch:
                    sorted_values, _ = torch.sort(input_values)
                else:
                    sorted_values = np.sort(input_values)
                
                # Update the grid based on input distribution
                # This is inspired by KANLayer's update_grid_from_samples method
                # but simplified for our use case
                num_segments = len(getattr(spline_func, 'points', []))
                if num_segments <= 1:
                    continue
                    
                # Create new grid points based on quantiles of the data
                if self.use_torch:
                    quantiles = torch.linspace(0, 1, num_segments)
                    grid_positions = torch.quantile(sorted_values, quantiles)
                else:
                    quantiles = np.linspace(0, 1, num_segments)
                    grid_positions = np.quantile(sorted_values, quantiles)
                
                # Mix with uniform grid based on grid_eps
                if self.use_torch:
                    min_val = sorted_values.min().item()
                    max_val = sorted_values.max().item()
                    uniform_grid = torch.linspace(min_val, max_val, num_segments)
                    new_grid = grid_eps * uniform_grid + (1 - grid_eps) * grid_positions
                else:
                    min_val = sorted_values.min()
                    max_val = sorted_values.max()
                    uniform_grid = np.linspace(min_val, max_val, num_segments)
                    new_grid = grid_eps * uniform_grid + (1 - grid_eps) * grid_positions
                
                # Update the spline function's grid points
                # Note: This would require modifying the spline function implementation
    
    @staticmethod
    def create(genome, config):
        """Create a KANNetwork from a genome and configuration."""
        # Get connections from genome
        connections = [(key, conn) for key, conn in genome.connections.items() if conn.enabled]
        
        # Create layers based on feed forward structure
        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys,
                                   [key for key, _ in connections])
        
        # Determine if we should use torch based on config
        use_torch = getattr(config, 'use_torch', False)
        
        # Process node evaluations
        node_evals = []
        
        for layer in layers:
            for node_key in layer:
                # Find connections to this node
                node_connections = []
                for (in_node, out_node), conn in connections:
                    if out_node == node_key:
                        # Create spline function for this connection
                        spline_func = conn.get_spline_function(
                            implementation='torch' if use_torch else 'numpy'
                        )
                        
                        node_connections.append((in_node, spline_func, conn.weight, conn.scale, conn.bias))
                
                # Get node info
                ng = genome.nodes[node_key]
                act_func = config.genome_config.activation_defs.get(ng.activation)
                agg_func = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                
                # Add node evaluation
                node_evals.append((node_key, act_func, agg_func, ng.bias, ng.response, node_connections))
                
        return KANNetwork(config.genome_config.input_keys, config.genome_config.output_keys, 
                         node_evals, use_torch=use_torch)