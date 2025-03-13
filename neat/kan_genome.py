import torch
import numpy as np
from neat.genome import DefaultGenome, DefaultGenomeConfig
from neat.config import ConfigParameter
from neat.genes import DefaultNodeGene, DefaultConnectionGene
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from random import gauss, choice

class KANNodeGene(DefaultNodeGene):
    """Node gene for Kolmogorov-Arnold Networks.
    
    Represents a summation node in the KAN architecture with enhanced functionality.
    """
    _gene_attributes = [
        FloatAttribute('bias'),
        FloatAttribute('response'),
        StringAttribute('activation'),
        StringAttribute('aggregation'),
        BoolAttribute('enabled')
    ]
    
    def __init__(self, key):
        super().__init__(key)
        self.aggregation = 'sum'  # KAN nodes are summation nodes by default
        self.enabled = True
        self.scale = 1.0  # Scale factor for node output

class SplineSegmentGene:
    """Represents a single point in a spline grid.
    
    Each segment defines a point in the grid for piecewise polynomial interpolation.
    """
    
    def __init__(self, key, value=0.0, grid_position=0.0):
        self.key = key
        self.value = value
        self.grid_position = grid_position if not isinstance(grid_position, tuple) else grid_position[2]
        
    def distance(self, other, config):
        return abs(self.value - other.value) / config.spline_coefficient_range
        
    def mutate(self, config):
        """Mutate the value of this spline segment."""
        if np.random.random() < config.spline_mutation_rate:
            self.value += np.random.normal(0, config.spline_mutation_power)
            self.value = max(min(self.value, config.spline_max_value), config.spline_min_value)

class KANConnectionGene(DefaultConnectionGene):
    """Connection gene for Kolmogorov-Arnold Networks.
    
    Represents a spline-based connection between nodes with enhanced functionality
    inspired by the KANLayer implementation.
    """
    _gene_attributes = [
        FloatAttribute('weight'),
        BoolAttribute('enabled'),
        FloatAttribute('scale'),
        FloatAttribute('bias'),
    ]
    
    def __init__(self, key, weight=0.0, enabled=True, scale=1.0, bias=0.0):
        super().__init__(key)
        self.weight = weight
        self.enabled = enabled
        self.scale = scale  # Scale factor for spline output (like scale_sp in KANLayer)
        self.bias = bias    # Bias for spline output
        self.spline_segments = {}  # Dictionary mapping grid positions to spline segment genes
        self.grid = None    # Will store the grid for this connection
        
    def mutate(self, config):
        """Mutate this connection gene's attributes."""
        # Standard weight mutation
        if np.random.random() < config.weight_mutate_rate:
            if np.random.random() < config.weight_replace_rate:
                self.weight = np.random.normal(0, config.weight_init_stdev)
            else:
                self.weight += np.random.normal(0, config.weight_mutate_power)
                self.weight = max(min(self.weight, config.weight_max_value), config.weight_min_value)
                
        # Enable/disable mutation
        if np.random.random() < config.enabled_mutate_rate:
            self.enabled = not self.enabled
            
        # KAN-specific mutations
        if np.random.random() < config.scale_mutation_rate:
            self.scale += np.random.normal(0, config.scale_mutation_power)
            self.scale = max(min(self.scale, config.scale_max_value), config.scale_min_value)
            
        if np.random.random() < config.kan_bias_mutation_rate:
            self.bias += np.random.normal(0, config.kan_bias_mutation_power)
            self.bias = max(min(self.bias, config.kan_bias_max_value), config.kan_bias_min_value)
        
        # Mutate spline segments
        for segment in list(self.spline_segments.values()):
            segment.mutate(config)
            
        # Add new spline segment
        if (np.random.random() < config.spline_add_prob and 
            len(self.spline_segments) < config.max_spline_segments):
            self._add_random_spline_segment(config)
            
        # Delete a spline segment
        if (np.random.random() < config.spline_delete_prob and 
            len(self.spline_segments) > config.min_spline_segments):
            to_delete = np.random.choice(list(self.spline_segments.keys()))
            del self.spline_segments[to_delete]
                
    def add_spline_segment(self, key, grid_position, value=None):
        """Add a new spline segment to this connection."""
        if value is None:
            value = np.random.normal(0.0, 0.1)
            
        # Ensure grid_position is a number
        if isinstance(grid_position, tuple):
            grid_position = grid_position[2]
            
        self.spline_segments[key] = SplineSegmentGene(key, value, grid_position)
        
    def _add_random_spline_segment(self, config):
        """Add a new spline segment at a random position."""
        # Extract existing positions
        existing_positions = sorted([seg.grid_position for seg in self.spline_segments.values()])
        
        if len(existing_positions) <= 1:
            # If there are 0 or 1 positions, create a random new one
            new_pos = np.random.uniform(config.spline_range_min, config.spline_range_max)
        else:
            # Choose a random gap between existing positions
            idx = np.random.choice(len(existing_positions) - 1)
            new_pos = (existing_positions[idx] + existing_positions[idx + 1]) / 2
        
        # Add new segment
        in_node, out_node = self.key
        new_key = (in_node, out_node, new_pos)
        self.add_spline_segment(new_key, new_pos)
        
    def get_spline_function(self, implementation='numpy'):
        """Return a callable spline function based on the spline segments.
        
        Args:
            implementation: str, either 'numpy' or 'torch' to determine the implementation
        """
        # Sort grid points by position
        points = sorted([(seg.grid_position, seg.value) 
                         for seg in self.spline_segments.values()])
        
        if implementation == 'torch':
            # Return a PyTorch implementation
            def spline_func(x):
                x_tensor = x if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)
                
                # Special cases
                if len(points) == 0:
                    return torch.zeros_like(x_tensor)
                if len(points) == 1:
                    return torch.ones_like(x_tensor) * points[0][1]
                    
                result = torch.zeros_like(x_tensor)
                
                # Find position in the grid for each point in x
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    
                    # Find points in this segment range
                    mask = (x_tensor >= x1) & (x_tensor <= x2)
                    if not mask.any():
                        continue
                        
                    # Linear interpolation
                    t = (x_tensor[mask] - x1) / (x2 - x1) if x2 != x1 else 0
                    result[mask] = y1 + t * (y2 - y1)
                
                # Handle boundary points
                result[x_tensor < points[0][0]] = points[0][1]
                result[x_tensor > points[-1][0]] = points[-1][1]
                
                return result
                
        else:  # numpy implementation
            def spline_func(x):
                x_array = np.asarray(x)
                
                # Special cases
                if len(points) == 0:
                    return np.zeros_like(x_array)
                if len(points) == 1:
                    return np.ones_like(x_array) * points[0][1]
                    
                result = np.zeros_like(x_array)
                
                # Find position in the grid
                for i in range(len(points) - 1):
                    x1, y1 = points[i]
                    x2, y2 = points[i + 1]
                    
                    # Find points in this segment range
                    mask = (x_array >= x1) & (x_array <= x2)
                    if not np.any(mask):
                        continue
                        
                    # Linear interpolation
                    t = (x_array[mask] - x1) / (x2 - x1) if x2 != x1 else 0
                    result[mask] = y1 + t * (y2 - y1)
                
                # Handle boundary points
                result[x_array < points[0][0]] = points[0][1]
                result[x_array > points[-1][0]] = points[-1][1]
                
                return result
                
        return spline_func
            
    def distance(self, other, config):
        """Return the genetic distance between this connection gene and the other."""
        d = abs(self.weight - other.weight)
        
        # Add distance for scale and bias
        d += abs(self.scale - other.scale) / config.scale_coefficient_range
        d += abs(self.bias - other.bias) / config.bias_coefficient_range
        
        # Add distance for spline segments
        common_grid_positions = set()
        for seg1 in self.spline_segments.values():
            for seg2 in other.spline_segments.values():
                if abs(seg1.grid_position - seg2.grid_position) < 1e-6:
                    common_grid_positions.add(seg1.grid_position)
        
        if common_grid_positions:
            spline_dist = 0.0
            for pos in common_grid_positions:
                seg1 = next((seg for seg in self.spline_segments.values() if abs(seg.grid_position - pos) < 1e-6), None)
                seg2 = next((seg for seg in other.spline_segments.values() if abs(seg.grid_position - pos) < 1e-6), None)
                if seg1 and seg2:
                    spline_dist += abs(seg1.value - seg2.value) / config.spline_coefficient_range
                    
            d += spline_dist / len(common_grid_positions)
        
        return d * config.compatibility_weight_coefficient

class KANGenomeConfig(DefaultGenomeConfig):
    """Configuration for KANGenome class."""
    
    def __init__(self, params):
        # Set node and connection gene types
        if 'node_gene_type' not in params:
            params['node_gene_type'] = KANNodeGene
        if 'connection_gene_type' not in params:
            params['connection_gene_type'] = KANConnectionGene
            
        super().__init__(params)
        
        # Additional KAN-specific parameters
        kan_params = [
            # KAN initialization parameters
            ConfigParameter('scale_init_mean', float, 1.0),
            ConfigParameter('scale_init_stdev', float, 0.1),
            ConfigParameter('scale_replace_rate', float, 0.1),  # ADDED
            ConfigParameter('kan_bias_init_mean', float, 0.0),
            ConfigParameter('kan_bias_init_stdev', float, 0.1),
            ConfigParameter('kan_bias_replace_rate', float, 0.1),  # ADDED
            ConfigParameter('spline_init_mean', float, 0.0),
            ConfigParameter('spline_init_stdev', float, 0.1),
            ConfigParameter('spline_replace_rate', float, 0.1),  # ADDED
            
            # Other KAN parameters
            ConfigParameter('scale_coefficient_range', float, 5.0),
            ConfigParameter('bias_coefficient_range', float, 5.0),
            ConfigParameter('spline_coefficient_range', float, 5.0),
            ConfigParameter('scale_mutation_rate', float, 0.1),
            ConfigParameter('scale_mutation_power', float, 0.5),
            ConfigParameter('scale_min_value', float, -5.0),
            ConfigParameter('scale_max_value', float, 5.0),
            ConfigParameter('kan_bias_mutation_rate', float, 0.1),
            ConfigParameter('kan_bias_mutation_power', float, 0.5),
            ConfigParameter('kan_bias_min_value', float, -5.0),
            ConfigParameter('kan_bias_max_value', float, 5.0),
            ConfigParameter('spline_mutation_rate', float, 0.2),
            ConfigParameter('spline_mutation_power', float, 0.5),
            ConfigParameter('spline_min_value', float, -5.0),
            ConfigParameter('spline_max_value', float, 5.0),
            ConfigParameter('spline_add_prob', float, 0.1),
            ConfigParameter('spline_delete_prob', float, 0.05),
            ConfigParameter('initial_spline_segments', int, 3),
            ConfigParameter('min_spline_segments', int, 2),
            ConfigParameter('max_spline_segments', int, 10),
            ConfigParameter('spline_range_min', float, -1.0),
            ConfigParameter('spline_range_max', float, 1.0),
        ]
        
        self._params += kan_params
        
        # Set attributes directly
        for p in kan_params:
            setattr(self, p.name, p.interpret(params))

class KANGenome(DefaultGenome):
    """Genome class for Kolmogorov-Arnold Networks."""
    
    @classmethod
    def parse_config(cls, param_dict):
        return KANGenomeConfig(param_dict)
    
    def __init__(self, key):
        super().__init__(key)
        self.nodes = {}  # Dictionary of node genes (key: node_id, value: KANNodeGene)
        self.connections = {}  # Dictionary of connection genes
        
    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""
        super().configure_new(config)
        
        # Configure initial splines for all connections
        for conn_key, conn in self.connections.items():
            # Initialize spline segments
            if isinstance(config.initial_spline_segments, int) and config.initial_spline_segments > 0:
                for i in range(config.initial_spline_segments):
                    if config.initial_spline_segments <= 1:
                        grid_pos = 0.0
                    else:
                        grid_pos = config.spline_range_min + i * (
                            (config.spline_range_max - config.spline_range_min) / 
                            (config.initial_spline_segments - 1)
                        )
                    seg_key = (conn_key[0], conn_key[1], grid_pos)
                    value = np.random.normal(0.0, 0.1)
                    conn.add_spline_segment(seg_key, grid_pos, value)
    
    def configure_crossover(self, parent1, parent2, config):
        """Configure this genome as a crossover of the two parent genomes."""
        super().configure_crossover(parent1, parent2, config)
        
        # Handle spline segments
        for conn_key, conn in self.connections.items():
            # Check if this connection exists in either parent
            p1_conn = parent1.connections.get(conn_key)
            p2_conn = parent2.connections.get(conn_key)
            
            if p1_conn and p2_conn:  # Connection in both parents
                # Inherit spline segments from either parent
                p1_spline_segments = getattr(p1_conn, 'spline_segments', {})
                p2_spline_segments = getattr(p2_conn, 'spline_segments', {})
                
                # Get all grid positions from both parents
                all_grid_positions = set()
                for seg in p1_spline_segments.values():
                    all_grid_positions.add(seg.grid_position)
                for seg in p2_spline_segments.values():
                    all_grid_positions.add(seg.grid_position)
                
                # For each grid position, inherit from either parent
                for pos in all_grid_positions:
                    p1_seg = next((seg for seg in p1_spline_segments.values() 
                                  if abs(seg.grid_position - pos) < 1e-6), None)
                    p2_seg = next((seg for seg in p2_spline_segments.values() 
                                  if abs(seg.grid_position - pos) < 1e-6), None)
                    
                    if p1_seg and p2_seg:  # Position in both parents
                        # Randomly choose which parent's segment to inherit
                        chosen_seg = np.random.choice([p1_seg, p2_seg])
                        seg_key = (conn_key[0], conn_key[1], pos)
                        conn.add_spline_segment(seg_key, pos, chosen_seg.value)
                    elif p1_seg:  # Position only in parent 1
                        seg_key = (conn_key[0], conn_key[1], pos)
                        conn.add_spline_segment(seg_key, pos, p1_seg.value)
                    elif p2_seg:  # Position only in parent 2
                        seg_key = (conn_key[0], conn_key[1], pos)
                        conn.add_spline_segment(seg_key, pos, p2_seg.value)
            elif p1_conn:  # Connection only in parent1
                p1_spline_segments = getattr(p1_conn, 'spline_segments', {})
                for seg in p1_spline_segments.values():
                    seg_key = (conn_key[0], conn_key[1], seg.grid_position)
                    conn.add_spline_segment(seg_key, seg.grid_position, seg.value)
            elif p2_conn:  # Connection only in parent2
                p2_spline_segments = getattr(p2_conn, 'spline_segments', {})
                for seg in p2_spline_segments.values():
                    seg_key = (conn_key[0], conn_key[1], seg.grid_position)
                    conn.add_spline_segment(seg_key, seg.grid_position, seg.value)
            else:  # New connection
                # Initialize with default spline segments
                self._initialize_connection_splines(conn, config)
    
    def _initialize_connection_splines(self, conn, config):
        """Initialize spline segments for a connection."""
        for i in range(config.initial_spline_segments):
            if config.initial_spline_segments <= 1:
                grid_pos = 0.0
            else:
                grid_pos = config.spline_range_min + i * (
                    (config.spline_range_max - config.spline_range_min) / 
                    (config.initial_spline_segments - 1)
                )
            seg_key = (conn.key[0], conn.key[1], grid_pos)
            value = np.random.normal(0.0, 0.1)
            conn.add_spline_segment(seg_key, grid_pos, value)
    
    def mutate(self, config):
        """Mutate this connection gene's attributes."""
        # Standard weight mutation
        if np.random.random() < config.weight_mutate_rate:
            if np.random.random() < config.weight_replace_rate:
                self.weight = gauss(0, config.weight_init_stdev)
            else:
                self.weight += gauss(0, config.weight_mutate_power)
                self.weight = max(min(self.weight, config.weight_max_value), 
                                config.weight_min_value)
                
        # Enable/disable mutation
        if np.random.random() < config.enabled_mutate_rate:
            self.enabled = not self.enabled
            
        # KAN-specific mutations - with replacement option
        if hasattr(config, 'scale_mutation_rate') and np.random.random() < config.scale_mutation_rate:
            if hasattr(config, 'scale_replace_rate') and np.random.random() < config.scale_replace_rate:
                # Complete replacement
                self.scale = gauss(config.scale_init_mean, config.scale_init_stdev)
            else:
                # Small mutation
                self.scale += gauss(0, config.scale_mutation_power)
                self.scale = max(min(self.scale, config.scale_max_value), config.scale_min_value)
            
        if hasattr(config, 'kan_bias_mutation_rate') and np.random.random() < config.kan_bias_mutation_rate:
            if hasattr(config, 'kan_bias_replace_rate') and np.random.random() < config.kan_bias_replace_rate:
                # Complete replacement
                self.bias = gauss(config.kan_bias_init_mean, config.kan_bias_init_stdev)
            else:
                # Small mutation
                self.bias += gauss(0, config.kan_bias_mutation_power)
                self.bias = max(min(self.bias, config.kan_bias_max_value), config.kan_bias_min_value)
        
        # Mutate spline segments
        for segment in list(self.spline_segments.values()):
            if np.random.random() < config.spline_mutation_rate:
                if hasattr(config, 'spline_replace_rate') and np.random.random() < config.spline_replace_rate:
                    # Complete replacement
                    segment.value = gauss(config.spline_init_mean, config.spline_init_stdev)
                else:
                    # Small mutation
                    segment.value += gauss(0, config.spline_mutation_power)
                    segment.value = max(min(segment.value, config.spline_max_value), 
                                    config.spline_min_value)
            
        # Add new spline segment
        if (hasattr(config, 'spline_add_prob') and 
            np.random.random() < config.spline_add_prob and 
            len(self.spline_segments) < config.max_spline_segments):
            self._add_random_spline_segment(config)
            
        # Delete a spline segment
        if (hasattr(config, 'spline_delete_prob') and 
            np.random.random() < config.spline_delete_prob and 
            len(self.spline_segments) > getattr(config, 'min_spline_segments', 2)):
            to_delete = choice(list(self.spline_segments.keys()))
            del self.spline_segments[to_delete]
    
    def distance(self, other, config):
        """Return the genetic distance between this genome and the other."""
        distance = super().distance(other, config)
        return distance  # Connection distance already includes spline distance