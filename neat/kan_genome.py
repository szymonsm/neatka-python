from neat.genome import DefaultGenome, DefaultGenomeConfig
from neat.config import ConfigParameter
from neat.genes import DefaultNodeGene, DefaultConnectionGene
import numpy as np
from random import random, choice, gauss

# KAN-specific gene classes
class KANNodeGene(DefaultNodeGene):
    """Node gene for Kolmogorov-Arnold Networks.
    Represents a summation node in the KAN architecture."""
    
    def __init__(self, key):
        super().__init__(key)
        self.aggregation = 'sum'  # KAN nodes are summation nodes by default

class SplineSegmentGene:
    """Represents a single point in a spline grid."""
    
    def __init__(self, key, value=0.0, grid_position=0.0):
        self.key = key  # This should be (in_node, out_node, position)
        self.value = value
        # Make sure grid_position is a number, not a tuple
        if isinstance(grid_position, tuple):
            self.grid_position = grid_position[2]  # Extract the numeric position
        else:
            self.grid_position = grid_position
        
    def distance(self, other, config):
        return abs(self.value - other.value) / config.genome_config.spline_coefficient_range

class KANConnectionGene(DefaultConnectionGene):
    """Connection gene for Kolmogorov-Arnold Networks.
    Represents a spline-based connection between nodes."""
    
    def __init__(self, key, weight=0.0, enabled=True, scale=1.0, bias=0.0):
        super().__init__(key)
        self.weight = weight
        self.enabled = enabled
        self.scale = scale
        self.bias = bias
        self.spline_segments = {}  # Dictionary mapping grid positions to spline segment genes
        
    def mutate(self, config):
        super().mutate(config)
        
        # Mutate scale
        if random() < config.scale_mutation_rate:
            self.scale += gauss(0, config.scale_mutation_power)
            self.scale = max(min(self.scale, config.scale_max_value), config.scale_min_value)
            
        # Mutate bias
        if random() < config.kan_bias_mutation_rate:
            self.bias += gauss(0, config.kan_bias_mutation_power)
            self.bias = max(min(self.bias, config.kan_bias_max_value), config.kan_bias_min_value)
        
        # Mutate spline segments
        for key, segment in self.spline_segments.items():
            if random() < config.spline_mutation_rate:
                segment.value += gauss(0, config.spline_mutation_power)
                segment.value = max(min(segment.value, config.spline_max_value), 
                                    config.spline_min_value)
                
    def add_spline_segment(self, key, grid_position, value=None):
        """Add a new spline segment to this connection."""
        if value is None:
            value = gauss(0.0, 0.1)
        # Ensure grid_position is a number
        if isinstance(grid_position, tuple):
            grid_position = grid_position[2]  # Extract the numeric position
        self.spline_segments[key] = SplineSegmentGene(key, value, grid_position)

class KANGenomeConfig(DefaultGenomeConfig):
    """Configuration for KANGenome class."""
    
    def __init__(self, params):
        # Ensure node_gene_type and connection_gene_type are set
        if 'node_gene_type' not in params:
            params['node_gene_type'] = KANNodeGene  # Use KANNodeGene by default
            
        if 'connection_gene_type' not in params:
            params['connection_gene_type'] = KANConnectionGene  # Use KANConnectionGene by default

        super().__init__(params)
        
        # Additional KAN-specific parameters
        self._params += [
            ConfigParameter('scale_coefficient_range', float, 5.0),
            ConfigParameter('bias_coefficient_range', float, 5.0),
            ConfigParameter('spline_coefficient_range', float, 5.0),
            ConfigParameter('scale_mutation_rate', float, 0.1),
            ConfigParameter('scale_mutation_power', float, 0.5),
            ConfigParameter('scale_min_value', float, -5.0),
            ConfigParameter('scale_max_value', float, 5.0),
            ConfigParameter('kan_bias_mutation_rate', float, 0.1),  # RENAMED
            ConfigParameter('kan_bias_mutation_power', float, 0.5), # RENAMED
            ConfigParameter('kan_bias_min_value', float, -5.0),     # RENAMED
            ConfigParameter('kan_bias_max_value', float, 5.0),      # RENAMED
            ConfigParameter('spline_mutation_rate', float, 0.2),
            ConfigParameter('spline_mutation_power', float, 0.5),
            ConfigParameter('spline_min_value', float, -5.0),
            ConfigParameter('spline_max_value', float, 5.0),
            ConfigParameter('spline_add_prob', float, 0.1),
            ConfigParameter('spline_delete_prob', float, 0.05),
            ConfigParameter('initial_spline_segments', int, 3),
            ConfigParameter('max_spline_segments', int, 10),
            ConfigParameter('spline_range_min', float, -1.0),
            ConfigParameter('spline_range_max', float, 1.0),
        ]

        for p in self._params:
            setattr(self, p.name, p.interpret(params))

class KANGenome(DefaultGenome):
    """Genome class for Kolmogorov-Arnold Networks."""
    
    @classmethod
    def parse_config(cls, param_dict):
        # Always use the actual class objects, ignore what's in the config file
        param_dict['node_gene_type'] = KANNodeGene
        param_dict['connection_gene_type'] = KANConnectionGene
        return KANGenomeConfig(param_dict)
    
    def __init__(self, key):
        super().__init__(key)
        self.nodes = {}  # Dictionary of node genes (key: node_id, value: KANNodeGene)
        self.connections = {}  # Dictionary of connection genes (key: (in_node_id, out_node_id), value: KANConnectionGene)
        self.spline_segments = {}  # Dictionary mapping connection key to dictionary of spline segments
        
    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""
        super().configure_new(config)
        
        # Configure initial splines for all connections
        for conn_key in list(self.connections.keys()):
            conn = self.connections[conn_key]
            self.spline_segments[conn_key] = {}
            
            # Create initial spline segments
            for i in range(config.initial_spline_segments):
                grid_pos = config.spline_range_min + i * (
                    (config.spline_range_max - config.spline_range_min) / 
                    (config.initial_spline_segments - 1)
                )
                seg_key = (conn_key[0], conn_key[1], grid_pos)
                value = gauss(0.0, 0.1)
                conn.add_spline_segment(seg_key, grid_pos, value)
    
    def configure_crossover(self, parent1, parent2, config):
        """Configure this genome as a crossover of the two parent genomes."""
        super().configure_crossover(parent1, parent2, config)
        
        # Handle spline segments
        for conn_key, conn in self.connections.items():
            self.spline_segments[conn_key] = {}
            
            # Check if this connection exists in either parent
            p1_conn = parent1.connections.get(conn_key)
            p2_conn = parent2.connections.get(conn_key)
            
            if p1_conn and p2_conn:  # Connection in both parents
                # Inherit spline segments from either parent
                for grid_pos in set(list(p1_conn.spline_segments.keys()) + list(p2_conn.spline_segments.keys())):
                    if grid_pos in p1_conn.spline_segments and grid_pos in p2_conn.spline_segments:
                        # Get from either parent
                        seg = choice([p1_conn.spline_segments[grid_pos], p2_conn.spline_segments[grid_pos]])
                        conn.spline_segments[grid_pos] = SplineSegmentGene(
                            (conn_key[0], conn_key[1], grid_pos), 
                            seg.value, 
                            grid_pos
                        )
                    elif grid_pos in p1_conn.spline_segments:
                        seg = p1_conn.spline_segments[grid_pos]
                        conn.spline_segments[grid_pos] = SplineSegmentGene(
                            (conn_key[0], conn_key[1], grid_pos), 
                            seg.value, 
                            grid_pos
                        )
                    else:
                        seg = p2_conn.spline_segments[grid_pos]
                        conn.spline_segments[grid_pos] = SplineSegmentGene(
                            (conn_key[0], conn_key[1], grid_pos), 
                            seg.value, 
                            grid_pos
                        )
            elif p1_conn:  # Connection only in parent1
                for grid_pos, seg in p1_conn.spline_segments.items():
                    conn.spline_segments[grid_pos] = SplineSegmentGene(
                        (conn_key[0], conn_key[1], grid_pos), 
                        seg.value, 
                        grid_pos
                    )
            elif p2_conn:  # Connection only in parent2
                for grid_pos, seg in p2_conn.spline_segments.items():
                    conn.spline_segments[grid_pos] = SplineSegmentGene(
                        (conn_key[0], conn_key[1], grid_pos), 
                        seg.value, 
                        grid_pos
                    )
            else:  # New connection
                # Create initial spline segments
                for i in range(config.initial_spline_segments):
                    grid_pos = config.spline_range_min + i * (
                        (config.spline_range_max - config.spline_range_min) / 
                        (config.initial_spline_segments - 1)
                    )
                    seg_key = (conn_key[0], conn_key[1], grid_pos)
                    value = gauss(0.0, 0.1)
                    conn.add_spline_segment(seg_key, grid_pos, value)
    
    def mutate(self, config):
        """Mutate this genome."""
        super().mutate(config)
        
        # Additional mutations for KAN-specific elements
        for conn_key, conn in self.connections.items():
            # Mutate scale and bias
            if hasattr(config, 'scale_mutation_rate') and random() < config.scale_mutation_rate:
                conn.scale += gauss(0, config.scale_mutation_power)
                conn.scale = max(min(conn.scale, config.scale_max_value), config.scale_min_value)
                
            if hasattr(config, 'kan_bias_mutation_rate') and random() < config.kan_bias_mutation_rate:
                conn.bias += gauss(0, config.kan_bias_mutation_power)
                conn.bias = max(min(conn.bias, config.kan_bias_max_value), config.kan_bias_min_value)
            
            # Mutate existing spline segments
            for grid_pos, segment in list(conn.spline_segments.items()):
                if hasattr(config, 'spline_mutation_rate') and random() < config.spline_mutation_rate:
                    segment.value += gauss(0, config.spline_mutation_power)
                    segment.value = max(min(segment.value, config.spline_max_value), 
                                        config.spline_min_value)
            
            # Add a new spline segment with some probability
            if hasattr(config, 'spline_add_prob') and random() < config.spline_add_prob and len(conn.spline_segments) < config.max_spline_segments:
                # Find a position between existing grid points
                # If grid_position is storing tuples instead of numbers, extract the numeric value
                existing_positions = []
                for seg in conn.spline_segments.values():
                    if isinstance(seg.grid_position, tuple):
                        # If grid_position is a tuple like (in_node, out_node, pos), extract pos
                        existing_positions.append(seg.grid_position[2])  # Assuming position is the 3rd element
                    else:
                        # It's already a number
                        existing_positions.append(seg.grid_position)
                existing_positions = sorted(existing_positions)
                
                if len(existing_positions) <= 1:
                    # If there are 0 or 1 positions, create a random new one
                    new_pos = random() * (config.spline_range_max - config.spline_range_min) + config.spline_range_min
                else:
                    # Choose a random gap between existing positions
                    idx = choice(range(len(existing_positions) - 1))
                    new_pos = (existing_positions[idx] + existing_positions[idx + 1]) / 2
                
                # Add new segment
                new_key = (conn_key[0], conn_key[1], new_pos)
                value = gauss(0.0, 0.1)
                conn.add_spline_segment(new_key, new_pos, value)
            
            # Delete a spline segment with some probability
            if hasattr(config, 'spline_delete_prob') and random() < config.spline_delete_prob and len(conn.spline_segments) > 2:
                # Don't delete all segments - keep at least 2
                to_delete = choice(list(conn.spline_segments.keys()))
                del conn.spline_segments[to_delete]
    
    def distance(self, other, config):
        """Return the genetic distance between this genome and the other."""
        distance = super().distance(other, config)
        
        # Add distance component for spline segments
        if not self.connections or not other.connections:
            return distance
            
        # Get connections that exist in both genomes
        common_connections = set(self.connections.keys()) & set(other.connections.keys())
        if not common_connections:
            return distance
            
        spline_distance = 0.0
        for key in common_connections:
            conn1 = self.connections[key]
            conn2 = other.connections[key]
            
            # CHANGE HERE: Access attributes directly from config, not config.genome_config
            if hasattr(config, 'scale_coefficient_range'):
                # Compare scale and bias
                spline_distance += abs(conn1.scale - conn2.scale) / config.scale_coefficient_range
                spline_distance += abs(conn1.bias - conn2.bias) / config.bias_coefficient_range
            
            # Compare spline segments at matching grid positions
            if hasattr(conn1, 'spline_segments') and hasattr(conn2, 'spline_segments'):
                common_positions = set(seg.grid_position for seg in conn1.spline_segments.values()) & \
                                set(seg.grid_position for seg in conn2.spline_segments.values())
                                
                if common_positions and hasattr(config, 'spline_coefficient_range'):
                    seg_dist = 0.0
                    for pos in common_positions:
                        seg1 = next((seg for seg in conn1.spline_segments.values() if seg.grid_position == pos), None)
                        seg2 = next((seg for seg in conn2.spline_segments.values() if seg.grid_position == pos), None)
                        if seg1 and seg2:
                            seg_dist += abs(seg1.value - seg2.value) / config.spline_coefficient_range
                    
                    if common_positions:  # Avoid division by zero
                        spline_distance += seg_dist / len(common_positions)
            
        # Normalize by number of common connections
        if common_connections:  # Avoid division by zero
            spline_distance /= len(common_connections)
        
        # Add this component to total distance
        distance += spline_distance
        
        return distance