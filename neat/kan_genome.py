import torch
import random
import numpy as np
from neat.genome import DefaultGenome, DefaultGenomeConfig
from neat.config import ConfigParameter
from neat.genes import DefaultNodeGene, DefaultConnectionGene, BaseGene
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.nn.kan import SplineFunctionImpl

class KANNodeGene(DefaultNodeGene):
    """Node gene for Kolmogorov-Arnold Networks.
    
    KAN nodes sum their inputs like standard NEAT nodes.
    """
    _gene_attributes = [FloatAttribute('bias'),
                       FloatAttribute('response'),
                       StringAttribute('activation', options=''),
                       StringAttribute('aggregation', options='')]
                       
    def __init__(self, key):
        super().__init__(key)
        self.aggregation = 'sum'  # KAN nodes are summation nodes by default

class SplineSegmentGene(BaseGene):
    """Represents a single point in a spline grid."""
    _gene_attributes = [FloatAttribute('value'),
                       FloatAttribute('grid_position')]
    
    def __init__(self, key, value=0.0, grid_position=0.0):
        super().__init__(key)
        self.value = value
        self.grid_position = grid_position
    
    def distance(self, other, config):
        """Calculate genetic distance between spline segments."""
        return abs(self.value - other.value) / config.spline_coefficient_range
    
    def crossover(self, gene2):
        """Crosses over with another gene."""
        # Randomly select from either parent
        if random.random() > 0.5:
            return SplineSegmentGene(self.key, self.value, self.grid_position)
        else:
            return SplineSegmentGene(self.key, gene2.value, gene2.grid_position)
            
    def __str__(self):
        return f"SplineSegmentGene(key={self.key}, grid_pos={self.grid_position:.3f}, value={self.value:.3f})"
        
    def __lt__(self, other):
        """Enable sorting based on grid position."""
        return self.grid_position < other.grid_position

class KANConnectionGene(DefaultConnectionGene):
    """Connection gene for Kolmogorov-Arnold Networks.
    
    Represents a spline-based connection between nodes.
    """
    _gene_attributes = [FloatAttribute('weight'),
                       BoolAttribute('enabled'),
                       FloatAttribute('scale'),
                       FloatAttribute('bias')]
    
    def __init__(self, key, weight=0.0, enabled=True, scale=1.0, bias=0.0):
        super().__init__(key)
        self.weight = weight
        self.enabled = enabled
        self.scale = scale
        self.bias = bias
        self.spline_segments = {}  # Dictionary mapping segment keys to SplineSegmentGenes
        
    def mutate(self, config):
        """Mutate this connection gene's attributes."""
        # Call the parent class's mutate method for weight and enabled
        super().mutate(config)
            
        # KAN-specific mutations
        # Mutate scale
        if random.random() < config.scale_mutation_rate:
            if random.random() < config.scale_replace_rate:
                # Complete replacement
                self.scale = random.gauss(config.scale_init_mean, config.scale_init_stdev)
            else:
                # Small mutation
                self.scale += random.gauss(0, config.scale_mutation_power)
                self.scale = max(min(self.scale, config.scale_max_value), config.scale_min_value)
            
        # Mutate bias
        if random.random() < config.kan_bias_mutation_rate:
            if random.random() < config.kan_bias_replace_rate:
                # Complete replacement
                self.bias = random.gauss(config.kan_bias_init_mean, config.kan_bias_init_stdev)
            else:
                # Small mutation
                self.bias += random.gauss(0, config.kan_bias_mutation_power)
                self.bias = max(min(self.bias, config.kan_bias_max_value), config.kan_bias_min_value)
        
        # Mutate spline segments
        for segment in list(self.spline_segments.values()):
            if random.random() < config.spline_mutation_rate:
                if random.random() < config.spline_replace_rate:
                    # Complete replacement
                    segment.value = random.gauss(config.spline_init_mean, config.spline_init_stdev)
                else:
                    # Small mutation
                    segment.value += random.gauss(0, config.spline_mutation_power)
                    segment.value = max(min(segment.value, config.spline_max_value), config.spline_min_value)
        
        # Add a new spline segment with some probability
        if (random.random() < config.spline_add_prob and 
            len(self.spline_segments) < config.max_spline_segments):
            self._add_random_spline_segment(config)
            
        # Delete a spline segment with some probability
        if (random.random() < config.spline_delete_prob and 
            len(self.spline_segments) > config.min_spline_segments):
            
            # Don't delete all segments - keep at least the minimum required
            keys_list = list(self.spline_segments.keys())
            if keys_list:
                to_delete = random.choice(keys_list)
                del self.spline_segments[to_delete]
                
    def _add_random_spline_segment(self, config):
        """Add a new spline segment at a position between existing segments."""
        # Get existing grid positions
        existing_positions = sorted([seg.grid_position for seg in self.spline_segments.values()])
        
        # Choose a position for the new segment
        if not existing_positions:
            # If no segments exist, create initial ones
            grid_min = getattr(config, 'spline_range_min', -1.0)
            grid_max = getattr(config, 'spline_range_max', 1.0)
            new_pos = random.uniform(grid_min, grid_max)
        elif len(existing_positions) == 1:
            # Only one position exists, add to either side
            grid_min = getattr(config, 'spline_range_min', -1.0)
            grid_max = getattr(config, 'spline_range_max', 1.0)
            
            if random.random() > 0.5 and existing_positions[0] > grid_min:
                # Add to the left
                new_pos = random.uniform(grid_min, existing_positions[0])
            else:
                # Add to the right
                new_pos = random.uniform(existing_positions[0], grid_max)
        else:
            # Choose a random gap between positions
            idx = random.randint(0, len(existing_positions) - 2)
            new_pos = random.uniform(existing_positions[idx], existing_positions[idx+1])
            
        # Create a key for the new segment
        seg_key = (*self.key, new_pos)  # (in_node, out_node, grid_pos)
        
        # Add the new segment
        value = random.gauss(0.0, config.spline_init_stdev)
        self.spline_segments[seg_key] = SplineSegmentGene(seg_key, value, new_pos)
    
    def add_spline_segment(self, key, grid_position, value=None):
        """Add a new spline segment to this connection."""
        if value is None:
            value = random.uniform(-0.1, 0.1)  # Default to small random value
        self.spline_segments[key] = SplineSegmentGene(key, value, grid_position)
        
    def get_spline_function(self):
        """Return a callable spline function for this connection."""
        from neat.nn.kan import SplineFunctionImpl
        
        # Extract control points
        control_points = []
        for seg in self.spline_segments.values():
            control_points.append((seg.grid_position, seg.value))
            
        # If we have no points, add two default points
        if not control_points:
            control_points = [(-1.0, 0.0), (1.0, 0.0)]
            
        return SplineFunctionImpl(control_points)
    
    def distance(self, other, config):
        """Return the genetic distance between this connection gene and the other."""
        # Start with the standard connection distance (weight and enabled)
        d = abs(self.weight - other.weight) / config.weight_max_value
        if self.enabled != other.enabled:
            d += 1.0
            
        # Add distance for KAN-specific attributes
        d += abs(self.scale - other.scale) / config.scale_coefficient_range
        d += abs(self.bias - other.bias) / config.bias_coefficient_range
        
        # Calculate spline segment distance
        spline_dist = 0.0
        spline_count = 0
        
        # Find common grid positions and calculate distance
        all_positions = set()
        for seg in self.spline_segments.values():
            all_positions.add(seg.grid_position)
        for seg in other.spline_segments.values():
            all_positions.add(seg.grid_position)
            
        for pos in all_positions:
            # Find segments at this position in both genes, if they exist
            seg1 = next((seg for seg in self.spline_segments.values() if abs(seg.grid_position - pos) < 1e-6), None)
            seg2 = next((seg for seg in other.spline_segments.values() if abs(seg.grid_position - pos) < 1e-6), None)
            
            if seg1 and seg2:
                # Both genes have a segment at this position
                spline_dist += abs(seg1.value - seg2.value) / config.spline_coefficient_range
            else:
                # Only one gene has a segment at this position
                spline_dist += 1.0  # Maximum distance
                
            spline_count += 1
            
        # Add normalized spline distance component if there were any splines
        if spline_count > 0:
            d += spline_dist / spline_count
            
        return d * config.compatibility_weight_coefficient
    
    def copy(self):
        """Return a copy of this gene with copied spline segments."""
        new_gene = KANConnectionGene(self.key, self.weight, self.enabled, self.scale, self.bias)
        
        # Copy spline segments
        for key, seg in self.spline_segments.items():
            new_gene.spline_segments[key] = SplineSegmentGene(seg.key, seg.value, seg.grid_position)
            
        return new_gene
    
    def crossover(self, gene2):
        """Perform crossover with another gene."""
        # Create a new gene with attributes from the fitter parent (self)
        new_gene = self.copy()
        
        # Handle spline segments from both parents
        for key, seg2 in gene2.spline_segments.items():
            grid_pos = seg2.grid_position
            
            # Find if there's a matching segment in the first parent
            seg1 = next((seg for seg in self.spline_segments.values() 
                       if abs(seg.grid_position - grid_pos) < 1e-6), None)
            
            if seg1:
                # Both parents have a segment at this position, do crossover
                if random.random() > 0.5:
                    # Keep segment from first parent (already copied)
                    pass
                else:
                    # Use segment from second parent
                    new_key = (*self.key, grid_pos)
                    new_gene.spline_segments[new_key] = SplineSegmentGene(
                        new_key, seg2.value, grid_pos)
            else:
                # Only second parent has a segment at this position
                # Inherit with 50% probability
                if random.random() > 0.5:
                    new_key = (*self.key, grid_pos)
                    new_gene.spline_segments[new_key] = SplineSegmentGene(
                        new_key, seg2.value, grid_pos)
        
        return new_gene

class KANGenomeConfig(DefaultGenomeConfig):
    """Configuration for KANGenome class."""
    
    def __init__(self, params):
        # Set node and connection gene types
        params['node_gene_type'] = KANNodeGene
        params['connection_gene_type'] = KANConnectionGene
            
        # Initialize the parent class
        super().__init__(params)
        
        # Additional KAN-specific parameters
        kan_params = [
            # KAN initialization parameters
            ConfigParameter('scale_init_mean', float, 1.0),
            ConfigParameter('scale_init_stdev', float, 0.1),
            ConfigParameter('scale_replace_rate', float, 0.1),
            ConfigParameter('kan_bias_init_mean', float, 0.0),
            ConfigParameter('kan_bias_init_stdev', float, 0.1),
            ConfigParameter('kan_bias_replace_rate', float, 0.1),
            ConfigParameter('spline_init_mean', float, 0.0),
            ConfigParameter('spline_init_stdev', float, 0.1),
            ConfigParameter('spline_replace_rate', float, 0.1),
            
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
        """Parse the configuration dictionary for a KANGenome."""
        return KANGenomeConfig(param_dict)
    
    def __init__(self, key):
        """Initialize a new KANGenome with the given key."""
        super().__init__(key)
    
    def configure_new(self, config):
        """Configure a new genome based on the given configuration."""
        super().configure_new(config)
        
        # Configure initial splines for all connections
        for conn_key, conn in self.connections.items():
            # Initialize scale and bias with configured values
            conn.scale = random.gauss(config.scale_init_mean, config.scale_init_stdev)
            conn.bias = random.gauss(config.kan_bias_init_mean, config.kan_bias_init_stdev)
            
            # Create initial spline segments
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
                
            # Create a key for this segment
            seg_key = (conn.key[0], conn.key[1], grid_pos)
            
            # Initialize with small random value
            value = random.gauss(config.spline_init_mean, config.spline_init_stdev)
            conn.add_spline_segment(seg_key, grid_pos, value)
    
    def configure_crossover(self, parent1, parent2, config):
        """Configure this genome as a crossover of the two parent genomes."""
        super().configure_crossover(parent1, parent2, config)
        
        # Handle KAN-specific attributes for connections
        for conn_key, conn in self.connections.items():
            # Check if this connection exists in either parent
            p1_conn = parent1.connections.get(conn_key)
            p2_conn = parent2.connections.get(conn_key)
            
            if p1_conn and p2_conn:
                # Connection exists in both parents
                # Inherit scale and bias from the fitter parent (or random)
                if parent1.fitness > parent2.fitness:
                    conn.scale = p1_conn.scale
                    conn.bias = p1_conn.bias
                elif parent2.fitness > parent1.fitness:
                    conn.scale = p2_conn.scale
                    conn.bias = p2_conn.bias
                else:
                    # Equal fitness, choose randomly
                    if random.random() > 0.5:
                        conn.scale = p1_conn.scale
                        conn.bias = p1_conn.bias
                    else:
                        conn.scale = p2_conn.scale
                        conn.bias = p2_conn.bias
                
                # Perform crossover of spline segments
                p1_spline_segments = getattr(p1_conn, 'spline_segments', {})
                p2_spline_segments = getattr(p2_conn, 'spline_segments', {})
                
                # Get all grid positions from both parents
                all_positions = set()
                for seg in p1_spline_segments.values():
                    all_positions.add(seg.grid_position)
                for seg in p2_spline_segments.values():
                    all_positions.add(seg.grid_position)
                
                # For each position, perform crossover
                for pos in all_positions:
                    seg1 = next((seg for seg in p1_spline_segments.values() 
                               if abs(seg.grid_position - pos) < 1e-6), None)
                    seg2 = next((seg for seg in p2_spline_segments.values() 
                               if abs(seg.grid_position - pos) < 1e-6), None)
                    
                    if seg1 and seg2:
                        # Both parents have a segment at this position
                        if parent1.fitness > parent2.fitness:
                            # Take segment from fitter parent
                            seg_key = (conn.key[0], conn.key[1], pos)
                            conn.add_spline_segment(seg_key, pos, seg1.value)
                        elif parent2.fitness > parent1.fitness:
                            # Take segment from fitter parent
                            seg_key = (conn.key[0], conn.key[1], pos)
                            conn.add_spline_segment(seg_key, pos, seg2.value)
                        else:
                            # Equal fitness, choose randomly
                            if random.random() > 0.5:
                                seg_key = (conn.key[0], conn.key[1], pos)
                                conn.add_spline_segment(seg_key, pos, seg1.value)
                            else:
                                seg_key = (conn.key[0], conn.key[1], pos)
                                conn.add_spline_segment(seg_key, pos, seg2.value)
                    elif seg1:
                        # Only first parent has this position
                        if random.random() > 0.5:  # 50% chance to inherit
                            seg_key = (conn.key[0], conn.key[1], pos)
                            conn.add_spline_segment(seg_key, pos, seg1.value)
                    elif seg2:
                        # Only second parent has this position
                        if random.random() > 0.5:  # 50% chance to inherit
                            seg_key = (conn.key[0], conn.key[1], pos)
                            conn.add_spline_segment(seg_key, pos, seg2.value)
                
            elif p1_conn:
                # Only first parent has this connection
                conn.scale = p1_conn.scale
                conn.bias = p1_conn.bias
                
                # Inherit spline segments
                p1_spline_segments = getattr(p1_conn, 'spline_segments', {})
                for seg in p1_spline_segments.values():
                    seg_key = (conn_key[0], conn_key[1], seg.grid_position)
                    conn.add_spline_segment(seg_key, seg.grid_position, seg.value)
                    
            elif p2_conn:
                # Only second parent has this connection
                conn.scale = p2_conn.scale
                conn.bias = p2_conn.bias
                
                # Inherit spline segments
                p2_spline_segments = getattr(p2_conn, 'spline_segments', {})
                for seg in p2_spline_segments.values():
                    seg_key = (conn_key[0], conn_key[1], seg.grid_position)
                    conn.add_spline_segment(seg_key, seg.grid_position, seg.value)
                    
            else:  # New connection
                # Initialize with default spline segments
                self._initialize_connection_splines(conn, config)
    
    def mutate(self, config):
        """Mutate this genome."""
        # Call the parent class mutate method, which will also call mutate on all connections
        super().mutate(config)
        # KANConnectionGene.mutate already handles spline mutations
    
    def distance(self, other, config):
        """Return the genetic distance between this genome and the other."""
        distance = super().distance(other, config)
        return distance  # Connection distance already includes spline distance