import random
import numpy as np
from neat.genome import DefaultGenome, DefaultGenomeConfig
from neat.config import ConfigParameter
from neat.genes import DefaultNodeGene, DefaultConnectionGene, BaseGene
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute
from neat.nn.kan import SplineFunctionImpl
from neat.graphs import creates_cycle

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
    
    def crossover(self, gene2, config):
        """Crosses over with another gene."""
        # Randomly select from either parent
        if random.random() > config.kan_segment_crossover_better_fitness_rate:
            return SplineSegmentGene(self.key, self.value, self.grid_position)
        else:
            return SplineSegmentGene(self.key, gene2.value, gene2.grid_position)
        
    def mutate(self, config):
        if random.random() < config.spline_mutate_rate:
            if random.random() < config.spline_replace_rate:
                # Complete replacement
                self.value = random.gauss(config.spline_init_mean, config.spline_init_stdev)
            else:
                # Small mutation
                self.value += random.gauss(0, config.spline_mutate_power)
                self.value = max(min(self.value, config.spline_max_value), config.spline_min_value)
            
    def __str__(self):
        return f"SplineSegmentGene(key={self.key}, grid_pos={self.grid_position:.3f}, value={self.value:.3f})"
        
    def __lt__(self, other):
        """Enable sorting based on grid position."""
        return self.grid_position < other.grid_position

class KANConnectionGene(DefaultConnectionGene):
    """Connection gene for Kolmogorov-Arnold Networks.
    
    Represents a spline-based connection between nodes.
    """
    _gene_attributes = [BoolAttribute('enabled'),
                       FloatAttribute('weight_s'),
                       FloatAttribute('weight_b')]
    
    def __init__(self, key, enabled=True, weight_s=1.0, weight_b=1.0):
        super().__init__(key)
        self.enabled = enabled
        self.weight_s = weight_s
        self.weight_b = weight_b
        self.spline_segments = {}  # Dictionary mapping segment keys to SplineSegmentGenes
        
    def mutate(self, config):
        """Mutate this connection gene's attributes."""
        # Call the parent class's mutate method for weight and enabled
        super().mutate(config)
            
        # KAN-specific mutations
        # Mutate weight for spline
        if random.random() < config.weight_s_mutate_rate:
            if random.random() < config.weight_s_replace_rate:
                # Complete replacement
                self.weight_s = random.gauss(config.weight_s_init_mean, config.weight_s_init_stdev)
            else:
                # Small mutation
                self.weight_s += random.gauss(0, config.weight_s_mutate_power)
                self.weight_s = max(min(self.weight_s, config.weight_s_max_value), config.weight_s_min_value)

        # Mutate weight for a basis function
        if random.random() < config.weight_b_mutate_rate:
            if random.random() < config.weight_b_replace_rate:
                # Complete replacement
                self.weight_b = random.gauss(config.weight_b_init_mean, config.weight_b_init_stdev)
            else:
                # Small mutation
                self.weight_b += random.gauss(0, config.weight_b_mutate_power)
                self.weight_b = max(min(self.weight_b, config.weight_b_max_value), config.weight_b_min_value)
        
        # Mutate spline segments
        for segment in list(self.spline_segments.values()):
            segment.mutate(config)
        
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
        d = 0.0
        if self.enabled != other.enabled:
            d += 1.0
            
        # Add distance for KAN-specific attributes
        d += abs(self.weight_s - other.weight_s) / config.weight_s_coefficient_range
        d += abs(self.weight_b - other.weight_b) / config.weight_b_coefficient_range
        
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
        new_gene = KANConnectionGene(self.key, self.enabled, self.weight_s, self.weight_b)
        
        # Copy spline segments
        for key, seg in self.spline_segments.items():
            new_gene.spline_segments[key] = SplineSegmentGene(seg.key, seg.value, seg.grid_position)
            
        return new_gene
    
    def crossover(self, gene2, config):
        """Perform crossover with another gene."""
        # Create a new gene with attributes from the fitter parent (self)
        new_gene = self.copy()
        
        # Handle spline segments from both parents
        for key, seg2 in gene2.spline_segments.items():
            grid_pos = seg2.grid_position
            
            # Find if there's a matching segment in the first parent
            seg1 = next((seg for seg in self.spline_segments.values() 
                       if abs(seg.grid_position - grid_pos) < config.kan_segments_distance_treshold), None)
            
            if seg1:
                new_seg = seg1.crossover(seg2, config)
                new_gene.spline_segments[new_seg.key] = new_seg
            else:
                if random.random() > config.kan_connection_crossover_add_segment_rate:
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
            ConfigParameter('weight_s_init_mean', float, 1.0),
            ConfigParameter('weight_s_init_stdev', float, 0.1),
            ConfigParameter('weight_s_replace_rate', float, 0.1),
            ConfigParameter('weight_b_init_mean', float, 0.0),
            ConfigParameter('weight_b_init_stdev', float, 0.1),
            ConfigParameter('weight_b_replace_rate', float, 0.1),
            ConfigParameter('spline_init_mean', float, 0.0),
            ConfigParameter('spline_init_stdev', float, 0.1),
            ConfigParameter('spline_replace_rate', float, 0.1),
            
            # Other KAN parameters
            ConfigParameter('weight_s_coefficient_range', float, 5.0),
            ConfigParameter('weight_b_coefficient_range', float, 5.0),
            ConfigParameter('spline_coefficient_range', float, 5.0),
            ConfigParameter('weight_s_mutate_rate', float, 0.1),
            ConfigParameter('weight_s_mutate_power', float, 0.5),
            ConfigParameter('weight_s_min_value', float, -5.0),
            ConfigParameter('weight_s_max_value', float, 5.0),
            ConfigParameter('weight_b_mutate_rate', float, 0.1),
            ConfigParameter('weight_b_mutate_power', float, 0.5),
            ConfigParameter('weight_b_min_value', float, -5.0),
            ConfigParameter('weight_b_max_value', float, 5.0),
            ConfigParameter('spline_mutate_rate', float, 0.2),
            ConfigParameter('spline_mutate_power', float, 0.5),
            ConfigParameter('spline_min_value', float, -5.0),
            ConfigParameter('spline_max_value', float, 5.0),
            ConfigParameter('spline_add_prob', float, 0.1),
            ConfigParameter('spline_delete_prob', float, 0.05),
            ConfigParameter('initial_spline_segments', int, 3),
            ConfigParameter('min_spline_segments', int, 2),
            ConfigParameter('max_spline_segments', int, 10),
            ConfigParameter('spline_range_min', float, -1.0),
            ConfigParameter('spline_range_max', float, 1.0),
            ConfigParameter('kan_segments_distance_treshold', float, 1e-3),
            ConfigParameter('kan_connection_crossover_add_segment_rate', float, 0.5),
            ConfigParameter('kan_segment_crossover_better_fitness_rate', float, 0.75) # 75% chance to inherit from fitter parent
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
            conn.weight_s = random.gauss(config.weight_s_init_mean, config.weight_s_init_stdev)
            conn.weight_b = random.gauss(config.weight_b_init_mean, config.weight_b_init_stdev)
            
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
    
    @staticmethod
    def create_connection(config, input_id, output_id):
        """
        Creates a new connection gene between the specified nodes.
        Initialize with KAN-specific attributes.
        """
        connection = config.connection_gene_type((input_id, output_id))
        connection.init_attributes(config)
        
        # Explicitly initialize KAN-specific attributes
        connection.weight_s = random.gauss(config.weight_s_init_mean, config.weight_s_init_stdev)
        connection.weight_b = random.gauss(config.weight_b_init_mean, config.weight_b_init_stdev)
        
        return connection

    def mutate_add_node(self, config):
        """
        Adds a new node by splitting an existing connection.
        Uses weight_s and weight_b instead of weight.
        """
        if not self.connections:
            if config.check_structural_mutation_surer():
                self.mutate_add_connection(config)
            return

        # Choose a random connection to split
        conn_to_split = random.choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(config, new_node_id)
        self.nodes[new_node_id] = ng

        # Disable this connection and create two new connections joining its nodes via
        # the given node.
        conn_to_split.enabled = False

        i, o = conn_to_split.key
        
        # Create input connection to the new node with default weights
        self.add_connection(config, i, new_node_id, 1.0, 1.0, True)
        
        # Create output connection from the new node with the same weights as the split connection
        self.add_connection(config, new_node_id, o, conn_to_split.weight_s, conn_to_split.weight_b, True)

    def add_connection(self, config, input_key, output_key, weight_s=1.0, weight_b=0.0, enabled=True):
        """
        Adds a new connection between specified nodes with given weight_s and weight_b values.
        """
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)
        
        key = (input_key, output_key)
        connection = config.connection_gene_type(key)
        connection.init_attributes(config)
        
        # Set the weights explicitly
        connection.weight_s = weight_s
        connection.weight_b = weight_b
        connection.enabled = enabled
        
        # Initialize spline segments for the new connection
        self.connections[key] = connection
        self._initialize_connection_splines(connection, config)
        
        return connection

    def mutate_add_connection(self, config):
        """
        Attempt to add a new connection, using weight_s and weight_b instead of weight.
        The only restriction is that the output node cannot be one of the network input pins.
        """
        possible_outputs = list(self.nodes)
        out_node = random.choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = random.choice(possible_inputs)

        # Don't duplicate connections.
        key = (in_node, out_node)
        if key in self.connections:
            # Just enable the connection if it exists but is disabled
            if config.check_structural_mutation_surer():
                self.connections[key].enabled = True
            return

        # Don't allow connections between two output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return

        # For feed-forward networks, avoid creating cycles.
        if config.feed_forward and creates_cycle(list(self.connections), key):
            return

        # Create the connection with default KAN values
        weight_s = random.gauss(config.weight_s_init_mean, config.weight_s_init_stdev)
        weight_b = random.gauss(config.weight_b_init_mean, config.weight_b_init_stdev)
        
        # Add the connection using our updated method
        self.add_connection(config, in_node, out_node, weight_s, weight_b, True)

    def mutate_delete_node(self, config):
        """
        Delete a node if possible. Returns the key of the deleted node or -1 if no node was deleted.
        This method remains mostly unchanged as it doesn't directly interact with weights.
        """
        # Do nothing if there are no non-output nodes.
        available_nodes = [k for k in self.nodes if k not in config.output_keys]
        if not available_nodes:
            return -1

        del_key = random.choice(available_nodes)

        connections_to_delete = set()
        for k, v in self.connections.items():
            if del_key in v.key:
                connections_to_delete.add(v.key)

        for key in connections_to_delete:
            del self.connections[key]

        del self.nodes[del_key]

        return del_key

    def mutate_delete_connection(self):
        """
        Delete a random connection if any exist.
        This method doesn't need changes as it doesn't directly interact with weights.
        """
        if self.connections:
            key = random.choice(list(self.connections.keys()))
            del self.connections[key]

    def configure_crossover(self, parent1, parent2, config):
        """Configure this genome as a crossover of the two parent genomes."""

        better_parent = parent1 if parent1.fitness > parent2.fitness else parent2
        worse_parent = parent1 if parent1.fitness <= parent2.fitness else parent2

        # Inherit connection genes
        for key, cg1 in better_parent.connections.items():
            cg2 = worse_parent.connections.get(key)
            if cg2 is None:
                # Excess or disjoint gene: copy from the fittest parent.
                self.connections[key] = cg1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.connections[key] = cg1.crossover(cg2, config)

            if len(self.connections[key].spline_segments) < config.min_spline_segments:
                # Add segments until we reach minimum
                self._ensure_min_spline_segments(self.connections[key], config)
                
            # IMPORTANT FIX: Ensure connection doesn't exceed max_spline_segments
            if len(self.connections[key].spline_segments) > config.max_spline_segments:
                # Remove excess segments, keeping the ones with highest absolute values
                self._reduce_to_max_spline_segments(self.connections[key], config)

        # Inherit node genes
        parent1_set = better_parent.nodes
        parent2_set = worse_parent.nodes

        for key, ng1 in parent1_set.items():
            ng2 = parent2_set.get(key)
            assert key not in self.nodes
            if ng2 is None:
                # Extra gene: copy from the fittest parent
                self.nodes[key] = ng1.copy()
            else:
                # Homologous gene: combine genes from both parents.
                self.nodes[key] = ng1.crossover(ng2)
    
    def _ensure_min_spline_segments(self, conn, config):
        """Ensure a connection has at least min_spline_segments."""
        # Add segments until we reach the minimum
        while len(conn.spline_segments) < config.min_spline_segments:
            # Get existing positions
            existing_positions = sorted([seg.grid_position for seg in conn.spline_segments.values()])
            
            # Find a good position to add a new segment
            new_pos = None
            
            if not existing_positions:
                # No existing positions - add at midpoint
                new_pos = (config.spline_range_min + config.spline_range_max) / 2
            elif len(existing_positions) == 1:
                # Just one position - add at opposite end
                if existing_positions[0] < (config.spline_range_min + config.spline_range_max) / 2:
                    new_pos = config.spline_range_max - 0.1
                else:
                    new_pos = config.spline_range_min + 0.1
            else:
                # Find largest gap and add in middle
                largest_gap = 0
                gap_pos = None
                
                # Check gaps between positions
                for i in range(len(existing_positions) - 1):
                    gap = existing_positions[i+1] - existing_positions[i]
                    if gap > largest_gap:
                        largest_gap = gap
                        gap_pos = (existing_positions[i] + existing_positions[i+1]) / 2
                
                # Check gap at beginning
                if existing_positions[0] - config.spline_range_min > largest_gap:
                    largest_gap = existing_positions[0] - config.spline_range_min
                    gap_pos = (config.spline_range_min + existing_positions[0]) / 2
                    
                # Check gap at end
                if config.spline_range_max - existing_positions[-1] > largest_gap:
                    largest_gap = config.spline_range_max - existing_positions[-1]
                    gap_pos = (existing_positions[-1] + config.spline_range_max) / 2
                
                new_pos = gap_pos
                
            # Create and add the new segment if we found a valid position
            if new_pos is not None:
                seg_key = (conn.key[0], conn.key[1], new_pos)
                value = random.gauss(config.spline_init_mean, config.spline_init_stdev)
                conn.add_spline_segment(seg_key, new_pos, value)
    
    def _reduce_to_max_spline_segments(self, conn, config):
        """Reduce spline segments to max_spline_segments, keeping most significant ones."""
        if len(conn.spline_segments) <= config.max_spline_segments:
            return
            
        # Sort segments by absolute value (importance)
        segments_by_importance = sorted(
            conn.spline_segments.items(),
            key=lambda item: abs(item[1].value), 
            reverse=True
        )
        
        # Keep only the most important segments
        to_keep = segments_by_importance[:config.max_spline_segments]
        
        # Create a new dictionary with just these segments
        new_segments = {key: seg for key, seg in to_keep}
        conn.spline_segments = new_segments
    
    def mutate(self, config):
        """Mutate this genome."""
        # Call the parent class mutate method, which will also call mutate on all connections
        super().mutate(config)
        
        # Additional check after mutation to ensure segment count constraints are met
        for conn in self.connections.values():
            if conn.enabled:
                conn.mutate(config)
                # Ensure min segments
                if len(conn.spline_segments) < config.min_spline_segments:
                    self._ensure_min_spline_segments(conn, config)
                
                # Ensure max segments
                if len(conn.spline_segments) > config.max_spline_segments:
                    self._reduce_to_max_spline_segments(conn, config)

    
    def distance(self, other, config):
        """Return the genetic distance between this genome and the other."""
        distance = super().distance(other, config)
        return distance  # Connection distance already includes spline distance