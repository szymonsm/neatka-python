"""Handles node and connection genes."""
import warnings
from random import random

from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute


# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.
# TODO: Evaluate using __slots__ for performance/memory usage improvement.


class BaseGene(object):
    """
    Handles functions shared by multiple types of genes (both node and connection),
    including crossover and calling mutation methods.
    """

    def __init__(self, key):
        self.key = key

    def __str__(self):
        attrib = ['key'] + [a.name for a in self._gene_attributes]
        attrib = [f'{a}={getattr(self, a)}' for a in attrib]
        return f'{self.__class__.__name__}({", ".join(attrib)})'

    def __lt__(self, other):
        assert isinstance(self.key, type(other.key)), f"Cannot compare keys {self.key!r} and {other.key!r}"
        return self.key < other.key

    @classmethod
    def parse_config(cls, config, param_dict):
        pass

    @classmethod
    def get_config_params(cls):
        params = []
        if not hasattr(cls, '_gene_attributes'):
            setattr(cls, '_gene_attributes', getattr(cls, '__gene_attributes__'))
            warnings.warn(
                f"Class '{cls.__name__!s}' {cls!r} needs '_gene_attributes' not '__gene_attributes__'",
                DeprecationWarning)
        for a in cls._gene_attributes:
            params += a.get_config_params()
        return params

    @classmethod
    def validate_attributes(cls, config):
        for a in cls._gene_attributes:
            a.validate(config)

    def init_attributes(self, config):
        for a in self._gene_attributes:
            setattr(self, a.name, a.init_value(config))

    def mutate(self, config):
        for a in self._gene_attributes:
            v = getattr(self, a.name)
            setattr(self, a.name, a.mutate_value(v, config))

    def copy(self):
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            setattr(new_gene, a.name, getattr(self, a.name))

        return new_gene

    def crossover(self, gene2):
        """ Creates a new gene randomly inheriting attributes from its parents."""
        assert self.key == gene2.key

        # Note: we use "a if random() > 0.5 else b" instead of choice((a, b))
        # here because `choice` is substantially slower.
        new_gene = self.__class__(self.key)
        for a in self._gene_attributes:
            if random() > 0.5:
                setattr(new_gene, a.name, getattr(self, a.name))
            else:
                setattr(new_gene, a.name, getattr(gene2, a.name))

        return new_gene


# TODO: Should these be in the nn module?  iznn and ctrnn can have additional attributes.


class DefaultNodeGene(BaseGene):
    _gene_attributes = [FloatAttribute('bias'),
                        FloatAttribute('response'),
                        StringAttribute('activation', options=''),
                        StringAttribute('aggregation', options='')]

    def __init__(self, key):
        assert isinstance(key, int), f"DefaultNodeGene key must be an int, not {key!r}"
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.bias - other.bias) + abs(self.response - other.response)
        if self.activation != other.activation:
            d += 1.0
        if self.aggregation != other.aggregation:
            d += 1.0
        return d * config.compatibility_weight_coefficient


# TODO: Do an ablation study to determine whether the enabled setting is
# important--presumably mutations that set the weight to near zero could
# provide a similar effect depending on the weight range, mutation rate,
# and aggregation function. (Most obviously, a near-zero weight for the
# `product` aggregation function is rather more important than one giving
# an output of 1 from the connection, for instance!)
class DefaultConnectionGene(BaseGene):
    _gene_attributes = [FloatAttribute('weight'),
                        BoolAttribute('enabled')]

    def __init__(self, key):
        assert isinstance(key, tuple), f"DefaultConnectionGene key must be a tuple, not {key!r}"
        BaseGene.__init__(self, key)

    def distance(self, other, config):
        d = abs(self.weight - other.weight)
        if self.enabled != other.enabled:
            d += 1.0
        return d * config.compatibility_weight_coefficient
    
import numpy as np
from random import random, choice, gauss, uniform
# from neat.genes import BaseGene, DefaultNodeGene, DefaultConnectionGene
from neat.attributes import FloatAttribute, BoolAttribute, StringAttribute

class KANNodeGene(DefaultNodeGene):
    """Node gene for Kolmogorov-Arnold Networks.
    
    Represents a summation node in the KAN architecture.
    """
    def __init__(self, key):
        super().__init__(key)
        self.aggregation = 'sum'  # KAN nodes are summation nodes by default

class SplineSegmentGene(BaseGene):
    """Represents a single point in a spline grid."""
    _gene_attributes = [
        FloatAttribute('value'),
        FloatAttribute('grid_position'),
    ]
    
    def __init__(self, key, value=0.0, grid_position=0.0):
        super().__init__(key)
        self.value = value
        self.grid_position = grid_position
        
    def distance(self, other, config):
        return abs(self.value - other.value) / config.spline_coefficient_range

class KANConnectionGene(DefaultConnectionGene):
    """Connection gene for Kolmogorov-Arnold Networks.
    
    Represents a spline-based connection between nodes.
    """
    _gene_attributes = [
        FloatAttribute('weight'),
        BoolAttribute('enabled'),
        FloatAttribute('scale'),
        FloatAttribute('bias'),
    ]
    
    def __init__(self, key, weight=0.0, enabled=True, scale=1.0, bias=0.0):
        super().__init__(key, weight, enabled)
        self.scale = scale
        self.bias = bias
        self.spline_segments = {}  # Dictionary mapping grid positions to spline segment genes
        
    def mutate(self, config):
        super().mutate(config)
        
        # Mutate scale
        if random() < config.scale_mutate_rate:
            self.scale += gauss(0, config.scale_mutate_power)
            self.scale = max(min(self.scale, config.scale_max_value), config.scale_min_value)
            
        # Mutate bias
        if random() < config.kan_bias_mutate_rate:
            self.bias += gauss(0, config.kan_bias_mutate_power)
            self.bias = max(min(self.bias, config.kan_bias_max_value), config.kan_bias_min_value)
        
        # Mutate spline segments
        for key, segment in self.spline_segments.items():
            if random() < config.spline_mutate_rate:
                segment.value += gauss(0, config.spline_mutate_power)
                segment.value = max(min(segment.value, config.spline_max_value), config.spline_min_value)
                
    def add_spline_segment(self, key, grid_position, value=None):
        """Add a new spline segment to this connection."""
        if value is None:
            value = uniform(-1.0, 1.0)
        self.spline_segments[key] = SplineSegmentGene(key, value, grid_position)
        
    def get_spline_function(self):
        """Return a callable spline function based on the spline segments."""
        # Sort grid points by position
        points = sorted([(seg.grid_position, seg.value) for seg in self.spline_segments.values()])
        
        # Simple linear interpolation between points
        def spline_func(x):
            # Special cases
            if len(points) == 0:
                return 0.0
            if len(points) == 1:
                return points[0][1]
                
            # Find position in the grid
            if x <= points[0][0]:
                return points[0][1]
            if x >= points[-1][0]:
                return points[-1][1]
                
            # Linear interpolation between points
            for i in range(len(points)-1):
                x1, y1 = points[i]
                x2, y2 = points[i+1]
                
                if x1 <= x <= x2:
                    # Linear interpolation: y = y1 + (y2-y1)*(x-x1)/(x2-x1)
                    t = (x - x1) / (x2 - x1) if x2 != x1 else 0
                    return y1 + t * (y2 - y1)
            
            # Should never reach here
            return 0.0
            
    def distance(self, other, config):
        """Return the genetic distance between this connection gene and the other."""
        d = super().distance(other, config)
        
        # Add distance for scale and bias
        d += abs(self.scale - other.scale) / config.scale_coefficient_range
        d += abs(self.bias - other.bias) / config.bias_coefficient_range
        
        # Add distance for spline segments (if they share grid positions)
        spline_dist = 0.0
        count = 0
        for pos in set(self.spline_segments.keys()) & set(other.spline_segments.keys()):
            seg1 = self.spline_segments[pos]
            seg2 = other.spline_segments[pos]
            spline_dist += seg1.distance(seg2, config)
            count += 1
            
        if count > 0:
            d += spline_dist / count
            
        return d
