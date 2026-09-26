import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geometry.conversion import cartesian_to_spherical, spherical_to_cartesian
from geometry.metrics import backreaction
from operators.base import SphericalOperator, IdentityOperator

class GeometricActivationLayer(nn.Module):
    def __init__(self, operator=None, backreaction_mode='vector', store_backreaction=True, strict_conversion=False):
        super().__init__()
        self.operator = operator if operator is not None else IdentityOperator()
        self.backreaction_mode = backreaction_mode
        self.store_backreaction = store_backreaction
        self.strict_conversion = strict_conversion
        self.last_backreaction = None
        self.add_module('operator', self.operator)

    def forward(self, x, return_backreaction=False):
        original = x
        spherical = cartesian_to_spherical(x)
        transformed_spherical = self.operator(spherical
        transformed_cartesian = spherical_to_cartesian(transformed_spherical, strict=self.strict_conversion)
                                              
        if self.backreaction_mode != 'none':
            reaction = backreaction(original, transformed_cartesian, mode=self.backreaction_mode)
            if self.store_backreaction:
                self.last_backreaction = reaction
        else:
            reaction = None
            self.last_backreaction = None
        
        if return_backreaction:
            return transformed_cartesian, reaction
        else:
            return transformed_cartesian

    def get_last_backreaction(self):
        return self.last_backreaction

    def extra_repr(self):
        return f"operator={self.operator.__class__.__name__}, backreaction_mode={self.backreaction_mode}"

class GeometricSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
    
    def forward(self, x):
        backreactions = []
        for layer in self.layers:
            if isinstance(layer, GeometricActivationLayer):
                x = layer(x)
                if layer.store_backreaction:
                    reaction = layer.get_last_backreaction()
                    if reaction is not None:
                        backreactions.append(reaction)
            else:
                x = layer(x)
        return x
    
    def get_backreactions(self):
        backreactions = []        
        for layer in self.layers:
            if isinstance(layer, GeometricActivationLayer) and layer.store_backreaction:
                reaction = layer.get_last_backreaction()
                if reaction is not None:
                    backreactions.append(reaction)
        return backreactions
