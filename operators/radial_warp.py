import torch
import torch.nn as nn
from .base import SphericalOperator as Operator

class RadialWarpOperator(Operator):
    def __init__(self, warp_type='polynomial', parameters=None, custom_function=None, trainable=False):
        super().__init__(trainable=trainable)
        self.warp_type = warp_type
        default_parameters = {
            'polynomial': {'a': 1.0, 'n': 1.0, 'b': 0.0},
            'exponential': {'a': 1.0, 'b': 0.1, 'c': 0.0},
            'sinusoidal': {'a': 1.0, 'b': 1.0, 'c': 0.0}
        }
        parameters = parameters or default_parameters.get(warp_type, {})
        if warp_type == 'polynomial':
            a = parameters.get('a', 1.0)
            n = parameters.get('n', 1.0)
            b = parameters.get('b', 0.0)
            
            if trainable:
                self.a = nn.Parameter(torch.tensor([a], dtype=torch.float32))
                self.n = nn.Parameter(torch.tensor([n], dtype=torch.float32))
                self.b = nn.Parameter(torch.tensor([b], dtype=torch.float32))
            else:
                self.register_buffer('a', torch.tensor([a], dtype=torch.float32))
                self.register_buffer('n', torch.tensor([n], dtype=torch.float32))
                self.register_buffer('b', torch.tensor([b], dtype=torch.float32))
                
        elif warp_type == 'exponential':
            a = parameters.get('a', 1.0)
            b = parameters.get('b', 0.1)
            c = parameters.get('c', 0.0)
            
            if trainable:
                self.a = nn.Parameter(torch.tensor([a], dtype=torch.float32))
                self.b = nn.Parameter(torch.tensor([b], dtype=torch.float32))
                self.c = nn.Parameter(torch.tensor([c], dtype=torch.float32))
            else:
                self.register_buffer('a', torch.tensor([a], dtype=torch.float32))
                self.register_buffer('b', torch.tensor([b], dtype=torch.float32))
                self.register_buffer('c', torch.tensor([c], dtype=torch.float32))
                
        elif warp_type == 'sinusoidal':
            a = parameters.get('a', 1.0)
            b = parameters.get('b', 1.0)
            c = parameters.get('c', 0.0)
            
            if trainable:
                self.a = nn.Parameter(torch.tensor([a], dtype=torch.float32))
                self.b = nn.Parameter(torch.tensor([b], dtype=torch.float32))
                self.c = nn.Parameter(torch.tensor([c], dtype=torch.float32))
            else:
                self.register_buffer('a', torch.tensor([a], dtype=torch.float32))
                self.register_buffer('b', torch.tensor([b], dtype=torch.float32))
                self.register_buffer('c', torch.tensor([c], dtype=torch.float32))
                
        elif warp_type == 'custom':
            if custom_function is None:
                raise ValueError("Must provide a function for custom warp_type")
            self.custom_function = custom_function
            
        else:
            raise ValueError(f"Unknown warp type: {warp_type}")
        self.custom_function = custom_function

    def forward(self, coordinates):
        r = coordinates[..., 0:1]
        theta = coordinates[..., 1:2]
        phi = coordinates[..., 2:3]
        
        if self.warp_type == 'polynomial':
            r_new = self.a * torch.pow(r, self.n) + self.b
            
        elif self.warp_type == 'exponential':
            r_new = self.a * torch.exp(self.b * r) + self.c
            
        elif self.warp_type == 'sinusoidal':
            r_new = self.a * torch.sin(self.b * r) + self.c
            
        elif self.warp_type == 'custom':
            r_new = self.custom_function(r)
            
        r_new = torch.clamp(r_new, min=1e-8)
        return torch.cat([r_new, theta, phi], dim=-1)
