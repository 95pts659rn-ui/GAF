import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from operators.base import SphericalOperator
from geometry.conversion import cartesian_to_spherical, spherical_to_cartesian
from geometry.metrics import backreaction

class BlockwiseGeometricLayer(nn.Module):
    def __init__(self, input_dim, operator, backreaction_mode='norm', store_backreaction=True, strict_conversion=False):
        super().__init__()        
        self.input_dim = input_dim
        self.operator = operator
        self.backreaction_mode = backreaction_mode
        self.store_backreaction = store_backreaction
        self.strict_conversion = strict_conversion
        self.num_blocks = input_dim // 3
        self.residual_dim = input_dim % 3
        
        if self.residual_dim > 0:
            self.residual_to_3d = nn.Linear(self.residual_dim, 3)
            self.residual_from_3d = nn.Linear(3, self.residual_dim)
        else:
            self.residual_to_3d = None
            self.residual_from_3d = None
        self.last_backreaction = None
    
    def forward(self, x, return_backreaction=False):
        batch_size = x.shape[0]
        original = x
        block_size = 3 * self.num_blocks
        blocks = x[:, :block_size].reshape(batch_size, self.num_blocks, 3)
        transformed_blocks = []
        for i in range(self.num_blocks):
            block = blocks[:, i, :]
            spherical = cartesian_to_spherical(block)
            transformed_spherical = self.operator(spherical)
            transformed_cartesian = spherical_to_cartesian(transformed_spherical, strict=self.strict_conversion)
            transformed_blocks.append(transformed_cartesian)
        
        transformed_main = torch.stack(transformed_blocks, dim=1).reshape(batch_size, -1)
        if self.residual_dim > 0:
            residual = x[:, block_size:]  # [batch, residual_dim]
            # Project residual to 3D, transform, project back
            residual_3d = self.residual_to_3d(residual)
            # Transform residual
            spherical = cartesian_to_spherical(residual_3d)
            transformed_spherical = self.operator(spherical)
            transformed_residual_3d = spherical_to_cartesian(transformed_spherical, strict=self.strict_conversion)
            # Project back to residual dim
            transformed_residual = self.residual_from_3d(transformed_residual_3d)
            transformed = torch.cat([transformed_main, transformed_residual], dim=1)
        else:
            transformed = transformed_main
        
        # Compute backreaction if requested
        if return_backreaction and self.backreaction_mode != 'none':
            reaction = backreaction(original, transformed, mode=self.backreaction_mode)
            if self.store_backreaction:
                self.last_backreaction = reaction
        else:
            reaction = None
            self.last_backreaction = None
        
        if return_backreaction:
            return transformed, reaction
        else:
            return transformed, None
    
    def get_last_backreaction(self):
        return self.last_backreaction
    
    def extra_repr(self):
        return (f"input_dim={self.input_dim}, num_blocks={self.num_blocks}, residual_dim={self.residual_dim}, operator={self.operator.__class__.__name__}, backreaction_mode={self.backreaction_mode}")
