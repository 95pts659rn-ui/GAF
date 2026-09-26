import torch
import torch.nn as nn
from .base import SphericalOperator as Operator
from geometry.conversion import spherical_to_cartesian, cartesian_to_spherical

class RotatorOperator(Operator):
    def __init__(self, axis='z', angle=None, trainable=False):
        super().__init__(trainable=trainable)
        self.axis = axis.lower()
        if self.axis not in ['x', 'y', 'z']:
            raise ValueError(f"Axis must be 'x', 'y', or 'z', got {axis}")
        if trainable:
            _angle = angle if angle is not None else torch.rand(1).item() * 0.1
            self.angle = nn.Parameter(torch.tensor([_angle], dtype=torch.float32))
        else:
            self.register_buffer('angle', torch.tensor([angle or 0.0], dtype=torch.float32))

    def forward(self, coordinates):
        cartesian = spherical_to_cartesian(coordinates)
        x, y, z = cartesian[..., 0:1], cartesian[..., 1:2], cartesian[..., 2:3]
        angle = self.angle
        if self.axis == 'x':
            # Rotation around x-axis
            y_new = y * torch.cos(angle) - z * torch.sin(angle)
            z_new = y * torch.sin(angle) + z * torch.cos(angle)
            x_new = x
        elif self.axis == 'y':
            # Rotation around y-axis
            x_new = x * torch.cos(angle) + z * torch.sin(angle)
            z_new = -x * torch.sin(angle) + z * torch.cos(angle)
            y_new = y
        else:
            # Rotation around z-axis
            x_new = x * torch.cos(angle) - y * torch.sin(angle)
            y_new = x * torch.sin(angle) + y * torch.cos(angle)
            z_new = z

        spherical = cartesian_to_spherical(torch.cat([x_new, y_new, z_new], dim=-1))
        return spherical
