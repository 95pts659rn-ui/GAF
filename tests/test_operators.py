import numpy as np
import torch

from activations.geometric_activation import GeometricActivationLayer
from geometry.conversion import cartesian_to_spherical, spherical_to_cartesian
from operators.base import IdentityOperator
from operators.composite import CompositeOperator
from operators.radial_warp import RadialWarpOperator
from operators.rotator import RotatorOperator


def test_identity_operator_returns_input():
    operator = IdentityOperator()
    x = torch.tensor([[1.0, np.pi / 4, 0.0]], dtype=torch.float32)

    assert torch.allclose(operator(x), x)


def test_rotator_preserves_radius():
    operator = RotatorOperator(axis='z', angle=np.pi / 3)
    radii = torch.tensor([[1.0], [2.0], [0.5]], dtype=torch.float32)
    spherical = torch.cat([
        radii,
        torch.full((3, 1), np.pi / 4, dtype=torch.float32),
        torch.zeros((3, 1), dtype=torch.float32),
    ], dim=1)

    result = operator(spherical)

    assert torch.allclose(result[:, 0], radii.squeeze(), atol=1e-5)


def test_radial_warp_preserves_angles():
    operator = RadialWarpOperator(warp_type='exponential', parameters={'a': 1.0, 'b': 0.1, 'c': 0.0}, custom_function=None, trainable=False)
    spherical = torch.tensor([
        [1.0, 0.5, 1.0],
        [2.0, 1.5, 2.0],
        [0.5, 2.0, 3.0],
    ], dtype=torch.float32)

    result = operator(spherical)

    assert torch.allclose(result[:, 1], spherical[:, 1])
    assert torch.allclose(result[:, 2], spherical[:, 2])


def test_geometric_activation_forward_shape():
    layer = GeometricActivationLayer(operator=IdentityOperator(), backreaction_mode='vector')
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    output = layer(x)

    assert output.shape == x.shape


def test_geometric_activation_backreaction_return():
    layer = GeometricActivationLayer(operator=IdentityOperator(), backreaction_mode='norm')
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)

    output, backreaction = layer(x, return_backreaction=True)

    assert output.shape == x.shape
    assert backreaction.shape == (1,)


def test_composed_operator_basic():
    first = IdentityOperator()
    second = IdentityOperator()
    composed = CompositeOperator(first, second)
    x = torch.tensor([[1.0, np.pi / 2, 0.0]], dtype=torch.float32)

    result = composed(x)

    assert torch.allclose(result, x)
