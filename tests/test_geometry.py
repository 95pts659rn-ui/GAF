import numpy as np
import torch

from geometry.conversion import cartesian_to_spherical, spherical_to_cartesian


def test_cartesian_to_spherical_basic():
    x = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    spherical = cartesian_to_spherical(x)

    assert torch.allclose(spherical[0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(spherical[0, 1], torch.tensor(np.pi / 2), atol=1e-6)
    assert torch.allclose(spherical[0, 2], torch.tensor(0.0), atol=1e-6)


def test_spherical_to_cartesian_basic():
    spherical = torch.tensor([[1.0, np.pi / 2, 0.0]], dtype=torch.float32)
    cartesian = spherical_to_cartesian(spherical)

    assert torch.allclose(cartesian[0, 0], torch.tensor(1.0), atol=1e-6)
    assert torch.allclose(cartesian[0, 1], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(cartesian[0, 2], torch.tensor(0.0), atol=1e-6)


def test_round_trip_conversion():
    x = torch.randn(16, 3, dtype=torch.float32)
    spherical = cartesian_to_spherical(x)
    recovered = spherical_to_cartesian(spherical)

    assert torch.allclose(x, recovered, atol=1e-5)
