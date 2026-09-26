import numpy as np
import torch

from geometry.conversion import cartesian_to_spherical, spherical_to_cartesian
from geometry.metrics import backreaction


def test_backreaction_zero_for_identity():
    original = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    recovered = original.clone()

    assert torch.allclose(backreaction(original, recovered, mode='vector'), torch.zeros_like(original), atol=1e-6)
    assert torch.allclose(backreaction(original, recovered, mode='norm'), torch.zeros_like(torch.tensor([0.0])), atol=1e-6)
    assert torch.allclose(backreaction(original, recovered, mode='energy'), torch.zeros_like(torch.tensor([0.0])), atol=1e-6)


def test_backreaction_vector_mode():
    original = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    recovered = torch.tensor([[1.5, 2.5, 3.5]], dtype=torch.float32)

    result = backreaction(original, recovered, mode='vector')

    assert torch.allclose(result, original - recovered, atol=1e-6)
    assert result.shape == (1, 3)


def test_backreaction_norm_mode():
    original = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    recovered = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    result = backreaction(original, recovered, mode='norm')

    assert torch.allclose(result, torch.tensor([1.0]), atol=1e-6)
    assert result.shape == (1,)


def test_backreaction_energy_mode():
    original = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float32)
    recovered = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)

    result = backreaction(original, recovered, mode='energy')

    assert torch.allclose(result, torch.tensor([25.0]), atol=1e-6)
    assert result.shape == (1,)


def test_backreaction_divergence_mode():
    original = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
    ], dtype=torch.float32)
    recovered = torch.tensor([
        [1.0, 0.1, 0.1],
        [0.1, 1.0, 0.1],
        [0.1, 0.1, 1.0],
        [0.9, 0.9, 0.9],
    ], dtype=torch.float32)

    result = backreaction(original, recovered, mode='divergence')

    assert result.shape == (4,)
    assert torch.all((result >= 0) & (result <= 2))


def test_backreaction_batch_processing():
    batch_size = 50
    original = torch.randn(batch_size, 3)
    recovered = torch.randn(batch_size, 3)

    vector_result = backreaction(original, recovered, mode='vector')
    norm_result = backreaction(original, recovered, mode='norm')

    assert vector_result.shape == (batch_size, 3)
    assert norm_result.shape == (batch_size,)


def test_backreaction_symmetry():
    original = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    recovered = torch.tensor([[2.0, 3.0, 4.0]], dtype=torch.float32)

    result_1 = backreaction(original, recovered, mode='vector')
    result_2 = backreaction(recovered, original, mode='vector')

    assert torch.allclose(result_1, -result_2, atol=1e-6)


def test_backreaction_round_trip():
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    spherical = cartesian_to_spherical(x)
    recovered = spherical_to_cartesian(spherical)

    result = backreaction(x, recovered, mode='vector')

    assert torch.allclose(result, torch.zeros_like(result), atol=1e-5)
def test_backreaction_nonnegativity_norm():
    original = torch.randn(20, 3)
    recovered = torch.randn(20, 3)

    result = backreaction(original, recovered, mode='norm')

    assert torch.all(result >= 0)


def test_backreaction_triangle_inequality():
    x1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    x2 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    x3 = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)

    br_12 = backreaction(x1, x2, mode='norm')
    br_23 = backreaction(x2, x3, mode='norm')
    br_13 = backreaction(x1, x3, mode='norm')

    assert br_13 <= br_12 + br_23 + 1e-5
