from __future__ import annotations
from typing import Literal, Tuple
import torch
from geometry.cartesian_to_spherical import cartesian_to_spherical
from geometry.spherical_to_cartesian import spherical_to_cartesian
from operators.base_operator import SphericalOperator

BackreactionMode = Literal["vector", "norm", "energy"]

def _validate_cartesian_input(x: torch.Tensor) -> None:
    if not torch.is_tensor(x):
        raise TypeError("Input must be a torch.Tensor")
    if x.ndim < 1 or x.shape[-1] != 3:
        raise ValueError(
            "Input must have shape (..., 3), representing Cartesian (x, y, z) coordinates"
        )

def backreaction_orders(
    original: torch.Tensor,
    first: SphericalOperator,
    second: SphericalOperator,
    *,
    strict_conversion: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the two possible ordered compositions of two spherical operators.

    Given Cartesian input x, this evaluates:
        second(first(S(x)))
        first(second(S(x)))
    and converts both results back to Cartesian coordinates.
    The two returned tensors therefore represent the two counterfactual
    outcomes of applying the same pair of operators in opposite orders.

    Args:
        original:
            Cartesian tensor of shape (..., 3).
        first:
            Operator applied first in the first composition.
        second:
            Operator applied second in the first composition.
        strict_conversion:
            Whether spherical-to-Cartesian conversion should enforce
            coordinate-domain constraints.

    Returns:
        Tuple (first_then_second, second_then_first), both in Cartesian space.
    """
    _validate_cartesian_input(original)

    if not isinstance(first, SphericalOperator):
        raise TypeError("first must be a SphericalOperator")

    if not isinstance(second, SphericalOperator):
        raise TypeError("second must be a SphericalOperator")

    spherical = cartesian_to_spherical(original)
    first_then_second_spherical = second(first(spherical))
    second_then_first_spherical = first(second(spherical))
    first_then_second = spherical_to_cartesian(
        first_then_second_spherical,
        strict=strict_conversion
    )

    second_then_first = spherical_to_cartesian(
        second_then_first_spherical,
        strict=strict_conversion
    )
    
    return first_then_second, second_then_first

def backreaction(
    original: torch.Tensor,
    first: SphericalOperator,
    second: SphericalOperator,
    *,
    mode: BackreactionMode = "vector",
    strict_conversion: bool = False
    ) -> torch.Tensor:
    """
    Measure the geometric manifestation of operator backreaction.
    Backreaction is obtained by comparing the two ordered compositions:
        second(first(S(x)))
        first(second(S(x)))

    after both have been mapped back into Cartesian space.
    The fundamental geometric quantity is the displacement between those
    counterfactual outcomes:
        first_then_second - second_then_first

    Therefore:
        [first, second] = 0  ->  backreaction = 0
    up to floating-point error and the numerical behavior of the operators.

    Args:
        original:
            Cartesian tensor of shape (..., 3).
        first:
            Operator applied first in the reference ordering.
        second:
            Operator applied second in the reference ordering.
        mode:
            - "vector": Cartesian backreaction vector, shape (..., 3)
            - "norm": Euclidean magnitude, shape (...)
            - "energy": squared Euclidean magnitude, shape (...)
        strict_conversion:
            Whether spherical-to-Cartesian conversion should enforce
            coordinate-domain constraints.

    Returns:
        Backreaction measurement in the requested form.
    """
    first_then_second, second_then_first = backreaction_orders(
        original,
        first,
        second,
        strict_conversion=strict_conversion,
    )

    difference = first_then_second - second_then_first
    
    if mode == "vector":
        return difference
    if mode == "norm":
        return torch.linalg.vector_norm(difference, dim=-1)
    if mode == "energy":
        return torch.sum(difference.square(), dim=-1)

    raise ValueError(
        f"Unknown backreaction mode: {mode!r}. "
        "Expected 'vector', 'norm', or 'energy'."
    )