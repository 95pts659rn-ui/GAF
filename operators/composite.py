import torch
from .base import SphericalOperator as Operator
from .rotator import RotatorOperator
from .radial_warp import RadialWarpOperator

class CompositeOperator(Operator):
    def __init__(self, first, second):
        super().__init__(trainable=first.is_trainable() or second.is_trainable())
        self.first = first
        self.second = second

    def forward(self, coordinates):
        return self.second(self.first(coordinates))

def make_default():
    return CompositeOperator(first=RotatorOperator(axis="z", trainable=True), second=RadialWarpOperator(warp_type="polynomial", parameters={"a": 1.0, "n": 1.0, "b": 0.0}, trainable=True))
