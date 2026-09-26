import torch
from .base import SphericalOperator as Operator

class CompositeOperator(Operator):
    def __init__(self, first, second):
        super().__init__(trainable=first.is_trainable() or second.is_trainable())
        self.first = first
        self.second = second

    def forward(self, coordinates):
        return self.second(self.first(coordinates))

def make_default():
    pass
