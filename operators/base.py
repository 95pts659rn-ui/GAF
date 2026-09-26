import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class SphericalOperator(nn.Module, ABC):
    def __init__(self, trainable=False):
        super().__init__()
        self.trainable = trainable

    @abstractmethod
    def forward(self, coordinates):
        pass

    def is_trainable(self):
        return self.trainable

    def get_parameters(self):
        if not self.trainable:
            raise RuntimeError("This operator is not trainable and has no parameters")
        parameters = {}
        for name, parameter in self.named_parameters():
            parameters[name] = parameter
        return parameters

class IdentityOperator(SphericalOperator):
    def __init__(self):
        super().__init__(trainable=False)

    def forward(self, coordinates):
        return coordinates
