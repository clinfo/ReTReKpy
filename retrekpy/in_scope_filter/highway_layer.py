""" The ``retrekpy.in_scope_filter`` package ``highway_layer`` module. """

import torch

from typing import Callable


class HighwayLayer(torch.nn.Module):
    """ The 'HighwayLayer' class. """

    def __init__(
            self,
            dimension: int,
            n_layers: int,
            activation: Callable
    ) -> None:
        """ The constructor method of the class. """

        super().__init__()

        self.n_layers = n_layers

        self.transform = torch.nn.ModuleList([torch.nn.Linear(dimension, dimension) for _ in range(self.n_layers)])

        self.gate = torch.nn.ModuleList([torch.nn.Linear(dimension, dimension) for _ in range(self.n_layers)])

        self.activation = activation

    def forward(
            self,
            x: int
    ) -> torch.Tensor:
        """ The 'forward' method of the class. """

        for layer in range(self.n_layers):
            gate = torch.sigmoid(self.gate[layer](x))

            h = self.activation(self.transform[layer](x))

            x = h * gate + x * (1 - gate)

        return x
