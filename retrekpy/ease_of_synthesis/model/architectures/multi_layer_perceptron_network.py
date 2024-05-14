""" The ``retrekpy.ease_of_synthesis.architectures`` package ``multi_layer_perceptron_network`` module. """

import torch

from typing import Any, Dict, List

from kmol.core.helpers import SuperFactory
from kmol.model.architectures import AbstractNetwork


class MultiLayerPerceptronNetwork(AbstractNetwork):
    """ The 'MultiLayerPerceptronNetwork' class. """

    def __init__(
            self,
            in_features: int,
            hidden_features: int,
            out_features: int,
            layers_count: int,
            activation: str = "torch.nn.ReLU",
            dropout: float = 0.,
            use_batch_norm: bool = False,
    ) -> None:
        """ The constructor method class. """

        super().__init__()

        self.out_features = out_features
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        self.activation = activation

        layers = self._build_layer(in_features, hidden_features)

        for _ in range(layers_count - 2):
            layers += self._build_layer(hidden_features, hidden_features)

        layers.append(torch.nn.Linear(hidden_features, out_features))

        self.block = torch.nn.Sequential(*layers)

    def _build_layer(
            self,
            in_features: int,
            out_features: int
    ) -> List[torch.nn.Module]:
        """ The '_build_layer' method class. """

        layer = [torch.nn.Linear(in_features, out_features)]

        if self.use_batch_norm:
            layer.append(torch.nn.BatchNorm1d(out_features))

        layer.append(SuperFactory.reflect(self.activation)())

        if self.dropout:
            layer.append(torch.nn.Dropout(p=self.dropout))

        return layer

    def get_requirements(
            self
    ) -> List[str]:
        """ The 'get_requirements' method class. """

        return ["features"]

    def forward(
            self,
            data: Dict[str, Any]
    ) -> torch.Tensor:
        """ The 'forward' method class. """

        features = data[self.get_requirements()[0]]

        return self.block(features)
