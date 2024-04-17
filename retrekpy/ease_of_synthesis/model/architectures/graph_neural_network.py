""" The ``retrekpy.ease_of_synthesis.architectures`` package ``graph_neural_network`` module. """

import torch

from typing import Any, Dict

from kmol.model.architectures import GraphConvolutionalNetwork

from ..read_out import get_read_out


class GraphNeuralNetwork(GraphConvolutionalNetwork):
    """ The 'GraphNeuralNetwork' class. """

    def __init__(
            self,
            **kwargs
    ) -> None:
        """ The constructor method class. """

        self.read_out_name = kwargs.pop("read_out", "mean")

        self.read_out_kwargs = kwargs.pop("read_out_kwargs", {})

        hidden_features = kwargs["hidden_features"]

        self.read_out_kwargs["in_channels"] = hidden_features

        self.hidden_features = hidden_features

        self.p_dropout = kwargs.get("dropout", 0)

        self.out_features = kwargs["out_features"]

        super().__init__(**kwargs)

        self.read_out = get_read_out(self.read_out_name, self.read_out_kwargs)

        self.build_head()

    def build_head(
            self
    ) -> None:
        """ The 'build_head' method class. """

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.read_out.out_dim, self.hidden_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=self.p_dropout),
            torch.nn.Linear(self.hidden_features, self.out_features)
        )

    def forward(
            self,
            data: Dict[str, Any],
            return_atom_features: bool = False
    ) -> torch.Tensor:
        """ The 'forward' method class. """

        data = data[self.get_requirements()[0]]

        atom_features = data.x.float()

        for convolution in self.convolutions:
            atom_features = convolution(atom_features, data.edge_index, data.edge_attr, data.batch)

        graph_features = self.read_out(atom_features, data.batch)

        out = self.mlp(graph_features)

        return out
