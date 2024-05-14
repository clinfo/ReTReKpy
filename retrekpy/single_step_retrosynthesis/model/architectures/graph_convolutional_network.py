""" The ``retrekpy.single_step_retrosynthesis.model.architectures`` package ``graph_convolutional_network`` module. """

import torch

from typing import Any, Dict, List, Optional, Union

from kmol.core.helpers import SuperFactory
from kmol.model.architectures.abstract_network import AbstractNetwork
from kmol.model.layers import GraphConvolutionWrapper
from kmol.model.read_out import get_read_out


class CustomizedGraphConvolutionalNetwork(AbstractNetwork):
    """ The customized 'GraphConvolutionalNetwork' class. """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        graph_dense_features: int,
        mlp_features: int,
        out_features: int,
        dropout: float,
        layer_type: str = "torch_geometric.nn.GCNConv",
        layers_count: int = 2,
        is_residual: bool = True,
        norm_layer: Optional[str] = None,
        activation: str = "torch.nn.ReLU",
        read_out: Union[str, List[str]] = "sum",
        read_out_kwargs: Optional[Dict[str, Any]] = None,
        final_activation: Optional[str] = None,
        **kwargs
    ) -> None:
        """ The customized constructor method of the class. """

        super().__init__()

        self.out_features = out_features
        self.convolutions = torch.nn.ModuleList()

        self.convolutions.append(
            GraphConvolutionWrapper(
                in_features=in_features,
                out_features=hidden_features,
                dropout=dropout,
                layer_type=layer_type,
                is_residual=is_residual,
                norm_layer=norm_layer,
                activation=activation,
                **kwargs,
            )
        )

        for _ in range(layers_count - 1):
            self.convolutions.append(
                GraphConvolutionWrapper(
                    in_features=hidden_features,
                    out_features=hidden_features,
                    dropout=dropout,
                    layer_type=layer_type,
                    is_residual=is_residual,
                    norm_layer=norm_layer,
                    activation=activation,
                    **kwargs,
                )
            )

        self.graph_dense = torch.nn.Sequential(
            torch.nn.Linear(hidden_features, graph_dense_features, bias=False),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout)
        )

        if read_out_kwargs is None:
            read_out_kwargs = {}

        read_out_kwargs.update({"in_channels": hidden_features})

        self.read_out = get_read_out(read_out, read_out_kwargs)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(graph_dense_features, mlp_features),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(mlp_features, mlp_features),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(mlp_features, out_features)
        )

        if final_activation is not None:
            self.mlp.add_module("final_activation", SuperFactory.reflect(final_activation)())

        self.last_hidden_layer_name = "mlp.1"

    def get_requirements(
            self
    ) -> List[str]:
        """ The customized 'get_requirements' method of the class. """

        return ["graph"]

    def forward(
            self,
            data: Dict[str, Any]
    ) -> torch.Tensor:
        """ The customized 'forward' method of the class. """

        data = data[self.get_requirements()[0]]

        x = data.x.float()

        for convolution in self.convolutions:
            x = convolution(x, data.edge_index, data.edge_attr, data.batch)

        graph_dense_output = self.graph_dense(x)

        read_out = self.read_out(graph_dense_output, batch=data.batch)

        x = self.mlp(read_out)

        return x
