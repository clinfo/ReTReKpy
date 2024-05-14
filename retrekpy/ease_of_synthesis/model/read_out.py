""" The ``retrekpy.ease_of_synthesis`` package ``read_out`` module. """

import torch

from typing import Dict, Optional, Tuple, Union

from torch_geometric.nn.aggr import AttentionalAggregation, Set2Set
from torch_geometric.nn.pool import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax

from torch_scatter import scatter_add


class MaxReadOut(torch.nn.Module):
    """ The 'MaxReadOut' class. """

    def __init__(
            self,
            in_channels: int,
            **kwargs
    ) -> None:
        """ The constructor method class. """

        super().__init__()

        self.out_dim = in_channels

    def forward(
            self,
            x: torch.Tensor,
            batch: torch.LongTensor
    ) -> torch.Tensor:
        """ The 'forward' method class. """

        return global_max_pool(x, batch)


class SumReadOut(torch.nn.Module):
    """ The 'SumReadOut' class. """

    def __init__(
            self,
            in_channels: int,
            **kwargs
    ) -> None:
        """ The constructor method class. """

        super().__init__()

        self.out_dim = in_channels

    def forward(
            self,
            x: torch.Tensor,
            batch: torch.LongTensor
    ) -> torch.Tensor:
        """ The 'forward' method class. """

        return global_add_pool(x, batch)


class MeanReadOut(torch.nn.Module):
    """ The 'MeanReadOut' class. """

    def __init__(
            self,
            in_channels: int,
            **kwargs
    ) -> None:
        """ The constructor method class. """

        super().__init__()

        self.out_dim = in_channels

    def forward(
            self,
            x: torch.Tensor,
            batch: torch.LongTensor
    ) -> torch.Tensor:
        """ The 'forward' method class. """

        return global_mean_pool(x, batch)


class AttentionReadOut(AttentionalAggregation):
    """ The 'AttentionReadOut' class. """

    def __init__(
            self,
            in_channels: int,
            full: bool = True,
            **kwargs
    ) -> None:
        """ The constructor method class. """

        self.attention_out_dim = in_channels if full else 1

        gate_nn = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels, self.attention_out_dim),
        )

        super().__init__(gate_nn)

        self.out_dim = in_channels

    def forward(
            self,
            x: torch.Tensor,
            batch: torch.LongTensor,
            ptr: Optional[torch.Tensor] = None,
            dim_size: Optional[int] = None,
            dim: int = -2,
    ) -> torch.Tensor:
        """ The 'forward' method class. """

        size = int(batch.max().item() + 1)

        if self.attention_out_dim == 1:
            return super().forward(x, batch, size=size)

        else:
            x = x.unsqueeze(-1) if x.dim() == 1 else x
            gate = self.gate_nn(x).view(-1, self.attention_out_dim)
            x = self.nn(x) if self.nn is not None else x

            if not gate.dim() == x.dim() and gate.size(0) == x.size(0):
                raise ValueError(f"Wrong input dimension: {gate.shape}, {x.shape}")

            gate = softmax(gate, batch, num_nodes=size)
            out = scatter_add(gate * x, batch, dim=0, dim_size=size)

            return out


class MLPSumReadOut(torch.nn.Module):
    """ The 'MLPSumReadOut' class. """

    def __init__(
            self,
            in_channels: int,
            **kwargs
    ) -> None:
        """ The constructor method class. """

        super().__init__()

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_channels, in_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(in_channels, in_channels),
        )

        self.out_dim = in_channels

    def forward(
            self,
            x: torch.Tensor,
            batch: torch.LongTensor
    ) -> torch.Tensor:
        """ The 'forward' method class. """

        x = self.mlp(x)

        return global_add_pool(x, batch)


class Set2SetReadOut(Set2Set):
    """ The 'Set2SetReadOut' class. """

    def __init__(
            self,
            in_channels: int,
            processing_steps: int = 4,
            num_layers: int = 2,
            **kwargs
    ) -> None:
        """ The constructor method class. """

        super().__init__(in_channels, processing_steps, num_layers)

        self.out_dim = 2 * in_channels


class CombinedReadOut(torch.nn.Module):
    """ The 'CombinedReadOut' class. """

    def __init__(
            self,
            read_out_list: Tuple[str, ...],
            read_out_kwargs: dict
    ) -> None:
        """ The constructor method class. """

        super().__init__()

        self.read_outs = torch.nn.ModuleList(
            [get_read_out(f, read_out_kwargs) for f in read_out_list]
        )

        self.out_dim = sum([read_out.out_dim for read_out in self.read_outs])

    def forward(
            self,
            x: torch.Tensor,
            batch: torch.LongTensor
    ) -> torch.Tensor:
        """ The 'forward' method class. """

        return torch.cat([read_out(x, batch) for read_out in self.read_outs], dim=1)


READOUT_FUNCTIONS = {
    "max": MaxReadOut,
    "sum": SumReadOut,
    "mean": MeanReadOut,
    "set2set": Set2SetReadOut,
    "attention": AttentionReadOut,
    "mlp_sum": MLPSumReadOut,
}


def get_read_out(
        read_out: Union[str, Tuple[str, ...]],
        read_out_kwargs: Dict
) -> torch.nn.Module:
    """ The 'get_read_out' function. """

    if "in_channels" not in read_out_kwargs:
        raise ValueError("Can't instantiate read_out without `in_channels` argument")

    if isinstance(read_out, list):
        return CombinedReadOut(read_out, read_out_kwargs)

    else:
        read_out_fn = READOUT_FUNCTIONS.get(read_out, None)

        if read_out_fn is None:
            raise ValueError(f"Unknown read_out function : {read_out}")

        return read_out_fn(**read_out_kwargs)
