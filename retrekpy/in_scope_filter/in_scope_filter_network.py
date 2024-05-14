""" The ``retrekpy.in_scope_filter`` package ``in_scope_filter_network`` module. """

import torch

from .highway_layer import HighwayLayer


class InScopeFilterNetwork(torch.nn.Module):
    """ The 'InScopeFilterNetwork' class. """

    def __init__(
            self
    ) -> None:
        """ The constructor method of the class. """

        super().__init__()

        self.product_network = torch.nn.Sequential(
            torch.nn.Linear(16384, 1024),
            torch.nn.ELU(),
            torch.nn.Dropout(0.3),
            HighwayLayer(1024, 5, torch.nn.functional.elu)
        )

        self.reaction_layer = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.ELU()
        )

        self.cosine_sim = torch.nn.CosineSimilarity()

    def forward(
            self,
            reaction: torch.Tensor,
            product: torch.Tensor,
            logits: bool = False
    ) -> torch.Tensor:
        """ The 'forward' method of the class. """

        r = self.reaction_layer(reaction)
        p = self.product_network(product)
        sim = self.cosine_sim(p, r).view(-1, 1)

        out = 10 * sim

        if logits:
            return out

        else:
            return torch.sigmoid(out)
