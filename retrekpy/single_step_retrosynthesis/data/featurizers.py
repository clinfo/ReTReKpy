""" The ``retrekpy.single_step_retrosynthesis.data`` package ``featurizers`` module. """

import torch

from functools import partial
from itertools import chain
from typing import Callable, List, Optional

from rdkit.Chem import Atom, Bond, MolFromSmiles

from torch_geometric.data import Data as TorchGeometricData

from kmol.core.exceptions import FeaturizationError
from kmol.data.featurizers import AbstractDescriptorComputer, AbstractTorchGeometricFeaturizer
from kmol.data.resources import DataPoint


class CustomizedGraphFeaturizer(AbstractTorchGeometricFeaturizer):
    """ The customized 'GraphFeaturizer' class. """

    DEFAULT_ATOM_TYPES = ["B", "C", "N", "O", "F", "Na", "Si", "P", "S", "Cl", "K", "Br", "I"]

    def __init__(
        self,
        inputs: List[str],
        outputs: List[str],
        descriptor_calculator: AbstractDescriptorComputer,
        allowed_atom_types: Optional[List[str]] = None,
        should_cache: bool = False,
        rewrite: bool = True,
    ) -> None:
        """ The customized constructor method of the class. """

        super().__init__(inputs, outputs, should_cache, rewrite)

        if allowed_atom_types is None:
            allowed_atom_types = self.DEFAULT_ATOM_TYPES

        self._allowed_atom_types = allowed_atom_types
        self._descriptor_calculator = descriptor_calculator

    def _process(
            self,
            data: str,
            entry: DataPoint
    ) -> TorchGeometricData:
        """ The customized '_process' method of the class. """

        mol = MolFromSmiles(data)

        data = super()._process(data=data, entry=entry)

        molecule_features = self._descriptor_calculator.run(mol, entry)
        molecule_features = torch.FloatTensor(molecule_features).view(-1, len(molecule_features))

        data.molecule_features = molecule_features

        return data

    def process(
            self,
            data: str
    ) -> TorchGeometricData:
        """ The 'process' method of the class. """

        mol = MolFromSmiles(data)

        if mol is None:
            raise FeaturizationError("Could not featurize entry: [{}]".format(data))

        atom_features = self._get_vertex_features(mol)
        atom_features = torch.FloatTensor(atom_features).view(-1, len(atom_features[0]))

        edge_indices, edge_attributes = self._get_edge_features(mol)
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)
        edge_attributes = torch.FloatTensor(edge_attributes)

        if edge_indices.numel() > 0:
            permutation = (edge_indices[0] * atom_features.size(0) + edge_indices[1]).argsort()
            edge_indices, edge_attributes = edge_indices[:, permutation], edge_attributes[permutation]

        return TorchGeometricData(
            x=atom_features,
            edge_index=edge_indices,
            edge_attr=edge_attributes
        )

    def _featurize_atom(
            self,
            atom: Atom
    ) -> List[float]:
        """ The customized '_featurize_atom' method of the class. """

        return list(chain.from_iterable([
            featurizer(atom)
            for featurizer in self._list_atom_featurizers()
        ]))

    def _featurize_bond(
            self,
            bond: Bond
    ) -> List[float]:
        """ The customized '_featurize_bond' method of the class. """

        return list(chain.from_iterable([
            featurizer(bond)
            for featurizer in CustomizedGraphFeaturizer._list_bond_featurizers()
        ]))

    def _list_atom_featurizers(
            self
    ) -> List[Callable]:
        """ The customized '_list_atom_featurizers' method of the class. """

        from kmol.vendor.dgllife.utils.featurizers import (
            atom_type_one_hot,
            atom_degree_one_hot,
            atom_implicit_valence_one_hot,
            atom_formal_charge,
            atom_num_radical_electrons,
            atom_hybridization_one_hot,
            atom_is_aromatic,
            atom_total_num_H_one_hot,
        )

        return [
            partial(
                atom_type_one_hot,
                allowable_set=self._allowed_atom_types,
                encode_unknown=True
            ),
            atom_degree_one_hot,
            atom_implicit_valence_one_hot,
            atom_formal_charge,
            atom_num_radical_electrons,
            atom_hybridization_one_hot,
            atom_is_aromatic,
            atom_total_num_H_one_hot,
        ]

    @staticmethod
    def _list_bond_featurizers(
    ) -> List[Callable]:
        """ The customized '_list_bond_featurizers' method of the class. """

        from kmol.vendor.dgllife.utils.featurizers import (
            bond_type_one_hot,
            bond_is_conjugated,
            bond_is_in_ring,
            bond_stereo_one_hot,
        )

        return [
            bond_type_one_hot,
            bond_is_conjugated,
            bond_is_in_ring,
            bond_stereo_one_hot,
        ]
