""" The ``retrekpy.in_scope_filter`` package ``reaction_dataset`` module. """

import numpy
import pandas
import torch

from typing import Tuple, Union

from rdkit.Chem import Mol, MolFromSmiles, SanitizeMol
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from torch.utils.data import Dataset


class ReactionDataset(Dataset):
    """ The 'ReactionDataset' class. """

    def __init__(
            self,
            csv_path: str,
            reaction_column: str,
            product_column: str,
            label_column: str
    ) -> None:
        """ The constructor method of the class. """

        super().__init__()

        self.df = pandas.read_csv(csv_path)

        self.reaction_column = reaction_column
        self.product_column = product_column
        self.label_column = label_column

    def __len__(
            self
    ) -> int:
        """ The '__len__' method of the class. """

        return len(self.df)

    def __getitem__(
            self,
            idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ The '__getitem__' method of the class. """

        row = self.df.iloc[idx]

        reaction = torch.FloatTensor(ReactionDataset.get_reaction_count_fgp(row[self.reaction_column]))
        product = torch.log(torch.FloatTensor(ReactionDataset.get_count_fgp(row[self.product_column], n_bits=16384)) + 1)

        y = torch.FloatTensor([row[self.label_column]])

        return reaction, product, y

    @staticmethod
    def get_count_fgp(
            mol: Union[str, Mol],
            n_bits: int = 2048,
            radius: int = 2
    ) -> numpy.ndarray:
        """ The 'get_count_fgp' method of the class. """

        if isinstance(mol, str):
            mol = MolFromSmiles(mol)

        SanitizeMol(mol)

        bit_info = {}

        fgp = numpy.zeros(n_bits)

        GetMorganFingerprintAsBitVect(mol, radius, n_bits, bitInfo=bit_info)

        for bit_id, active in bit_info.items():
            fgp[bit_id] = len(active)

        return fgp

    @staticmethod
    def get_reaction_count_fgp(
            reaction: str,
            n_bits: int = 2048,
            radius: int = 2
    ) -> numpy.ndarray:
        """ The 'get_reaction_count_fgp' method of the class. """

        reactants = reaction.split(">")[0]
        product = reaction.split(">")[-1]

        p_fgp = ReactionDataset.get_count_fgp(product, n_bits, radius)
        r_fgp = ReactionDataset.get_count_fgp(reactants, n_bits, radius)

        return p_fgp - r_fgp
