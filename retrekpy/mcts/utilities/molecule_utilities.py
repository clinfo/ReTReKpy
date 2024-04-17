""" The ``retrekpy.mcts.utilities`` package ``molecule_utilities`` module. """

import numpy

from typing import List, Optional, Set

from rdkit.Chem import Mol, MolToInchiKey, MolToSmiles, SANITIZE_NONE, SanitizeMol
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

from .search_utilities import SearchUtilities


class MoleculeUtilities:
    """ The 'MoleculeUtilities' class. """

    @staticmethod
    def generate_ecfp(
            mol: Mol,
            radius: int = 2,
            bits: int = 2048
    ) -> numpy.ndarray:
        """
        Create Extended Connectivity FingerPrint.

        Input:
            mol (Mol):
            radius (int):
            bits (int):

        Output:
            (numpy.ndarray) Numpy array type ECFP.
        """

        fp = GetMorganFingerprintAsBitVect(mol, radius, nBits=bits).ToBitString()

        return numpy.asarray([[int(i) for i in list(fp)]])

    @staticmethod
    def generate_count_ecfp(
            mol: Mol,
            radius: int = 2,
            bits: int = 2048
    ) -> numpy.ndarray:
        """
        Create Extended Connectivity Fingerprint with counts.

        Input:
            mol (Mol):
            radius (int):
            bits (int):

        Output:
            (numpy.ndarray) Numpy array type ECFP with counts
        """

        SanitizeMol(mol)

        bit_info = {}

        fgp = numpy.zeros(bits)

        GetMorganFingerprintAsBitVect(mol, radius, bits, bitInfo=bit_info)

        for bit_id, active in bit_info.items():
            fgp[bit_id] = len(active)

        return fgp

    @staticmethod
    def generate_reaction_count_ecfp(
            product: Optional[Mol],
            reactants: List[Mol],
            radius: int = 2,
            bits: int = 2048,
            pre_computed_product_fgp: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """
        Create Extended Connectivity Fingerprint with counts of reaction.

        Input:
            product (Optional[Mol]):
            reactants (List[Mol]):
            radius (int):
            bits (int):
            pre_computed_product_fgp (Optional[numpy.ndarray]):

        Output:
            (numpy.ndarray) Numpy array type ECFP with counts.
        """

        if pre_computed_product_fgp is None:
            p_fgp = MoleculeUtilities.generate_count_ecfp(product, radius, bits)

        else:
            p_fgp = pre_computed_product_fgp

        r_fgp = numpy.sum([
            MoleculeUtilities.generate_count_ecfp(r, radius, bits)
            for r in reactants
        ], axis=0)

        return p_fgp - r_fgp

    @staticmethod
    def update_mol_condition(
            mol_conditions: List[int],
            mols: List[Mol],
            divided_mols: List[Mol],
            starting_materials: Set[str],
            idx: int
    ) -> None:
        """
        Update the molecule condition if the molecules in start materials

        Input:
            mol_conditions (List[int]):
            mols (List[Mol]):
            divided_mols (List[Mol]):
            starting_materials (Set[str]):
            idx (int):
        """

        mols.pop(idx)
        mol_conditions.pop(idx)

        for divided_mol in divided_mols:
            mols.append(divided_mol)

            if "inchi" in starting_materials:
                SanitizeMol(divided_mol)

                to_check = MolToInchiKey(divided_mol)

            else:
                to_check = MolToSmiles(divided_mol, canonical=True)

            if SearchUtilities.sequential_search(to_check, starting_materials):
                mol_conditions.append(1)

            else:
                mol_conditions.append(0)

    @staticmethod
    def get_unsolved_mol_condition_idx(
            mol_conditions: List[int]
    ) -> List[int]:
        """
        Get indexes of mol_conditions whose condition is 0.

        Input:
            mol_conditions (List[int]):

        Output:
            (List[int]):
        """

        unsolved_idxs = list()

        for i in range(len(mol_conditions)):
            if mol_conditions[i] == 0:
                unsolved_idxs.append(i)

        return unsolved_idxs

    @staticmethod
    def is_valid(
            mol: Mol
    ) -> bool:
        """
        Check whether Mol Object is valid.

        Input:
            mol (Mol):
        Returns:
            (bool) True if mol is valid otherwise False.
        """

        flag = SanitizeMol(mol, catchErrors=True)

        return True if flag == SANITIZE_NONE else False
