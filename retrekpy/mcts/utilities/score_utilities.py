""" The ``retrekpy.mcts.utilities`` package ``score_utilities`` module. """

from functools import reduce
from numpy import mean
from operator import mul
from statistics import mean
from typing import List, Set

from rdkit.Chem import GetSymmSSSR, Mol, MolToSmiles, MolFromSmarts


class ScoreUtilities:
    """ The 'ScoreUtilities' class. """

    @staticmethod
    def calculate_cdscore(
            product: Mol,
            reactants: List[Mol]
    ) -> float:
        """
        The 'calculate_cdscore' method of the class.

        Input:
            product (Mol):
            reactants (List[Mol]):

        Output:
            (float) Return 1 if a molecule was divided evenly otherwise 0 <= x < 1.
        """

        if len(reactants) == 1:
            return 0.0

        pro_atom_num = product.GetNumAtoms()

        rct_atom_nums = [m.GetNumAtoms() for m in reactants]

        scale_factor = pro_atom_num / len(rct_atom_nums)

        abs_errors = [abs(r - scale_factor) for r in rct_atom_nums]

        return 1 / (1 + mean(abs_errors))

    @staticmethod
    def calculate_asscore(
            mol_condition_before: List[int],
            mol_condition_after: List[int],
            num_divided_mols: int
    ) -> float:
        """
        The 'calculate_asscore' method of the class.

        Input:
            mol_condition_before (List[int]):
            mol_condition_after (List[int]):
            num_divided_mols (int):

        Output:
            (float) Return 1 if all divided molecules were starting materials otherwise 0 <= x < 1.
        """

        if num_divided_mols == 1:
            return 0.0

        return (mol_condition_after.count(1) - mol_condition_before.count(1)) / num_divided_mols

    @staticmethod
    def __get_num_ring(
            mol: Mol
    ) -> int:
        """ The '__get_num_ring' method of the class. """

        try:
            ring_num = mol.GetRingInfo().NumRings()

        except:
            mol.UpdatePropertyCache()

            GetSymmSSSR(mol)

            ring_num = mol.GetRingInfo().NumRings()

        return ring_num

    @staticmethod
    def calculate_rdscore(
            product: Mol,
            reactants: List[Mol]
    ) -> float:
        """
        The 'calculate_rdscore' method of the class.

        Input:
            product (Mol):
            reactants (List[Mol]):

        Output:
            (float) Return 1 if a number of rings in a product is reduced otherwise 0.
        """

        pro_ring_num = ScoreUtilities.__get_num_ring(product)

        rct_ring_nums = sum([ScoreUtilities.__get_num_ring(m) for m in reactants])

        rdscore = pro_ring_num - rct_ring_nums

        return 1. if rdscore > 0 else 0.0

    @staticmethod
    def calculate_stscore(
            reactants: List[Mol],
            reaction_template: str
    ) -> float:
        """
        The 'calculate_stscore' method of the class.

        Input:
            reactants (List[Mol]):
            reaction_template (str):

        Output:
            (float) Return 1 if each reactant has a respective substructure in reaction template otherwise 1 / number of
                the combination.
        """

        patts_for_rct = [
            MolFromSmarts(patt)
            for patt in reaction_template.split(">>")[0].split(".")
        ]

        match_patts = list()

        for rct, patt in zip(reactants, patts_for_rct):
            match_patts.append(len(rct.GetSubstructMatches(patt, useChirality=True)))

        match_patts = [1 if patt == 0 else patt for patt in match_patts]

        return 1 / reduce(mul, match_patts)

    @staticmethod
    def calculate_intermediate_score(
            mols: List[Mol],
            intermediates: Set[str]
    ) -> float:
        """ The 'calculate_intermediate_score' method of the class. """

        return mean([
            MolToSmiles(m) in intermediates
            for m in mols
        ])
