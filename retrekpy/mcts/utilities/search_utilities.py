""" The ``retrekpy.mcts.utilities`` package ``search_utilities`` module. """

from typing import List, Set

from rdkit.Chem import Mol, MolToSmiles

from ..chemistry_helper import ChemistryHelper


class SearchUtilities:
    """ The 'SearchUtilities' class. """

    @staticmethod
    def sequential_search(
            smiles: str,
            start_materials: Set[str]
    ) -> bool:
        """
        The 'sequential_search' method of the class.

        Input:
            smiles (str):
            start_materials (Set[str]):

        Output:
            (bool)
        """

        return True if smiles in start_materials else False

    @staticmethod
    def is_proved(
            mol_conditions: List[int]
    ) -> bool:
        """
        The 'is_proved' method of the class.

        Input:
            mol_conditions (List[int]):

        Output:
            (bool)
        """

        return all([i == 1 for i in mol_conditions])

    @staticmethod
    def is_terminal(
            mols: List[Mol],
            chemistry_helper: ChemistryHelper
    ) -> bool:
        """
        The 'is_terminal' method of the class.

        Input:
            mols (List[Mol]):
            chemistry_helper (ChemistryHelper):

        Output:
            (bool)
        """

        return chemistry_helper.is_terminal(mols)

    @staticmethod
    def is_loop_route(
            mols: List[Mol],
            node
    ) -> bool:
        """
        Check whether a molecule is in a route.

        Input:
            mols (List[Mol]):
            node (MCTSNode):

        Output:
            (bool) True if a molecule is in a route, otherwise False.
        """

        mols = [MolToSmiles(m) for m in mols]

        while node is not None:
            unresolved_mols = set(node.state.mols[i] for i, c in enumerate(node.state.mol_conditions) if c == 0)
            unresolved_mols = [MolToSmiles(m) for m in unresolved_mols]

            for m in mols:
                if m in unresolved_mols:
                    return True

            node = node.parent_node

        return False
