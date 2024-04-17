""" The ``retrekpy.mcts`` package ``mcts_state`` module. """

from numpy import mean
from typing import List, Optional

from rdkit.Chem import Mol
from rdkit.Chem.rdChemReactions import ChemicalReaction


class MCTSState:
    """ The 'MCTSState' class. """

    def __init__(
            self,
            mols: List[Mol],
            rxn_rule: Optional[ChemicalReaction] = None,
            mol_conditions: Optional[List[int]] = None,
            rxn_applied_mol_idx: Optional[int] = None,
            stscore: float = 0.0,
            cdscore: float = 0.0,
            rdscore: float = 0.0,
            asscore: float = 0.0,
            intermediate_score: float = 0.0,
            template_score: float = 0.0,
            knowledge: str = "all",
            knowledge_weights: Optional[List[float]] = None
    ) -> None:
        """
        The constructor method of the class.

        Input:
            mols (list[Mol Object]): RDKit Mol Object
            rxn_rule (Chem Reaction): RDKit Chemical Reaction
            mol_conditions (list[int]): A condition of molecules. "1" if a molecule is in building blocks "0" otherwise
            rxn_applied_mol_idx (int): The index of a reaction-applied molecule in mols
            stscore (float):
            cdscore (float):
            rdscore (float):
            asscore (float):
            intermediate_score (float):
            template_score (float):
            knowledge (str):
            knowledge_weights (Optional[List[float]]):
        """

        self.mols = mols
        self.rxn_rule = rxn_rule

        self.mol_conditions = [0] if mol_conditions is None else mol_conditions
        self.rxn_applied_mol_idx = None if rxn_applied_mol_idx is None else rxn_applied_mol_idx

        self.stscore = stscore
        self.cdscore = cdscore
        self.rdscore = rdscore
        self.asscore = asscore
        self.intermediate_score = intermediate_score
        self.template_score = template_score

        self.knowledge_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, ] if knowledge_weights is None else knowledge_weights

        knowledge_score = list()

        if "cdscore" in knowledge or "all" in knowledge:
            knowledge_score.append(self.knowledge_weights[0] * self.cdscore)

        if "rdscore" in knowledge or "all" in knowledge:
            knowledge_score.append(self.knowledge_weights[1] * self.rdscore)

        if "asscore" in knowledge or "all" in knowledge:
            knowledge_score.append(self.knowledge_weights[2] * self.asscore)

        if "stscore" in knowledge or "all" in knowledge:
            knowledge_score.append(self.knowledge_weights[3] * self.stscore)

        if "intermediate_score" in knowledge or "all" in knowledge:
            knowledge_score.append(self.knowledge_weights[4] * self.intermediate_score)

        if "template_score" in knowledge or "all" in knowledge:
            knowledge_score.append(self.knowledge_weights[5] * self.template_score)

        self.knowledge_score = mean(knowledge_score) if knowledge_score else 0
