""" The ``retrekpy.mcts.mcts`` package ``node`` module. """

import numpy

from copy import deepcopy
from logging import Logger
from math import sqrt
from random import choice
from typing import List, Optional, Set

from rdkit.Chem.rdChemReactions import ChemicalReaction

from .chemistry_helper import ChemistryHelper
from .mcts_state import MCTSState
from .utilities import MoleculeUtilities, ReactionUtilities, SearchUtilities


class MCTSNode:
    """ The 'MCTSNode' class. """

    def __init__(
            self,
            state: MCTSState,
            parent_node: Optional["MCTSNode"] = None,
            has_child: bool = False,
            depth: Optional[int] = None
    ) -> None:
        """
        The constructor method of the class.

        Input:
            state (MCTSState):
            parent_node (MCTSNode):
            has_child (bool): True if the Node has child Node or False otherwise.
            depth (int): A depth of the Node.
        """

        self.state = state
        self.parent_node = parent_node
        self.child_nodes = list()
        self.has_child = has_child
        self.depth = 0 if depth is None else depth

        self.node_probs = list()
        self.rxn_probs = 0.0
        self.total_scores = 0.0
        self.visits = 1
        self.max_length = 10

    def get_best_leaf(
            self,
            logger: Logger
    ) -> "MCTSNode":
        """
        Method to find best leaf of node based on scores/visits solely.

        Input:
            logger (Logger): The logger.

        Output:
            (MCTSNode) The leaf node with max total_scores/visits.
        """

        tmp_node = self

        while tmp_node.has_child:
            tmp_node = tmp_node.select_node(0, logger)

        return tmp_node

    def ucb(
            self,
            constant: float
    ) -> float:
        """
        Computes UCB for all child nodes.

        Args:
            constant (float): constant to use for UCB computation.

        Returns:
            (float) The UCB for all child nodes.
        """

        parent_visits = self.visits

        child_visits = numpy.array([node.visits for node in self.child_nodes])
        probs = numpy.array([node.node_probs[0] for node in self.child_nodes])

        knowledge_scores = numpy.array([node.state.knowledge_score for node in self.child_nodes])

        total_scores = numpy.array([node.total_scores for node in self.child_nodes])

        exploit = total_scores / child_visits
        explore = probs * sqrt(parent_visits) / (1 + child_visits)

        ucb = exploit + constant * explore + knowledge_scores

        return ucb

    def select_node(
            self,
            constant: int,
            logger: Logger
    ) -> "MCTSNode":
        """
        Selection implementation of MCTS. Define Q(st, a) to total_scores, N(st, a) to child_visits, N(st-1, a) to
        parent_visits and P(st, a) to p. p is a prior probability received from the expansion.

        Args:
            constant (int):
            logger (Logger): The logger.

        Returns:
            (MCTSNode) The node which has max ucb score.
        """

        ucb = self.ucb(constant)

        max_index = choice(numpy.where(ucb == ucb.max())[0])

        node_num = len(self.child_nodes)

        logger.debug(
            f"\n################ SELECTION ################\n"
            f"ucb_list:\n {ucb}\n"
            f"visit: \n{[self.child_nodes[i].visits for i in range(node_num)]}\n"
            f"child total scores: \n{[self.child_nodes[i].total_scores for i in range(node_num)]}\n"
            f"parent visits: {self.visits}\n"
            f"child node probs: \n{[self.child_nodes[i].node_probs for i in range(node_num)]}\n"
            f"############################################\n"
        )

        return self.child_nodes[max_index]

    def add_node(
            self,
            st: MCTSState,
            new_node_prob: numpy.ndarray
    ) -> "MCTSNode":
        """
        Add Node as child node to self.

        Args
            st (State):
            new_node_prob (numpy.ndarray):

        Returns:
            (MCTSNode) The child node which was added to the parent Node.
        """

        new_node = MCTSNode(
            state=st,
            parent_node=self,
            depth=self.depth+1
        )

        new_node.node_probs.append(new_node_prob)

        for p in self.node_probs:
            new_node.node_probs.append(deepcopy(p))

        self.child_nodes.append(new_node)

        if not self.has_child:
            self.has_child = True

        return new_node

    def rollout(
            self,
            reaction_util: ReactionUtilities,
            rxn_rules: List[ChemicalReaction],
            rollout_model,
            starting_materials: Set[str],
            rollout_depth: int,
            cum_prob_mod: bool,
            cum_prob_thresh: float,
            expansion_num: int,
            max_atom_num: int,
            chemistry_helper: Optional[ChemistryHelper] = None
    ) -> float:
        """
        Rollout implementation of MCTS.

        Input:
            reaction_util (ReactionUtilities):
            rxn_rules (List[ChemicalReaction]):
            rollout_model ():
            starting_materials (Set[str]):
            rollout_depth (int):
            cum_prob_mod (bool):
            cum_prob_thresh (float):
            expansion_num (int):
            max_atom_num (int):
            chemistry_helper (Optional[ChemistryHelper]):

        Output:
            A float type rollout score.
        """

        mol_cond = deepcopy(self.state.mol_conditions)
        mols = deepcopy(self.state.mols)

        rand_pred_rxns = list()

        # Before starting rollout, the state is first checked for being terminal or proved.
        unsolved_mols = [mols[i] for i in MoleculeUtilities.get_unsolved_mol_condition_idx(mol_cond)]

        if SearchUtilities.is_proved(mol_cond):
            return 10.0

        elif SearchUtilities.is_terminal(
            mols=unsolved_mols,
            chemistry_helper=chemistry_helper
        ):
            return -1.0

        else:
            for d in range(rollout_depth):
                rand_pred_rxns.clear()
                unsolved_indices = MoleculeUtilities.get_unsolved_mol_condition_idx(mol_cond)

                # Random pick a molecule from the unsolved molecules
                unsolved_idx = choice(unsolved_indices)
                rand_mol = mols[unsolved_idx]

                if rand_mol.GetNumAtoms() > max_atom_num:
                    return 0.0

                # Get top 10 reaction candidate from rand_mol
                rand_pred_rxns, self.rxn_probs = reaction_util.predict_reactions(
                    rxn_rules=rxn_rules,
                    model=rollout_model,
                    mol=rand_mol,
                    cum_prob_mod=cum_prob_mod,
                    cum_prob_thresh=cum_prob_thresh,
                    expansion_num=expansion_num,
                    top_number=10
                )

                # Random pick a reaction from the reaction candidate.
                rand_rxn_cand = choice(rand_pred_rxns)

                divided_mols_list = reaction_util.react_product_to_reactants(
                    product=rand_mol,
                    rxn_rule=rand_rxn_cand,
                    chemistry_helper=chemistry_helper
                )

                if not divided_mols_list:
                    continue

                MoleculeUtilities.update_mol_condition(
                    mol_conditions=mol_cond,
                    mols=mols,
                    divided_mols=choice(divided_mols_list),
                    starting_materials=starting_materials,
                    idx=unsolved_idx
                )

                if SearchUtilities.is_proved(
                    mol_conditions=mol_cond
                ):
                    break

            return mol_cond.count(1) / len(mol_cond)

    def update(
            self,
            score: float
    ) -> None:
        """
        Update implementation of MCTS.

        Input:
            score (float):
        """

        # The frequency of visits to the State.
        self.visits += 1

        prob = sum(self.node_probs)
        length_factor = self.depth - prob
        weight = max(.0, (self.max_length - length_factor) / self.max_length)

        q_score = score * weight

        self.total_scores += q_score
