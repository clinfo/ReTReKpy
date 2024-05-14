""" The ``retrekpy.mcts.mcts`` package ``mcts`` module. """

import logging
import os

from copy import deepcopy
from pathlib import Path
from time import time
from tqdm import tqdm
from typing import List, Optional, Set, Tuple, Union

from rdkit.Chem import Mol, MolToSmiles
from rdkit.Chem.rdChemReactions import ChemicalReaction

from .chemistry_helper import ChemistryHelper
from .mcts_node import MCTSNode
from .mcts_state import MCTSState
from .utilities import MoleculeUtilities, ReactionUtilities, ScoreUtilities, SearchUtilities


os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class MCTS:
    """ The 'MCTS' class. """

    def __init__(
            self,
            target_mol: Mol,
            max_atom_num: int,
            expansion_rules: Union[List[str], List[ChemicalReaction]],
            rollout_rules: Union[List[str], List[ChemicalReaction]],
            starting_materials: Set[str],
            intermediate_materials: Set[str],
            template_scores,
            knowledge: str,
            knowledge_weights: List[float],
            save_tree: bool,
            search_count: int,
            selection_constant: int,
            save_result_dir: str,
            cum_prob_mod: bool,
            cum_prob_thresh: float,
            expansion_num: int,
            rollout_depth: int
    ) -> None:
        """ The constructor method of the class. """

        self.TARGET_MOL = target_mol
        self.MAX_ATOM_NUM = max_atom_num
        self.EXPANSION_RULES = expansion_rules
        self.ROLLOUT_RULES = rollout_rules
        self.STARTING_MATERIALS = starting_materials
        self.INTERMEDIATE_MATERIALS = intermediate_materials
        self.ROOT_NODE = MCTSNode(MCTSState([self.TARGET_MOL]))
        self.TEMPLATE_SCORES = template_scores

        self.KNOWLEDGE = knowledge
        self.KNOWLEDGE_WEIGHTS = knowledge_weights
        self.SAVE_TREE = save_tree
        self.SEARCH_COUNT = search_count
        self.SELECTION_CONSTANT = selection_constant
        self.SAVE_RESULT_DIR = save_result_dir
        self.CUM_PROB_MOD = cum_prob_mod
        self.CUM_PROB_THRESH = cum_prob_thresh
        self.EXPANSION_NUM = expansion_num
        self.ROLLOUT_DEPTH = rollout_depth

    def _is_necessary_to_compute(
            self,
            knowledge_score: str
    ) -> bool:
        """ The '_is_necessary_to_compute' method of the class. """

        weight_index = {
            "cdscore": 0,
            "rdscore": 1,
            "asscore": 2,
            "stscore": 3,
            "intermediate_score": 4,
            "template_score": 5
        }[knowledge_score]

        return (knowledge_score in self.KNOWLEDGE or "all" in self.KNOWLEDGE) and (self.KNOWLEDGE_WEIGHTS[weight_index] > 0)

    def search(
            self,
            expansion_model,
            rollout_model,
            in_scope_model,
            logger: logging.Logger,
            chemistry_helper: Optional[ChemistryHelper] = None,
            time_limit: Optional[int] = None
    ) -> Tuple[MCTSNode, bool]:
        """
        Implementation of Monte Carlo Tree Search.

        Input:
            expansion_model:
            rollout_model:
            in_scope_model:
            logger (Logger):
            chemistry_helper (Optional[ChemistryHelper]):
            time_limit (Optional[int]):

        Output:
            Node class and True if a reaction route is found or Node class and False otherwise.
        """

        header = "self node\tparent node\tdepth\tscore\tRDScore\tCDScore\tSTScore\tASScore\tIntermediateScore\tTemplateScore"

        tree_info = [header] if self.SAVE_TREE else None

        start = time()

        reaction_util = ReactionUtilities()

        for c in tqdm(range(self.SEARCH_COUNT)):
            if time_limit and time() - start > time_limit:
                break

            if self.ROOT_NODE.visits != 0:
                logger.debug(
                    f"Count: {c} Root: visits: {self.ROOT_NODE.visits} "
                    f"Total scores: {self.ROOT_NODE.total_scores / self.ROOT_NODE.visits}"
                )

            # Selection
            tmp_node = self.ROOT_NODE

            while tmp_node.has_child:
                tmp_node = tmp_node.select_node(self.SELECTION_CONSTANT, logger)

            # Expansion
            for mol, mol_cond, mol_idx in zip(tmp_node.state.mols, tmp_node.state.mol_conditions, range(len(tmp_node.state.mols))):
                if mol_cond == 1:
                    continue

                if mol.GetNumAtoms() > self.MAX_ATOM_NUM:
                    MCTS.back_propagation(tmp_node, -1)
                    break

                new_rxn_rules, tmp_node.rxn_probs = reaction_util.predict_reactions(
                    rxn_rules=self.EXPANSION_RULES,
                    model=expansion_model,
                    mol=mol,
                    cum_prob_mod=self.CUM_PROB_MOD,
                    cum_prob_thresh=self.CUM_PROB_THRESH,
                    expansion_num=self.EXPANSION_NUM
                )

                for i in range(len(new_rxn_rules)):
                    divided_mols_list = reaction_util.react_product_to_reactants(
                        product=mol,
                        rxn_rule=new_rxn_rules[i],
                        chemistry_helper=chemistry_helper
                    )

                    if not divided_mols_list:
                        score = -1.0 / len(new_rxn_rules)

                        MCTS.back_propagation(tmp_node, score)

                        continue

                    divided_mols_list = reaction_util.filter_in_scope_reactions(
                        in_scope_model=in_scope_model,
                        product_mol=mol,
                        reactants_set_list=divided_mols_list
                    )

                    for divided_mols in divided_mols_list:
                        if SearchUtilities.is_loop_route(
                            mols=divided_mols,
                            node=tmp_node
                        ):
                            continue

                        new_mols = deepcopy(tmp_node.state.mols)
                        new_mol_conditions = deepcopy(tmp_node.state.mol_conditions)

                        logger.debug(f"A depth of new node: {tmp_node.depth}\n")
                        logger.debug(f"Reaction template: {new_rxn_rules[i]}")
                        logger.debug(f"Before mol condition: {new_mol_conditions}")
                        logger.debug([MolToSmiles(m) for m in new_mols])

                        MoleculeUtilities.update_mol_condition(
                            mol_conditions=new_mol_conditions,
                            mols=new_mols,
                            divided_mols=divided_mols,
                            starting_materials=self.STARTING_MATERIALS,
                            idx=mol_idx
                        )

                        logger.debug(f"After mol condition: {new_mol_conditions}")
                        logger.debug([MolToSmiles(m) for m in new_mols])

                        # Computing the knowledge scores.
                        cdscore, rdscore, stscore, asscore, intermediate_score, template_score = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

                        if self._is_necessary_to_compute("cdscore"):
                            cdscore = ScoreUtilities.calculate_cdscore(
                                product=mol,
                                reactants=divided_mols
                            )

                        if self._is_necessary_to_compute("rdscore"):
                            rdscore = ScoreUtilities.calculate_rdscore(
                                product=mol,
                                reactants=divided_mols
                            )

                        if self._is_necessary_to_compute("stscore"):
                            stscore = ScoreUtilities.calculate_stscore(
                                reactants=divided_mols,
                                reaction_template=new_rxn_rules[i]
                            )

                        if self._is_necessary_to_compute("asscore"):
                            asscore = ScoreUtilities.calculate_asscore(
                                mol_condition_before=tmp_node.state.mol_conditions,
                                mol_condition_after=new_mol_conditions,
                                num_divided_mols=len(divided_mols)
                            )

                        if self._is_necessary_to_compute("intermediate_score"):
                            intermediate_score = ScoreUtilities.calculate_intermediate_score(
                                mols=new_mols,
                                intermediates=self.INTERMEDIATE_MATERIALS
                            )

                        if self._is_necessary_to_compute("template_score"):
                            template_score = self.TEMPLATE_SCORES.get(new_rxn_rules[i], 0)

                        new_state = MCTSState(
                            mols=new_mols,
                            rxn_rule=new_rxn_rules[i],
                            mol_conditions=new_mol_conditions,
                            rxn_applied_mol_idx=mol_idx,
                            stscore=stscore,
                            cdscore=cdscore,
                            rdscore=rdscore,
                            asscore=asscore,
                            intermediate_score=intermediate_score,
                            template_score=template_score,
                            knowledge=self.KNOWLEDGE,
                            knowledge_weights=self.KNOWLEDGE_WEIGHTS
                        )

                        leaf_node = tmp_node.add_node(
                            st=new_state,
                            new_node_prob=tmp_node.rxn_probs[i]
                        )

                        if SearchUtilities.is_proved(
                            mol_conditions=new_mol_conditions
                        ):
                            MCTS.back_propagation(
                                node=leaf_node,
                                score=10.0
                            )

                            if self.SAVE_TREE:
                                tree_info.append(
                                    MCTS.get_node_info(
                                        node=leaf_node,
                                        ws=self.KNOWLEDGE_WEIGHTS
                                    )
                                )

                                with open(os.path.join(self.SAVE_RESULT_DIR, "tree_log.csv"), "w") as file_handle:
                                    file_handle.write("\n".join(tree_info))

                            return leaf_node, True

            if tmp_node.has_child:
                # Select most promising leaf node.
                leaf_node = tmp_node.select_node(
                    constant=self.SELECTION_CONSTANT,
                    logger=logger
                )

                # Rollout.
                score = leaf_node.rollout(
                    reaction_util=reaction_util,
                    rxn_rules=self.ROLLOUT_RULES,
                    rollout_model=rollout_model,
                    starting_materials=self.STARTING_MATERIALS,
                    rollout_depth=self.ROLLOUT_DEPTH,
                    cum_prob_mod=self.CUM_PROB_MOD,
                    cum_prob_thresh=self.CUM_PROB_THRESH,
                    expansion_num=self.EXPANSION_NUM,
                    max_atom_num=self.MAX_ATOM_NUM,
                    chemistry_helper=chemistry_helper
                )

                # Back propagation.
                MCTS.back_propagation(
                    node=leaf_node,
                    score=score
                )

                if self.SAVE_TREE:
                    tree_info.append(
                        MCTS.get_node_info(
                            node=leaf_node,
                            ws=self.KNOWLEDGE_WEIGHTS
                        )
                    )

            else:
                MCTS.back_propagation(
                    node=tmp_node,
                    score=-1
                )

        if self.SAVE_TREE:
            with open(os.path.join(self.SAVE_RESULT_DIR, "tree_log.csv"), "w") as file_handle:
                file_handle.write("\n".join(tree_info))

        # For returning the leaf node of the current best route.
        leaf_node = self.ROOT_NODE.get_best_leaf(
            logger=logger
        )

        return leaf_node, False

    @staticmethod
    def back_propagation(
            node: MCTSNode,
            score: float
    ) -> None:
        """
        The 'back_propagation' method of the class.

        Input:
            node (MCTSNode):
            score (float):
        """

        while node is not None:
            node.update(score)

            node = node.parent_node

    @staticmethod
    def save_route(
            nodes: List[MCTSNode],
            save_dir: str,
            is_proven: bool,
            ws: List[int]
    ) -> None:
        """
        Save the searched reaction route.

        Input:
            nodes (List[MCTSNode]): List of reaction route nodes.
            save_dir (str):
            is_proven (bool): Reaction route search has done or not.
            ws (List[int]): Knowledge weights. [cdscore, rdscore, asscore, stscore, intermediate_score, template_score]
        """

        is_proven = "proven" if is_proven else "not_proven"

        Path(os.path.join(save_dir, is_proven)).touch()

        mols_nodes = [".".join([MolToSmiles(mol) for mol in node.state.mols]) for node in nodes]

        state_save_path = os.path.join(save_dir, "state.sma")

        with open(state_save_path, "w") as file_handle:
            file_handle.write("\n".join(mols_nodes))

        reaction_save_path = os.path.join(save_dir, "reaction.sma")

        rxns = [node.state.rxn_rule for node in nodes if node.state.rxn_rule is not None]

        with open(reaction_save_path, "w") as file_handle:
            file_handle.write("\n".join(rxns))

        tree_save_path = os.path.join(save_dir, "best_tree_info.csv")

        tree_info = ["self node\tparent node\tdepth\tscore\tRDScore\tCDScore\tASScore\tIntermediateScore\tTemplateScore"]

        tree_info.extend([MCTS.get_node_info(node, ws) for node in nodes])

        with open(tree_save_path, "w") as file_handle:
            file_handle.write("\n".join(tree_info))

    @staticmethod
    def print_route(
            nodes: List[MCTSNode],
            is_proven: bool,
            logger: logging.Logger
    ) -> None:
        """
        Print the searched route.

        Input:
            nodes (List[MCTSNode]): List of reaction route nodes.
            is_proven (bool): Reaction route search has done or not.
            logger (Logger): The logger.
        """

        logger.info(
            msg="Reaction route search done." if is_proven else "[INFO] Can't find any route..."
        )

        route_summary = ""
        route_summary += "\n\n################### Starting Material(s) ###################"

        rxn_rule = None
        idx = -1

        for node in nodes:
            route_summary += (
                f"\n------ Visit frequency to node: {node.visits} --------\n"
                f"The total score: {node.total_scores / node.visits}\n"
                f"The node depth: {node.depth}\n"
            )

            if rxn_rule is not None:
                route_summary += f"[INFO] Apply reverse reaction rule: {rxn_rule}\n"

            rxn_rule = node.state.rxn_rule

            if idx != -1:
                route_summary += f"[INFO] Reaction applied molecule index: {idx}\n"

            idx = node.state.rxn_applied_mol_idx

            for i in range(len(node.state.mols)):
                route_summary += f"{i}: {MolToSmiles(node.state.mols[i])}\n"

        route_summary += "###################### Target Molecule #####################\n"

        logger.info(
            msg=route_summary
        )

    @staticmethod
    def get_node_info(
            node: MCTSNode,
            ws: List[float]
    ) -> str:
        """
        The 'get_node_info' method of the class.

        Input:
            node (MCTSNode):
            ws (List[float]): Knowledge weights. [cdscore, rdscore, asscore, stscore, intermediate_score, template_score]

        Output:
            (str) The node information for a searched tree analysis.
        """

        return (
            f"{id(node)}\t"
            f"{id(node.parent_node)}\t"
            f"{node.depth}\t"
            f"{node.total_scores / node.visits}\t"
            f"{node.state.rdscore}\t"
            f"{node.state.cdscore * ws[0]}\t"
            f"{node.state.stscore * ws[3]}\t"
            f"{node.state.asscore * ws[2]}\t"
            f"{node.state.intermediate_score * ws[4]}\t"
            f"{node.state.template_score * ws[5]}\t"
        )
