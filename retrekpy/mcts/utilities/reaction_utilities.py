""" The ``retrekpy.mcts.utilities`` package ``reaction_utilities`` module. """

import numpy
import os
import torch

from typing import List, Optional, Tuple

from rdkit.Chem import Mol, MolFromSmiles, MolToSmarts, MolToSmiles, RWMol
from rdkit.Chem.rdChemReactions import ChemicalReaction, ReactionFromSmarts, ReactionToSmarts

from .molecule_utilities import MoleculeUtilities

from ..chemistry_helper import ChemistryHelper


class ReactionUtilities:
    """ The 'ReactionUtilities' class. """

    def __init__(
            self,
            mol: Mol = None
    ) -> None:
        """
        The constructor method of the class.

        Input:
            mol (Mol Object):
        """

        self.mol = mol
        self.predict_reaction_cache = {}
        self.product_to_reactants_cache = {}
        self.rxn_candidates = []
        self.sorted_rxn_prob_list = None
        self.sorted_rxn_prob_idxs = None

    def react_product_to_reactants(
            self,
            product: Mol,
            rxn_rule: ChemicalReaction,
            chemistry_helper: Optional[ChemistryHelper] = None
    ) -> Optional[List[Mol]]:
        """
        The 'react_product_to_reactants' method of the class.

        Input:
            product (Mol):
            rxn_rule (ChemicalReaction):
            chemistry_helper (ChemistryHelper):

        Output:
            (Optional[List[Mol]])
        """

        return_list = list()

        if (product, rxn_rule) in self.product_to_reactants_cache:
            return self.product_to_reactants_cache[(product, rxn_rule)]

        if chemistry_helper:
            try:
                product_mol = MolToSmiles(product)

                reactants_list = chemistry_helper.react_product_to_reactants(
                    product_mol=product_mol,
                    rxn_template=rxn_rule
                )

                for reactants in reactants_list:
                    if reactants is None or None in reactants:
                        continue

                    reactants = [MolFromSmiles(m) for m in reactants]

                    if reactants and None not in reactants:
                        return_list.append(reactants)

                self.product_to_reactants_cache[(product, rxn_rule)] = return_list if return_list else None

                return self.product_to_reactants_cache[(product, rxn_rule)]

            except:
                self.product_to_reactants_cache[(product, rxn_rule)] = None

                return None

        if ChemicalReaction.Validate(rxn_rule)[1] == 1 or rxn_rule.GetNumReactantTemplates() != 1:
            return None

        reactants_list = rxn_rule.RunReactants([product, ])

        if not reactants_list:
            return None

        for reactants in reactants_list:
            for reactant in reactants:
                if not MoleculeUtilities.is_valid(reactant):
                    continue

            return_list.append(reactants)

        return return_list if return_list else None

    def set_reaction_candidates_and_probabilities(
            self,
            model,
            rxn_rules: List[ChemicalReaction],
            expansion_num: int
    ) -> None:
        """
        The 'set_reaction_candidates_and_probabilities' method of the class.

        Input:
            model (CustomizedPredictor):
            rxn_rules (list[Chemical Reaction]):
            expansion_num (int):
        """

        canonical_smiles = MolToSmiles(self.mol, canonical=True)

        if canonical_smiles in self.predict_reaction_cache:
            sorted_rxn_prob_list, sorted_rxn_prob_idxs = self.predict_reaction_cache[canonical_smiles]

        else:
            rxn_prob_list = model.predict(
                compound=self.mol
            )

            sorted_rxn_prob_idxs = numpy.argsort(-rxn_prob_list)[:expansion_num]
            sorted_rxn_prob_list = rxn_prob_list[sorted_rxn_prob_idxs][:expansion_num]

            self.predict_reaction_cache[canonical_smiles] = (sorted_rxn_prob_list, sorted_rxn_prob_idxs)

        self.sorted_rxn_prob_idxs = sorted_rxn_prob_idxs
        self.sorted_rxn_prob_list = sorted_rxn_prob_list

        self.rxn_candidates = self.get_reaction_candidates(rxn_rules, expansion_num)

    @staticmethod
    def get_reactions(
            rxn_rule_path: str,
            save_dir: str,
            use_reaction_complement: bool = False
    ) -> List[str]:
        """ The 'get_reactions' method of the class. """

        def complement_reaction(
                rxn_template_local: ChemicalReaction
        ) -> None:
            """ The 'complement_reaction' method of the class. """

            if rxn_template_local.GetNumProductTemplates() != 1:
                raise Exception("[ERROR] A reaction template has only one product template.")

            pro = rxn_template_local.GetProductTemplate(0)

            rw_pro = RWMol(pro)

            amaps_pro = {a.GetAtomMapNum() for a in pro.GetAtoms()}
            amaps_rcts = {a.GetAtomMapNum() for rct in rxn_template_local.GetReactants() for a in rct.GetAtoms()}
            amaps_not_in_rcts = amaps_pro.intersection(amaps_rcts)

            for amap in amaps_not_in_rcts:
                aidx = [a.GetIdx() for a in rw_pro.GetAtoms() if a.GetAtomMapNum() == amap][0]
                rw_pro.RemoveAtom(aidx)

            m = rw_pro.GetMol()

            if "." in MolToSmarts(m):
                return

            if (m.GetNumAtoms() == 0) or (m.GetNumAtoms() == 1 and m.GetAtomWithIdx(0).GetSymbol() in {"*", None}):
                return

            rxn_template_local.AddReactantTemplate(m)

        with open(rxn_rule_path) as file_handle:
            lines = [line.strip("\n") for line in file_handle.readlines()]

        if use_reaction_complement:
            rxn_templates = list()

            for line in lines:
                try:
                    rxn_templates.append(ReactionFromSmarts(line))

                except:
                    rxn_templates.append(line)

            for rxn_template in rxn_templates:
                if isinstance(rxn_template, ChemicalReaction):
                    complement_reaction(rxn_template)

            out_reactions = [
                ReactionToSmarts(rt) if isinstance(rt, ChemicalReaction) else rt
                for rt in rxn_templates
            ]

            basename, ext = os.path.splitext(os.path.basename(rxn_rule_path))

            with open(os.path.join(save_dir, f"{basename}_complemented{ext}"), "w") as file_handle:
                file_handle.writelines("\n".join(out_reactions))

            return out_reactions

        else:
            return lines

    @staticmethod
    def get_reverse_reactions(
            rxn_rule_path: str
    ) -> List[ChemicalReaction]:
        """
        The 'get_reverse_reactions' method of the class.

        Input:
            rxn_rule_path (str):

        Output:
            (List[ChemicalReaction])
        """

        with open(rxn_rule_path) as file_handle:
            lines = file_handle.readlines()

        split_rxn_rules = [line.strip().split(">>") for line in lines]

        reverse_rxn_str = [">>".join(split_rxn_rule[::-1]) for split_rxn_rule in split_rxn_rules]

        return [ReactionFromSmarts(r) for r in reverse_rxn_str]

    def get_reaction_candidates(
            self,
            rxn_rules: List[ChemicalReaction],
            expansion_num: int,
            top_number: Optional[int] = None,
            cum_prob_thresh: float = 0.0
    ) -> List[ChemicalReaction]:
        """
        The 'get_reaction_candidates' method of the class.

        Input:
            rxn_rules (List[ChemicalReaction]):
            expansion_num (int):
            top_number (Optional[int]):
            cum_prob_thresh (float):

        Output:
            (List[ChemicalReaction])
        """

        counter_limit = top_number if top_number is not None else expansion_num

        probs = self.sorted_rxn_prob_list[:counter_limit]
        idxs = self.sorted_rxn_prob_idxs[:counter_limit]

        if cum_prob_thresh:
            cum_probs = numpy.cumsum(probs)

            pruned = max(1, len(cum_probs[cum_probs < cum_prob_thresh]))

            probs = probs[:pruned]

            idxs = idxs[:pruned]

        rxn_cands = [rxn_rules[i] for i in idxs]

        if top_number is None:
            self.sorted_rxn_prob_list = probs

        return rxn_cands

    def predict_reactions(
            self,
            rxn_rules: List[ChemicalReaction],
            model,
            mol: Mol,
            cum_prob_mod: bool,
            cum_prob_thresh: float,
            expansion_num: int,
            top_number: Optional[int] = None
    ) -> Tuple[List[ChemicalReaction], numpy.ndarray]:
        """
        The 'predict_reactions' method of the class.

        Input:
            rxn_rules (List[ChemicalReaction]):
            model (CustomizedPredictor):
            mol (Mol):
            cum_prob_mod (bool):
            cum_prob_thresh (float):
            expansion_num (int):
            top_number (int): If not None, get top-N prediction values.

        Returns:
            (Tuple[List[ChemicalReaction], numpy.ndarray]) Lists of predicted chemical reactions and probabilities.
        """

        self.mol = mol

        self.set_reaction_candidates_and_probabilities(
            model=model,
            rxn_rules=rxn_rules,
            expansion_num=expansion_num
        )

        cum_prob_thresh = cum_prob_thresh if cum_prob_mod else 0

        if top_number is None:
            return self.get_reaction_candidates(
                rxn_rules=rxn_rules,
                expansion_num=expansion_num,
                cum_prob_thresh=cum_prob_thresh
            ), self.sorted_rxn_prob_list

        else:
            return self.get_reaction_candidates(
                rxn_rules=rxn_rules,
                expansion_num=expansion_num,
                top_number=top_number,
                cum_prob_thresh=cum_prob_thresh
            ), self.sorted_rxn_prob_list

    def filter_in_scope_reactions(
            self,
            in_scope_model,
            product_mol: Mol,
            reactants_set_list: List[List[Mol]]
    ) -> List[List[Mol]]:
        """ The 'filter_in_scope_reactions' method of the class. """

        if in_scope_model is None:
            return reactants_set_list

        product_input = torch.log(torch.FloatTensor(MoleculeUtilities.generate_count_ecfp(product_mol, bits=16384)) + 1)
        product_fgp = MoleculeUtilities.generate_count_ecfp(product_mol)

        reaction_input = list()

        for reactants in reactants_set_list:
            reaction_input.append(torch.FloatTensor(
                MoleculeUtilities.generate_reaction_count_ecfp(
                    product=None,
                    reactants=reactants,
                    pre_computed_product_fgp=product_fgp
                )
            ))

        product_input = product_input.repeat(len(reactants_set_list), 1)

        reaction_input = torch.stack(reaction_input)

        in_scope_probs = in_scope_model(reaction_input, product_input)

        valid_reactants = [r for p, r in zip(in_scope_probs, reactants_set_list) if p > 0.5]

        return valid_reactants
