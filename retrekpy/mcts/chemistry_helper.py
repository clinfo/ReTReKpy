""" The ``retrekpy.chemistry`` package `` chemistry_helper`` module. """

import multiprocessing

from typing import List, Optional, Tuple, Union

from rdkit.Chem import Mol

from ..utilities.data_utilities import DataUtilities
from ..utilities.format_conversion_utilities import FormatConversionUtilities
from ..utilities.reactivity_utilities import ReactivityUtilities


class ChemistryHelper:
    """ The chemistry helper class. """

    def __init__(
            self,
            reaction_rule_list_path: str,
            max_products: int = 2
    ) -> None:
        """
        Initialize the class with a rollout list of specified SMARTS reaction rules.

        Input:
            reaction_rule_list_path (str):
            max_products (int): Maximum number of products to consider when applying reverse reaction.
        """

        self.rollout_list = ChemistryHelper.__get_reaction_list(reaction_rule_list_path)

        self.max_products = max_products

    @staticmethod
    def __get_reaction_list(
            reaction_rule_list_path: str,
            file_ext: str = ".sma",
            file_sep: str = "\t",
            header: Optional[int] = None
    ) -> List[str]:
        """
        Read the file containing the list of available reaction rules in SMARTS string format.

        Input:
            reaction_rule_list_path (str): The path to the file containing the rules.
            file_ext (str):
            file_sep (str):
            header (Optional[int]):

        Output:
            (List[str]): The list of reaction rules in SMARTS string format.
        """

        try:
            rollout_list = DataUtilities.read_dataset(
                dataset_file_path=reaction_rule_list_path,
                dataset_file_extension=file_ext,
                separator=file_sep,
                header=header
            )

            return rollout_list.iloc[:, 0].values.tolist()

        except Exception as exception:
            raise Exception(
                "Exception occurred during the opening of the reaction rules SMARTS file. "
                "Detailed exception message: \n{}".format(exception)
            )

    @staticmethod
    def __get_mol_list(
            input_mols: Union[List[str], Tuple[str]],
            use_num_cores: int = 1
    ) -> List[Mol]:
        """
        Convert a list of molecules from SMILES strings to RDKit Mol objects.

        Input:
            input_mols (Union[List[str], Tuple[str]]): The collection of molecule SMILES strings that need to be
                converted.
            use_num_cores (int): The number of cores available for this task.

        Output:
            (List[Mol]): The list of the successfully converted RDKit Mol objects.
        """

        try:
            if use_num_cores == 1:
                final_mol_list = [
                    FormatConversionUtilities.smiles_to_mol(smiles=mol_smiles)
                    for mol_smiles in input_mols
                ]

            else:
                if use_num_cores <= multiprocessing.cpu_count():
                    use_num_cores = use_num_cores

                else:
                    use_num_cores = multiprocessing.cpu_count()

                with multiprocessing.Pool(use_num_cores) as process_pool:
                    final_mol_list = [
                        processed_entry
                        for processed_entry in process_pool.map(FormatConversionUtilities.smiles_to_mol, input_mols)
                    ]

            # Before returning, eliminate the SMILES strings which failed to convert to RDKit Mol objects.
            return [mol for mol in final_mol_list if mol is not None]

        except Exception as exception:
            raise Exception(
                "Exception occurred during the conversion of the molecules given in the specified list. "
                "Detailed exception message: \n{}".format(exception)
            )

    def is_terminal(
            self,
            input_mols: Union[List[str], Tuple[str]]
    ) -> bool:
        """
        Check whether a list of reactants represents a terminal state in the synthesis tree.

        Input:
            input_mols (Union[List[str], Tuple[str]]): The collection of reactant SMILES strings.

        Output:
            (bool): The flag indicating whether the combination of reactants is terminal.
        """

        for mol in input_mols:
            for rxn_rule in self.rollout_list:
                try:
                    # If a reaction rule can be applied on a molecule from the specified list, return False.
                    if len(self.react_product_to_reactants(mol, rxn_rule)) != 0:
                        return False

                except:
                    pass

        return True

    def react_product_to_reactants(
            self,
            product_mol: Union[str, Mol],
            rxn_template: str,
            use_library: str = "rdkit",
            return_type: str = "str"
    ) -> Union[List[Tuple[Optional[str], ...]], List[Tuple[Optional[Mol], ...]]]:
        """
        Apply a given reaction rule template to a specific product molecule(s).

        Input:
            product_mol (Union[str, Mol]): The product SMILES string or RDKit Mol object.
            rxn_template (str):
            use_library (str):
            return_type (str):

        Output:
            (Union[List[Tuple[Optional[str], ...]], List[Tuple[Optional[Mol], ...]]]): The list of potential reactant
                combinations generated by PyReactor.
        """

        if not isinstance(product_mol, str) and not isinstance(product_mol, Mol):
            raise Exception(
                "The specified product molecule needs to be either a SMILES string or a RDKit Mol object."
            )

        try:
            if use_library == "rdkit":
                reactants_suggestions = ReactivityUtilities.reverse_apply(
                    product_mols=product_mol,
                    rxn_template=rxn_template,
                    max_products=self.max_products,
                    return_type="str"
                )

            elif use_library == "rdchiral":
                reactants_suggestions = ReactivityUtilities.reverse_apply_rdchiral(
                    product_mols=product_mol,
                    rxn_template=rxn_template,
                    max_products=self.max_products,
                    return_type=return_type
                )

            else:
                raise Exception(
                    "The currently supported libraries are 'rdkit' or 'rdchiral'. Got: '{}'.".format(use_library)
                )

            if isinstance(product_mol, str):
                product_smiles = FormatConversionUtilities.smiles_to_canonical_smiles(product_mol)

            else:
                product_smiles = FormatConversionUtilities.mol_to_canonical_smiles(product_mol)

            valid_reactants_suggestions = list()

            for reactants in reactants_suggestions:
                potential_products = ReactivityUtilities.forward_apply(
                    reactant_mols=reactants,
                    rxn_template=rxn_template,
                    return_type=return_type
                )

                for products in potential_products:
                    if product_smiles in products:
                        valid_reactants_suggestions.append(reactants)

                        break

            return valid_reactants_suggestions

        except Exception as exception:
            raise Exception(
                "The reverse application of the reaction rule template was unsuccessful. "
                "Detailed exception message:\n{}".format(exception)
            )
