""" The ``retrekpy.utilities`` package ``atom_mapping_utilities`` module. """

import os
import sys

from typing import Optional

from indigo import Indigo

from rdchiral.template_extractor import extract_from_reaction

from .format_conversion_utilities import FormatConversionUtilities


class AtomMappingUtilities:
    """ The class containing a multiprocessing-friendly wrapper for the Epam Indigo chemical reaction mapping API."""

    @staticmethod
    def atom_map_reaction(
            rxn_smiles: str,
            timeout_period: int,
            existing_mapping: str = "discard",
            verbose: bool = False
    ) -> Optional[str]:
        """
        Atom map a reaction SMILES string using the Epam Indigo reaction atom mapper API. Any existing mapping will be
        handled according to the value of the parameter 'existing_mapping'. Because it can be a time-consuming process,
        a timeout occurs after 'timeout_period' ms.

        Input:
            rxn_smiles (str): A reaction SMILES string representing a chemical reaction which is going to be mapped.
            timeout_period (int): A timeout which occurs after the set number of ms.
            existing_mapping (str): Method to handle any existing mapping: 'discard', 'keep', 'alter' or 'clear'.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Optional[str]): The mapped reaction SMILES string.
        """

        try:
            # Instantiate the Indigo class object and set the timeout period.
            indigo_mapper = Indigo()
            indigo_mapper.setOption("aam-timeout", timeout_period)

            # Return the atom mapping of the reaction SMILES string.
            rxn = indigo_mapper.loadReaction(rxn_smiles)
            rxn.automap(existing_mapping)

            return rxn.smiles()

        # If an exception occurs, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during atom mapping of the reaction SMILES. "
                    "Detailed message: {}".format(exception)
                )

            return None

    @staticmethod
    def extract_reaction_template(
            rxn_smiles: str,
            verbose: bool = False
    ) -> Optional[str]:
        """
        Extract a reaction template from a SMILES string using the RDChiral library. This function relies on the
        GetSubstructMatches function from RDKit and if the reaction contains many large molecules, the process can take
        a lot of time.

        Input:
            rxn_smiles (str): A reaction SMILES string from which the template is going to be extracted.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Optional[str]): The extracted reaction template in the form of a SMARTS string.
        """

        try:
            # Parse the reaction roles from the reaction SMILES.
            reactant_smiles, _, product_smiles = FormatConversionUtilities.rxn_smiles_to_rxn_roles(rxn_smiles)

            reactant_side = ".".join(reactant_smiles)
            product_side = ".".join(product_smiles)

            if not verbose:
                # Prevent function from printing junk.
                old_stdout = sys.stdout
                sys.stdout = open(os.devnull, "w")

                # Extract the templates from the reaction SMILES using RDChiral.
                reaction_template = extract_from_reaction({
                    "reactants": reactant_side,
                    "products": product_side,
                    "_id": "0"
                })

                sys.stdout = old_stdout

            else:
                # Extract the templates from the reaction SMILES using RDChiral.
                reaction_template = extract_from_reaction({
                    "reactants": reactant_side,
                    "products": product_side,
                    "_id": "0"
                })

            # Return the reaction SMARTS result if the processing finished correctly.
            if reaction_template is not None and "reaction_smarts" in reaction_template.keys():
                # Because RDChiral returns the switched template, switch the order of the reactants and products.
                reactant_side, _, product_side = reaction_template["reaction_smarts"].split(">")
                final_reaction_template = ">>".join([product_side, reactant_side])

                return final_reaction_template

            else:
                return None

        # If an exception occurs, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during the reaction rule template extraction from the reaction SMILES. "
                    "Detailed message: {}".format(exception)
                )

            return None
