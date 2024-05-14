""" The ``retrekpy.utilities`` package ``format_conversion_utils`` module. """

from typing import List, Optional, Tuple

from rdkit.Chem import Mol, MolFromSmarts, MolFromSmiles, MolToSmarts, MolToSmiles, SanitizeMol
from rdkit.Chem.rdChemReactions import ReactionFromSmarts, ReactionToSmarts, ReactionToSmiles, SanitizeRxn


class FormatConversionUtilities:
    """ The class containing a group of methods for handling the conversion of chemical formats using RDKit. """

    @staticmethod
    def smiles_to_mol(
            smiles: str,
            verbose: bool = False
    ) -> Optional[Mol]:
        """
        Convert a SMILES string to a RDKit Mol object. Returns None if either the conversion or the sanitization of the
        SMILES string fail.

        Input:
            smiles (str): A SMILES string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Optional[Mol]): An RDKit Mol object representing the given SMILES string.
        """

        mol = None

        # Try to convert the SMILES string into a RDKit Mol object and sanitize it.
        try:
            mol = MolFromSmiles(smiles)

            SanitizeMol(mol)

            return mol

        # If an exception occurs, print the error message if indicated, and return None.
        except Exception as exception:
            if verbose:
                if mol is None:
                    print("Exception occurred during the conversion process of ", end="")

                else:
                    print("Exception occurred during the sanitization of ", end="")

                print("'{}'. Detailed exception message:\n{}".format(smiles, exception))

            return None

    @staticmethod
    def smarts_to_mol(
            smarts: str,
            verbose: bool = False
    ) -> Optional[Mol]:
        """
        Convert a SMARTS string to a RDKit Mol object. Returns None if either the conversion or the sanitization of the
        SMARTS string fail.

        Input:
            smarts (str): A SMARTS string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Optional[Mol]): An RDKit Mol object representing the given SMARTS string.
        """

        smarts_mol = None

        # Try to convert the SMARTS string into a RDKit Mol object and sanitize it.
        try:
            smarts_mol = MolFromSmarts(smarts)

            return smarts_mol

        # If an exception occurs, print the error message if indicated, and return None.
        except Exception as exception:
            if verbose:
                if smarts_mol is None:
                    print("Exception occurred during the conversion process of ", end="")

                else:
                    print("Exception occurred during the sanitization of ", end="")

                print("'{}'. Detailed exception message:\n{}".format(smarts, exception))

            return None

    @staticmethod
    def smiles_to_canonical_smiles(
            smiles: str,
            verbose: bool = False
    ) -> Optional[str]:
        """
        Convert a SMILES string to a Canonical SMILES string. Returns None if either the conversion to the Canonical
        SMILES string fails.

        Input:
            smiles (str): A SMILES string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Optional[str]): A Canonical SMILES string representing the given chemical structure.
        """

        try:
            return MolToSmiles(FormatConversionUtilities.smiles_to_mol(smiles), canonical=True)

        # If an exception occurs, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during the conversion of '{}' to Canonical SMILES. "
                    "Detailed message: {}".format(smiles, exception)
                )

            return None

    @staticmethod
    def mol_to_canonical_smiles(
            mol: Mol,
            verbose: bool = False
    ) -> Optional[str]:
        """
        Convert a RDKit Mol object to a Canonical SMILES string. Returns None if either the conversion to the Canonical
        SMILES string fails.

        Input:
            mol (AllChem.Mol): An RDKit Mol object representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Optional[str]): A Canonical SMILES string representing the given chemical structure.
        """

        try:
            return MolToSmiles(mol, canonical=True)

        # If an exception occurs, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during the conversion of the RDKit Mol object to Canonical SMILES. "
                    "Detailed message: {}".format(exception)
                )

            return None

    @staticmethod
    def mol_to_smarts(
            mol: Mol,
            verbose: bool = False
    ) -> Optional[str]:
        """
        Convert a RDKit Mol object to a SMARTS string. Returns None if either the conversion to the SMARTS string fails.

        Input:
            mol (Mol): An RDKit Mol object representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Optional[str]): A SMARTS string representing the given chemical structure.
        """

        try:
            return MolToSmarts(mol)

        # If an exception occurs, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during the conversion of the RDKit Mol object to SMARTS. "
                    "Detailed message: {}".format(exception)
                )

            return None

    @staticmethod
    def rxn_smarts_to_rxn_smiles(
            rxn_smarts: str,
            verbose: bool = False
    ) -> Optional[str]:
        """
        Convert a reaction SMARTS string to a Canonical reaction SMILES string.

        Input:
            rxn_smarts (str): A reaction SMARTS string representing a chemical reaction.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Optional[str]): A Canonical reaction SMILES string.
        """

        try:
            rxn = ReactionFromSmarts(rxn_smarts)

            SanitizeRxn(rxn)

            return ReactionToSmiles(rxn, canonical=True)

        # If an exception occurs, print the message if indicated, and return None for each of the reaction roles.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during the conversion for reaction SMARTS '{}'. "
                    "Detailed message: {}".format(rxn_smarts, exception)
                )

            return None

    @staticmethod
    def rxn_smiles_to_rxn_smarts(
            rxn_smiles: str,
            verbose: bool = False
    ) -> Optional[str]:
        """
        Convert a reaction SMILES string to a reaction SMARTS string.

        Input:
            rxn_smarts (str): A reaction SMILES string representing a chemical reaction.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Optional[str]): A reaction SMARTS string.
        """

        try:
            rxn = ReactionFromSmarts(rxn_smiles)

            SanitizeRxn(rxn)

            return ReactionToSmarts(rxn)

        # If an exception occurs, print the message if indicated, and return None for each of the reaction roles.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during the conversion for reaction SMILES '{}'."
                    "Detailed message: {}".format(rxn_smiles, exception)
                )

            return None

    @staticmethod
    def rxn_smiles_to_rxn_roles(
            rxn_smiles: str,
            verbose: bool = False
    ) -> Tuple[Optional[List[str]], Optional[List[str]], Optional[List[str]]]:
        """
        Parse the reaction roles strings from the reaction SMILES string.

        Input:
            rxn_smiles (str): A SMILES string representing a chemical reaction.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Tuple[Optional[List[str]], Optional[List[str]], Optional[List[str]]]): A 3-tuple containing the Canonical
                SMILES strings for the reactants, agents and products, respectively.
        """

        try:
            # Split the reaction SMILES string by the '>' symbol to obtain the reactants and products.
            # Sometimes, extended SMILES can have additional information at the end separated by a whitespace.
            reactant_smiles = [r_smi for r_smi in rxn_smiles.split(">")[0].split(".") if r_smi != ""]
            agent_smiles = [a_smi for a_smi in rxn_smiles.split(">")[1].split(".") if a_smi != ""]
            product_smiles = [p_smi for p_smi in rxn_smiles.split(">")[2].split(" ")[0].split(".") if p_smi != ""]

            return reactant_smiles, agent_smiles, product_smiles

        # If an exception occurs, print the message if indicated, and return None for each of the reaction roles.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during the parsing of the reaction roles for '{}'. "
                    "Detailed message: {}".format(rxn_smiles, exception)
                )

            return None, None, None
