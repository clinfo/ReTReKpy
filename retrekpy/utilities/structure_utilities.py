""" The ``retrekpy.utilities`` package ``structure_utilities`` module. """

from typing import Optional

from molvs import Standardizer

from rdkit.Chem import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize

from .format_conversion_utilities import FormatConversionUtilities


class StructureUtilities:
    """ The class containing a group of methods for handling the correctness of molecular structures. """

    @staticmethod
    def remove_salts(
            smiles: str,
            salts_file_path: str,
            apply_ad_hoc_stripper: bool = False,
            verbose: bool = False
    ) -> Optional[str]:
        """
        Remove salts from a SMILES string using the RDKit salt stripper. Returns None if the RDKit salt removal process
        fails.

        Input:
            smiles (str): A SMILES string representing a chemical structure.
            salt_list_file_path (str): A path string to a user-defined list of salt SMILES in .txt format.
            apply_ad_hoc_stripper (bool): A bool value indicating if the ad-hoc salt stripper will be applied.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Optional[str]): A Canonical SMILES string representing the given chemical structure without salts.
        """

        try:
            # Apply the RDKit salt stripper to remove the defined salt molecules.
            salt_remover = SaltRemover.SaltRemover(defnFilename=salts_file_path)

            no_salt_smiles = FormatConversionUtilities.mol_to_canonical_smiles(
                salt_remover.StripMol(
                    FormatConversionUtilities.smiles_to_mol(smiles)
                )
            )

            # If there are some salts left behind, apply the 'ad hoc' salt stripper based on the symbol '.'.
            # This is risky and should only be applied if the SMILES string is one molecule, not on reaction SMILES.
            if apply_ad_hoc_stripper and "." in no_salt_smiles:
                no_salt_smiles = FormatConversionUtilities.smiles_to_canonical_smiles(
                    sorted(no_salt_smiles.split("."), key=len, reverse=True)[0]
                )

            # If nothing is left behind because all molecule parts are defined as salts, return None.
            if no_salt_smiles == "":
                return None

            else:
                return no_salt_smiles

        # If an exception occurs, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during stripping of the salts from '{}'. "
                    "Detailed exception message:\n{}".format(smiles, exception)
                )

            return None

    @staticmethod
    def normalize_structure(
            smiles: str,
            verbose: bool = False
    ) -> Optional[str]:
        """
        Use RDKit to normalize the specified molecule and return it as canonical SMILES. Returns None if the RDKit
        normalization process fails.

        Input:
            smiles (str): A SMILES string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Optional[str]): A Canonical SMILES string representing the normalized chemical structure.
        """

        try:
            mol = rdMolStandardize.Normalize(FormatConversionUtilities.smiles_to_mol(smiles))

            return FormatConversionUtilities.mol_to_canonical_smiles(mol)

        # If an exception occurs, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during the normalization of '{}'. "
                    "Detailed exception message:\n{}".format(smiles, exception)
                )

            return None

    @staticmethod
    def molvs_standardize(
            smiles: str,
            verbose: bool = False
    ) -> Optional[str]:
        """
        Use MolVS and RDKit to standardize the specified molecule and return it as canonical SMILES. Returns None if the
        standardization process fails.

        Input:
            smiles (str): A SMILES string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Optional[str]): A Canonical SMILES string representing the standardized chemical structure.
        """

        try:
            standardizer = Standardizer()

            standardized_mol = standardizer.standardize(FormatConversionUtilities.smiles_to_mol(smiles))

            return FormatConversionUtilities.mol_to_canonical_smiles(standardized_mol)

        # If an exception occurs, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during the standardization of '{}'. "
                    "Detailed exception message:\n{}".format(smiles, exception)
                )

            return None
