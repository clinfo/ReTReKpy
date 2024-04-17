""" The ``retrekpy.utilities`` package ``sa_score_utilities`` module. """

import os
import sys

from rdkit.Chem import RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))

import sascorer

from typing import Optional

from rdkit.Chem import SanitizeMol

from .format_conversion_utilities import FormatConversionUtilities


class SaScoreUtilities:
    """ The class containing a group of methods for handling the SA_Score calculations for molecular structures. """

    @staticmethod
    def calculate_sa_score(
            smiles: str,
            verbose: bool = False
    ) -> Optional[float]:
        """
        Calculates the SA_Score value for a given SMILES string. Returns None if the calculation fails for any reason.

        Input:
            smiles (str): A SMILES string representing a chemical structure.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (Optional[float]): A floating point value representing the SA_Score value for the input SMILES string.
        """

        # Try converting the current SMILES string into a RDKit Mol object and calculate the SA_Score value.
        try:
            mol = FormatConversionUtilities.smiles_to_mol(smiles)

            SanitizeMol(mol)

            return sascorer.calculateScore(mol)

        # If an exception occurs, print the message if indicated, and return None.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during the SA_Score calculation for '{}'. "
                    "Detailed exception message:\n{}".format(smiles, exception)
                )

            return None
