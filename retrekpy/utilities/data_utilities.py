""" The ``retrekpy.utilities`` package ``data_utilities`` module. """

import pandas

from typing import Optional


class DataUtilities:
    """ The class containing a group of methods for handling dataset related tasks. """

    @staticmethod
    def read_dataset(
            dataset_file_path: str,
            dataset_file_extension: str,
            separator: str = "\t",
            header: Optional[int] = 0,
            verbose: bool = False
    ) -> pandas.DataFrame:
        """
        Read a dataset according to the specified parameters as a Pandas dataframe.

        Input:
            dataset_file_path (str): The path to the dataset file which is going to be read.
            dataset_file_extension (str): The extension of the dataset file.
            separator (str): The row separator of the dataset file.
            header (Optional[int]): The header of the dataset file.
            verbose (bool): A bool value indicating if potential error messages are printed.

        Output:
            (pandas.DataFrame): A Pandas dataframe object containing the read dataset.
        """

        # Try to read the dataset using Pandas.
        try:
            if dataset_file_extension in [
                ".csv", ".tsv", ".smi", ".smiles", ".sma", ".smarts", ".rsmi", ".rsmiles", ".txt",
            ]:
                final_dataset = pandas.read_csv(
                    dataset_file_path,
                    sep=separator,
                    header=header,
                    low_memory=False
                )

            elif dataset_file_extension == ".pkl":
                final_dataset = pandas.read_pickle(dataset_file_path)

            else:
                raise ValueError(
                    "Currently supported dataset file formats are: '.csv', '.tsv', '.smi', '.smiles', '.sma',"
                    "'.smarts', '.rsmi', '.rsmiles', '.txt', '.pkl'. Got value: '{}'.".format(dataset_file_extension)
                )

            # If indicated, print the shape and preview of the dataset.
            if verbose:
                print("Dataset shape: {}".format(final_dataset.shape))
                print(final_dataset.head(10))

            return final_dataset

        # If an exception occurs, print the message if indicated, and propagate the exception.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during the reading of the input dataset as a Pandas dataframe. "
                    "Detailed message: {}".format(exception)
                )

            raise

    @staticmethod
    def save_dataset(
            dataset_df: pandas.DataFrame,
            output_folder_path: str,
            output_file_name: str,
            output_file_extension: str,
            separator: str = "\n",
            header: bool = True,
            verbose: bool = False
    ) -> None:
        """
        Save a Pandas dataframe to in the specified format to a specified location.

        Input:
            dataset_df (pd.DataFrame): The Pandas dataframe object containing the dataset which is going to be saved.
            output_folder_path (str): The path to the folder where the output dataset file is going to be saved.
            output_file_name (str): The name of the output dataset file.
            output_file_extension (str): The extension of the output dataset file.
            separator (str): The row separator of the output dataset file.
            header (bool): The header of the output dataset file.
            verbose (bool): A bool value indicating if potential error messages are printed.
        """

        # Try to save the dataset using Pandas in the specified format.
        try:
            if output_file_extension in [
                ".csv", ".tsv", ".smi", ".smiles", ".sma", ".smarts", ".rsmi", ".rsmiles", ".txt",
            ]:
                dataset_df.to_csv(
                    output_folder_path + output_file_name + output_file_extension,
                    sep=separator,
                    header=header,
                    index=False
                )

            elif output_file_extension == ".pkl":
                dataset_df.to_pickle(output_folder_path + output_file_name + output_file_extension)

            else:
                raise ValueError(
                    "Currently supported dataset file formats are: '.csv', '.tsv', '.smi', '.smiles', '.sma', "
                    "'.smarts', '.rsmi', '.rsmiles', '.txt', '.pkl'. Got value: '{}'.".format(output_file_extension)
                )

            if verbose:
                print(
                    "The dataset has been successfully saved under: '{}'".format(
                        output_folder_path + output_file_name + output_file_extension
                    )
                )

        # If an exception occurs, print the message if indicated, and propagate the exception.
        except Exception as exception:
            if verbose:
                print(
                    "Exception occurred during the saving of the dataset as a Pandas dataframe. "
                    "Detailed message: {}".format(exception)
                )

            raise
