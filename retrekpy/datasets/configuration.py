""" The ``retrekpy.datasets`` package ``configuration`` module. """

from json import load
from typing import NamedTuple, Optional


class TemplateExtractionConfiguration(NamedTuple):
    """ The class containing the necessary configuration parameters for template extraction. """

    # Path to the input dataset file.    
    input_file_path: str
    # Extension of the input dataset file.
    input_file_extension: str
    # Row separator of the input dataset file.
    input_file_separator: str
    # Name of the column containing the ID's of the dataset entries.
    id_column: str
    # Name of the column containing the reaction SMILES strings of the dataset entries.
    rxn_smiles_column: str

    # Path to the folder where the output dataset file should be saved.
    output_folder_path: str
    # Name of the output dataset file.
    output_file_name: str
    # Extension of the output dataset file.
    output_file_extension: str
    # Row separator of the output dataset file.
    output_file_separator: str
    # Name of the new column that will be generated for the cleaned reaction SMILES strings.
    cleaned_smiles_column: str
    # Name of the new column that will be generated for the mapped reaction SMILES strings.
    mapped_smiles_column: str
    # Name of the new column that will be generated for the reaction template SMARTS strings
    rxn_template_smarts_column: str

    # Path to the '*.txt' file containing additional user-defined salts.
    salts_file_path: str

    # Timeout period in ms after which the atom mapping should be stopped for a single reaction SMILES.
    atom_mapping_timeout: int
    # Way to handle any previous mapping in the reaction SMILES string.
    handle_existing_mapping: str

    # Timeout period in ms after which the template extraction should be stopped for a single reaction SMILES.
    extract_template_timeout: int
    # The length of reaction SMILES which is not in danger of having a timeout, and can be processed faster. 
    extract_template_threshold: int
    # The number of occurrences for a single reaction SMARTS based on which it is included in the final result.
    template_occurrence_threshold: int

    # Flag to signal if multiprocessing should be used or not.
    use_multiprocessing: bool


class PreProcessingConfiguration(NamedTuple):
    """ The class containing the necessary configuration parameters for data pre-processing. """

    # Path to the input dataset file.    
    input_file_path: str
    # Extension of the input dataset file.
    input_file_extension: str
    # Row separator of the input dataset file.
    input_file_separator: str
    # Name of the column containing the ID's of the dataset entries.
    id_column: str
    # Name of the column containing the SMILES strings of the dataset entries.
    smiles_column: str

    # Path to the folder where the output dataset file should be saved.
    output_folder_path: str
    # Name of the output dataset file.
    output_file_name: str
    # Extension of the output dataset file.
    output_file_extension: str
    # Row separator of the output dataset file.
    output_file_separator: str
    # Name of the new column that will be generated for the canonical SMILES strings.
    processed_smiles_column: str
    # Name of the new column that will be generated for the SA_Score values.
    sa_score_column: str

    # Path to the '*.txt' file containing additional user-defined salts.
    salts_file_path: str
    # Path to the '*.sma' file containing the user-defined unwanted elements.
    unwanted_elements_file_path: str

    # Flag to signal if multiprocessing should be used or not.
    use_multiprocessing: bool


class Configuration(NamedTuple):
    """ Class containing the necessary methods to load configuration files. """

    # Configuration parameters for the automatic template extraction task.
    template_extraction_configuration: TemplateExtractionConfiguration

    # Configuration parameters for the dataset pre-processing task.
    pre_processing_configuration: PreProcessingConfiguration

    @classmethod
    def load_configuration(
            cls,
            file_path: Optional[str]
    ) -> "Configuration":
        """ Load the configuration parameters from the specified configuration file. """

        with open(file_path) as read_handle:
            settings = load(read_handle)

            if "template_extraction_configuration" not in settings or "data_pre_processing_configuration" not in settings:
                raise ValueError("Mandatory setting groups are missing from the configuration file.")

            return cls(
                template_extraction_configuration=TemplateExtractionConfiguration(**settings["template_extraction_configuration"]),
                pre_processing_configuration=PreProcessingConfiguration(**settings["pre_processing_configuration"])
            )
