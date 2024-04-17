""" The ``scripts.single_step_retrosynthesis`` directory ``create_dataset`` script. """

import logging
import pandas

from argparse import ArgumentParser, Namespace
from math import floor
from pathlib import Path
from typing import Union

from sklearn.model_selection import StratifiedKFold, train_test_split


def get_script_arguments(
) -> Namespace:
    """
    Get the script arguments.

    :returns: The script arguments.
    """

    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "-f",
        "--dataset_file_path",
        type=str,
        required=True,
        help="The path to the dataset '*.csv' file."
    )

    argument_parser.add_argument(
        "-d",
        "--output_directory_path",
        type=str,
        required=True,
        help="The path to the output directory."
    )

    argument_parser.add_argument(
        "-t",
        "--minimum_number_of_reaction_template_occurrences",
        default=1,
        type=int,
        required=False,
        help="The minimum number of chemical reaction template occurrences."
    )

    argument_parser.add_argument(
        "-c",
        "--number_of_cross_validation_splits",
        default=5,
        type=int,
        required=False,
        help="The number of cross-validation splits."
    )

    argument_parser.add_argument(
        "-v",
        "--validation_percentage",
        default=0.15,
        type=float,
        required=False,
        help="The percentage of the dataset that should be utilized for validation."
    )

    argument_parser.add_argument(
        "-s",
        "--random_seed",
        default=42,
        type=int,
        required=False,
        help="The random seed value."
    )

    return argument_parser.parse_args()


def get_script_logger(
        level: Union[int, str]
) -> logging.Logger:
    """
    Get the script logger.

    :parameter level: The logger level.

    :returns: The script logger.
    """

    logger = logging.getLogger(
        name=__name__
    )

    logger.setLevel(
        level=level
    )

    logger.propagate = False

    logger_formatter = logging.Formatter(
        fmt="[{asctime:s}] {levelname:s}: \"{message:s}\"",
        style="{"
    )

    logger_stream_handler = logging.StreamHandler()

    logger_stream_handler.setLevel(
        level=level
    )

    logger_stream_handler.setFormatter(
        fmt=logger_formatter
    )

    logger.addHandler(
        hdlr=logger_stream_handler
    )

    return logger


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    #  Initialize the script.
    # ------------------------------------------------------------------------------------------------------------------

    script_arguments = get_script_arguments()

    script_logger = get_script_logger(
        level=logging.INFO
    )

    script_logger.info(
        msg="The 'scripts.single_step_retrosynthesis.create_dataset' script has been initialized."
    )

    # ------------------------------------------------------------------------------------------------------------------
    #  Pre-process the chemical reaction template dataset.
    # ------------------------------------------------------------------------------------------------------------------

    dataset = pandas.read_csv(
        filepath_or_buffer=script_arguments.dataset_file_path
    )

    dataset = dataset.loc[~dataset["multi_part_core"], :]

    dataset = dataset[dataset["template_count"] >= script_arguments.minimum_number_of_reaction_template_occurrences]

    dataset["class"] = pandas.factorize(dataset["retro_template"])[0]

    # ------------------------------------------------------------------------------------------------------------------
    #  Generate the cross-validation splits.
    # ------------------------------------------------------------------------------------------------------------------

    number_of_validation_samples = len(dataset.index) - floor(len(dataset.index) * (1 - script_arguments.validation_percentage))

    cross_validation_splitter = StratifiedKFold(
        n_splits=script_arguments.number_of_cross_validation_splits,
        shuffle=True,
        random_state=script_arguments.random_seed
    )

    single_sample_dataset = dataset[dataset["template_count"] == 1]

    script_logger.info(
        msg="Number of classes with n == 1 samples (Training): {0:d}.".format(single_sample_dataset["class"].nunique())
    )

    not_enough_samples_dataset = dataset[dataset["template_count"].between(2, script_arguments.number_of_cross_validation_splits, "left")]

    script_logger.info(
        msg="{0:s} {1:s}".format(
            "Number of classes with 2 <= n < {0:d} samples (Training and Validation): {1:d}.".format(
                script_arguments.number_of_cross_validation_splits,
                not_enough_samples_dataset["class"].nunique()
            ),
            "Number of samples: {0:d}.".format(len(not_enough_samples_dataset.index))
        )
    )

    enough_samples_dataset = dataset[dataset["template_count"] >= script_arguments.number_of_cross_validation_splits]

    script_logger.info(
        msg="{0:s} {1:s}".format(
            "Number of classes with n >= {0:d} samples (Training, Validation and Testing): {1:d}.".format(
                script_arguments.number_of_cross_validation_splits,
                enough_samples_dataset["class"].nunique()
            ),
            "Number of samples: {0:d}.".format(
                len(enough_samples_dataset.index)
            )
        )
    )

    for cross_validation_split_index, (training_and_validation_indices, testing_indices) in enumerate(cross_validation_splitter.split(
        X=enough_samples_dataset["main_product"].values,
        y=enough_samples_dataset["class"].values
    )):
        testing_dataset = enough_samples_dataset.iloc[testing_indices].sample(
            frac=1.0,
            random_state=script_arguments.random_seed
        ).reset_index(
            drop=True
        )

        training_dataset = pandas.concat(
            objs=[enough_samples_dataset.iloc[training_and_validation_indices], not_enough_samples_dataset, ]
        )

        training_dataset, validation_dataset = train_test_split(
            training_dataset,
            test_size=number_of_validation_samples,
            stratify=training_dataset["class"],
            random_state=script_arguments.random_seed
        )

        training_dataset = pandas.concat(
            objs=[training_dataset, single_sample_dataset, ]
        ).sample(
            frac=1.0,
            random_state=script_arguments.random_seed
        )

        validation_dataset = validation_dataset.sample(
            frac=1.0,
            random_state=script_arguments.random_seed
        )

        training_and_validation_dataset = pandas.concat(
            objs=[training_dataset, validation_dataset, ]
        ).reset_index(
            drop=True
        )

        script_logger.info(
            msg="{0:s} {1:s} {2:s}".format(
                "The cross-validation split {0:d} has been successfully generated.".format(cross_validation_split_index + 1),
                "Number of classes: {0:d} / {1:d} / {2:d}.".format(
                    training_dataset["class"].nunique(),
                    validation_dataset["class"].nunique(),
                    testing_dataset["class"].nunique()
                ),
                "Number of samples: {0:d} ({1:.2f}%) / {2:d} ({3:.2f}%) / {4:d} ({5:.2f}%).".format(
                    len(training_dataset.index),
                    100 * len(training_dataset.index) / len(dataset.index),
                    len(validation_dataset.index),
                    100 * len(validation_dataset.index) / len(dataset.index),
                    len(testing_dataset.index),
                    100 * len(testing_dataset.index) / len(dataset.index)
                )
            )
        )

        training_and_validation_dataset.to_csv(
            path_or_buf=Path(
                script_arguments.output_directory_path,
                "{0:s}_training_and_validation_split_{1:d}.csv".format(
                    script_arguments.dataset_file_path.split("/")[-1].split(".")[0],
                    cross_validation_split_index + 1
                )
            ).resolve().as_posix(),
            index=False
        )
    
        testing_dataset.to_csv(
            path_or_buf=Path(
                script_arguments.output_directory_path,
                "{0:s}_test_split_{1:d}.csv".format(
                    script_arguments.dataset_file_path.split("/")[-1].split(".")[0],
                    cross_validation_split_index + 1
                )
            ).resolve().as_posix(),
            index=False
        )

        script_logger.info(
            msg="The cross-validation split {0:d} has been successfully stored at: '{1:s}'.".format(
                cross_validation_split_index + 1,
                script_arguments.output_directory_path
            )
        )
