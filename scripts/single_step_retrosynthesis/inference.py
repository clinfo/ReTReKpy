""" The ``scripts.single_step_retrosynthesis`` directory ``inference`` script. """

from retrekpy.kmol_patch.data.loaders import PatchedListLoader

from kmol.data import loaders

loaders.ListLoader = PatchedListLoader

import numpy
import logging

from argparse import ArgumentParser, Namespace
from typing import Union

from kmol.core.config import Config

from retrekpy.single_step_retrosynthesis.model.architectures import CustomizedGraphConvolutionalNetwork
from retrekpy.single_step_retrosynthesis.model.executors import CustomizedPredictor


def get_script_arguments(
) -> Namespace:
    """
    Get the script arguments.

    :returns: The script arguments.
    """

    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "-s",
        "--compound_smiles",
        type=str,
        required=True,
        help="The chemical compound SMILES string."
    )

    argument_parser.add_argument(
        "-c",
        "--configuration_file_path",
        type=str,
        required=True,
        help="The path to the configuration '*.json' file."
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
    script_arguments = get_script_arguments()

    script_logger = get_script_logger(
        level=logging.INFO
    )

    configuration = Config.from_file(
        file_path=script_arguments.configuration_file_path,
        job_command="infer"
    )

    customized_predictor = CustomizedPredictor(configuration)

    predictions = customized_predictor.predict(
        compound=script_arguments.compound_smiles
    )

    script_logger.info(
        msg="The Top-10 predictions for the SMILES string '{0:s}' are: [{1:s}].".format(
            script_arguments.compound_smiles,
            ", ".join([
                "(class={0:d}, softmax_probability={1:.5f})".format(prediction[0], prediction[1] * -1)
                for prediction in zip(*(numpy.argsort(-predictions)[:10], numpy.sort(-predictions)[:10]))
            ])
        )
    )
