""" The ``scripts.single_step_retrosynthesis`` directory ``test`` script. """

from retrekpy.kmol_patch.data.loaders import PatchedListLoader

from kmol.data import loaders

loaders.ListLoader = PatchedListLoader

import logging

from argparse import ArgumentParser, Namespace
from typing import Union

from kmol.core.config import Config
from kmol.data.streamers import GeneralStreamer

from retrekpy.single_step_retrosynthesis.model.executors import CustomizedEvaluator
from retrekpy.single_step_retrosynthesis.model.architectures import CustomizedGraphConvolutionalNetwork


def get_script_arguments(
) -> Namespace:
    """
    Get the script arguments.

    :returns: The script arguments.
    """

    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "-c",
        "--configuration_file_path",
        type=str,
        default="../../configurations/single_step_retrosynthesis/test_configuration.json",
        required=False,
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
        job_command="test"
    )

    general_streamer = GeneralStreamer(
        config=configuration
    )

    test_data_loader = general_streamer.get(
        split_name=configuration.test_split,
        batch_size=configuration.batch_size,
        shuffle=False,
        mode=GeneralStreamer.Mode.TEST
    )

    customized_evaluator = CustomizedEvaluator(configuration)

    customized_evaluator_output = customized_evaluator.evaluate(
        data_loader=test_data_loader
    )

    script_logger.info(
        msg="The '{0:s}' metric value is {1:f}.".format(
            configuration.target_metric,
            customized_evaluator_output.__dict__[configuration.target_metric][0]
        )
    )
