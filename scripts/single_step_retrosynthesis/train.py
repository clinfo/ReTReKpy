""" The ``scripts.single_step_retrosynthesis`` directory ``train`` script. """

from retrekpy.kmol_patch.data.loaders import PatchedListLoader

from kmol.data import loaders

loaders.ListLoader = PatchedListLoader

from argparse import ArgumentParser, Namespace

from kmol.core.config import Config
from kmol.data.streamers import GeneralStreamer

from retrekpy.single_step_retrosynthesis.model.executors import CustomizedTrainer
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
        required=True,
        help="The path to the configuration '*.json' file."
    )

    return argument_parser.parse_args()


if __name__ == "__main__":
    script_arguments = get_script_arguments()

    configuration = Config.from_file(
        file_path=script_arguments.configuration_file_path,
        job_command="train"
    )

    general_streamer = GeneralStreamer(
        config=configuration
    )

    training_data_loader = general_streamer.get(
        split_name=configuration.train_split,
        batch_size=configuration.batch_size,
        shuffle=True,
        mode=GeneralStreamer.Mode.TRAIN
    )

    validation_data_loader = general_streamer.get(
        split_name=configuration.validation_split,
        batch_size=configuration.batch_size,
        shuffle=True,
        mode=GeneralStreamer.Mode.TEST
    )

    custom_trainer = CustomizedTrainer(configuration)

    custom_trainer.run(
        data_loader=training_data_loader,
        val_loader=validation_data_loader,
    )
