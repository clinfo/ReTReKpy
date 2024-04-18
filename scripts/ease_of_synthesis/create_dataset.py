""" The ``scripts.ease_of_synthesis`` directory ``create_dataset`` script. """

from argparse import ArgumentParser, Namespace

from retrekpy.ease_of_synthesis.data.reader import Reader


def get_script_arguments(
) -> Namespace:
    """
    Get the script arguments.

    :returns: The script arguments.
    """

    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        required=True
    )

    argument_parser.add_argument(
        "-t",
        "--uspto_template_path",
        type=str,
        required=True
    )

    argument_parser.add_argument(
        "-c",
        "--template_column",
        type=str,
        required=True
    )

    argument_parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        required=True
    )

    return argument_parser.parse_args()


if __name__ == "__main__":
    script_arguments = get_script_arguments()

    reader = Reader(
        uspto_data_path=script_arguments.uspto_template_path,
        template_column=script_arguments.template_column
    )

    df = reader.extract(script_arguments.data_folder)

    df.to_csv(
        script_arguments.output_path,
        index=False
    )
