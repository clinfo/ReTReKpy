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
        default="/nasa/shared_homes/haris/development/riken_retrek_improvement/deliverables/mcts/data/processed/chembl_eval_1000_results",
        required=False
    )

    argument_parser.add_argument(
        "-t",
        "--uspto_template_path",
        type=str,
        default="/nasa/shared_homes/haris/development/riken_retrek_improvement/deliverables/single_step_retrosynthesis/data/processed/uspto_grants_templates_training_and_validation_split_5.csv",
        required=False
    )

    argument_parser.add_argument(
        "-c",
        "--template_column",
        type=str,
        default="forward_template",
        required=False
    )

    argument_parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default="/nasa/shared_homes/haris/development/riken_retrek_improvement/deliverables/ease_of_synthesis/data/processed/chembl_eval_1000_results_600.csv",
        required=False
    )

    return argument_parser.parse_args()


if __name__ == "__main__":
    script_arguments = get_script_arguments()

    reader = Reader(
        uspto_data_path=script_arguments.uspto_template_path,
        template_column=script_arguments.template_column
    )

    df = reader.extract(script_arguments.data_folder)

    # df = df[df["time"] <= 600]

    df.to_csv(
        script_arguments.output_path,
        index=False
    )

    print(df.columns)
    print(len(df.index))

    from numpy import mean

    print(len(df[df["solved"] == 0].values))
    print(len(df[df["solved"] == 1].values))
    print(mean(df["time"].values))
