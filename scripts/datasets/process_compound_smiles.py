""" The ``scripts.datasets`` directory ``process_compound_smiles`` module. """

from argparse import ArgumentParser

from retrekpy.datasets import Configuration, PreProcessing


def parse_user_args():
    """ Parse the arguments specified by the user during input. """

    parser = ArgumentParser("Data pre-processing")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )

    parser.add_argument(
        "-smi",
        "--smiles",
        default="",
        type=str,
        help="Process only a single SMILES string."
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show messages which occur during processing."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_user_args()

    config = Configuration.load_configuration(args.config)

    data_pre_processing = PreProcessing(config.pre_processing_configuration)

    if args.smiles != "":
        print("Processed SMILES and SA_Score: {}".format(data_pre_processing.pre_process_entry(args.smiles, verbose=args.verbose)))

    else:
        data_pre_processing.pre_process_data(verbose=args.verbose)
