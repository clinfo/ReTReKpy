""" The ``scripts.ease_of_synthesis`` directory ``train`` script. """

from retrekpy.kmol_patch.data.loaders import PatchedListLoader

from kmol.data import loaders

loaders.ListLoader = PatchedListLoader

import numpy
import pandas
import torch

from argparse import ArgumentParser, Namespace
from pathlib import Path

from torch.utils.data import DataLoader, Subset

from kmol.core.config import Config

from retrekpy.ease_of_synthesis.data.dataset import CSVDataset, NoSetDeviceCollater
from retrekpy.ease_of_synthesis.model.executors import Predictor
from retrekpy.ease_of_synthesis.model.architectures import GraphNeuralNetwork


def get_script_arguments(
) -> Namespace:
    """
    Get the script arguments.

    :returns: The script arguments.
    """

    argument_parser = ArgumentParser()

    argument_parser.add_argument(
        "-c",
        "--classification-model",
        type=str,
        required=True,
        help="Path config file of classification model."
    )

    argument_parser.add_argument(
        "-r",
        "--regression-model",
        type=str,
        required=False,
        help="Path to config file of regression model."
    )

    argument_parser.add_argument(
        "-d",
        "--data",
        type=str,
        required=True,
        help="Path to file containing one smiles per line or single smiles string."
    )

    argument_parser.add_argument(
        "-s",
        "--save-path",
        type=str,
        required=True
    )

    argument_parser.add_argument(
        "-f",
        "--featurizer",
        type=str,
        choices=[
            "graph",
            "ecfp",
            "mordred"
        ],
        default="graph"
    )

    return argument_parser.parse_args()


def to_loader(
        subset: Subset,
        batch_size: int,
        shuffle: bool,
        num_workers: int
) -> DataLoader:
    """ The 'to_loader' function. """

    return DataLoader(
        dataset=subset,
        collate_fn=NoSetDeviceCollater().apply,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


if __name__ == "__main__":
    script_arguments = get_script_arguments()

    csv_dataset = CSVDataset(
        input_path=None,
        target_columns=[],
        featurizer=script_arguments.featurizer,
        use_cache=False
    )

    if Path(script_arguments.data).exists():
        with Path(script_arguments.data).open() as f:
            smiles = [line.split("\n")[0] for line in f.readlines()]

        csv_dataset.data = pandas.DataFrame({"smiles": smiles})

    else:
        csv_dataset.data = pandas.DataFrame({"smiles": [script_arguments.data]})

    print(f"Number of smiles to process: {len(csv_dataset)}")

    data_loader = to_loader(
        subset=csv_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    # Is solved prediction
    solved_preds = Predictor(
        config=Config.from_file(script_arguments.classification_model, "classification")
    ).run(data_loader=data_loader)[:, 0]

    solved_preds = torch.sigmoid(solved_preds)  # converting logits to probability

    results = pandas.DataFrame({
        "smiles": csv_dataset.data.smiles.values,
        "solved": solved_preds
    })

    if script_arguments.regression_model is not None:
        # Keep only samples predicted to be solved for regression
        to_filter_out = solved_preds < 0.5

        reg_config = Config.from_file(script_arguments.regression_model, "regression")

        regression_preds = Predictor(config=reg_config).run(data_loader=data_loader)

        for i, output in enumerate(reg_config.loader["target_columns"]):
            preds = regression_preds[:, i]
            preds[to_filter_out.tolist()] = numpy.nan
            results[output] = preds

    print(results)

    results.to_csv(script_arguments.save_path, index=False)

    print(f"Results saved to: {script_arguments.save_path}")
