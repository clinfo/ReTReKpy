""" The ``scripts.ease_of_synthesis`` directory ``train`` script. """

from retrekpy.kmol_patch.data.loaders import PatchedListLoader

from kmol.data import loaders

loaders.ListLoader = PatchedListLoader

from argparse import ArgumentParser, Namespace
from typing import Dict, List

from torch.utils.data import DataLoader, Subset

from kmol.core.config import Config
from kmol.core.helpers import SuperFactory
from kmol.data.splitters import AbstractSplitter

from retrekpy.ease_of_synthesis.data.dataset import CSVDataset, NoSetDeviceCollater
from retrekpy.ease_of_synthesis.model.executors import BayesianOptimizer, Evaluator, Trainer
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
        "--config",
        type=str,
        required=True,
    )

    argument_parser.add_argument(
        "-t",
        "--task",
        choices=[
            "train",
            "eval",
            "bayesian_opt",
        ],
        default="train",
        required=False
    )

    argument_parser.add_argument(
        "-n",
        "--num_workers",
        type=int,
        default=16
    )

    argument_parser.add_argument(
        "-e",
        "--eval_output_path",
        type=str,
        default=""
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


def init_splits(
        dataset: Subset,
        configuration: Config
) -> Dict[str, List[int]]:
    """ The 'init_splits' function. """

    splitter = SuperFactory.create(AbstractSplitter, configuration.splitter)

    return splitter.apply(
        data_loader=dataset
    )


def init_loader(
        dataset,
        configuration: Config,
        splits: Dict[str, List[int]],
        split_name: str,
        num_workers: int
) -> DataLoader:
    """ The 'init_loader' function. """

    subset = Subset(
        dataset=dataset,
        indices=splits[split_name]
    )

    return to_loader(
        subset=subset,
        batch_size=configuration.batch_size,
        shuffle=split_name == "train",
        num_workers=num_workers
    )


if __name__ == "__main__":
    script_arguments = get_script_arguments()

    script_configuration = Config.from_file(script_arguments.config, script_arguments.task)

    csv_dataset = CSVDataset(
        cache_location=script_configuration.cache_location,
        **script_configuration.loader
    )

    dataset_splits = init_splits(
        dataset=csv_dataset,
        configuration=script_configuration
    )

    if script_arguments.task == "train":
        train_loader = init_loader(
            dataset=csv_dataset,
            configuration=script_configuration,
            splits=dataset_splits,
            split_name="train",
            num_workers=script_arguments.num_workers
        )

        val_loader = init_loader(
            dataset=csv_dataset,
            configuration=script_configuration,
            splits=dataset_splits,
            split_name="validation",
            num_workers=script_arguments.num_workers
        )

        trainer = Trainer(script_configuration)

        trainer.run(
            train_loader=train_loader,
            val_loader=val_loader
        )

    elif script_arguments.task == "eval":
        test_loader = init_loader(
            dataset=csv_dataset,
            configuration=script_configuration,
            splits=dataset_splits,
            split_name="test",
            num_workers=script_arguments.num_workers
        )

        if script_arguments.eval_output_path:
            script_configuration = script_configuration.cloned_update(
                output_path=script_arguments.eval_output_path
            )

        evaluator = Evaluator(script_configuration)

        evaluator.run(
            data_loader=test_loader
        )

    elif script_arguments.task == "bayesian_opt":
        train_loader = init_loader(
            dataset=csv_dataset,
            configuration=script_configuration,
            splits=dataset_splits,
            split_name="train",
            num_workers=script_arguments.num_workers
        )

        val_loader = init_loader(
            dataset=csv_dataset,
            configuration=script_configuration,
            splits=dataset_splits,
            split_name="validation",
            num_workers=script_arguments.num_workers
        )

        executor = BayesianOptimizer(script_arguments.config)

        executor.run(
            train_loader=train_loader,
            val_loader=val_loader
        )
