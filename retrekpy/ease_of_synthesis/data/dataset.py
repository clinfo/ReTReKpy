""" The ``retrekpy.ease_of_synthesis.data`` package ``dataset`` module. """

import logging
import numpy
import pandas
import torch

from joblib import delayed, Parallel
from tqdm import tqdm
from typing import Dict, List

from rdkit.Chem import MolFromSmiles

from torch.utils.data import DataLoader, Subset

from kmol.core.exceptions import FeaturizationError
from kmol.core.helpers import CacheManager
from kmol.data.resources import Batch, DataPoint, GeneralCollater

from kmol.data.featurizers import (
    AbstractDescriptorComputer,
    AbstractFeaturizer,
    CircularFingerprintFeaturizer,
    GraphFeaturizer,
    MordredDescriptorComputer,
    RdkitDescriptorComputer,
)


class NoSetDeviceCollater(GeneralCollater):
    """ The 'NoSetDeviceCollater' class. """

    def __init__(
            self
    ) -> None:
        """ The constructor method class. """

        super().__init__()

    def apply(
            self,
            batch: List[DataPoint]
    ) -> Batch:
        """ The 'apply' method class. """

        batch = self._unpack(batch)

        for key, values in batch.inputs.items():
            batch.inputs[key] = self._collater.collate(values)

        return batch


class DescriptorFeaturizer(AbstractFeaturizer):
    """ The 'DescriptorFeaturizer' class. """

    def __init__(
            self,
            inputs: List[str],
            outputs: List[str],
            descriptor_calculator: AbstractDescriptorComputer,
            should_cache: bool = False,
            rewrite: bool = True
    ) -> None:
        """ The constructor method class. """

        super().__init__(inputs, outputs, should_cache, rewrite)

        self._descriptor_calculator = descriptor_calculator

    def _process(
            self,
            data: str,
            entry: DataPoint
    ) -> torch.Tensor:
        """ The '_process' method class. """

        mol = MolFromSmiles(data)

        molecule_features = self._descriptor_calculator.run(mol, entry)

        return torch.FloatTensor(molecule_features)


class CSVDataset(torch.utils.data.Dataset):
    """ The 'CSVDataset' class. """

    def __init__(
            self,
            input_path: str,
            target_columns: List[str],
            smiles_column: str = "smiles",
            featurizer: str = "graph",
            use_cache: bool = True,
            num_workers: int = 16,
            cache_location: str = "/tmp/"
    ) -> None:
        """ The constructor method class. """

        super().__init__()

        self.data = pandas.read_csv(input_path) if input_path is not None else input_path

        self.input_columns = [smiles_column]

        self.target_columns = target_columns

        self._cache_manager = CacheManager(cache_location=cache_location)

        self.cache = {}

        if featurizer == "graph":
            self.featurizer = GraphFeaturizer(
                inputs=[smiles_column],
                outputs=["graph"],
                rewrite=False,
                descriptor_calculator=RdkitDescriptorComputer()
            )

        elif featurizer == "ecfp":
            self.featurizer = CircularFingerprintFeaturizer(
                inputs=[smiles_column],
                outputs=["features"]
            )

        elif featurizer == "mordred":
            self.featurizer = DescriptorFeaturizer(
                inputs=[smiles_column],
                outputs=["features"],
                descriptor_calculator=MordredDescriptorComputer(),
                should_cache=True
            )

        else:
            raise ValueError(f"Unknown featurizer type: {featurizer}. Use one of : 'graph', 'ecfp', 'mordred'.")

        if use_cache:
            logging.info("Caching dataset...")
            self.cache = self._cache_manager.execute_cached_operation(
                processor=self._prepare_cache, arguments={"num_workers": num_workers}, cache_key={
                    "input_path": input_path,
                    "target_columns": target_columns,
                    "smiles_column": smiles_column,
                    "featurizer": featurizer
                }
            )

    def _prepare_cache(
            self,
            num_workers
    ) -> Dict[int, Subset]:
        """ The '_prepare_cache' method class. """

        all_ids = self.list_ids()
        chunk_size = len(all_ids) // num_workers
        chunks = [all_ids[i: i+chunk_size] for i in range(0, len(all_ids), chunk_size)]
        chunks = [Subset(self, chunk) for chunk in chunks]
        dataset = sum(Parallel(n_jobs=len(chunks))(delayed(self._prepare_chunk)(chunk) for chunk in chunks), [])

        return {sample.id_: sample for sample in dataset}

    def _prepare_chunk(
            self,
            loader: DataLoader
    ) -> List[DataPoint]:
        """ The '_prepare_chunk' method class. """

        dataset = []

        with tqdm(total=len(loader)) as progress_bar:
            for sample in loader:
                try:
                    dataset.append(sample)

                except FeaturizationError as e:
                    logging.warning(e)

                progress_bar.update()

        return dataset

    def __getitem__(
            self,
            idx: int
    ) -> DataPoint:
        """ The '__getitem__' method class. """

        if idx in self.cache:
            return self.cache[idx]

        entry = self.data.iloc[idx]

        sample = DataPoint(
            id_=idx,
            labels=self.target_columns,
            inputs={**entry[self.input_columns]},
            outputs=entry[self.target_columns].to_list()
        )

        self.featurizer.run(sample)

        return sample

    def __len__(
            self
    ) -> int:
        """ The '__len__' method class. """

        return len(self.data)

    def list_ids(
            self
    ) -> numpy.ndarray:
        """ The 'list_ids' method class. """

        return numpy.arange(len(self)).tolist()
