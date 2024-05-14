""" The ``retrekpy.ease_of_synthesis.data`` package ``reader`` module. """

import numpy
import pandas

from pathlib import Path
from tqdm import tqdm
from typing import Dict, NamedTuple, Union


class TemplateStat(NamedTuple):
    """ The 'TemplateStat' class. """

    minimum: float
    maximum: float
    average: float


class Reader:
    """ The 'Reader' class. """

    def __init__(
            self,
            uspto_data_path: str,
            template_column: str,
            filter_multi_part_core: bool = True
    ) -> None:
        """ The constructor method class. """

        templates = pandas.read_csv(uspto_data_path)

        if filter_multi_part_core:
            templates["multi_part_core"] = templates[template_column].apply(lambda x: "." in x.split(">>")[-1])
            templates = templates[~templates["multi_part_core"]]

        self.template_counts = templates[template_column].value_counts()

    @staticmethod
    def read_smiles(
            compound_folder: Path
    ) -> str:
        """ The 'read_smiles' method class. """

        f = (compound_folder / "state.sma").open()

        return f.read().split("\n")[-1]

    @staticmethod
    def read_solved(
            compound_folder: Path
    ) -> int:
        """ The 'read_solved' method class. """

        return int((compound_folder / "proven").exists())

    @staticmethod
    def read_time(
            compound_folder: Path
    ) -> float:
        """ The 'read_time' method class. """

        f = (compound_folder / "time.txt").open()

        return float(f.read())

    @staticmethod
    def read_steps(
            compound_folder: Path
    ) -> int:
        """ The 'read_steps' method class. """

        f = (compound_folder / "state.sma").open()

        return len(f.read().split("\n")) - 1

    def get_template_stats(
            self,
            compound_folder: Path
    ) -> TemplateStat:
        """ The 'get_template_stats' method class. """

        f = (compound_folder / "reaction.sma").open()

        reactions = f.read().split("\n")

        if reactions[0] != "":
            counts = self.template_counts[reactions].values

        else:
            counts = numpy.array([0])

        return TemplateStat(
            minimum=counts.min(),
            maximum=counts.max(),
            average=int(counts.mean())
        )

    def read_compound(
            self,
            compound_folder: Path
    ) -> Dict[str, Union[bool, int, str, float]]:
        """ The 'get_template_stats' method class. """

        smiles = Reader.read_smiles(compound_folder)
        solved = Reader.read_solved(compound_folder)
        time = Reader.read_time(compound_folder)

        if solved:
            steps = Reader.read_steps(compound_folder)
            template_counts_stats = self.get_template_stats(compound_folder)

        else:
            steps = numpy.nan

            template_counts_stats = TemplateStat(
                minimum=numpy.nan,
                maximum=numpy.nan,
                average=numpy.nan
            )

        return {
            "smiles": smiles,
            "solved": solved,
            "time": time,
            "steps": steps,
            "template_min_count": template_counts_stats.minimum,
            "template_max_count": template_counts_stats.maximum,
            "template_mean_count": template_counts_stats.average,
        }

    def extract(
            self,
            results_folder: Union[str, Path],
            add_log_transforms: bool = True
    ) -> pandas.DataFrame:
        """ The 'get_template_stats' method class. """

        if isinstance(results_folder, str):
            results_folder = Path(results_folder)

        data = []

        for compound_folder in tqdm(results_folder.glob("*")):
            sample = self.read_compound(compound_folder)
            data.append(sample)

        df = pandas.DataFrame(data)

        if add_log_transforms:
            df["log_time"] = numpy.log(df.time)
            df["log_template_min_count"] = numpy.log(df.template_min_count)
            df["log_template_max_count"] = numpy.log(df.template_max_count)
            df["log_template_mean_count"] = numpy.log(df.template_mean_count)

        return df
