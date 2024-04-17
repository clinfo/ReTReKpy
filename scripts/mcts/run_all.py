""" The ``scripts.mcts`` directory ``run_all`` script. """

from warnings import filterwarnings

filterwarnings(
    action="ignore",
    category=FutureWarning
)

import logging
import os
import torch

from argparse import ArgumentParser, Namespace
from datetime import datetime
from json import dump, load
from pathlib import Path
from random import seed
from time import time
from typing import Optional, Union

from rdkit.Chem import MolFromMolFile
from rdkit.RDLogger import DisableLog

from kmol.core.config import Config

from retrekpy.in_scope_filter.in_scope_filter_network import InScopeFilterNetwork

from retrekpy.mcts.chemistry_helper import ChemistryHelper
from retrekpy.mcts import MCTS
from retrekpy.mcts.utilities import ReactionUtilities

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
        "-t",
        "--targets",
        type=str,
        default="/nasa/shared_homes/haris/development/riken_retrek_improvement/deliverables/mcts/data/original/chembl_eval_27000",
        required=False,
        help="Path to the target molecule file."
    )

    argument_parser.add_argument(
        "-r",
        "--save_result_dir",
        type=str,
        default="/nasa/shared_homes/haris/development/riken_retrek_improvement/deliverables/mcts/data/processed/chembl_eval_27000_results",
        required=False,
        help="Path to the result directory."
    )

    argument_parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="/nasa/shared_homes/haris/development/riken_retrek_improvement/ReTReKpy/configurations/mcts/run.json",
        required=False,
        help="Path to the config file."
    )

    argument_parser.add_argument(
        "-mc",
        "--model_config",
        type=str,
        default="/nasa/shared_homes/haris/development/riken_retrek_improvement/ReTReKpy/configurations/single_step_retrosynthesis/inference_configuration.json",
        required=False,
        help="Path to the config file."
    )

    argument_parser.add_argument(
        "-d",
        "--debug",
        action="store_true",
        help="Debug mode."
    )

    argument_parser.add_argument(
        "-s",
        "--random_seed",
        type=int,
        default=42,
        help="Fix the random seed value."
    )

    return argument_parser.parse_args()


def get_script_logger(
        name: str,
        level: Union[int, str],
        output_file_path: str
) -> logging.Logger:
    """
    Get the script logger.

    :parameter name: The logger name.
    :parameter level: The logger level.
    :parameter output_file_path: The path to the output file.

    :returns: The script logger.
    """

    logger = logging.getLogger(
        name=name
    )

    logger.setLevel(
        level=level
    )

    logger.propagate = False

    logger_formatter = logging.Formatter(
        fmt="[{asctime:s}] {levelname:s}: \"{message:s}\"",
        style="{"
    )

    logger_file_handler = logging.FileHandler(
        filename=output_file_path
    )

    logger_file_handler.setLevel(
        level=level
    )

    logger_file_handler.setFormatter(
        fmt=logger_formatter
    )

    logger.addHandler(
        hdlr=logger_file_handler
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


def load_model(
        configuration_path: str
) -> CustomizedPredictor:
    """ The 'load_model' function. """

    configuration = Config.from_file(
        file_path=configuration_path,
        job_command="infer"
    )

    return CustomizedPredictor(configuration)


def load_in_scope_model(
        checkpoint_file_path: str
) -> Optional[InScopeFilterNetwork]:
    """ The 'load_in_scope_model' function. """

    if checkpoint_file_path is not None:
        model = InScopeFilterNetwork()
        model.load_state_dict(torch.load(checkpoint_file_path, map_location=torch.device("cpu")))
        model.eval()

        return model

    else:
        return None


if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    script_arguments = get_script_arguments()

    with open(script_arguments.config) as file_handle:
        script_configuration = load(
            fp=file_handle
        )

    if not script_arguments.debug:
        DisableLog("rdApp.*")

    if script_arguments.random_seed is not None:
        seed(script_arguments.random_seed)

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    expansion_rules = ReactionUtilities.get_reactions(
        rxn_rule_path=script_configuration["expansion_rules"],
        save_dir=script_arguments.save_result_dir
    )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    rollout_rules = ReactionUtilities.get_reactions(
        rxn_rule_path=script_configuration["rollout_rules"],
        save_dir=script_arguments.save_result_dir
    )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    with open(script_configuration["starting_material"]) as file_handle:
        starting_materials = set([line.strip() for line in file_handle.readlines()])

        if all([len(starting_material) == 27 for starting_material in starting_materials]):
            starting_materials.add("inchi")

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if script_configuration["intermediate_material"] is not None:
        with open(script_configuration["intermediate_material"]) as file_handle:
            intermediate_materials = set([line.strip() for line in file_handle.readlines()])

    else:
        intermediate_materials = set()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if script_configuration["template_scores"]:
        with open(script_configuration["template_scores"]) as file_handle:
            template_scores = load(file_handle)

    else:
        template_scores = set()

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    expansion_model = load_model(
        configuration_path=script_arguments.model_config
    )

    rollout_model = load_model(
        configuration_path=script_arguments.model_config
    )

    in_scope_filter_model = load_in_scope_model(
        checkpoint_file_path=script_configuration["in_scope_model"]
    )

    # ------------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    finished_indices = set()

    for dir_name in os.listdir("/nasa/shared_homes/haris/development/riken_retrek_improvement/deliverables/mcts/data/processed/chembl_eval_27000_results"):
        finished_indices.add(
            int(dir_name.split("_")[1])
        )

    for file_name in os.listdir(script_arguments.targets):
        if file_name.endswith(".mol") and int(file_name.split(".")[0].split("_")[1]) not in finished_indices:
            # ----------------------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------

            script_arguments.target = Path(script_arguments.targets, file_name).resolve().as_posix()

            script_configuration["save_result_dir"] = Path(
                script_arguments.save_result_dir,
                "{0:s}_{1:s}".format(
                    script_arguments.target.split("/")[-1].split(".")[0],
                    datetime.now().strftime(
                        format="%Y%m%d%H%M"
                    )
                )
            ).as_posix()

            Path(script_configuration["save_result_dir"]).mkdir(
                exist_ok=True
            )

            with open(
                file=Path(script_configuration["save_result_dir"], "parameters.json"),
                mode="w"
            ) as file_handle:
                dump(
                    obj={key: repr(value) for key, value in script_configuration.items()},
                    fp=file_handle,
                    indent=2
                )

            script_logger = get_script_logger(
                name=file_name,
                level=logging.DEBUG if script_arguments.debug else logging.INFO,
                output_file_path=Path(script_configuration["save_result_dir"], "run.log").resolve().as_posix()
            )

            # ----------------------------------------------------------------------------------------------------------
            # ----------------------------------------------------------------------------------------------------------

            chemistry_helper = ChemistryHelper(
                reaction_rule_list_path=script_configuration["rollout_rules"]
            )

            try:
                target_mol = MolFromMolFile(script_arguments.target)

                if target_mol is None:
                    raise ValueError("Can't read the input molecule file. Please check it.")

                mcts = MCTS(
                    target_mol=target_mol,
                    max_atom_num=script_configuration["max_atom_num"],
                    expansion_rules=expansion_rules,
                    rollout_rules=rollout_rules,
                    starting_materials=starting_materials,
                    intermediate_materials=intermediate_materials,
                    template_scores=template_scores,
                    knowledge=script_configuration["knowledge"],
                    knowledge_weights=script_configuration["knowledge_weights"],
                    save_tree=script_configuration["save_tree"],
                    search_count=script_configuration["search_count"],
                    selection_constant=script_configuration["selection_constant"],
                    save_result_dir=script_configuration["save_result_dir"],
                    cum_prob_mod=script_configuration["cum_prob_mod"],
                    cum_prob_thresh=script_configuration["cum_prob_thresh"],
                    expansion_num=script_configuration["expansion_num"],
                    rollout_depth=script_configuration["rollout_depth"]
                )

                script_logger.info(f"[INFO] knowledge type: {script_configuration['knowledge']}")
                script_logger.info("[INFO] start search")

                start = time()

                leaf_node, is_proven = mcts.search(
                    expansion_model=expansion_model,
                    rollout_model=rollout_model,
                    in_scope_model=in_scope_filter_model,
                    logger=script_logger,
                    chemistry_helper=chemistry_helper,
                    time_limit=script_configuration["time_limit"]
                )

                elapsed_time = time() - start

                script_logger.info(f"[INFO] done in {elapsed_time:5f} s")

                with open(Path(script_configuration["save_result_dir"], "time.txt"), "w") as file_handle:
                    file_handle.write(f"{elapsed_time}")

                nodes = list()

                if leaf_node is None:
                    Path(script_configuration["save_result_dir"], "not_proven").touch()

                    raise Exception("Can't apply any predicted reaction templates to the target compound.")

                while leaf_node.parent_node is not None:
                    nodes.append(leaf_node)
                    leaf_node = leaf_node.parent_node

                else:
                    nodes.append(leaf_node)

                MCTS.print_route(
                    nodes=nodes,
                    is_proven=is_proven,
                    logger=script_logger
                )

                MCTS.save_route(
                    nodes=nodes,
                    save_dir=script_configuration["save_result_dir"],
                    is_proven=is_proven,
                    ws=script_configuration["knowledge_weights"]
                )

            except Exception as exception_handle:
                script_logger.exception(
                    msg=exception_handle
                )
