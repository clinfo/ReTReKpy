""" The ``scripts.datasets`` directory ``create_reaction_template_dataset`` script. """

from warnings import simplefilter

simplefilter(
    action="ignore",
    category=FutureWarning
)

from rdkit.RDLogger import DisableLog

DisableLog("rdApp.*")

import numpy
import os
import pandas

from argparse import ArgumentParser
from random import seed
from tqdm import tqdm

from rdkit.Chem import MolFromSmarts, MolFromSmiles, MolToSmiles, SanitizeMol

from retrekpy.gln import GLNUtilities


def get_parser():
    """ Parse user-specified arguments. """

    arg_parser = ArgumentParser(
        description="description",
        usage="usage"
    )

    arg_parser.add_argument(
        "-i",
        "--input_file_path",
        required=True,
        type=str,
        help="Path of the USPTO Grants or Applications .rsmi file."
    )

    arg_parser.add_argument(
        "-o",
        "--output_folder_path",
        required=True,
        type=str,
        help="Path of the folder where the outputs are stored."
    )

    arg_parser.add_argument(
        "-seed",
        "--random_seed",
        required=False,
        type=int,
        default=101,
        help="Random seed value."
    )

    arg_parser.add_argument(
        "-cores",
        "--num_cores",
        required=False,
        type=int,
        default=1,
        help="Number of CPU cores for multiprocessing."
    )

    return arg_parser.parse_args()


def post_process_reaction_templates(
        uspto_rsmi_file_path: str,
        save_folder_path: str
) -> None:
    """ Post-process the cleaned and extracted USPTO reaction templates. """

    uspto_version = "grants" if "1976" in uspto_rsmi_file_path else "applications"

    train = pandas.read_csv(
        os.path.join(save_folder_path, f"uspto_{uspto_version}_single_product_train.csv")
    )[[
        'document_id', 'paragraph_num', 'publication_year', 'original_reaction_smiles', 'new_reaction_smiles',
        'retro_templates',
    ]]

    validation = pandas.read_csv(
        os.path.join(save_folder_path, f"uspto_{uspto_version}_single_product_val.csv")
    )[[
        'document_id', 'paragraph_num', 'publication_year', 'original_reaction_smiles', 'new_reaction_smiles',
        'retro_templates',
    ]]

    test = pandas.read_csv(
        os.path.join(save_folder_path, f"uspto_{uspto_version}_single_product_test.csv")
    )[[
        'document_id', 'paragraph_num', 'publication_year', 'original_reaction_smiles', 'new_reaction_smiles',
        'retro_templates',
    ]]

    train.columns = [
        "patent_id", "paragraph_num", "publication_year", "original_reaction_smiles", "new_reaction_smiles",
        "retro_template",
    ]

    validation.columns = [
        "patent_id", "paragraph_num", "publication_year", "original_reaction_smiles", "new_reaction_smiles",
        "retro_template",
    ]

    test.columns = [
        "patent_id", "paragraph_num", "publication_year", "original_reaction_smiles", "new_reaction_smiles",
        "retro_template",
    ]

    clean_data = pandas.concat([train, validation, test])

    clean_data["forward_template"] = [
        x.split(">>")[1] + ">>" + x.split(">>")[0]
        for x in clean_data["retro_template"].values
    ]

    clean_data["multi_part_core"] = [
        len(x.split(">")[0].split(".")) > 1
        for x in clean_data["retro_template"].values
    ]

    clean_data = clean_data[[
        "patent_id", "paragraph_num", "publication_year", "original_reaction_smiles", "new_reaction_smiles",
        "forward_template", "retro_template", "multi_part_core"
    ]]

    unmapped_products, publication_year = [], []

    for data_tuple in tqdm(
        clean_data.values,
        total=len(clean_data.index),
        ascii=True,
        ncols=120,
        desc="Post-processing extracted reaction templates"
    ):
        try:
            mol = MolFromSmiles(data_tuple[4].split(">")[2])

            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)

            SanitizeMol(mol)

            can_sm = MolToSmiles(mol, canonical=True)

            unmapped_products.append(can_sm)

        except:
            unmapped_products.append(None)

    clean_data["main_product"] = unmapped_products
    clean_data = clean_data.dropna(subset=["main_product"])
    clean_data = clean_data.drop_duplicates(subset=["main_product", "forward_template", "retro_template"])

    clean_data["template_count"] = clean_data.groupby("forward_template")["forward_template"].transform("count")

    product_template_lengths = []
    failed_ctr = 0

    for retro_template in clean_data["retro_template"].values:
        length = 0

        for core in retro_template.split(">")[2].split("."):
            try:
                length += len(MolFromSmarts(core).GetAtoms())

            except:
                failed_ctr += 1
                length += 1000

        product_template_lengths.append(length)

    clean_data["product_template_length"] = product_template_lengths

    clean_data = clean_data.sort_values(by=["publication_year"])

    clean_data = clean_data[[
        "patent_id", "paragraph_num", "publication_year", "original_reaction_smiles", "new_reaction_smiles",
        "forward_template", "retro_template", "main_product", "multi_part_core", "product_template_length",
        "template_count"
    ]]

    clean_data.to_csv(
        os.path.join(save_folder_path, f"uspto_{uspto_version}_reaction_templates_dataset.csv"),
        index=False
    )


def generate_kgcn_data(
        uspto_rsmi_file_path: str,
        save_folder_path: str,
        allow_multi_part_cores: bool = False,
        min_frequency: int = None,
        max_product_template_length: int = None
) -> None:
    """ Reads the dataset and generates the kGCN-ready dataset. """

    uspto_version = "grants" if "1976" in uspto_rsmi_file_path else "applications"

    source_data = pandas.read_csv(
        os.path.join(save_folder_path, f"uspto_{uspto_version}_reaction_templates_dataset.csv"),
        low_memory=False
    )

    if not allow_multi_part_cores:
        source_data = source_data[~source_data["multi_part_core"]]

    if min_frequency is not None:
        source_data = source_data[source_data["template_count"] >= min_frequency]

    if max_product_template_length is not None:
        source_data = source_data[source_data["product_template_length"] <= max_product_template_length]

    kgcn_data = source_data[["main_product", "forward_template", "publication_year"]]

    kgcn_data.columns = ["product", "reaction_core", "max_publication_year"]

    kgcn_data.to_csv(
        os.path.join(save_folder_path, f"uspto_{uspto_version}_kgcn_dataset.csv"),
        index=False
    )


def clean_up_files(
        uspto_rsmi_file_path: str,
        save_folder_path: str
) -> None:
    """ Clean up the additionally generated files. """

    uspto_version = "grants" if "1976" in uspto_rsmi_file_path else "applications"

    file_names_to_delete = [
        f"uspto_{uspto_version}_raw_train.csv",
        f"uspto_{uspto_version}_raw_val.csv",
        f"uspto_{uspto_version}_raw_test.csv",
        f"uspto_{uspto_version}_single_product_train.csv",
        f"uspto_{uspto_version}_single_product_val.csv",
        f"uspto_{uspto_version}_single_product_test.csv",
        f"uspto_{uspto_version}_failed_template_train.csv",
        f"uspto_{uspto_version}_failed_template_val.csv",
        f"uspto_{uspto_version}_failed_template_test.csv"
    ]

    for file_name in os.listdir(save_folder_path):
        if file_name in file_names_to_delete:
            os.remove(os.path.join(save_folder_path, file_name))


if __name__ == "__main__":

    args = get_parser()

    seed(args.random_seed)
    numpy.random.seed(args.random_seed)

    reaction_templates_folder_path = os.path.join(args.output_folder_path, "dataset_0_reaction_templates")

    if not os.path.exists(reaction_templates_folder_path):
        os.mkdir(reaction_templates_folder_path)

    print("(1/5) Running the GLN 'clean_uspto.py' script code:")
    GLNUtilities.gln_clean_uspto(args.input_file_path, reaction_templates_folder_path)

    print("\n(2/5) Running the GLN 'build_raw_template.py' script code:")
    GLNUtilities.gln_build_raw_template(args.input_file_path, reaction_templates_folder_path, args.num_cores)

    seed(args.random_seed)
    numpy.random.seed(args.random_seed)

    print("\n(3/5) Running the reaction template post-processing code:")
    post_process_reaction_templates(args.input_file_path, reaction_templates_folder_path)

    print("\n(4/5) Running the kGCN model dataset generation code:")
    generate_kgcn_data(args.input_file_path, reaction_templates_folder_path)

    print("\n(5/5) Running the clean up code:")
    clean_up_files(args.input_file_path, reaction_templates_folder_path)