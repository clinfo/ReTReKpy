""" The ``scripts.datasets`` directory ``process_reaction_smiles`` module. """

from argparse import ArgumentParser

from retrekpy.datasets import Configuration, TemplateExtraction


def parse_user_args():
    """ Parse the arguments specified by the user during input. """

    parser = ArgumentParser("Template extraction")

    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file."
    )

    parser.add_argument(
        "-t",
        "--task",
        default="all",
        type=str,
        choices=["clean", "map", "ext_tmplt", "all", ],
        help="Task to perform."
    )

    parser.add_argument(
        "-rsmi",
        "--rsmiles",
        default="",
        type=str,
        help="Extract the template for a single reaction SMILES string."
    )

    parser.add_argument(
        "-sd",
        "--save_extended_dataset",
        action="store_true",
        help="Save the extended dataset."
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

    template_extraction = TemplateExtraction(config.template_extraction_configuration)

    if args.rsmiles != "":
        print("\nOriginal Reaction SMILES: {}".format(args.rsmiles))

        if args.task == "clean":
            print("Cleaned Reaction SMILES: {}".format(
                template_extraction.clean_entry(args.rsmiles, verbose=args.verbose)
            ))

        elif args.task == "map":
            print("Atom-mapped Reaction SMILES: {}".format(
                template_extraction.atom_map_entry(args.rsmiles, verbose=args.verbose)
            ))

        elif args.task == "ext_tmplt":
            print("Extracted Template SMARTS: {}".format(
                template_extraction.extract_reaction_template_from_entry(args.rsmiles, verbose=args.verbose)
            ))

        elif args.task == "all":
            cleaned_rsmiles = template_extraction.clean_entry(args.rsmiles, verbose=args.verbose)

            print("Cleaned Reaction SMILES: {}".format(cleaned_rsmiles))

            mapped_rsmiles = template_extraction.atom_map_entry(cleaned_rsmiles, verbose=args.verbose)

            print("Atom-mapped Reaction SMILES: {}".format(mapped_rsmiles))

            print("Extracted Template SMARTS: {}".format(
                template_extraction.extract_reaction_template_from_entry(mapped_rsmiles, verbose=args.verbose)
            ))

    else:
        template_extraction.extract_reaction_templates(
            save_extended_dataset=args.save_extended_dataset,
            verbose=args.verbose
        )
