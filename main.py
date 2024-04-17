""" The min script. """

import pandas

if __name__ == "__main__":
    training_dataset = pandas.read_csv(
        filepath_or_buffer="/nasa/shared_homes/haris/development/riken_retrek_improvement/deliverables/single_step_retrosynthesis/data/processed/uspto_grants_templates_training_and_validation_split_5.csv"
    )

    test_dataset = pandas.read_csv(
        filepath_or_buffer="/nasa/shared_homes/haris/development/riken_retrek_improvement/deliverables/single_step_retrosynthesis/data/processed/uspto_grants_templates_test_split_5.csv"
    )

    print(training_dataset.columns)
    print()
    print(len(training_dataset.index))
    print(len(training_dataset[training_dataset["template_count"] == 1].index))
    print()
    print(len(test_dataset.index))
