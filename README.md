# ReTReK: Retrosynthesis Planning Application using Retrosynthesis Knowledge (Python)

Welcome to the **ReTReKpy** project !!!

## Environment
This version of the ReTReKpy repository is utilizing the [***kMoL***](https://github.com/elix-tech/kmol) library. The
execution environment can be established using [***conda***](https://docs.conda.io/en/latest) and
[***pip***](https://pip.pypa.io/en/stable) as follows:

```bash
git clone https://github.com/elix-tech/kmol.git

cd kmol

make create-env

conda activate kmol

conda install -c conda-forge -c ljn917 molvs rdchiral_cpp -y

pip install epam.indigo quantulum3

cd /.../ReTReKpy

pip install --no-build-isolation -e . --user
```

**NOTE:** If [this issue](https://github.com/pytorch/pytorch/issues/123097) is encountered during the set-up of the
environment, please add `- mkl=2024.0` at the end of the `/.../kmol/environment.yml` file before running the
`make create-env` command.

### Docker
The execution environment can be established using [***Docker***](https://www.docker.com) as follows:

```bash
git clone https://github.com/elix-tech/kmol.git

cd kmol

make build-docker

cd /.../ReTReKpy

docker build . -t retrekpy

alias retrekpy_docker='docker run --rm -it --gpus=all --ipc=host --volume="$(pwd)"/:/opt/elix/kmol/ReTReKpy/ retrekpy'
```

Now, the `python` command can be replaced with the `retrekpy_docker` command.

## Scripts

### Datasets
The following script is utilized to **create the chemical reaction template dataset**:

```bash
python -m scripts.datasets.create_reaction_template_dataset
  --input_file_path  # The path of the USPTO Grants or Applications '*.rsmi' file.
  --output_folder_path  # The path to the output directory.
  --random_seed  # The random seed value.
  --num_cores  # The number of CPU cores for multiprocessing.
```

The following script is utilized to **create the chemical compound evaluation dataset**:

```bash
python -m scripts.datasets.create_evaluation_dataset
  --input_file_path  # The path of any raw ChEMBL '*.txt' file.
  --output_folder_path  # The path to the output directory.
  --salts_file_path  # The path to the salts file.
  --min_atoms  # The minimum atoms cutoff value.
  --max_atoms  # The maximum atoms cutoff value.
  --fp_similarity_cutoff  # The fingerprint similarity cutoff value during Butina clustering.
  --small_dataset_size  # The size of the small evaluation dataset.
  --large_dataset_size  # The size of the large evaluation dataset.
  --outlier_percentage  # The percentage of outliers included in the final evaluation datasets.
  --random_seed  # The random seed value.
  --num_cores  # The number of CPU cores for multiprocessing.
```

The following script is utilized to **create the additional information dataset**:

```bash
python -m scripts.datasets.create_negative_reaction_dataset
  --input_folder_path  # The path of the USPTO Grants or Applications folder.
  --output_folder_path  # The path to the output directory.
  --random_seed  # The random seed value.
  --num_cores  # The number of CPU cores for multiprocessing.
```

The following script is utilized to **create the in-scope filter training datasets**:

```bash
python -m scripts.datasets.create_negative_reaction_dataset
  --input_ai_file_path  # The path of the additional information dataset file.
  --input_rt_file_path  # The path of the reaction template dataset file.
  --output_folder_path  # The path to the output directory.
  --yield_confidence  # The reaction yield confidence value.
  --yield_cutoff  # The reaction yield percentage cutoff value.
  --min_template_frequency  # The minimum reaction template frequency.
  --rt_virtual_negatives  # The number of virtual negative examples generated using random reaction templates.
  --rp_virtual_negatives  # The number of virtual negative examples generated using random perturbations.
  --random_seed  # The random seed value.
  --num_cores  # The number of CPU cores for multiprocessing.
```

### Single-step Retrosynthesis
The following script is utilized to **create the single-step retrosynthesis model dataset**:

```bash
python -m scripts.single_step_retrosynthesis.create_dataset
  --dataset_file_path  # The path to the '*.csv' dataset file.
  --output_directory_path  # The path to the output directory.
  --minimum_number_of_reaction_template_occurrences # The minimum number of chemical reaction template occurrences.
  --number_of_cross_validation_splits  # The number of cross-validation splits.
  --validation_percentage  # The percentage of the dataset that should be utilized for validation.
  --random_seed  # The random seed value.
```

The following scripts are utilized to **train, test, and infer the single-step retrosynthesis model**:

```bash
python -m scripts.single_step_retrosynthesis.train
  --configuration_file_path  # The path to the configuration file.

python -m scripts.single_step_retrosynthesis.test
  --configuration_file_path  # The path to the configuration file.

python -m scripts.single_step_retrosynthesis.inference
  --compound_smiles  # The chemical compound SMILES string.
  --configuration_file_path  # The path to the configuration file.
```

### In-scope Filter
The following script is utilized to **train the in-scope filter model**:

```bash
python -m scripts.in_scope_filter.train
  --csv_path  # The path to the '*.csv' dataset file.
  --reaction_column  # The name of the chemical reaction column from the '*.csv' dataset file.
  --product_column # The name of the chemical reaction product column from the '*.csv' dataset file.
  --label_column  # The name of the label column from the '*.csv' dataset file.
  --pos_weight  # The positive weight for the BinaryCrossEntropy loss calculation.
  --learning_rate  # The learning rate value.
  --weight_decay  # The weight decay value.
  --epochs  # The number of epochs.
  --batch_size  # The batch size.
  --num_workers  # The number number of workers utilized to loading the dataset. 
  --use_cuda  # The indicator of whether CUDA should be utilized.
  --save_path  # The path to the output directory.
```

### MCTS
The following script is utilized to **run the Monte Carlo Tree Search**:

```bash
python -m scripts.mcts.run
  --target  # The path to the target chemical compound '*.mol' file.
  --save_result_dir  # The path to the output directory.
  --config  # The path to the configuration file.
  --debug  # The indicator of whether to enable the debug mode.
  --random_seed  # The random seed value.

python -m scripts.mcts.run_all
  --targets  # The path to the directory containing the target chemical compound '*.mol' files.
  --save_result_dir  # The path to the output directory.
  --config  # The path to the configuration file.
  --debug  # The indicator of whether to enable the debug mode.
  --random_seed  # The random seed value.
```

### Ease-of-synthesis
The following script is utilized to **create the ease-of-synthesis model dataset**:

```bash
python -m scripts.ease_of_synthesis.create_dataset
  --data_folder  # The path to the directory containing the results of the ReTReK approach.
  --uspto_template_path  # The path to the '*.csv' dataset file used for the training of the single-step retrosynthesis model.
  --template_column  # The name of the forward chemical reaction template column from the '*.csv' dataset file.
  --output_path  # The output directory path where the newly-created dataset should be stored.
```

The following scripts are utilized to **train and infer the ease-of-synthesis model**:

```bash
python -m scripts.ease_of_synthesis.train
  --config  # The path to the configuration file.
  --task # The indicator of the task: "train", "eval", or "bayesian_opt". 
  --num_workers # The number number of workers utilized to loading the dataset. 
  --eval_output_path # The path to a custom directory to store the evaluation results.

python -m scripts.ease_of_synthesis.inference
  --classification_model  # The path to the binary classification model configuration file.
  --regression_model  # The path to the regression model configuration file.
  --data  # The chemical compound SMILES string or the path to a file containing one chemical compound SMILES string per each line.
  --save_path  # The output directory path where the results should be stored.
  --featurizer  # The indicator of the featurizer that should be utilized: "ecfp", "graph", or "mordred". 
```

## Contact
If you have any additional questions or comments, please feel free to reach out via
[**GitHub Issues**](https://github.com/clinfo/ReTReKpy/issues) or via **e-mail**:

- **Shoichi Ishida:** [ishida.sho.nm@yokohama-cu.ac.jp](mailto:ishida.sho.nm@yokohama-cu.ac.jp)
- **Ryosuke Kojima:** [kojima.ryosuke.8e@kyoto-u.ac.jp ](mailto:kojima.ryosuke.8e@kyoto-u.ac.jp)

## Citation
```
@article{
    Ishida2022,
    author = {Shoichi Ishida and Kei Terayama and Ryosuke Kojima and Kiyosei Takasu and Yasushi Okuno},
    title = {AI-Driven Synthetic Route Design Incorporated with Retrosynthesis Knowledge},
    journal = {Journal of Chemical Information and Modeling},
    volume = {62},
    number = {6},
    pages = {1357-1367},
    year = {2022},
    doi = {https://doi.org/10.1021/acs.jcim.1c01074}
}
```
