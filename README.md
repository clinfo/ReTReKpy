# ReTReK: Retrosynthesis Planning Application using Retrosynthesis Knowledge

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
python -m scripts.datasets.create_reaction_template_dataset [--script_arguments]
```

The following script is utilized to **create the chemical compound evaluation dataset**:

```bash
python -m scripts.datasets.create_evaluation_dataset [--script_arguments]
```

The following script is utilized to **create the in-scope filter training dataset**:

```bash
python -m scripts.datasets.create_negative_reaction_dataset [--script_arguments]
```

### Single-step Retrosynthesis
The following script is utilized to **create the single-step retrosynthesis model dataset**:

```bash
python -m scripts.single_step_retrosynthesis.create_dataset [--script_arguments]
```

The following scripts are utilized to **train, test, and infer the single-step retrosynthesis model**:

```bash
python -m scripts.single_step_retrosynthesis.train [--script_arguments]
python -m scripts.single_step_retrosynthesis.test [--script_arguments]
python -m scripts.single_step_retrosynthesis.inference [--script_arguments]
```

### In-scope Filter
The following script is utilized to **train the in-scope filter model**:

```bash
python -m scripts.in_scope_filter.train [--script_arguments]
```

### MCTS
The following script is utilized to **run the Monte Carlo Tree Search**:

```bash
python -m scripts.mcts.run [--script_arguments]
python -m scripts.mcts.run_all [--script_arguments]
```

### Ease-of-synthesis
The following script is utilized to **create the ease-of-synthesis model dataset**:

```bash
python -m scripts.ease_of_synthesis.create_dataset [--script_arguments]
```

The following scripts are utilized to **train and infer the single-step retrosynthesis model**:

```bash
python -m scripts.ease_of_synthesis.train [--script_arguments]
python -m scripts.ease_of_synthesis.inference [--script_arguments]
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
