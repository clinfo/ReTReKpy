# ReTReK: ReTrosynthesis planning application using Retrosynthesis Knowledge (Elix final deliverable)
**(MANUSCRIPT IN PREPARATION)**  
This package provides a data-driven computer-aided synthesis planning tool using retrosynthesis knowledge.
In this package, the model of ReTReK was trained with US Patent dataset instead of Reaxys reaction dataset. 

<div align="center">
  <img src="./images/ReTReK_summary.jpg">
</div>

## Dependancy

### Environment
- Ubuntu: 20.04

### Package
- python: 3.7
- tensorflow-gpu: 1.13.1
- keras-gpu: 2.3.1
- cudatoolkit: 11
- RDKit
- [RDChiral](https://github.com/connorcoley/rdchiral)
- [ODDT](https://github.com/oddt/oddt)
- [mendeleev](https://github.com/lmmentel/mendeleev)
- [MolVS](https://github.com/mcs07/MolVS)
- PyTorch: 1.13
- py4j: 0.10.8.1 (for backward compatibility. not used)
- tqdm
- [kGCN](https://github.com/clinfo/kGCN)

## Setup

> **Note**
> Mamba is recommended over Conda for faster installation.

```bash
mamba create -n retrekpy python=3.7
mamba activate retrekpy
mamba install -c conda-forge -c anaconda -c ljn917 -c pytorch rdkit rdchiral_cpp keras-gpu=2.3.1 tensorflow-gpu=1.13.1 cudatoolkit=11 tqdm oddt mendeleev py4j=0.10.8.1 molvs zipp=3.15 pytorch=1.13
```


## Example usage

Note: The order of the knowledge arguments corresponds to that of the knowledge_weight arguments. 
```bash
# use all knowledge
python run.py --config config/sample.json --knowledge all --knowledge_weights 1.0 1.0 1.0 1.0 1.0 1.0

# use CDScore with a weight of 2.0
python run.py --config config/sample.json --knowledge cdscore --knowledge_weights 2.0 0.0 0.0 0.0 0.0 0.0
```

## Terms
### Convergent Disconnection Score (CDScore)
CDScore aims to favor convergent synthesis, which is known as an efficient strategy in multi-step chemical synthesis. 

### Available Substances Score (ASScore)
For the same purpose of CDScore, how many available substances generated in a reaction step is calculated.

### Ring Disconnection Score (RDScore)
A ring construction strategy is preferred if a target compounds has complex ring structures.
 
### Selective Transformation Score (STScore)
A synthetic reaction with few by-products is generally preferred in terms of its yield.

### Intermediate Score (IntermediateScore)
IntermediateScore favors states that include molecules from a user defined list of intermediate compounds.

### Template Score (TemplateScore)
TemplateScore allows to prioritize some reaction templates, it should be given as a JSON list in the format
`template: template_score (between 0 and 1)`. Templates not included will have a default score of 0.

## Contact
Shoichi Ishida: ishida.sho.nm@yokohama-cu.ac.jp

## Reference
