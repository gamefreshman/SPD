# *ShEPhERD*
This repository contains the code to train and sample from *ShEPhERD*'s diffusion generative model, which learns the joint distribution over 3D molecular structures and their shapes, electrostatics, and pharmacophores. At inference, *ShEPhERD* can be used to generate new molecules in their 3D conformations that exhibit target 3D interaction profiles.

Note that *ShEPhERD* has a sister repository, [shepherd-score](https://github.com/coleygroup/shepherd-score), that contains the code to generate/optimize conformers, extract interaction profiles, align molecules via their 3D interaction profiles, score 3D similarity, and evaluate samples from *ShEPhERD* by their validity, 3D similarity to a reference structure, etc. Both repositories are self-contained and have different installation requirements. The few dependencies on [shepherd-score](https://github.com/coleygroup/shepherd-score) that are necessary to train or to sample from *ShEPhERD* have been copied into `shepherd_score_utils/` for user convenience.

The preprint can be found on arXiv: [ShEPhERD: Diffusing shape, electrostatics, and pharmacophores for bioisosteric drug design](https://arxiv.org/abs/2411.04130)

### **Important** notice for current repository
This repository has undergone a major refactor to accommodate inference with PyTorch 2.5, primarily for ease-of-use. To maintain reproducibility for training and inference, the original code can be found under commit `c3d5ec0` or the Release titled "Publication code v0.1.0". The model checkpoints used for publication can be found in those binaries or at the following Dropbox [link](https://www.dropbox.com/scl/fo/rgn33g9kwthnjt27bsc3m/ADGt-CplyEXSU7u5MKc0aTo?rlkey=fhi74vkktpoj1irl84ehnw95h&e=1&st=wn46d6o2&dl=0) where training data can also be found. The checkpoints were converted with `python -m pytorch_lightning.utilities.upgrade_checkpoint <chkpt_path>`
Slight changes have also been made to the training code to adhere to Pytorch Lightning >2.0 and new versions of PyTorch Geometric.

We would like to acknowledge Matthew Cox for his contributions in updating this codebase.

<p align="center">
  <img width="400" src="./docs/images/shepherd_logo.svg">
</p>

<sub><sup>1</sup> **ShEPhERD**: **S**hape, **E**lectrostatics, and **Ph**armacophores **E**xplicit **R**epresentation **D**iffusion</sub>

## Table of Contents
1. [File Structure](##file-structure)
2. [Environment](##environment)
3. [Training and inference data](##training-and-inference-data)
4. [Training](##training)
5. [Inference](##inference)
6. [Evaluations](##evaluations)

## File Structure

```
.
├── src/                                        # source code package
│   └── shepherd/
│       ├── lightning_module.py                 # pytorch-lightning modules
│       ├── datasets.py                         # torch_geometric dataset class (for training)
│       ├── inference.py                        # inference functions
│       ├── extract.py                          # for extracting field properties
│       ├── shepherd_score_utils/               # dependencies from shepherd-score Github repository
│       └── model/
│           ├── equiformer_operations.py        # select E3NN operations from (original) Equiformer
│           ├── equiformer_v2_encoder.py        # slightly customized Equiformer-V2 module
│           ├── model.py                        # module definitions and forward passes
│           ├── utils/                          # misc. functions for forward passes
│           ├── egnn/                           # customized re-implementation of EGNN
│           └── equiformer_v2/                  # clone of equiformer_v2 with slight modifications
├── training/                                   # training scripts and configs
│   ├── train.py                                # main training script
│   ├── parameters/                             # hyperparameter specifications for all models in preprint
│   └── jobs/                                   # empty dir to hold outputs from train.py
├── data/
│   ├── shepherd_chkpts/                        # trained model checkpoints (from pytorch lightning)
│   └── conformers/                             # conditional target structures for experiments, and (sample) training data
├── examples/                                   # examples and experiments
│   ├── RUNME_conditional_generation_MOSESaq.ipynb  # Jupyter notebook for conditional generation
│   ├── RUNME_unconditional_generation.ipynb    # Jupyter notebook for unconditional generation
│   ├── basic_inference/                        # basic inference example
│   └── paper_experiments/                      # inference scripts for all experiments in preprint
├── docs/
│   └── images/
├── docker/                                     # Docker configuration
│   ├── Dockerfile                              # Docker image definition
│   └── shepherd_env.yml                        # conda environment for Docker
├── pyproject.toml                              # Python project configuration
├── setup.py                                    # package setup script
├── environment.yml                             # conda environment requirements
├── LICENSE                                     # license file
├── CHANGELOG.md                                # changelog
└── README.md
```


## Environment

### Requirements
```
python>=3.9
rdkit>=2023.03,<2025.03
torch>=2.5.1
numpy>1.2,<2.0
open3d>=0.18
xtb>=6.6
pandas==2.2.3
```

### Example Environment set-up

`environment.yml` contains the updated conda environment for *ShEPhERD* and compatibility with PyTorch >=2.5.

**We** followed these steps to create a suitable conda environment, which worked on our Linux system. Please note that this exact installation procedure may depend on your system, particularly your cuda version.

```
conda create -n shepherd python=3.9
conda activate shepherd
pip install uv

# download pytorch considering your cuda version
uv pip install torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

uv pip install pytorch-lightning
uv pip install pandas==2.2.3

uv pip install rdkit==2024.09.6 open3d matplotlib jupyterlab
# There may be issues the environment does not set up xTB properly.
#  If this is the case, please install from source.
conda install xtb

# cd to this repo and do a developer install
#  This will install additional requirements found in .toml and not covered above
pip install -e .
```


## Training and inference data
`data/conformers/` contains the 3D structures of the natural products, PDB ligands, and fragments that we used in our experiments in the preprint. It also includes the 100 test-set structures from GDB-17 that we used in our conditional generation evaluations. 

`data/conformers/gdb/example_molblock_charges.pkl` contains *sample* training data from our *ShEPhERD*-GDB-17 training dataset.
`data/conformers/moses_aq/example_molblock_charges.pkl` contains *sample* training data from our *ShEPhERD*-MOSES_aq training dataset.

The full training data for both datasets (<10GB each) can be accessed from this Dropbox link: [https://www.dropbox.com/scl/fo/rgn33g9kwthnjt27bsc3m/ADGt-CplyEXSU7u5MKc0aTo?rlkey=fhi74vkktpoj1irl84ehnw95h&e=1&st=wn46d6o2&dl=0](https://www.dropbox.com/scl/fo/rgn33g9kwthnjt27bsc3m/ADGt-CplyEXSU7u5MKc0aTo?rlkey=fhi74vkktpoj1irl84ehnw95h&e=1&st=wn46d6o2&dl=0)


## Training
`training/train.py` is our main training script. It can be run from the command line by specifying a parameter file and a seed. All of our parameter files are held in `training/parameters/`. To run training, first `cd` into the `training` directory. As an example, one may re-train the P(x1,x3,x4) model on ShEPhERD-MOSES-aq by calling:

```
cd training
python train.py params_x1x3x4_diffusion_mosesaq_20240824 0
```

The trained checkpoints in `data/shepherd_chkpts/` were obtained after training each model for ~2 weeks on 2 V100 gpus. Note that the checkpoints found in this folder have been converted for PyTorch Lightning v2.5. The original, unmodified checkpoints can be found in the original "Publication" release binaries or at the aforementioned data Dropbox link.


## Inference

The simplest way to run inference is to follow the Jupyter notebooks `examples/RUNME_unconditional_generation.ipynb` and `examples/RUNME_conditional_generation_MOSESaq.ipynb`. 

`examples/paper_experiments/` also contain scripts that we used to run the experiments in our preprint. Some of the scripts (`examples/paper_experiments/run_inference_*_unconditional_*_.py`) take a few additional command-line arguments, which are detailed in those corresponding scripts by argparse commands.

The inference script now supports conditional generation of molecules that contain a superset of the target profile's pharmacophores via partial inpainting. [1/13/2025]


## Evaluations

This repository does *not* contain the code to evaluate samples from *ShEPhERD* (e.g., evaluate their validity, RMSD upon relaxation, 3D similarity to a target structure, etc). All such evaluations can be found in the sister repository: https://github.com/coleygroup/shepherd-score. These repositories were made separate so that the functions within [shepherd-score](https://github.com/coleygroup/shepherd-score) can be used for more general-purpose applications in ligand-based drug design. We also encourage others to use [shepherd-score](https://github.com/coleygroup/shepherd-score) to evaluate other 3D generative models besides *ShEPhERD*.


## License

This project is licensed under the MIT License -- see [LICENSE](./LICENSE) file for details.

## Citation
If you use or adapt *ShEPhERD* or [shepherd-score](https://github.com/coleygroup/shepherd-score) in your work, please cite us:

```bibtex
@article{adamsShEPhERD2024,
  title = {{{ShEPhERD}}: {{Diffusing}} Shape, Electrostatics, and Pharmacophores for Bioisosteric Drug Design},
  author = {Adams, Keir and Abeywardane, Kento and Fromer, Jenna and Coley, Connor W.},
  year = {2024},
  number = {arXiv:2411.04130},
  eprint = {2411.04130},
  publisher = {arXiv},
  doi = {10.48550/arXiv.2411.04130},
  archiveprefix = {arXiv}
}
```
