[![arXiv](https://img.shields.io/badge/arXiv-2605.00099-b31b1b.svg)](https://arxiv.org/abs/2605.00099)

# Provable and scalable quantum Gaussian processes

Code and data repository for the article:

> **"Provable and scalable quantum Gaussian processes for quantum learning"**,<br>
> *J. Jäger, P. Braccia, P. Bermejo, M. G. Algaba, D. García-Martín, and M. Cerezo (2026).* 

Preprint available on arXiv: [arXiv:2605.00099](https://arxiv.org/abs/2605.00099)


## Organization

The repository is organized by the experiments and studies presented in the paper.

### Top-Level Directory

 * Prefixes `01` to `04`: Studies featured in the main text (corresponding to the four Results figures).
 * Prefixes `05` to `07`: Studies featured in the Supplemental Information.
 * `src/` & `config/`: General source code and configurations shared across multiple studies.

### Study Directory Structure

Some study directories are further divided into subdirectories for specific tasks (e.g., `a_Regression`/`b_Classification` with the XXX bond-alternating model, or `a_BO_2D_Synthetic`/`b_BO_Quantum_Sensing` for Bayesian optimization).

Within each study and/or sub-study folder, you will find:
 * `data/`: Datasets generated for the paper.
 * `plots/`: Output figures corresponding to the paper.
 * `notebooks/` and/or `scripts/`: Jupyter notebooks and code scripts for data generation and analysis.

If applicable, you may also find:
 * `results/`: Intermediate or final data from computationally intensive processing steps.
 * `src/` & `config/`: Source code and configurations specific *only* to that study.
 * `README.md`: Study-specific information and instructions.

### Code Naming Conventions

Workflows are generally split into data generation and analysis (which covers QGP training, evaluation, and plotting). If result processing is highly computationally intensive, it is split further.
 * **Data generation:** Files are named or prefixed with `data_generation`.
 * **Analysis & Processing:** Files are prefixed with `qgp_` or `results_plotting_` (if split) in the filename.


## Usage

### Jupyter Notebooks
Notebooks (`.ipynb`) can be executed directly in Jupyter and contain either Julia or Python code. Experiment configurations are typically defined in the top cells.

*Note:* Preset configurations match the exact settings presented in the paper. You may adapt these parameters when testing the code to limit computational demands.

### Julia Scripts
Julia scripts (`.jl`) are executed from the command line and expect experiment configurations (e.g., the number of qubits) either from a JSON file in the `config/` folder of the study directory or as command-line arguments.

**Example:**
```bash
julia data_generation.jl 10 2
```
where the Julia script `data_generation.jl` receives two arguments set to `10` and `2`.

*Note:* The exact arguments and configurations required to reproduce the paper's results are either contained in the study-specific `config/` folder (and automatically read by the Julia script) or provided in the study-specific READMEs.


## Installation

To run the scripts in this repository, you will need to install the required dependencies for both Python and Julia.

### Python Requirements
To ensure full reproducibility, this project uses **Python 3.11**. We recommend using [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage your Python environment. For example, create and activate a new Conda environment with Python 3.11:
```bash
conda create -n qgp_env python=3.11
conda activate qgp_env
```
where `qgp_env` is simply the name of the environment and may be adjusted as desired.

Then, install the required Python packages via `pip`:
```bash
pip install -r requirements.txt
```

### Julia Requirements

To install the required Julia packages, simply run the included setup script from your terminal:

```bash
julia setup.jl
```
*Note:* The first time you install and use Julia packages, they may take a few minutes to precompile.

**Included Julia Dependencies:**

For reference, the setup.jl script will automatically install the following packages into your environment:

 * AbstractGPs
 * ApproximateGPs
 * Arpack
 * Combinatorics
 * Dierckx
 * Distributions
 * DoubleFloats
 * Interpolations
 * ITensorMPS
 * ITensors
 * JLD2
 * JSON
 * KernelFunctions
 * LaTeXStrings
 * LogExpFunctions
 * NLopt
 * NPZ
 * Optim
 * Plots
 * ProgressMeter
 * PyCall
 * PyPlot
 * SpecialFunctions
 * StatsBase
 * Yao
 * Zygote
 

# Citation
If you use this repository (code or data) in your research, please cite the accompanying paper:
```bibtex
@misc{jaeger2026provablescalablequantumgaussian,
      title={Provable and scalable quantum Gaussian processes for quantum learning}, 
      author={Jonas Jäger and Paolo Braccia and Pablo Bermejo and Manuel G. Algaba and Diego García-Martín and M. Cerezo},
      year={2026},
      eprint={2605.00099},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2605.00099}, 
}
```