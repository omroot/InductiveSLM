
# Inductive Triplet Fine-Tuning for Small Language Models



## Introduction
Welcome! This repository explores methods for fine-tuning small language models to enhance their inductive reasoning capabilities. Our goal is to provide reproducible workflows and code for researchers and practitioners interested in improving model performance on inductive tasks.


## Installation
Clone the repository and install dependencies using Conda:

```zsh
git clone https://github.com/omroot/InductiveSLM.git
cd InductiveSLM
conda env create -f environment.yml  # If environment.yml is avai   lable
conda activate inductive-slm         # Replace with your environment name
pip install -r requirements.txt      # If using requirements.txt
```

## Environment Setup & JupyterLab Integration
To use your Conda environment with JupyterLab:

1. Activate your environment:
	```zsh
	conda activate <your_env_name>
	```
2. Install JupyterLab and ipykernel:
	```zsh
	conda install jupyterlab ipykernel
	```
3. Add your environment as a Jupyter kernel:
	```zsh
	python -m ipykernel install --user --name <your_env_name> --display-name "Python (<your_env_name>)"
	```
4. Launch JupyterLab:
	```zsh
	jupyter lab
	```

## Usage
See the provided Jupyter notebooks and scripts in the `src/` directory for examples of data loading, model training, and evaluation. Adjust paths and parameters as needed for your experiments.

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

