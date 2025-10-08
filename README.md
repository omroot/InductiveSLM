# Inductive Triplet Fine-Tuning for Small Language Models

## Introduction
This repository implements methods for fine-tuning small language models to enhance their inductive reasoning capabilities using triplet-based training data. The project focuses on improving model performance on inductive reasoning tasks through LoRA (Low-Rank Adaptation) fine-tuning.

## Paper & Dataset

📄 **Paper**: [Inductive Triplet Fine-Tuning for Small Language Models](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5529459)

📊 **IR-Triplet Dataset**:
- **Local**: Available in this repository at `cache/raw_data/ir_triplets/ir_triplets.json`
- **HuggingFace**: [Coming soon]

### Citation

If you use the **IR-Triplet benchmark** or this codebase in your research, please cite:

```bibtex
@article{missaoui2025inductive,
  title={Inductive Triplet Fine-Tuning for Small Language Models},
  author={Missaoui, Oualid},
  journal={SSRN Electronic Journal},
  year={2025},
  doi={10.2139/ssrn.5529459},
  url={https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5529459}
}
```

## Key Features
- **Triplet-based Training**: Converts inductive reasoning datasets into observation-question-answer triplets
- **LoRA Fine-tuning**: Memory-efficient fine-tuning using PEFT (Parameter-Efficient Fine-Tuning)
- **Multi-dataset Support**: Includes processors for IR-Triplets and DEER datasets
- **Comprehensive Evaluation**: Automated metrics calculation (ROUGE, BLEU) for both in-distribution and out-of-distribution validation
- **Model Analysis**: Integration with WeightWatcher for model quality diagnostics

## Installation

1. Clone the repository:
```bash
git clone https://github.com/omroot/InductiveSLM.git
cd InductiveSLM
```

2. Install dependencies:
```bash
# Basic installation (recommended for most users)
pip install -r requirements.txt

# Optional: If you have CUDA and need Mamba models
pip install -r requirements-cuda.txt
```

**Note**: The basic installation works on all platforms (CPU, GPU, ARM64). CUDA-specific packages (mamba-ssm, selective_scan_cuda) are optional and only needed for Mamba models.

3. Configure your environment:
   - Copy the template file and add your HuggingFace token:
     ```bash
     cp .env_template .env
     # Edit .env and add your HuggingFace token
     ```
   - Get your HuggingFace token from: https://huggingface.co/settings/tokens
   - Update the paths in `.env` to match your local setup

## Project Structure
```
InductiveSLM/
├── notebooks/          # Jupyter notebooks for experiments
│   └── main.ipynb     # Main training and evaluation workflow
├── src/
│   ├── config.py      # Model and training configuration
│   ├── settings.py    # Path settings
│   ├── models/        # Model training and inference
│   ├── preprocess/    # Dataset preprocessing (DEER, triplets)
│   └── utils/         # Utility functions for I/O
├── cache/
│   ├── raw_data/      # Raw datasets
│   │   ├── ir_triplets/    # IR-Triplet benchmark dataset
│   │   │   └── ir_triplets.json
│   │   └── deer/      # DEER dataset
│   └── models/        # Model outputs and evaluation results
└── requirements.txt   # Python dependencies
```

## Usage

### IR-Triplet Dataset Dashboard

Explore the IR-Triplet benchmark interactively:
```bash
streamlit run dashboard_ir_triplets.py
```

Features:
- Browse triplets with pagination
- Filter by reasoning form
- View statistics and distributions
- Export filtered data (JSON, CSV, Markdown)

### Quick Start - Single Model

Evaluate a single model:
```bash
python run_evaluation.py
```

### Batch Evaluation - Multiple Models

Evaluate all models in the model list:
```bash
python run_batch_evaluation.py
```

Or select specific models:
```bash
python run_selective_evaluation.py --small  # Evaluate small models only
python run_selective_evaluation.py 0 1 2    # Evaluate models at indices 0, 1, 2
```

See [BATCH_EVALUATION_GUIDE.md](BATCH_EVALUATION_GUIDE.md) for detailed instructions.

### Jupyter Notebooks

Launch JupyterLab and open `notebooks/main.ipynb`:
```bash
jupyter lab
```

The main notebook demonstrates:
1. Loading and preprocessing inductive reasoning datasets
2. Converting DEER dataset to triplet format
3. Fine-tuning a base model with LoRA
4. Evaluating baseline vs fine-tuned model performance
5. Analyzing model quality with WeightWatcher

### Configuration
Edit `src/config.py` to customize:
- Model selection (`MODEL_ID`)
- LoRA hyperparameters (rank, alpha, dropout)
- Training parameters (learning rate, batch size, epochs)
- Output directories

### Key Components

#### Data Preprocessing
```python
from src.preprocess.deer import DeerToTriplets
from src.utils.io.read import RawDataReader

rdr = RawDataReader(Settings.paths.RAW_DATA_PATH)
deer_dataset = rdr.read_deer()
converter = DeerToTriplets()
converter.process(deer_dataset)
```

#### Model Fine-tuning
```python
from src.models.sft.lora import finetune_model_with_lora

model = finetune_model_with_lora(
    model_id=MODEL_ID,
    train_dataset=train_ds,
    tokenizer=tokenizer,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    # ... other parameters
)
```

#### Evaluation
```python
from src.models.evaluate import eval_metrics, print_compare
from src.models.inference.inference import generate_answers_with

predictions = generate_answers_with(model, tokenizer, observations, questions)
metrics = eval_metrics(predictions, references, rouge, bleu)
```

## Datasets

### IR-Triplet Benchmark

The **IR-Triplet benchmark** is a novel dataset for evaluating inductive reasoning in language models. It consists of triplet-based inductive reasoning tasks designed to test models' ability to generalize from observations.

📊 **Access**:
- **In this repository**: `cache/raw_data/ir_triplets/ir_triplets.json`
- **HuggingFace**: [Coming soon]

📄 **Paper**: [Inductive Triplet Fine-Tuning for Small Language Models](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5529459)

⚠️ **Citation Required**: If you use the IR-Triplet benchmark in your research, please cite the paper (see [Citation](#citation) section above).

### Supported Datasets

The project currently supports:
- **IR-Triplets**: Our inductive reasoning triplet dataset (primary benchmark)
- **DEER**: Dataset for evaluating reasoning (converted to triplet format for out-of-distribution testing)

## Dependencies
Key dependencies include:
- `transformers`: HuggingFace models and training
- `peft`: LoRA implementation
- `bitsandbytes`: Quantization support
- `evaluate`, `rouge_score`, `sacrebleu`: Evaluation metrics
- `weightwatcher`: Model quality analysis
- `torch`: Deep learning framework

See `requirements.txt` for the complete list.

## Environment Setup for JupyterLab

1. Activate your environment:
```bash
conda activate <your_env_name>
```

2. Install JupyterLab and ipykernel:
```bash
conda install jupyterlab ipykernel
```

3. Add your environment as a Jupyter kernel:
```bash
python -m ipykernel install --user --name <your_env_name> --display-name "Python (<your_env_name>)"
```

4. Launch JupyterLab:
```bash
jupyter lab
```

## Acknowledgments

This work is based on the research presented in:

**Inductive Triplet Fine-Tuning for Small Language Models**
Available at: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5529459

If you use this codebase or the IR-Triplet benchmark, please cite the paper.

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.

**Note**: While the code is MIT licensed, if you use the **IR-Triplet dataset benchmark**, please cite the associated research paper.
