# Inductive Triplet Fine-Tuning for Small Language Models

## Introduction
This repository implements methods for fine-tuning small language models to enhance their inductive reasoning capabilities using triplet-based training data. The project focuses on improving model performance on inductive reasoning tasks through LoRA (Low-Rank Adaptation) fine-tuning.

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
pip install -r requirements.txt
```

3. Configure your environment:
   - Create a `.env` file with your HuggingFace token:
     ```
     HUGGINGFACE_HUB_TOKEN=your_token_here
     ```

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
├── cache/             # Cached data
└── requirements.txt   # Python dependencies
```

## Usage

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
The project supports:
- **IR-Triplets**: Inductive reasoning triplet dataset
- **DEER**: Dataset for evaluating reasoning (converted to triplet format)

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

## License
This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
