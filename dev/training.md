# Training Guide

This guide explains how to train your own EarlyModernNER adapters.

## Prerequisites

- Python 3.9+
- CUDA-compatible GPU with 16GB+ VRAM
- Training dependencies: `pip install -e ".[train]"`

## Training Data Format

Training data must be in chat format (JSONL) with the following structure:

```json
{
  "messages": [
    {"role": "system", "content": "You are EarlyModernNER specialized in extracting TOPONYM entities..."},
    {"role": "user", "content": "Please extract TOPONYM entities from the following text.\n\nTEXT:\n..."},
    {"role": "assistant", "content": "{\"entities\": [{\"text\": \"Jamaica\", \"type\": \"TOPONYM\"}]}"}
  ]
}
```

Each line is one training example. The assistant response must be valid JSON.

## Preparing Training Data

Use the provided script to convert annotated documents to chat format:

```bash
python dev/prepare_training_data.py \
    --input data/annotated_docs.jsonl \
    --output data/training/train_chat.jsonl \
    --entity-type TOPONYM
```

### Input Format

Annotated documents should be JSONL with this structure:

```json
{
  "doc_id": "doc_001",
  "text": "The sugar trade between Jamaica and Bristol...",
  "entities": [
    {"text": "Jamaica", "label": "TOPONYM"},
    {"text": "Bristol", "label": "TOPONYM"}
  ]
}
```

## Configuration

Training is configured via YAML files. See `dev/config/template.yaml` for all options.

Key parameters:

```yaml
# Model
base_model_name: "Qwen/Qwen3-4B-Instruct-2507"

# Data
train_file: "data/training/train_chat.jsonl"
eval_file: "data/training/dev_chat.jsonl"
output_dir: "outputs/my_adapter"

# Training
num_train_epochs: 2
learning_rate: 2.0e-4
per_device_train_batch_size: 2
gradient_accumulation_steps: 4

# LoRA
lora_r: 64
lora_alpha: 16
```

## Running Training

```bash
python dev/train_lora.py --config dev/config/your_config.yaml
```

Training progress is logged to the console. Checkpoints are saved to the output directory.

## Generating Synthetic Training Data

To improve precision, generate synthetic hard negatives:

```bash
python dev/generate_synthetic_training.py
```

This creates:
- **Hard negatives:** Examples with blocklisted terms where the correct output is empty
- **Hard positives:** Examples with commonly missed entities

The script outputs to `data/training/{entity_type}/train_chat_augmented.jsonl`.

## Evaluation

Evaluate your trained adapter:

```bash
python dev/evaluate.py --predictions your_predictions.jsonl
```

This compares predictions against the gold standard and reports:
- Precision, Recall, F1 per entity type
- Common false positives and false negatives
- Partial match handling

## Training Tips

### Entity-Specific Training

Train separate adapters for each entity type (ensemble approach):
- More specialized models perform better than one general model
- Allows different hyperparameters per entity type
- Enables priority-based merging of results

### Epochs

- **2 epochs** works well for most entity types
- **4 epochs** may help for rarer entity types (TOPONYM)
- Watch for overfitting on small datasets

### Learning Rate

- Start with `2.0e-4`
- Reduce to `1.0e-4` if training is unstable

### Blocklists

Maintain blocklists of common false positives:
- `data/blocklists/toponym_blocklist.txt`
- `data/blocklists/person_blocklist.txt`
- etc.

Use the review tool to identify candidates:
```bash
streamlit run dev/review_training_entities.py
```

## Model Architecture

EarlyModernNER uses:
- **Base model:** Qwen3-4B-Instruct (4 billion parameters)
- **Fine-tuning:** QLoRA (4-bit quantization + LoRA adapters)
- **LoRA targets:** All attention and MLP projection layers

This allows training on consumer GPUs while maintaining quality.

## Trained Adapter Structure

After training, your adapter directory contains:
```
my_adapter/
├── adapter_config.json
├── adapter_model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── README.md
```

To use your adapter, point `--model-dir` to the parent directory containing entity-specific subdirectories.
