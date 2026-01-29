# EarlyModernNER

Named Entity Recognition for Early Modern English documents (1500-1800).

## Overview

EarlyModernNER extracts four types of entities from historical texts:

| Entity Type | Description | Examples |
|-------------|-------------|----------|
| **TOPONYM** | Place names | London, Jamaica, West Indies |
| **PERSON** | Individual people | Oliver Cromwell, Governor Modyford |
| **ORGANIZATION** | Institutions | East India Company, Parliament |
| **COMMODITY** | Trade goods & materials | sugar, tobacco, silk |

## Performance

Evaluated on 100 gold-standard annotated documents:

| Entity Type | Precision | Recall | F1 |
|-------------|-----------|--------|-----|
| TOPONYM | 0.93 | 0.82 | 0.87 |
| PERSON | 0.93 | 0.69 | 0.80 |
| ORGANIZATION | 0.93 | 0.46 | 0.62 |
| COMMODITY | 0.85 | 0.80 | 0.83 |
| **Overall** | **0.89** | **0.77** | **0.83** |

## Quick Start

### Installation

```bash
pip install earlymodernner
```

Or install from source:
```bash
git clone https://github.com/polayj/earlymodernner.git
cd earlymodernner
pip install -e .
```

Model adapters (~680MB total) are automatically downloaded from Hugging Face Hub on first use.

### Usage

```bash
# Process a single file
python -m earlymodernner --input document.txt --output results.jsonl

# Process a directory
python -m earlymodernner --input /path/to/docs/ --output results.jsonl

# Output as CSV
python -m earlymodernner --input docs/ --output results.csv --csv

# Pre-download adapters (optional, for offline use)
python -m earlymodernner --download
```

### Output Format

**JSONL** (default):
```json
{
  "doc_id": "document_name",
  "text": "The sugar trade between Jamaica and Bristol...",
  "entities": [
    {"text": "Jamaica", "type": "TOPONYM"},
    {"text": "Bristol", "type": "TOPONYM"},
    {"text": "sugar", "type": "COMMODITY"}
  ]
}
```

**CSV** (with `--csv`):
```csv
doc_id,entity_text,entity_type
document_name,Jamaica,TOPONYM
document_name,Bristol,TOPONYM
document_name,sugar,COMMODITY
```

## Requirements

- Python 3.9+
- CUDA-compatible GPU with 8GB+ VRAM
- See `requirements.txt` for dependencies

## Project Structure

```
earlymodernner/
├── earlymodernner/          # Main package
│   ├── __main__.py          # CLI entry point
│   ├── pipeline.py          # Inference pipeline
│   ├── constants.py         # Entity types & prompts
│   └── adapters/            # Trained LoRA adapters
├── dev/                     # Training & development tools
│   ├── train_lora.py        # Training script
│   ├── evaluate.py          # Evaluation script
│   ├── training.md          # Training documentation
│   └── config/              # Training configurations
├── docs/                    # Documentation
│   ├── usage.md             # Detailed usage guide
│   └── corpus.md            # Training corpus details
└── results/                 # Default output directory
```

## Documentation

- **[Usage Guide](docs/usage.md)** - Detailed usage instructions, input/output formats
- **[Training Corpus](docs/corpus.md)** - Data sources and annotation process
- **[Training Guide](dev/training.md)** - How to train your own adapters

## How It Works

EarlyModernNER uses an **ensemble approach** with four specialized models:

1. Each entity type has its own fine-tuned LoRA adapter
2. Documents are processed by all four adapters
3. Results are merged using priority-based cascade (TOPONYM → COMMODITY → PERSON → ORGANIZATION)
4. Overlapping entities are resolved by giving priority to higher-performing models

**Technical details:**
- Base model: Qwen3-4B-Instruct
- Fine-tuning: QLoRA (4-bit quantization)
- Training: Silver-standard annotations + synthetic hard negatives

## Citation

```bibtex
@software{earlymodernner,
  title = {EarlyModernNER: Named Entity Recognition for Early Modern English},
  author = {Polay, Jacob},
  year = {2026},
  url = {https://github.com/polayj/earlymodernner}
}
```

## License

MIT License

## Author

Jacob Polay, MA Student, University of Saskatchewan

## Acknowledgments

- Built on [Qwen](https://github.com/QwenLM/Qwen) models
- Uses [PEFT](https://github.com/huggingface/peft) for efficient fine-tuning
- Training data from Old Bailey Online, PCEEC2, Royal Society Corpus, EEBO, and Archive.org
