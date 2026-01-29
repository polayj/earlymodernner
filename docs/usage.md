# Usage Guide

This guide covers how to use EarlyModernNER for extracting named entities from Early Modern English documents.

## Installation

```bash
# Install from PyPI
pip install earlymodernner

# Or install from source
git clone https://github.com/polayj/earlymodernner.git
cd earlymodernner
pip install -e .
```

### Requirements

- Python 3.9+
- CUDA-compatible GPU with at least 8GB VRAM (for inference)
- ~16GB VRAM recommended for training

### Model Adapters

Model adapters (~680MB total) are automatically downloaded from Hugging Face Hub on first use and cached in `~/.cache/earlymodernner/`.

To pre-download adapters (e.g., for offline use):
```bash
python -m earlymodernner --download
```

## Basic Usage

### Command Line

```bash
# Process a single file
python -m earlymodernner --input document.txt --output results.jsonl

# Process all files in a directory
python -m earlymodernner --input /path/to/docs/ --output results.jsonl

# Output as CSV instead of JSONL
python -m earlymodernner --input docs/ --output results.csv --csv
```

### Command Line Options

| Option | Description |
|--------|-------------|
| `--input`, `-i` | Input file or directory (required) |
| `--output`, `-o` | Output file path (required) |
| `--csv` | Output as CSV instead of JSONL |
| `--entity-types` | Entity types to extract (default: all four) |
| `--model-dir` | Custom directory for LoRA adapters |
| `--device` | Device to run on: `cuda` or `cpu` (default: cuda) |
| `--verbose` | Show detailed progress |

### Extract Specific Entity Types

```bash
# Only extract toponyms and commodities
python -m earlymodernner --input docs/ --output results.jsonl --entity-types TOPONYM COMMODITY

# Only extract people
python -m earlymodernner --input docs/ --output results.jsonl --entity-types PERSON
```

## Input Formats

EarlyModernNER accepts several input formats:

### Plain Text (`.txt`)
```
The sugar trade between Jamaica and Bristol flourished in the 18th century.
Governor Modyford established new plantations across the island.
```

### Markdown (`.md`)
Standard markdown files. The entire content is processed as text.

### XML (`.xml`)
XML files are read as plain text. Tags are preserved in the input.

### JSONL (`.jsonl`)
For batch processing, use JSONL format with `doc_id` and `text` fields:
```json
{"doc_id": "doc_001", "text": "The sugar trade between Jamaica and Bristol..."}
{"doc_id": "doc_002", "text": "Governor Modyford established new plantations..."}
```

## Output Formats

### JSONL (Default)

Each line contains a JSON object with the document and extracted entities:

```json
{
  "doc_id": "document_name",
  "text": "The original document text...",
  "entities": [
    {"text": "Jamaica", "type": "TOPONYM"},
    {"text": "Bristol", "type": "TOPONYM"},
    {"text": "sugar", "type": "COMMODITY"}
  ],
  "entity_counts": {
    "TOPONYM": 2,
    "PERSON": 0,
    "ORGANIZATION": 0,
    "COMMODITY": 1
  }
}
```

### CSV (with `--csv` flag)

Flat format suitable for spreadsheets and databases:

```csv
doc_id,entity_text,entity_type
document_name,Jamaica,TOPONYM
document_name,Bristol,TOPONYM
document_name,sugar,COMMODITY
```

## Use Cases

### Processing OCR'd PDFs

If you have OCR'd PDFs as text files:

```bash
# Assuming your OCR output is in txt files
python -m earlymodernner --input ocr_output/ --output ner_results.jsonl
```

For best results:
- Clean obvious OCR errors before processing
- Split very long documents into manageable chunks
- Historical spelling variations are generally handled well

### Extracting Entities for a Database

For importing into a knowledge graph or database:

```bash
# Export as CSV for easy database import
python -m earlymodernner --input documents/ --output entities.csv --csv
```

The CSV output can be directly imported into:
- SQL databases
- Neo4j (for knowledge graphs)
- Pandas DataFrames for further analysis

### Processing Large Collections

For large document collections:

```bash
# Process with verbose output to monitor progress
python -m earlymodernner --input large_corpus/ --output results.jsonl --verbose
```

Processing speed is approximately 2-5 seconds per document depending on length and GPU.

## Python API

You can also use EarlyModernNER programmatically:

```python
from pathlib import Path
from earlymodernner.pipeline import run_pipeline

# Run the full pipeline
run_pipeline(
    input_path=Path("documents/"),
    output_path=Path("results.jsonl"),
    entity_types=["TOPONYM", "PERSON", "ORGANIZATION", "COMMODITY"],
    verbose=True
)
```

## Troubleshooting

### Out of Memory Errors

If you encounter CUDA out of memory errors:
- Ensure no other GPU processes are running
- Try processing fewer documents at a time
- Use `--device cpu` (much slower but works with limited VRAM)

### Slow Processing

- Ensure you're using GPU (`--device cuda`)
- Check that CUDA is properly installed: `python -c "import torch; print(torch.cuda.is_available())"`

### Missing Entities

The model prioritizes precision over recall. If entities are being missed:
- Check that the text is in Early Modern English
- Very short entity mentions may not be captured
- Unusual historical spellings might not be recognized

## Performance

Tested on NVIDIA RTX 5070 Ti (16GB VRAM):
- Training: ~16GB VRAM
- Inference: ~8GB VRAM
- Speed: ~2-5 seconds per document

Minimum recommended: 8GB VRAM GPU for inference.
