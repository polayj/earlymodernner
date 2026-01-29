"""
Upload EarlyModernNER adapters to Hugging Face Hub.

Usage:
    python dev/upload_to_hub.py --token YOUR_HF_TOKEN --create-repo

Prerequisites:
    1. Create a Hugging Face account at https://huggingface.co/
    2. Get an access token at https://huggingface.co/settings/tokens (with Write access)
"""

import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, login

# Default settings
DEFAULT_REPO_ID = "jacobpol/earlymodernner-adapters"
ADAPTER_DIR = Path(__file__).parent.parent / "earlymodernner" / "adapters"

ADAPTER_NAMES = [
    # "toponym_lora",  # Already uploaded
    # "person_lora",   # Already uploaded
    "organization_lora",
    "commodity_lora",
]


def main():
    parser = argparse.ArgumentParser(description="Upload adapters to Hugging Face Hub")
    parser.add_argument(
        "--token",
        type=str,
        required=True,
        help="Hugging Face access token (get from https://huggingface.co/settings/tokens)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--adapter-dir",
        type=Path,
        default=ADAPTER_DIR,
        help="Directory containing adapter folders",
    )
    parser.add_argument(
        "--create-repo",
        action="store_true",
        help="Create the repository if it doesn't exist",
    )
    args = parser.parse_args()

    # Login with token
    print("Logging in to Hugging Face Hub...")
    login(token=args.token)

    api = HfApi(token=args.token)

    # Create repo if requested
    if args.create_repo:
        print(f"Creating repository: {args.repo_id}")
        create_repo(
            repo_id=args.repo_id,
            repo_type="model",
            exist_ok=True,
        )

    # Upload each adapter
    for adapter_name in ADAPTER_NAMES:
        adapter_path = args.adapter_dir / adapter_name

        if not adapter_path.exists():
            print(f"Warning: Adapter not found: {adapter_path}")
            continue

        print(f"\nUploading {adapter_name}...")

        # Upload the entire adapter folder
        api.upload_folder(
            folder_path=str(adapter_path),
            path_in_repo=adapter_name,
            repo_id=args.repo_id,
            repo_type="model",
        )

        print(f"  Uploaded: {adapter_name}")

    # Upload a README for the HF repo
    readme_content = """---
language:
- en
license: mit
tags:
- ner
- named-entity-recognition
- early-modern-english
- historical-texts
- lora
- peft
base_model: Qwen/Qwen3-4B-Instruct-2507
---

# EarlyModernNER Adapters

LoRA adapters for Named Entity Recognition in Early Modern English documents (1500-1800).

## Adapters

| Adapter | Entity Type | Precision | Recall | F1 |
|---------|-------------|-----------|--------|-----|
| `toponym_lora` | Place names | 0.93 | 0.82 | 0.87 |
| `person_lora` | People | 0.93 | 0.69 | 0.80 |
| `organization_lora` | Institutions | 0.93 | 0.46 | 0.62 |
| `commodity_lora` | Trade goods | 0.85 | 0.80 | 0.83 |

## Usage

These adapters are used by the [EarlyModernNER](https://github.com/polayj/earlymodernner) package:

```bash
pip install earlymodernner
python -m earlymodernner --input your_docs/ --output results.jsonl
```

Adapters are automatically downloaded on first use.

## Base Model

All adapters are trained on **Qwen/Qwen3-4B-Instruct-2507** using QLoRA (4-bit quantization).

## License

MIT License

## Author

Jacob Polay, University of Saskatchewan
"""

    print("\nUploading README.md...")
    api.upload_file(
        path_or_fileobj=readme_content.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"\nDone! Adapters uploaded to: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
