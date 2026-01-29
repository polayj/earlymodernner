"""
Main pipeline for EarlyModernNER.

Orchestrates: preprocessing → inference → output
"""

import csv
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from .constants import ENTITY_TYPES, SYSTEM_PROMPTS
from .normalization import normalize_entity_text


# Default model paths
DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B-Instruct-2507"
HF_ADAPTER_REPO = "jacobpol/earlymodernner-adapters"
DEFAULT_ADAPTER_NAMES = {
    "TOPONYM": "toponym_lora",
    "PERSON": "person_lora",
    "ORGANIZATION": "organization_lora",
    "COMMODITY": "commodity_lora",
}

# Default cache location for downloaded adapters
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "earlymodernner"


def get_adapter_path(adapter_name: str, cache_dir: Optional[Path] = None) -> Path:
    """
    Get the path to an adapter, downloading from Hugging Face Hub if needed.

    Checks in order:
    1. User-specified cache directory
    2. Default cache directory (~/.cache/earlymodernner)
    3. Downloads from HF Hub if not found

    Returns path to the adapter directory.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    adapter_path = cache_dir / adapter_name

    # Check if already downloaded
    if adapter_path.exists() and (adapter_path / "adapter_config.json").exists():
        return adapter_path

    # Need to download
    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub required to download adapters. "
            "Install with: pip install huggingface_hub"
        )

    print(f"  Downloading {adapter_name} from Hugging Face Hub...")

    # Download the specific adapter subfolder
    downloaded_path = snapshot_download(
        repo_id=HF_ADAPTER_REPO,
        allow_patterns=[f"{adapter_name}/*"],
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )

    adapter_path = Path(downloaded_path) / adapter_name

    if not adapter_path.exists():
        raise RuntimeError(f"Failed to download adapter: {adapter_name}")

    return adapter_path


def download_all_adapters(cache_dir: Optional[Path] = None, verbose: bool = False):
    """
    Pre-download all adapters from Hugging Face Hub.

    Useful for offline use or to avoid download delays during processing.
    """
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    if not HF_HUB_AVAILABLE:
        raise ImportError(
            "huggingface_hub required to download adapters. "
            "Install with: pip install huggingface_hub"
        )

    print(f"Downloading all adapters to: {cache_dir}")

    downloaded_path = snapshot_download(
        repo_id=HF_ADAPTER_REPO,
        local_dir=cache_dir,
        local_dir_use_symlinks=False,
    )

    print(f"Adapters downloaded to: {downloaded_path}")
    return Path(downloaded_path)


def preprocess_input(input_path: Path, verbose: bool = False) -> List[Dict]:
    """
    Convert input files to internal document format.

    Supports: .txt, .md, .xml, .jsonl

    Returns list of dicts with 'doc_id' and 'text' fields.
    """
    documents = []

    if input_path.is_file():
        files = [input_path]
    else:
        # Gather all supported files from directory
        files = []
        for pattern in ["*.txt", "*.md", "*.xml", "*.jsonl"]:
            files.extend(input_path.glob(pattern))
        files = sorted(files)

    if not files:
        raise ValueError(f"No supported files found in {input_path}")

    if verbose:
        print(f"Found {len(files)} files to process")

    for filepath in files:
        try:
            if filepath.suffix == ".jsonl":
                # JSONL: each line is a document
                with open(filepath, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if line.strip():
                            doc = json.loads(line)
                            if "doc_id" not in doc:
                                doc["doc_id"] = f"{filepath.stem}_{i}"
                            if "text" in doc:
                                documents.append(doc)
            else:
                # Plain text files (txt, md, xml)
                with open(filepath, 'r', encoding='utf-8') as f:
                    text = f.read()

                if text.strip():
                    documents.append({
                        "doc_id": filepath.stem,
                        "text": text,
                    })

        except Exception as e:
            if verbose:
                print(f"Warning: Could not read {filepath}: {e}")

    if verbose:
        print(f"Loaded {len(documents)} documents")

    return documents


def extract_json_from_output(output: str) -> dict:
    """Extract JSON object from model output."""
    import re

    output = output.strip()

    # Try direct parse first
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in output
    match = re.search(r'\{[^{}]*"entities"[^{}]*\[.*?\][^{}]*\}', output, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"entities": []}


def normalize_entities(entities: List[Dict], entity_type: str, source_text: str) -> List[Dict]:
    """Normalize and validate extracted entities."""
    if not isinstance(entities, list):
        return []

    normalized = []
    seen = set()

    for e in entities:
        if not isinstance(e, dict):
            continue

        text = e.get("text", "")
        if not text or not isinstance(text, str):
            continue

        text = text.strip()
        text_lower = text.lower()

        # Skip duplicates
        if text_lower in seen:
            continue
        seen.add(text_lower)

        # Filter hallucinations - entity must appear in source text
        source_lower = source_text.lower()
        if text_lower not in source_lower:
            # Try with hyphen/space normalization
            text_normalized = text_lower.replace("-", " ").replace("  ", " ")
            source_normalized = source_lower.replace("-", " ").replace("  ", " ")
            if text_normalized not in source_normalized:
                continue

        normalized.append({
            "text": text,
            "type": entity_type,
        })

    return normalized


def build_prompt(text: str, entity_type: str, tokenizer) -> str:
    """Build chat prompt for entity extraction."""
    system_prompt = SYSTEM_PROMPTS[entity_type]
    user_content = f"Please extract {entity_type} entities from the following text.\n\nTEXT:\n{text}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )


def load_model_with_adapter(base_model_name: str, adapter_path: str, device: str = "cuda"):
    """Load base model with LoRA adapter."""
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library required. Install with: pip install peft")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto" if device == "cuda" else device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    return model, tokenizer


def run_entity_extraction(
    model,
    tokenizer,
    documents: List[Dict],
    entity_type: str,
    max_new_tokens: int = 2048,
    verbose: bool = False,
) -> Dict[str, List[Dict]]:
    """Run inference for a single entity type on all documents."""

    print(f"\n  Running {entity_type} extraction on {len(documents)} documents...")

    results = {}

    for i, doc in enumerate(documents):
        doc_id = doc["doc_id"]
        text = doc["text"]

        if not text:
            results[doc_id] = []
            continue

        prompt = build_prompt(text, entity_type, tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_p=None,
                top_k=None,
                temperature=None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        obj = extract_json_from_output(generated)
        entities = normalize_entities(obj.get("entities", []), entity_type, text)
        results[doc_id] = entities

        # Progress every 10 docs
        if (i + 1) % 10 == 0:
            print(f"    Processed {i + 1}/{len(documents)} documents")

    print(f"  {entity_type} complete: {sum(len(v) for v in results.values())} entities extracted")

    return results


def merge_entity_results(
    all_results: Dict[str, Dict[str, List[Dict]]],
    documents: List[Dict],
) -> List[Dict]:
    """Merge results from all entity-type models using priority-based cascade."""

    # Priority order: strongest model first
    PRIORITY_ORDER = ["TOPONYM", "COMMODITY", "PERSON", "ORGANIZATION"]

    merged = []

    for doc in documents:
        doc_id = doc["doc_id"]

        claimed_texts = set()
        unique_entities = []

        for entity_type in PRIORITY_ORDER:
            if entity_type not in all_results:
                continue

            entities = all_results[entity_type].get(doc_id, [])

            for e in entities:
                text_lower = e["text"].lower().strip()
                text_normalized = text_lower.replace("-", " ").replace("  ", " ")

                # Check if already claimed by higher-priority model
                is_claimed = False
                for claimed in claimed_texts:
                    if text_normalized == claimed or text_normalized in claimed:
                        is_claimed = True
                        break

                if not is_claimed:
                    unique_entities.append(e)
                    claimed_texts.add(text_normalized)

        merged.append({
            "doc_id": doc_id,
            "text": doc["text"],
            "entities": unique_entities,
            "entity_counts": {
                etype: len([e for e in unique_entities if e["type"] == etype])
                for etype in ENTITY_TYPES
            }
        })

    return merged


def write_output(results: List[Dict], output_path: Path, as_csv: bool = False):
    """Write results to file."""

    if as_csv:
        # Flatten entities for CSV output
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["doc_id", "entity_text", "entity_type"])

            for doc in results:
                for entity in doc["entities"]:
                    writer.writerow([doc["doc_id"], entity["text"], entity["type"]])
    else:
        # JSONL output
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in results:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')


def run_pipeline(
    input_path: Path,
    output_path: Path,
    entity_types: List[str],
    model_dir: Optional[str] = None,
    output_csv: bool = False,
    batch_size: int = 1,
    device: str = "cuda",
    verbose: bool = False,
):
    """
    Main pipeline: preprocess → inference → output

    Args:
        input_path: File or directory of documents to process
        output_path: Output file path
        entity_types: List of entity types to extract
        model_dir: Directory containing LoRA adapters (None = use bundled)
        output_csv: If True, output CSV instead of JSONL
        batch_size: Documents per batch (currently unused, for future)
        device: cuda or cpu
        verbose: Show detailed progress
    """

    print("=" * 60)
    print("EarlyModernNER")
    print("=" * 60)

    start_time = time.time()

    # Step 1: Preprocess input
    print(f"\nLoading documents from: {input_path}")
    documents = preprocess_input(input_path, verbose=verbose)
    print(f"  Found {len(documents)} documents")

    if not documents:
        print("No documents to process.")
        return

    # Step 2: Determine model paths
    adapter_paths = {}

    for etype in entity_types:
        etype = etype.upper()
        adapter_name = DEFAULT_ADAPTER_NAMES.get(etype)
        if not adapter_name:
            print(f"  Warning: Unknown entity type: {etype}")
            continue

        if model_dir:
            # User specified a custom directory
            adapter_path = Path(model_dir) / adapter_name
            if adapter_path.exists():
                adapter_paths[etype] = str(adapter_path)
            else:
                print(f"  Warning: Adapter not found for {etype}: {adapter_path}")
        else:
            # Download from HF Hub (or use cached version)
            try:
                adapter_path = get_adapter_path(adapter_name)
                adapter_paths[etype] = str(adapter_path)
            except Exception as e:
                print(f"  Warning: Could not get adapter for {etype}: {e}")

    if not adapter_paths:
        raise ValueError(f"No adapters found in {model_dir}")

    print(f"\nEntity types to extract: {list(adapter_paths.keys())}")

    # Step 3: Run inference for each entity type
    all_results = {}

    for entity_type, adapter_path in adapter_paths.items():
        print(f"\n{'='*40}")
        print(f"Loading {entity_type} model...")

        model, tokenizer = load_model_with_adapter(
            DEFAULT_BASE_MODEL,
            adapter_path,
            device=device,
        )

        print(f"  Running extraction on {len(documents)} documents...")
        results = run_entity_extraction(
            model, tokenizer, documents, entity_type,
            verbose=verbose,
        )

        all_results[entity_type] = results

        # Free memory
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

    # Step 4: Merge results
    print(f"\n{'='*40}")
    print("Merging results...")
    merged_results = merge_entity_results(all_results, documents)

    # Step 5: Write output
    print(f"\nWriting output to: {output_path}")
    write_output(merged_results, output_path, as_csv=output_csv)

    # Summary
    elapsed = time.time() - start_time
    total_entities = sum(len(r["entities"]) for r in merged_results)

    print(f"\n{'='*60}")
    print("COMPLETE")
    print("=" * 60)
    print(f"Documents processed: {len(merged_results)}")
    print(f"Total entities extracted: {total_entities}")

    entity_counts = defaultdict(int)
    for r in merged_results:
        for e in r["entities"]:
            entity_counts[e["type"]] += 1

    print("\nBy type:")
    for etype in ENTITY_TYPES:
        if etype in adapter_paths:
            print(f"  {etype}: {entity_counts[etype]}")

    print(f"\nTime elapsed: {elapsed:.1f}s ({elapsed/len(documents):.2f}s per doc)")
    print(f"Output: {output_path}")
    print("=" * 60)
