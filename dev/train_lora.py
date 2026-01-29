#!/usr/bin/env python3
"""
Train EarlyModernNER using QLoRA on Qwen models.

Supports both multi-entity training (all entity types) and single-entity
training for ensemble models.

Usage:
    # Standard multi-entity training
    python scripts/train_lora.py --config config/qwen3_4b_ner_lora.yaml

    # Single-entity training for ensemble (COMMODITY specialist)
    python scripts/train_lora.py --config config/qwen3_4b_ner_lora.yaml --entity-type COMMODITY

    # Single-entity with custom output directory
    python scripts/train_lora.py --config config/qwen3_4b_ner_lora.yaml --entity-type COMMODITY --output-suffix _commodity
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from earlymodernner.inference import ner_inference_for_text
from earlymodernner.metrics import compute_unique_entity_f1
from earlymodernner.constants import ALLOWED_ENTITY_TYPES


# ---------- Custom Data Collator ----------

@dataclass
class DataCollatorForCausalLM:
    """
    Data collator that pads input_ids, attention_mask, and labels together.
    Handles variable-length sequences for causal language modeling.
    """
    tokenizer: Any
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extract individual fields
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]

        # Find max length in batch
        max_len = max(len(ids) for ids in input_ids)
        if self.max_length:
            max_len = min(max_len, self.max_length)

        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_len = ((max_len + self.pad_to_multiple_of - 1)
                      // self.pad_to_multiple_of * self.pad_to_multiple_of)

        # Pad each sequence
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for ids, mask, labs in zip(input_ids, attention_mask, labels):
            padding_length = max_len - len(ids)

            # Pad input_ids
            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            batch_input_ids.append(padded_ids)

            # Pad attention_mask
            padded_mask = mask + [0] * padding_length
            batch_attention_mask.append(padded_mask)

            # Pad labels (use -100 for padding tokens)
            padded_labels = labs + [-100] * padding_length
            batch_labels.append(padded_labels)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


# ---------- Config dataclass ----------

@dataclass
class TrainConfig:
    """Training configuration for EarlyModernNER."""

    # Model & data
    base_model_name: str
    train_file: str
    eval_file: str
    output_dir: str

    # Sequence / batching
    max_seq_length: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    eval_accumulation_steps: int

    # Training schedule
    num_train_epochs: int
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    weight_decay: float
    max_grad_norm: float

    # Precision & memory (QLoRA)
    use_4bit: bool
    bnb_4bit_compute_dtype: str
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool

    gradient_checkpointing: bool
    tf32: bool

    # LoRA config
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]

    # Logging & eval
    logging_steps: int
    save_steps: int
    eval_steps: int
    save_total_limit: int
    evaluation_strategy: str
    do_eval: bool

    # Misc
    seed: int
    report_to: List[str]


def load_train_config(path: str) -> TrainConfig:
    """Load training configuration from YAML file."""
    print(f"Loading config from: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    config = TrainConfig(**cfg_dict)

    # Print config summary
    print("\nTraining Configuration:")
    print(f"  Model: {config.base_model_name}")
    print(f"  Train file: {config.train_file}")
    print(f"  Eval file: {config.eval_file}")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Max seq length: {config.max_seq_length}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Effective batch size: {config.per_device_train_batch_size * config.gradient_accumulation_steps}")
    print(f"  4-bit quantization: {config.use_4bit}")
    print(f"  LoRA rank: {config.lora_r}\n")

    return config


# ---------- Data loading ----------

def load_chat_dataset(path: str):
    """
    Load chat-style dataset from JSONL.

    Args:
        path: Path to JSONL file where each line is {"messages": [...]}

    Returns:
        HuggingFace Dataset
    """
    print(f"Loading dataset from: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    dataset = load_dataset(
        "json",
        data_files=path,
        split="train",
    )

    print(f"  Loaded {len(dataset)} examples")
    return dataset


def filter_chat_dataset_by_entity_type(
    dataset,
    entity_type: str,
) -> "Dataset":
    """
    Filter a chat dataset to only include examples with the specified entity type.

    Modifies the assistant response to only include entities of the target type.

    Args:
        dataset: HuggingFace Dataset with 'messages' field
        entity_type: Entity type to filter for (e.g., 'COMMODITY')

    Returns:
        Filtered dataset
    """
    print(f"  Filtering for entity type: {entity_type}")

    def filter_and_transform(example):
        messages = example["messages"]

        # Find assistant message and parse entities
        new_messages = []
        has_target_entity = False

        for msg in messages:
            if msg["role"] == "assistant":
                try:
                    content = json.loads(msg["content"])
                    entities = content.get("entities", [])

                    # Filter to only target entity type
                    filtered = [e for e in entities if e.get("type") == entity_type]

                    if filtered:
                        has_target_entity = True

                    # Update assistant content
                    new_content = json.dumps({"entities": filtered}, ensure_ascii=False)
                    new_messages.append({"role": "assistant", "content": new_content})
                except (json.JSONDecodeError, TypeError):
                    new_messages.append(msg)
            else:
                new_messages.append(msg)

        return {
            "messages": new_messages,
            "_has_target_entity": has_target_entity,
        }

    # Apply filter and transformation
    transformed = dataset.map(filter_and_transform)

    # Keep only examples that had the target entity type
    filtered = transformed.filter(lambda x: x.get("_has_target_entity", False))

    # Remove the helper field
    filtered = filtered.remove_columns(["_has_target_entity"])

    print(f"  Filtered: {len(dataset)} -> {len(filtered)} examples")
    return filtered


# ---------- Tokenization / labeling ----------

def make_tokenize_fn(tokenizer, cfg: TrainConfig):
    """
    Create a tokenization function that masks labels for system+user messages.

    Only the assistant's JSON response should contribute to the loss.

    Args:
        tokenizer: HuggingFace tokenizer
        cfg: Training configuration

    Returns:
        Function that tokenizes a chat example
    """

    def tokenize_example(example: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize a single chat example with label masking."""
        messages = example["messages"]

        # Apply chat template to get the full formatted conversation
        # This includes system, user, and assistant messages
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Get the conversation without the assistant reply
        # This is system + user only
        non_assistant = [m for m in messages if m["role"] != "assistant"]
        prefix_text = tokenizer.apply_chat_template(
            non_assistant,
            tokenize=False,
            add_generation_prompt=True,  # Adds the assistant prompt token(s)
        )

        # Tokenize full conversation
        full_encoding = tokenizer(
            full_text,
            max_length=cfg.max_seq_length,
            truncation=True,
            padding=False,  # We'll pad in the collator
        )

        # Tokenize prefix to find where assistant starts
        prefix_encoding = tokenizer(
            prefix_text,
            max_length=cfg.max_seq_length,
            truncation=True,
            padding=False,
        )

        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]

        # Create labels (copy of input_ids)
        labels = input_ids.copy()

        # Find prefix length more robustly
        # The assistant response starts after the prefix
        prefix_len = len(prefix_encoding["input_ids"])

        # Mask all tokens before the assistant response (system + user)
        # We want loss only on assistant tokens
        for i in range(min(prefix_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    return tokenize_example


# ---------- Model + LoRA setup ----------

def create_model_and_tokenizer(cfg: TrainConfig):
    """
    Create quantized model with LoRA adapters and tokenizer.

    Args:
        cfg: Training configuration

    Returns:
        Tuple of (model, tokenizer)
    """
    print("Setting up model and tokenizer...\n")

    # Configure 4-bit quantization (QLoRA)
    if cfg.use_4bit:
        compute_dtype = getattr(torch, cfg.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        print(f"4-bit quantization config:")
        print(f"  Compute dtype: {cfg.bnb_4bit_compute_dtype}")
        print(f"  Quant type: {cfg.bnb_4bit_quant_type}")
        print(f"  Double quant: {cfg.bnb_4bit_use_double_quant}\n")
    else:
        bnb_config = None

    # Load tokenizer
    print(f"Loading tokenizer: {cfg.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    # Set padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  Pad token: {tokenizer.pad_token}")
    print(f"  EOS token: {tokenizer.eos_token}\n")

    # Load model
    print(f"Loading model: {cfg.base_model_name}")
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "use_cache": False,  # Required for gradient checkpointing
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_name,
        **model_kwargs,
    )

    # Prepare model for k-bit training
    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Enable gradient checkpointing if configured
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

    # Configure LoRA
    print("\nConfiguring LoRA:")
    print(f"  Rank (r): {cfg.lora_r}")
    print(f"  Alpha: {cfg.lora_alpha}")
    print(f"  Dropout: {cfg.lora_dropout}")
    print(f"  Target modules: {cfg.lora_target_modules}")

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    print("\nTrainable parameters:")
    model.print_trainable_parameters()
    print()

    return model, tokenizer


# ---------- Evaluation Callback ----------

class NERMetricsCallback(TrainerCallback):
    """
    Custom callback to compute NER F1 metrics during training.

    Runs inference on a sample of dev set and computes strict span-based F1.
    """

    def __init__(self, eval_docs: List[Dict], tokenizer, num_samples: int = 20, entity_type: str = None):
        """
        Args:
            eval_docs: List of gold dev documents with 'text' and 'entities'
            tokenizer: Tokenizer for the model
            num_samples: Number of docs to evaluate (default 20 for speed)
            entity_type: If set, use entity-specific prompt (TOPONYM, PERSON, etc.)
        """
        self.eval_docs = eval_docs[:num_samples]  # Sample for speed
        self.tokenizer = tokenizer
        self.num_samples = min(num_samples, len(eval_docs))
        self.entity_type = entity_type

    def on_evaluate(self, args, state, control, model, **kwargs):
        """Run NER inference and compute F1 after each evaluation."""
        print("\n" + "=" * 70)
        print(f"Running NER evaluation on {self.num_samples} dev samples...")
        print("=" * 70)

        # Unwrap model if it's wrapped in DDP/FSDP/etc
        unwrapped_model = model
        if hasattr(model, 'module'):
            unwrapped_model = model.module

        # Ensure model is in eval mode
        was_training = unwrapped_model.training
        unwrapped_model.eval()

        # Run inference on sample docs
        pred_docs = []
        for i, doc in enumerate(self.eval_docs):
            try:
                # Run NER inference (shorter generation for speed during training)
                result = ner_inference_for_text(
                    doc["text"],
                    unwrapped_model,
                    self.tokenizer,
                    max_new_tokens=128,  # Shorter for speed during training
                    temperature=0.0,
                    entity_type=self.entity_type,  # Use entity-specific prompt if set
                )

                pred_docs.append({
                    "doc_id": doc.get("doc_id", f"dev_{i}"),
                    "text": doc["text"],
                    "entities": result["entities"],
                })
            except Exception as e:
                print(f"WARNING: Failed to process doc {i}: {e}")
                # Add empty prediction to keep counts aligned
                pred_docs.append({
                    "doc_id": doc.get("doc_id", f"dev_{i}"),
                    "text": doc["text"],
                    "entities": [],
                })

        # Restore training mode if it was on
        if was_training:
            unwrapped_model.train()

        # Compute metrics using Unique-Entity F1 (PRIMARY METRIC)
        scores = compute_unique_entity_f1(self.eval_docs, pred_docs)

        # Print results
        overall = scores["overall"]

        # Quick sanity check: show first prediction and RAW OUTPUT
        if len(pred_docs) > 0:
            first_pred = pred_docs[0]
            first_gold = self.eval_docs[0]

            # Run one more inference to get raw output for debugging
            try:
                debug_result = ner_inference_for_text(
                    first_gold["text"][:500],  # First 500 chars only
                    unwrapped_model,
                    self.tokenizer,
                    max_new_tokens=128,
                    temperature=0.0,
                    entity_type=self.entity_type,  # Use entity-specific prompt if set
                )
                print(f"\n=== DEBUG: Raw model output (first 500 chars of doc 0) ===")
                print(debug_result["raw_output"][:300])
                print("=== End raw output ===\n")
            except Exception as e:
                print(f"DEBUG: Failed to get raw output: {e}")

            print(f"Sample prediction (doc 0):")
            print(f"  Gold entities: {len(first_gold.get('entities', []))}")
            print(f"  Pred entities: {len(first_pred['entities'])}")
            if len(first_pred['entities']) > 0:
                print(f"  First pred: {first_pred['entities'][0]}")

        print(f"\nDev Set Metrics (n={self.num_samples}):")
        print(f"  Overall F1:        {overall['f1']:.4f}")
        print(f"  Overall Precision: {overall['precision']:.4f}")
        print(f"  Overall Recall:    {overall['recall']:.4f}")
        print(f"  TP: {overall['tp']}, FP: {overall['fp']}, FN: {overall['fn']}")

        # Print per-label scores
        print("\n  Per-Label F1:")
        for label in ["COMMODITY", "TOPONYM", "PERSON", "ORGANIZATION"]:
            if label in scores:
                f1 = scores[label]["f1"]
                print(f"    {label:15s}: {f1:.4f}")

        print("=" * 70 + "\n")

        # Log to wandb/tensorboard if available
        if state.is_world_process_zero:
            # These will be logged automatically by Trainer
            metrics = {
                "eval_ner_f1": overall["f1"],
                "eval_ner_precision": overall["precision"],
                "eval_ner_recall": overall["recall"],
            }
            # Add per-label F1
            for label in ["COMMODITY", "TOPONYM", "PERSON", "ORGANIZATION"]:
                if label in scores:
                    metrics[f"eval_ner_f1_{label.lower()}"] = scores[label]["f1"]

            # Log metrics manually
            for key, value in metrics.items():
                state.log_history.append({
                    "step": state.global_step,
                    key: value,
                })


# ---------- Main ----------

def main():
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train EarlyModernNER with QLoRA"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g., config/qwen3_4b_ner_lora.yaml)",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Run a quick test with limited steps",
    )
    parser.add_argument(
        "--entity-type",
        type=str,
        choices=list(ALLOWED_ENTITY_TYPES),
        help="Train on a single entity type only (for ensemble models)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix to add to output directory (e.g., '_commodity')",
    )
    args = parser.parse_args()

    # Print entity type mode
    if args.entity_type:
        print(f"\n*** SINGLE-ENTITY MODE: Training {args.entity_type} specialist ***\n")

    # Load config
    cfg = load_train_config(args.config)

    # Modify output directory if using entity-type or suffix
    if args.entity_type:
        suffix = args.output_suffix or f"_{args.entity_type.lower()}"
        cfg.output_dir = cfg.output_dir.rstrip("/") + suffix
    elif args.output_suffix:
        cfg.output_dir = cfg.output_dir.rstrip("/") + args.output_suffix

    # Create output directory
    os.makedirs(cfg.output_dir, exist_ok=True)

    # Save config to output dir
    config_save_path = os.path.join(cfg.output_dir, "train_config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(cfg.__dict__, f)
    print(f"Saved config to: {config_save_path}\n")

    # Set random seed
    torch.manual_seed(cfg.seed)

    # Enable TF32 if configured (Ampere GPUs and newer)
    if cfg.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        print("TF32 enabled for matmul\n")

    # Load chat datasets for training
    train_ds = load_chat_dataset(cfg.train_file)
    eval_ds = load_chat_dataset(cfg.eval_file)

    # Filter datasets by entity type if specified
    if args.entity_type:
        train_ds = filter_chat_dataset_by_entity_type(train_ds, args.entity_type)
        eval_ds = filter_chat_dataset_by_entity_type(eval_ds, args.entity_type)

    # Also load gold dev docs for NER evaluation
    print("Loading gold dev documents for NER evaluation...")
    gold_dev_path = cfg.eval_file.replace("_chat.jsonl", ".jsonl")
    gold_dev_docs = []
    with open(gold_dev_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gold_dev_docs.append(json.loads(line))
    print(f"  Loaded {len(gold_dev_docs)} gold dev documents")

    # Filter gold docs by entity type if specified
    if args.entity_type:
        for doc in gold_dev_docs:
            doc["entities"] = [
                e for e in doc.get("entities", [])
                if e.get("type") == args.entity_type
            ]
        # Keep only docs with at least one entity of the target type
        gold_dev_docs = [d for d in gold_dev_docs if d.get("entities")]
        print(f"  Filtered to {len(gold_dev_docs)} docs with {args.entity_type}")
    print()

    # Create model and tokenizer
    model, tokenizer = create_model_and_tokenizer(cfg)

    # Tokenize datasets
    print("Tokenizing datasets...")
    tokenize_fn = make_tokenize_fn(tokenizer, cfg)

    train_ds_tok = train_ds.map(
        tokenize_fn,
        remove_columns=train_ds.column_names,
        desc="Tokenizing train set",
    )

    eval_ds_tok = eval_ds.map(
        tokenize_fn,
        remove_columns=eval_ds.column_names,
        desc="Tokenizing eval set",
    )

    print(f"  Train samples: {len(train_ds_tok)}")
    print(f"  Eval samples: {len(eval_ds_tok)}\n")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_accumulation_steps=cfg.eval_accumulation_steps,
        num_train_epochs=cfg.num_train_epochs,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        max_grad_norm=cfg.max_grad_norm,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        eval_steps=cfg.eval_steps,
        save_total_limit=cfg.save_total_limit,
        eval_strategy=cfg.evaluation_strategy,
        do_eval=cfg.do_eval,
        report_to=cfg.report_to,
        bf16=True,   # Use bfloat16 for compute
        fp16=False,
        seed=cfg.seed,
        remove_unused_columns=False,
    )

    # Override for test run
    if args.test_run:
        print("Running in TEST MODE - limiting to 10 steps\n")
        training_args.max_steps = 10
        training_args.eval_steps = 5
        training_args.save_steps = 5

    # Data collator for dynamic padding
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        padding=True,
        max_length=cfg.max_seq_length,
    )

    # Create NER evaluation callback
    ner_callback = NERMetricsCallback(
        eval_docs=gold_dev_docs,
        tokenizer=tokenizer,
        num_samples=10,  # Evaluate on 10 docs during training (faster)
        entity_type=args.entity_type,  # Use entity-specific prompt if set
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_ds_tok,
        eval_dataset=eval_ds_tok,
        args=training_args,
        data_collator=data_collator,
        callbacks=[ner_callback],
    )

    # Train
    print("=" * 70)
    print("Starting training...")
    print("=" * 70)
    print()

    trainer.train()

    # Save final model
    print("\n" + "=" * 70)
    print("Training complete! Saving model...")
    print("=" * 70)

    trainer.save_model(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

    print(f"\n✓ Model saved to: {cfg.output_dir}")
    print("\nTo use this model:")
    print(f"  from peft import AutoPeftModelForCausalLM")
    print(f"  model = AutoPeftModelForCausalLM.from_pretrained('{cfg.output_dir}')")


if __name__ == "__main__":
    main()
