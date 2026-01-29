"""Prepare entity-specific training data for ensemble models.

This script:
1. Merges HIPE-2022 data with existing gold data
2. Creates entity-specific training splits (one per entity type)
3. Generates 4 training configs for specialized models:
   - TOPONYM model
   - PERSON model
   - ORGANIZATION model
   - COMMODITY model

For each entity-type-specific model, we:
- Keep ALL documents but only include entities of that type
- This teaches the model to extract that specific entity type while
  correctly outputting empty lists for documents without that type
"""
import json
import random
from pathlib import Path
from collections import defaultdict
import shutil

# Paths
GOLD_TRAINING_DIR = Path("data/training")
HIPE_CONVERTED_DIR = Path("data/hipe2022/converted")
OUTPUT_DIR = Path("data/ensemble_training")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_OUTPUT_DIR = Path("earlymodernner/config")

# Entity types we train for
ENTITY_TYPES = ["TOPONYM", "PERSON", "ORGANIZATION", "COMMODITY"]

# System prompt for entity-specific models
def get_system_prompt(entity_type):
    """Get system prompt for single-entity-type extraction."""

    # Entity-specific prompts with precision-focused rules
    if entity_type == "PERSON":
        return """You are EarlyModernNER specialized in extracting PERSON entities.

Your ONLY task is to extract all PERSON entities from the TEXT.

PERSON includes: individual people (authors, merchants, officials, historical figures)

CRITICAL RULES:
1. Extract the exact text of each entity as it appears in the document.
2. "type" MUST be exactly "PERSON".
3. If no PERSON entities exist, return empty list.
4. Do NOT extract generic titles alone: "King", "Queen", "Captain", "Mr.", "Sir" without a name.
5. Do NOT extract nationalities (French, English, Spanish) as entities.
6. Do NOT extract place names as PERSON - places like Virginia, Carolina, Georgia are NOT people.
7. Do NOT extract organization names as PERSON - "East India Company", "Parliament" are NOT people.
8. Do NOT extract ship names as PERSON.
9. Only extract when you are CONFIDENT it refers to an individual human being.
10. When uncertain, do NOT extract - prefer precision over recall.

OUTPUT FORMAT (VERY IMPORTANT):
- Return a SINGLE JSON object.
- The object MUST have exactly one key: "entities".
- "entities" MUST be a list of objects with "text" and "type" fields.
- Do NOT output any explanations, comments, or markdown.
- Do NOT wrap JSON in backticks or code fences.

Example output:
{"entities": [{"text": "John Smith", "type": "PERSON"}]}

If no PERSON entities:
{"entities": []}
"""

    type_descriptions = {
        "TOPONYM": "place names (cities, ports, regions, islands, countries, rivers, seas)",
        "ORGANIZATION": "institutions (companies, guilds, courts, parishes, governments)",
        "COMMODITY": "goods, foodstuffs, agricultural products, spices, materials, trade goods"
    }

    desc = type_descriptions.get(entity_type, entity_type.lower())

    extra_rules = ""
    if entity_type == "COMMODITY":
        extra_rules = "\n6. Do NOT extract currency terms (money, guineas) as commodities."

    return f"""You are EarlyModernNER specialized in extracting {entity_type} entities.

Your ONLY task is to extract all {entity_type} entities from the TEXT.

{entity_type} includes: {desc}

CRITICAL RULES:
1. Extract the exact text of each entity as it appears in the document.
2. "type" MUST be exactly "{entity_type}".
3. If no {entity_type} entities exist, return empty list.
4. Do NOT extract generic terms like "King", "Court", "City", "Church" without specific names.
5. Do NOT extract nationalities (French, English, Spanish) as entities.{extra_rules}

OUTPUT FORMAT (VERY IMPORTANT):
- Return a SINGLE JSON object.
- The object MUST have exactly one key: "entities".
- "entities" MUST be a list of objects with "text" and "type" fields.
- Do NOT output any explanations, comments, or markdown.
- Do NOT wrap JSON in backticks or code fences.

Example output:
{{"entities": [{{"text": "London", "type": "{entity_type}"}}]}}

If no {entity_type} entities:
{{"entities": []}}
"""


def load_existing_gold():
    """Load existing gold training data in chat format."""
    docs = []
    train_file = GOLD_TRAINING_DIR / "train_chat.jsonl"
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))
    return docs


def load_hipe_data():
    """Load converted HIPE-2022 data."""
    docs = []
    hipe_file = HIPE_CONVERTED_DIR / "hipe2022_english_train.jsonl"
    if hipe_file.exists():
        with open(hipe_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))

    # Also load dev for more data
    hipe_dev = HIPE_CONVERTED_DIR / "hipe2022_english_dev.jsonl"
    if hipe_dev.exists():
        with open(hipe_dev, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    docs.append(json.loads(line))

    return docs


def extract_from_chat(chat_doc):
    """Extract text and entities from a chat-format document."""
    messages = chat_doc.get('messages', [])

    text = None
    entities = []

    for msg in messages:
        if msg['role'] == 'user':
            # Extract text from user message
            content = msg['content']
            if 'TEXT:' in content:
                text = content.split('TEXT:')[-1].strip()
        elif msg['role'] == 'assistant':
            # Parse entities from assistant response
            try:
                response = json.loads(msg['content'])
                entities = response.get('entities', [])
            except json.JSONDecodeError:
                pass

    return text, entities


def create_entity_specific_example(text, entities, entity_type):
    """Create a chat-format example for a specific entity type."""

    # Filter entities to only include the target type
    filtered_entities = [e for e in entities if e.get('type') == entity_type]

    system_prompt = get_system_prompt(entity_type)

    user_content = f"Please extract {entity_type} entities from the following text.\n\nTEXT:\n{text}"

    assistant_content = json.dumps({"entities": filtered_entities}, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }


def main():
    print("=" * 60)
    print("PREPARING ENSEMBLE TRAINING DATA")
    print("=" * 60)

    # Load existing gold data
    print("\nLoading existing gold training data...")
    gold_chat = load_existing_gold()
    print(f"  Gold chat examples: {len(gold_chat)}")

    # Load HIPE data
    print("\nLoading HIPE-2022 data...")
    hipe_docs = load_hipe_data()
    print(f"  HIPE documents: {len(hipe_docs)}")

    # Extract text/entities from gold chat format
    print("\nExtracting from chat format...")
    all_docs = []

    for chat in gold_chat:
        text, entities = extract_from_chat(chat)
        if text:
            all_docs.append({
                'text': text,
                'entities': entities,
                'source': 'gold'
            })

    # Add HIPE docs
    for doc in hipe_docs:
        all_docs.append({
            'text': doc.get('text', ''),
            'entities': doc.get('entities', []),
            'source': 'hipe'
        })

    print(f"  Total documents: {len(all_docs)}")

    # Count entities by type
    entity_counts = defaultdict(int)
    docs_with_type = defaultdict(int)

    for doc in all_docs:
        doc_types = set()
        for e in doc.get('entities', []):
            etype = e.get('type')
            if etype:
                entity_counts[etype] += 1
                doc_types.add(etype)
        for t in doc_types:
            docs_with_type[t] += 1

    print("\nEntity counts in combined data:")
    for etype in ENTITY_TYPES:
        print(f"  {etype}: {entity_counts[etype]:,} entities in {docs_with_type[etype]:,} docs")

    # Shuffle
    random.seed(42)
    random.shuffle(all_docs)

    # Split: 90% train, 5% dev, 5% test
    n = len(all_docs)
    train_end = int(n * 0.90)
    dev_end = int(n * 0.95)

    train_docs = all_docs[:train_end]
    dev_docs = all_docs[train_end:dev_end]
    test_docs = all_docs[dev_end:]

    print(f"\nSplits:")
    print(f"  Train: {len(train_docs)}")
    print(f"  Dev: {len(dev_docs)}")
    print(f"  Test: {len(test_docs)}")

    # Create entity-specific training data
    print("\nCreating entity-specific training data...")

    for entity_type in ENTITY_TYPES:
        type_dir = OUTPUT_DIR / entity_type.lower()
        type_dir.mkdir(exist_ok=True)

        type_train = []
        type_dev = []
        type_test = []

        # Process train
        for doc in train_docs:
            example = create_entity_specific_example(
                doc['text'],
                doc['entities'],
                entity_type
            )
            type_train.append(example)

        # Process dev
        for doc in dev_docs:
            example = create_entity_specific_example(
                doc['text'],
                doc['entities'],
                entity_type
            )
            type_dev.append(example)

        # Process test
        for doc in test_docs:
            example = create_entity_specific_example(
                doc['text'],
                doc['entities'],
                entity_type
            )
            type_test.append(example)

        # Count positive examples (docs with at least one entity of this type)
        train_positive = sum(1 for doc in train_docs if any(e.get('type') == entity_type for e in doc.get('entities', [])))

        # Write files
        with open(type_dir / "train_chat.jsonl", 'w', encoding='utf-8') as f:
            for ex in type_train:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        with open(type_dir / "dev_chat.jsonl", 'w', encoding='utf-8') as f:
            for ex in type_dev:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        with open(type_dir / "test_chat.jsonl", 'w', encoding='utf-8') as f:
            for ex in type_test:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')

        print(f"  {entity_type}:")
        print(f"    Train: {len(type_train)} ({train_positive} with entities)")
        print(f"    Dev: {len(type_dev)}")
        print(f"    Test: {len(type_test)}")

    # Create training configs for each entity type
    print("\nCreating training configs...")

    base_config = """# =========================================================================
# EarlyModernNER Ensemble Training - {entity_type} Specialist
# Model: Qwen3-4B-Instruct with QLoRA
# =========================================================================

# ========= Model & data =========
base_model_name: "Qwen/Qwen3-4B-Instruct-2507"
train_file: "data/ensemble_training/{entity_lower}/train_chat.jsonl"
eval_file: "data/ensemble_training/{entity_lower}/dev_chat.jsonl"
output_dir: "outputs/ensemble/{entity_lower}_lora"

# ========= Sequence / batching =========
max_seq_length: 2048
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
gradient_accumulation_steps: 4      # 2 * 4 = effective batch 8
eval_accumulation_steps: 1

# ========= Training schedule =========
num_train_epochs: 6
learning_rate: 2.0e-4
lr_scheduler_type: "cosine"
warmup_ratio: 0.03
weight_decay: 0.01
max_grad_norm: 1.0

# ========= Precision & memory (QLoRA) =========
use_4bit: true
bnb_4bit_compute_dtype: "bfloat16"
bnb_4bit_quant_type: "nf4"
bnb_4bit_use_double_quant: true

gradient_checkpointing: true
tf32: true

# ========= LoRA config =========
lora_r: 64
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - "q_proj"
  - "k_proj"
  - "v_proj"
  - "o_proj"
  - "gate_proj"
  - "up_proj"
  - "down_proj"

# ========= Logging & eval =========
logging_steps: 10
save_steps: 100
eval_steps: 100
save_total_limit: 3
evaluation_strategy: "steps"
do_eval: true

# ========= Misc =========
seed: 42
report_to:
  - "none"
"""

    for entity_type in ENTITY_TYPES:
        config_content = base_config.format(
            entity_type=entity_type,
            entity_lower=entity_type.lower()
        )

        config_file = CONFIG_OUTPUT_DIR / f"ensemble_{entity_type.lower()}.yaml"
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)

        print(f"  Created: {config_file.name}")

    # Save combined test set for ensemble evaluation
    combined_test = []
    for doc in test_docs:
        combined_test.append({
            'text': doc['text'],
            'entities': doc['entities'],
            'source': doc['source']
        })

    with open(OUTPUT_DIR / "test_combined.jsonl", 'w', encoding='utf-8') as f:
        for doc in combined_test:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')

    print(f"\nSaved combined test set: {len(combined_test)} docs")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train each specialized model:")
    for entity_type in ENTITY_TYPES:
        print(f"   python train_lora.py --config earlymodernner/config/ensemble_{entity_type.lower()}.yaml")
    print("\n2. Run ensemble inference with ensemble_inference.py (to be created)")


if __name__ == "__main__":
    main()
