"""Evaluate ensemble model against the main gold standard.

This script:
1. Loads gold standard documents from Z:/NER/gold
2. Compares predictions using text-based matching
3. Reports precision, recall, F1 for all entity types
"""
import json
import sys
from pathlib import Path
from collections import Counter


GOLD_DIR = Path(__file__).parent / "eval_sets" / "gold_standard"
OUTPUT_DIR = Path("evaluation_results")
OUTPUT_DIR.mkdir(exist_ok=True)

ENTITY_TYPES = ["TOPONYM", "PERSON", "ORGANIZATION", "COMMODITY"]


def load_gold_standard():
    """Load all gold standard documents, keyed by doc_id (without extension)."""
    docs = {}
    for file_path in sorted(GOLD_DIR.glob("*.json")):
        with open(file_path, 'r', encoding='utf-8') as f:
            doc = json.load(f)
        # Key by stem (no extension) so it matches prediction doc_ids that may differ in extension
        key = Path(doc['doc_id']).stem if '.' in doc['doc_id'] else doc['doc_id']
        docs[key] = {
            'doc_id': doc['doc_id'],
            'text': doc['text'],
            'entities': doc['entities'],
            'file': file_path.name
        }
    return docs


def normalize_text(text):
    """Normalize entity text for comparison.

    Handles common variations in historical texts:
    - Case differences
    - Hyphen vs space (east-india vs east india)
    - Leading articles (the East India Company vs East India Company)
    - Multiple spaces
    """
    import re
    t = text.lower().strip()
    # Remove leading "the "
    if t.startswith("the "):
        t = t[4:]
    # Normalize hyphens to spaces
    t = t.replace("-", " ")
    # Collapse multiple spaces
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def evaluate_predictions(gold_docs, predictions, entity_type):
    """Evaluate predictions against gold using text-based matching.

    Uses partial matching: if a prediction is a substring of a gold entity
    (or vice versa), it counts as a TP rather than FP+FN.
    E.g., "Lords of Trade" predicted vs "Lords of Trade and Plantations" in gold = TP.

    gold_docs: dict keyed by doc_id stem -> gold doc
    predictions: list of prediction dicts (each with 'doc_id' field)
    """

    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    false_positives = Counter()
    false_negatives = Counter()
    true_positives = Counter()
    partial_matches = Counter()

    skipped = 0
    for pred_doc in predictions:
        # Match by doc_id (strip extension to normalize)
        pred_id = pred_doc['doc_id']
        key = Path(pred_id).stem if '.' in pred_id else pred_id
        gold_doc = gold_docs.get(key)
        if gold_doc is None:
            skipped += 1
            continue

        # Get gold entities of target type (gold uses 'label')
        gold_entities = set(
            normalize_text(e['text'])
            for e in gold_doc['entities']
            if e.get('label') == entity_type
        )

        # Get predicted entities of target type (predictions use 'type')
        pred_entities = set(
            normalize_text(e['text'])
            for e in pred_doc.get('entities', [])
            if e.get('type') == entity_type
        )

        # Step 1: Exact matches
        tp = gold_entities & pred_entities
        remaining_fp = pred_entities - gold_entities
        remaining_fn = gold_entities - pred_entities

        # Step 2: Partial matches (substring relationships)
        matched_fp = set()
        matched_fn = set()
        for pred_item in remaining_fp:
            for gold_item in remaining_fn:
                # Prediction is substring of gold, or gold is substring of prediction
                if pred_item in gold_item or gold_item in pred_item:
                    matched_fp.add(pred_item)
                    matched_fn.add(gold_item)
                    partial_matches[f"{pred_item} ~ {gold_item}"] += 1
                    break  # One match per prediction

        # Partial matches count as TP
        tp = tp | matched_fp  # Use predicted text as the TP key
        remaining_fp = remaining_fp - matched_fp
        remaining_fn = remaining_fn - matched_fn

        overall_tp += len(tp)
        overall_fp += len(remaining_fp)
        overall_fn += len(remaining_fn)

        for item in tp:
            true_positives[item] += 1
        for item in remaining_fp:
            false_positives[item] += 1
        for item in remaining_fn:
            false_negatives[item] += 1

    # Calculate metrics
    precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': overall_tp,
        'fp': overall_fp,
        'fn': overall_fn,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_positives': true_positives,
        'partial_matches': partial_matches,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate on gold standard")
    parser.add_argument("--predictions", required=True, help="Predictions JSONL file")
    parser.add_argument("--entity-type", default=None, help="Entity type to evaluate (default: all)")
    parser.add_argument("--show-errors", type=int, default=20, help="Number of errors to show per type")
    args = parser.parse_args()

    # Determine which entity types to evaluate
    if args.entity_type:
        entity_types = [args.entity_type.upper()]
    else:
        entity_types = ENTITY_TYPES

    print("=" * 70)
    print("EVALUATING ON GOLD STANDARD (100 documents)")
    print("=" * 70)

    # Load gold
    print(f"\nLoading gold standard from {GOLD_DIR}...")
    gold_docs = load_gold_standard()
    print(f"  Loaded {len(gold_docs)} documents")

    # Count gold entities by type
    print("\n  Gold entity counts:")
    for etype in ENTITY_TYPES:
        count = sum(
            sum(1 for e in doc['entities'] if e.get('label') == etype)
            for doc in gold_docs.values()
        )
        print(f"    {etype}: {count}")

    # Load predictions
    print(f"\nLoading predictions from {args.predictions}...")
    predictions = []
    with open(args.predictions, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    print(f"  Loaded {len(predictions)} predictions")

    # Check coverage (doc_id based, not positional)
    pred_keys = set(
        Path(d['doc_id']).stem if '.' in d['doc_id'] else d['doc_id']
        for d in predictions
    )
    gold_keys = set(gold_docs.keys())
    unmatched_preds = pred_keys - gold_keys
    unmatched_gold = gold_keys - pred_keys
    if unmatched_preds:
        print(f"\n  WARNING: {len(unmatched_preds)} predictions have no matching gold doc")
    if unmatched_gold:
        print(f"\n  WARNING: {len(unmatched_gold)} gold docs have no matching prediction")

    # Evaluate each entity type
    print("\n" + "=" * 70)
    print("RESULTS BY ENTITY TYPE")
    print("=" * 70)

    all_results = {}
    total_tp, total_fp, total_fn = 0, 0, 0

    for entity_type in entity_types:
        results = evaluate_predictions(gold_docs, predictions, entity_type)
        all_results[entity_type] = results

        total_tp += results['tp']
        total_fp += results['fp']
        total_fn += results['fn']

        print(f"\n--- {entity_type} ---")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1:        {results['f1']:.4f}")
        print(f"  TP: {results['tp']}, FP: {results['fp']}, FN: {results['fn']}")

        if results.get('partial_matches'):
            print(f"\n  Partial matches (counted as TP): {sum(results['partial_matches'].values())}")
            for match, count in results['partial_matches'].most_common(10):
                print(f"    {match}: {count}")

        if args.show_errors > 0 and (results['fp'] > 0 or results['fn'] > 0):
            if results['false_positives']:
                print(f"\n  Top {min(args.show_errors, len(results['false_positives']))} FALSE POSITIVES:")
                for item, count in results['false_positives'].most_common(args.show_errors):
                    print(f"    {item}: {count}")

            if results['false_negatives']:
                print(f"\n  Top {min(args.show_errors, len(results['false_negatives']))} FALSE NEGATIVES:")
                for item, count in results['false_negatives'].most_common(args.show_errors):
                    print(f"    {item}: {count}")

    # Overall summary
    if len(entity_types) > 1:
        overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0

        print("\n" + "=" * 70)
        print("OVERALL (ALL ENTITY TYPES)")
        print("=" * 70)
        print(f"\n  Precision: {overall_p:.4f}")
        print(f"  Recall:    {overall_r:.4f}")
        print(f"  F1:        {overall_f1:.4f}")
        print(f"  TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")

        # Comparison table
        print("\n" + "-" * 70)
        print("SUMMARY TABLE")
        print("-" * 70)
        print(f"{'Entity Type':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
        print("-" * 70)
        for entity_type in entity_types:
            r = all_results[entity_type]
            print(f"{entity_type:<15} {r['precision']:>10.4f} {r['recall']:>10.4f} {r['f1']:>10.4f} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6}")
        print("-" * 70)
        print(f"{'OVERALL':<15} {overall_p:>10.4f} {overall_r:>10.4f} {overall_f1:>10.4f} {total_tp:>6} {total_fp:>6} {total_fn:>6}")

    # Save results
    output_file = OUTPUT_DIR / "gold_standard_eval_results.json"
    save_data = {
        'entity_results': {
            etype: {
                'precision': r['precision'],
                'recall': r['recall'],
                'f1': r['f1'],
                'tp': r['tp'],
                'fp': r['fp'],
                'fn': r['fn']
            }
            for etype, r in all_results.items()
        }
    }
    if len(entity_types) > 1:
        save_data['overall'] = {
            'precision': overall_p,
            'recall': overall_r,
            'f1': overall_f1,
            'tp': total_tp,
            'fp': total_fp,
            'fn': total_fn
        }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)
    print(f"\n  Results saved to: {output_file}")


if __name__ == "__main__":
    main()
