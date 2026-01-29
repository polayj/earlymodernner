"""
Metrics for evaluating NER performance.

PRIMARY METRIC: Unique-Entity F1 based on (normalized_text, type) matching.
SECONDARY METRIC: Strict span-based evaluation where (start, end, type) must match exactly.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Set

from earlymodernner.normalization import normalize_entity_text


Span = Tuple[int, int, str]  # (start, end, type)
EntityPair = Tuple[str, str]  # (normalized_text, type)


def normalize_text(text: str) -> str:
    """
    Normalize entity text for fuzzy matching.

    Uses enhanced normalization with required rules + commodity/person upgrades.

    Args:
        text: Raw entity text

    Returns:
        Normalized text (Unicode NFKC, lowercase, whitespace collapsed, etc.)
    """
    return normalize_entity_text(
        text,
        commodity_singularize=True,  # Normalize commodity plurals (slaves → slave)
        person_normalize=True,       # Strip leading "the" from person names
    )


def _entities_to_unique_pairs(entities: List[Dict]) -> Set[EntityPair]:
    """
    Convert entity list to set of unique (normalized_text, type) pairs.

    Prefers text_normalized field if present (from new inference pipeline),
    otherwise normalizes the text field on the fly (backward compatibility).

    Args:
        entities: List of entity dicts with 'text' and 'type' fields
                 (optionally 'text_normalized' for new format)

    Returns:
        Set of (normalized_text, type) tuples
    """
    pairs = set()
    for e in entities:
        if "text" in e and "type" in e:
            # Prefer text_normalized if present (from new inference pipeline)
            norm_text = e.get("text_normalized")
            if not norm_text:
                # Fall back to normalizing text field (backward compatibility)
                norm_text = normalize_text(e["text"])
            if norm_text:  # Skip empty strings
                pairs.add((norm_text, e["type"]))
    return pairs


def compute_unique_entity_f1(
    gold_docs: List[Dict],
    pred_docs: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """
    Compute Unique-Entity F1 metrics (PRIMARY EVALUATION METRIC).

    Entities are matched based on (normalized_text, type) pairs.
    Duplicates within a document are counted only once.

    Args:
        gold_docs: List of gold documents with 'entities'
        pred_docs: List of predicted documents with 'entities'
                   Must be aligned 1:1 with gold_docs

    Returns:
        Dict with:
            - 'overall': {precision, recall, f1, tp, fp, fn}
            - per-label entries: e.g. 'COMMODITY': {precision, recall, f1, ...}
    """
    assert len(gold_docs) == len(pred_docs), (
        f"Gold and predicted docs must have same length: "
        f"gold={len(gold_docs)}, pred={len(pred_docs)}"
    )

    # Overall counts
    overall_tp = overall_fp = overall_fn = 0

    # Per-label counts
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # Process each document
    for g_doc, p_doc in zip(gold_docs, pred_docs):
        gold_pairs = _entities_to_unique_pairs(g_doc.get("entities", []))
        pred_pairs = _entities_to_unique_pairs(p_doc.get("entities", []))

        # Compute TP, FP, FN
        tp_pairs = gold_pairs & pred_pairs
        fp_pairs = pred_pairs - gold_pairs
        fn_pairs = gold_pairs - pred_pairs

        overall_tp += len(tp_pairs)
        overall_fp += len(fp_pairs)
        overall_fn += len(fn_pairs)

        # Update per-label counts
        for (text, label) in tp_pairs:
            per_label[label]["tp"] += 1

        for (text, label) in fp_pairs:
            per_label[label]["fp"] += 1

        for (text, label) in fn_pairs:
            per_label[label]["fn"] += 1

    # Build result dict
    results: Dict[str, Dict[str, float]] = {}
    results["overall"] = _precision_recall_f1(overall_tp, overall_fp, overall_fn)

    # Compute per-label scores
    for label, counts in per_label.items():
        scores = _precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
        results[label] = scores

    return results


def _doc_spans_from_entities(entities: List[Dict]) -> List[Span]:
    """
    Convert entity list to set of (start, end, type) tuples.

    Args:
        entities: List of entity dicts with start, end, type

    Returns:
        List of (start, end, type) tuples
    """
    spans = []
    for e in entities:
        start = int(e["start"])
        end = int(e["end"])
        etype = e["type"]
        spans.append((start, end, etype))
    return spans


def _precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Compute precision, recall, and F1 from counts.

    Args:
        tp: True positives
        fp: False positives
        fn: False negatives

    Returns:
        Dict with precision, recall, f1
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def compute_strict_span_scores(
    gold_docs: List[Dict],
    pred_docs: List[Dict],
) -> Dict[str, Dict[str, float]]:
    """
    Compute strict span-based evaluation metrics.

    A prediction is correct if (start, end, type) matches exactly.

    Args:
        gold_docs: List of gold documents with 'entities'
        pred_docs: List of predicted documents with 'entities'
                   Must be aligned 1:1 with gold_docs

    Returns:
        Dict with:
            - 'overall': {precision, recall, f1, tp, fp, fn}
            - per-label entries: e.g. 'COMMODITY': {precision, recall, f1, ...}
    """
    assert len(gold_docs) == len(pred_docs), (
        f"Gold and predicted docs must have same length: "
        f"gold={len(gold_docs)}, pred={len(pred_docs)}"
    )

    # Overall counts
    overall_tp = overall_fp = overall_fn = 0

    # Per-label counts
    per_label = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    # Process each document
    for g_doc, p_doc in zip(gold_docs, pred_docs):
        gold_spans = _doc_spans_from_entities(g_doc.get("entities", []))
        pred_spans = _doc_spans_from_entities(p_doc.get("entities", []))

        gold_set = set(gold_spans)
        pred_set = set(pred_spans)

        # Compute TP, FP, FN
        tp_spans = gold_set & pred_set
        fp_spans = pred_set - gold_set
        fn_spans = gold_set - pred_set

        overall_tp += len(tp_spans)
        overall_fp += len(fp_spans)
        overall_fn += len(fn_spans)

        # Update per-label counts
        for (s, e, t) in tp_spans:
            per_label[t]["tp"] += 1

        for (s, e, t) in fp_spans:
            per_label[t]["fp"] += 1

        for (s, e, t) in fn_spans:
            per_label[t]["fn"] += 1

    # Build result dict
    results: Dict[str, Dict[str, float]] = {}
    results["overall"] = _precision_recall_f1(overall_tp, overall_fp, overall_fn)

    # Compute per-label scores
    for label, counts in per_label.items():
        scores = _precision_recall_f1(counts["tp"], counts["fp"], counts["fn"])
        results[label] = scores

    return results


def print_unique_entity_report(scores: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted report for Unique-Entity F1 (PRIMARY METRIC).

    Args:
        scores: Output from compute_unique_entity_f1
    """
    print("\n" + "=" * 80)
    print("UNIQUE-ENTITY F1 EVALUATION (PRIMARY METRIC)")
    print("=" * 80)

    # Overall scores
    overall = scores["overall"]
    print(f"\nOverall Performance:")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  F1:        {overall['f1']:.4f}")
    print(f"  TP:        {overall['tp']}")
    print(f"  FP:        {overall['fp']}")
    print(f"  FN:        {overall['fn']}")

    # Per-label scores
    print(f"\nPer-Label Performance:")
    print(f"  {'Label':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("  " + "-" * 75)

    # Sort labels alphabetically
    labels = sorted([k for k in scores.keys() if k != "overall"])

    for label in labels:
        vals = scores[label]
        print(
            f"  {label:<15} "
            f"{vals['precision']:>10.4f} "
            f"{vals['recall']:>10.4f} "
            f"{vals['f1']:>10.4f} "
            f"{vals['tp']:>6} "
            f"{vals['fp']:>6} "
            f"{vals['fn']:>6}"
        )

    print("\n" + "=" * 80)


def print_evaluation_report(scores: Dict[str, Dict[str, float]]) -> None:
    """
    Print a formatted evaluation report for strict span matching (SECONDARY METRIC).

    Args:
        scores: Output from compute_strict_span_scores
    """
    print("\n" + "=" * 80)
    print("STRICT SPAN EVALUATION (SECONDARY/DIAGNOSTIC METRIC)")
    print("=" * 80)

    # Overall scores
    overall = scores["overall"]
    print(f"\nOverall Performance:")
    print(f"  Precision: {overall['precision']:.4f}")
    print(f"  Recall:    {overall['recall']:.4f}")
    print(f"  F1:        {overall['f1']:.4f}")
    print(f"  TP:        {overall['tp']}")
    print(f"  FP:        {overall['fp']}")
    print(f"  FN:        {overall['fn']}")

    # Per-label scores
    print(f"\nPer-Label Performance:")
    print(f"  {'Label':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("  " + "-" * 75)

    # Sort labels alphabetically
    labels = sorted([k for k in scores.keys() if k != "overall"])

    for label in labels:
        vals = scores[label]
        print(
            f"  {label:<15} "
            f"{vals['precision']:>10.4f} "
            f"{vals['recall']:>10.4f} "
            f"{vals['f1']:>10.4f} "
            f"{vals['tp']:>6} "
            f"{vals['fp']:>6} "
            f"{vals['fn']:>6}"
        )

    print("\n" + "=" * 80)


def compute_per_doc_scores(
    gold_docs: List[Dict],
    pred_docs: List[Dict],
) -> List[Dict[str, any]]:
    """
    Compute per-document metrics using unique entity pairs.

    Useful for identifying worst-performing documents.

    Args:
        gold_docs: List of gold documents
        pred_docs: List of predicted documents

    Returns:
        List of dicts with doc_id and per-doc metrics
    """
    assert len(gold_docs) == len(pred_docs)

    per_doc_results = []

    for g_doc, p_doc in zip(gold_docs, pred_docs):
        gold_pairs = _entities_to_unique_pairs(g_doc.get("entities", []))
        pred_pairs = _entities_to_unique_pairs(p_doc.get("entities", []))

        tp = len(gold_pairs & pred_pairs)
        fp = len(pred_pairs - gold_pairs)
        fn = len(gold_pairs - pred_pairs)

        scores = _precision_recall_f1(tp, fp, fn)
        scores["doc_id"] = g_doc.get("doc_id", "UNKNOWN")
        scores["num_gold_entities"] = len(gold_pairs)
        scores["num_pred_entities"] = len(pred_pairs)

        per_doc_results.append(scores)

    return per_doc_results


def find_worst_documents(
    per_doc_scores: List[Dict],
    n: int = 10,
    metric: str = "f1"
) -> List[Dict]:
    """
    Find the worst-performing documents.

    Args:
        per_doc_scores: Output from compute_per_doc_scores
        n: Number of worst documents to return
        metric: Metric to sort by ('f1', 'precision', or 'recall')

    Returns:
        List of worst n documents, sorted by metric (ascending)
    """
    sorted_docs = sorted(per_doc_scores, key=lambda x: x[metric])
    return sorted_docs[:n]
