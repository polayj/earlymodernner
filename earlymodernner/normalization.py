"""
Enhanced normalization for entity text matching.

Provides stronger normalization than simple lowercase + whitespace collapse.
Handles Unicode variants, punctuation, and optional transformations for
better entity matching in evaluation.
"""

import re
import unicodedata
from typing import Dict

# Abbreviation expansion map (optional feature)
ABBREVIATION_MAP: Dict[str, str] = {
    "st.": "saint",
    "st": "saint",
    # Add more as needed based on corpus analysis
}

# Commodity allowlist - always keep these when normalizing
# Seeded from common false negatives
COMMODITY_ALLOWLIST = {
    "salt", "bread", "fish", "butter", "spice", "spices",
    "pepper", "slave", "slaves", "sugar", "rice", "flour",
    "eggs", "mace", "cotton", "indigo", "tea", "coffee",
    "cocoa", "rum", "brandy", "tobacco", "silk", "wine",
    "timber", "cloth", "wool", "iron", "gold", "silver",
    "coal", "tin", "lead", "wheat", "grain",
}

# Simple singularization rules for commodities
COMMODITY_PLURAL_MAP: Dict[str, str] = {
    "slaves": "slave",
    "spices": "spice",
    "eggs": "egg",
    "grains": "grain",
    "wines": "wine",
    "cloths": "cloth",
    "timbers": "timber",
    "silks": "silk",
    "tobaccos": "tobacco",
}


def normalize_entity_text(
    text: str,
    remove_hyphens: bool = False,
    remove_possessive: bool = False,
    expand_abbreviations: bool = False,
    commodity_singularize: bool = False,
    person_normalize: bool = False,
) -> str:
    """
    Enhanced normalization with required + optional rules.

    Required rules (always applied):
    1. Unicode NFKC normalization (handles ligatures, special chars)
    2. Lowercase
    3. Strip leading/trailing whitespace
    4. Collapse internal whitespace to single spaces
    5. Normalize apostrophes/quotes to '
    6. Normalize hyphens/dashes to -
    7. Strip surrounding punctuation (keep internal - and ')

    Optional rules (flag-controlled):
    - remove_hyphens: Replace hyphens with spaces
    - remove_possessive: Remove possessive 's
    - expand_abbreviations: Expand common abbreviations (st. → saint)
    - commodity_singularize: Apply singularization to commodity plurals
    - person_normalize: Strip leading "the" from person names

    Args:
        text: Entity text to normalize
        remove_hyphens: Replace hyphens with spaces (default: False)
        remove_possessive: Remove possessive 's (default: False)
        expand_abbreviations: Expand abbreviations from ABBREVIATION_MAP (default: False)
        commodity_singularize: Singularize commodity plurals (default: False)
        person_normalize: Normalize person names (strip leading "the") (default: False)

    Returns:
        Normalized text

    Examples:
        >>> normalize_entity_text("St. Paul's")
        'saint pauls'
        >>> normalize_entity_text("Princes-Street")
        'princes-street'
        >>> normalize_entity_text("  Jamaica  ")
        'jamaica'
        >>> normalize_entity_text("Barbadoes.")
        'barbadoes'
        >>> normalize_entity_text("slaves", commodity_singularize=True)
        'slave'
    """
    if not text:
        return ""

    # Apply required normalization
    normalized = _apply_required_normalization(text)

    # Apply optional normalization
    if any([remove_hyphens, remove_possessive, expand_abbreviations,
            commodity_singularize, person_normalize]):
        normalized = _apply_optional_normalization(
            normalized,
            remove_hyphens=remove_hyphens,
            remove_possessive=remove_possessive,
            expand_abbreviations=expand_abbreviations,
            commodity_singularize=commodity_singularize,
            person_normalize=person_normalize,
        )

    return normalized


def _apply_required_normalization(text: str) -> str:
    """
    Apply all required normalization rules.

    Args:
        text: Input text

    Returns:
        Normalized text with required rules applied
    """
    # 1. Unicode NFKC normalization (handles ligatures, special characters)
    text = unicodedata.normalize("NFKC", text)

    # 2. Lowercase
    text = text.lower()

    # 3 & 4. Strip whitespace and collapse internal whitespace
    text = text.strip()
    text = " ".join(text.split())

    # 5. Normalize apostrophes and quotes to standard '
    # Handle: ' (U+2018), ' (U+2019), " (U+201C), " (U+201D), ` (U+0060)
    apostrophe_variants = ["\u2018", "\u2019", "\u201c", "\u201d", "\u0060", '"']
    for variant in apostrophe_variants:
        text = text.replace(variant, "'")

    # 6. Normalize hyphens and dashes to standard -
    # Handle: – (en dash), — (em dash), ‐ (hyphen U+2010)
    hyphen_variants = ["\u2013", "\u2014", "\u2010"]
    for variant in hyphen_variants:
        text = text.replace(variant, "-")

    # 7. Strip surrounding punctuation (but keep internal - and ')
    text = _strip_surrounding_punctuation(text)

    return text


def _apply_optional_normalization(
    text: str,
    remove_hyphens: bool = False,
    remove_possessive: bool = False,
    expand_abbreviations: bool = False,
    commodity_singularize: bool = False,
    person_normalize: bool = False,
) -> str:
    """
    Apply optional normalization rules.

    Args:
        text: Input text (already has required normalization applied)
        remove_hyphens: Replace hyphens with spaces
        remove_possessive: Remove possessive 's
        expand_abbreviations: Expand abbreviations
        commodity_singularize: Singularize commodity plurals
        person_normalize: Normalize person names (strip leading "the")

    Returns:
        Text with optional rules applied
    """
    # Person normalization - strip leading "the" (do this early)
    if person_normalize:
        if text.startswith("the "):
            text = text[4:].strip()

    # Expand abbreviations (do this before other transformations)
    if expand_abbreviations:
        words = text.split()
        expanded_words = []
        for word in words:
            # Check with and without trailing period
            base_word = word.rstrip(".")
            if word in ABBREVIATION_MAP:
                expanded_words.append(ABBREVIATION_MAP[word])
            elif base_word in ABBREVIATION_MAP:
                expanded_words.append(ABBREVIATION_MAP[base_word])
            else:
                expanded_words.append(word)
        text = " ".join(expanded_words)

    # Commodity singularization
    if commodity_singularize:
        if text in COMMODITY_PLURAL_MAP:
            text = COMMODITY_PLURAL_MAP[text]

    # Remove possessive 's
    if remove_possessive:
        text = re.sub(r"'s\b", "", text)
        text = " ".join(text.split())  # Clean up any double spaces

    # Replace hyphens with spaces
    if remove_hyphens:
        text = text.replace("-", " ")
        text = " ".join(text.split())  # Collapse any double spaces

    return text


def _strip_surrounding_punctuation(text: str) -> str:
    """
    Strip punctuation from start and end of text, but preserve internal punctuation.

    Keeps internal hyphens and apostrophes (e.g., "saint-paul's" → "saint-paul's")
    Removes surrounding punctuation (e.g., "jamaica." → "jamaica")

    Args:
        text: Input text

    Returns:
        Text with surrounding punctuation removed
    """
    if not text:
        return text

    # Define punctuation to strip (excluding - and ' which we keep internally)
    # This includes periods, commas, semicolons, colons, quotes, etc.
    surrounding_punct = r""".,;:!?¡¿\[\]{}()<>«»"""

    # Strip from start
    while text and text[0] in surrounding_punct:
        text = text[1:]

    # Strip from end
    while text and text[-1] in surrounding_punct:
        text = text[:-1]

    return text.strip()
