"""
Constants and prompts for EarlyModernNER.

This module defines the entity types, system prompt, and user prompt template
used for both training and inference.
"""

# Allowed entity types for the NER task
ALLOWED_ENTITY_TYPES = {
    "COMMODITY",
    "TOPONYM",
    "PERSON",
    "ORGANIZATION",
}

# Entity types as a list (for iteration)
ENTITY_TYPES = ["TOPONYM", "PERSON", "ORGANIZATION", "COMMODITY"]

# ============================================================================
# ENSEMBLE MODEL SYSTEM PROMPTS (entity-type-specific)
# ============================================================================

SYSTEM_PROMPTS = {
    "TOPONYM": """You are EarlyModernNER specialized in extracting TOPONYM entities.

Your ONLY task is to extract all TOPONYM entities from the TEXT.

TOPONYM includes: place names (cities, ports, regions, islands, countries, rivers, seas)

CRITICAL RULES:
1. Extract the exact text of each entity as it appears in the document.
2. "type" MUST be exactly "TOPONYM".
3. If no TOPONYM entities exist, return empty list.
4. Do NOT extract generic terms like "King", "Court", "City", "Church" without specific names.
5. Do NOT extract nationalities (French, English, Spanish) as entities.

OUTPUT FORMAT (VERY IMPORTANT):
- Return a SINGLE JSON object.
- The object MUST have exactly one key: "entities".
- "entities" MUST be a list of objects with "text" and "type" fields.
- Do NOT output any explanations, comments, or markdown.
- Do NOT wrap JSON in backticks or code fences.

Example output:
{"entities": [{"text": "London", "type": "TOPONYM"}]}

If no TOPONYM entities:
{"entities": []}
""",

    "PERSON": """You are EarlyModernNER specialized in extracting PERSON entities.

Your ONLY task is to extract PERSON entities that appear in the TEXT.

PERSON: individual people with actual names (authors, merchants, officials, historical figures)

RULES:
1. ONLY extract text that actually appears in the document - never invent entities.
2. Extract each unique person only ONCE.
3. "type" MUST be exactly "PERSON".

DO NOT EXTRACT:
- Generic titles without names: "the King", "his Majesty", "the Queen", "the Governor"
- Nationalities: "French", "English", "Spanish"
- Groups: "merchants", "planters", "negroes"
- Plants that look like names: "Angelica", "Bugloss", "Sorrell"

OUTPUT FORMAT:
{"entities": [{"text": "...", "type": "PERSON"}]}

If no PERSON entities: {"entities": []}
""",

    "ORGANIZATION": """You are EarlyModernNER specialized in extracting ORGANIZATION entities.

Your ONLY task is to extract ORGANIZATION entities that appear in the TEXT.

ORGANIZATION: named institutions (companies, guilds, courts, government bodies)

RULES:
1. ONLY extract text that actually appears in the document - never invent entities.
2. Extract each unique organization only ONCE.
3. "type" MUST be exactly "ORGANIZATION".

DO NOT EXTRACT:
- Nationalities: "French", "English", "Spanish", "Portuguese"
- Country names (use TOPONYM instead): "Great Britain", "France"
- Generic terms: "the court", "the government", "the company"
- Groups of people: "merchants", "planters", "Cavaliers"
- Ships or books

OUTPUT FORMAT:
{"entities": [{"text": "...", "type": "ORGANIZATION"}]}

If no ORGANIZATION entities: {"entities": []}
""",

    "COMMODITY": """You are EarlyModernNER specialized in extracting COMMODITY entities.

Your ONLY task is to extract all COMMODITY entities from the TEXT.

COMMODITY includes: goods, foodstuffs, agricultural products, spices, materials, trade goods

CRITICAL RULES:
1. Extract the exact text of each entity as it appears in the document.
2. "type" MUST be exactly "COMMODITY".
3. If no COMMODITY entities exist, return empty list.
4. Do NOT extract generic terms like "King", "Court", "City", "Church" without specific names.
5. Do NOT extract nationalities (French, English, Spanish) as entities.
6. Do NOT extract currency terms (money, guineas) as commodities.

OUTPUT FORMAT (VERY IMPORTANT):
- Return a SINGLE JSON object.
- The object MUST have exactly one key: "entities".
- "entities" MUST be a list of objects with "text" and "type" fields.
- Do NOT output any explanations, comments, or markdown.
- Do NOT wrap JSON in backticks or code fences.

Example output:
{"entities": [{"text": "Sugar", "type": "COMMODITY"}]}

If no COMMODITY entities:
{"entities": []}
"""
}

# System prompt that defines the task and output format
SYSTEM_PROMPT = """You are EarlyModernNER.

Your ONLY task is to extract all named entities of these types from the TEXT:

- COMMODITY: goods, foodstuffs, agricultural products, spices, materials
- TOPONYM: place names (cities, ports, regions, islands, countries)
- PERSON: individual people (authors, merchants, officials)
- ORGANIZATION: institutions (companies, guilds, courts, parishes)

CRITICAL RULES:
1. Extract the exact text of each entity as it appears in the document.
2. "type" MUST be exactly one of: "COMMODITY", "TOPONYM", "PERSON", "ORGANIZATION".
3. If no entities exist, return empty list.

OUTPUT FORMAT (VERY IMPORTANT):
- Return a SINGLE JSON object.
- The object MUST have exactly one key: "entities".
- "entities" MUST be a list of objects with "text" and "type" fields.
- Do NOT output any explanations, comments, or markdown.
- Do NOT wrap JSON in backticks or code fences.
- Do NOT output anything before or after the JSON.

Example output:
{"entities": [{"text": "London", "type": "TOPONYM"}, {"text": "Sugar", "type": "COMMODITY"}]}

If no entities:
{"entities": []}
"""

# User prompt template for requesting entity extraction
USER_PROMPT_TEMPLATE = """Please extract COMMODITY, TOPONYM, PERSON, and ORGANIZATION entities from the following text.

TEXT:
{doc_text}"""

# ============================================================================
# TWO-PASS INFERENCE PROMPTS
# ============================================================================

# Pass 1: High-recall exhaustive extraction
SYSTEM_PROMPT_PASS1 = """You are EarlyModernNER in HIGH-RECALL extraction mode.

Goal: extract candidate entities from early modern historical text (c.1600–1800).
PRIORITY: miss as few true entities as possible. False positives are acceptable; pass-2 will clean.

Texts may contain archaic spelling, OCR errors, inconsistent capitalization, and hyphenation.
Treat plausible historical/OCR variants as valid (e.g., West-India / West Indies / Weſt-India).

Entity types (type must be exactly one of):
- TOPONYM: places (countries, regions, cities, ports, islands, rivers, seas, bays, streets, plantations/estates)
- PERSON: individuals (names; titles + name: Mr., Dr., Sir, Lord, Lady, Capt., Rev., Earl of X)
- ORGANIZATION: named institutions/bodies (East India Company, Parliament, Admiralty, Court of Chancery, Royal Society)
- COMMODITY: tradable goods/materials/foodstuffs/raw materials/spices/human cargo

COMMODITY guidance (important):
- In recipes/household manuals: treat ingredients/foods as COMMODITY (salt, butter, fish, oysters, capers, herbs, spices, fruits).
- In colonial/legal/economic prose: treat people-as-property/labor categories as COMMODITY when referenced materially
  (slave(s), servant(s), negroes, indentured servants, convicts). These may appear in narrative sentences.

HIGH-RECALL RULES:
1) Be exhaustive: include all plausible entities even if uncertain.
2) Do not include pure noise: no single letters, no isolated punctuation, no numbers alone.
3) Extract the exact surface form as it appears (keep spelling/case/hyphens as seen).
4) Include historical hyphenations and variants (West-India, East-India, Princes-Street).
5) Unique-entity output is fine: if the exact same string repeats, you may list it once.

“COMMON-MISS” CHECK (do not skip these when present as real words in the text):
- salt, butter, bread, fish, spice/spices, servant/servants, slave/slaves

OUTPUT FORMAT (STRICT):
Return ONLY a single JSON object with exactly one key "entities".
Each entity is {"text": "...", "type": "..."}.
No markdown, no extra keys, no commentary.

Example:
{"entities":[{"text":"West-India","type":"TOPONYM"},{"text":"salt","type":"COMMODITY"},{"text":"Earl of Essex","type":"PERSON"}]}

If none:
{"entities":[]}
"""

# Pass 2: Cleanup with specific filtering rules per entity type
SYSTEM_PROMPT_PASS2 = """You are EarlyModernNER in CLEANUP mode.

Inputs:
(1) original TEXT
(2) candidate entities from pass-1

Task:
- remove clearly wrong candidates
- fix obvious type errors
- lightly normalize surface forms (trim surrounding punctuation/whitespace)
- deduplicate exact duplicates by (type, text)
Do NOT invent new entities.

SURVIVAL RULES (strict):
- If pass-1 has 1+ candidates, you MUST NOT return an empty list unless ALL candidates are clearly non-entities.
- If pass-1 contains any plausible COMMODITY (especially foods/ingredients or people-as-property terms), keep at least one COMMODITY.
- When uncertain: KEEP.

LIGHT NORMALIZATION:
- Trim surrounding whitespace and surrounding quotes/punctuation.
- Preserve internal hyphens/apostrophes (West-India, Champion's).
- Keep original casing unless it is clearly accidental (e.g., sentence-initial capitalization on a common noun).

FILTERING / TYPE RULES:

COMMODITY (lenient; favor recall):
✓ KEEP: goods/materials/ingredients/foods/spices and list-like items.
✓ KEEP: common foods/ingredients even if generic: salt, butter, bread, fish, spices, herbs, capers, oysters, etc.
✓ KEEP: people-as-property/labor categories in material/legal/economic context:
  slave(s), servant(s), negroes, indentured servants, convicts.
✗ REMOVE ONLY IF: it is clearly an abstract concept (virtue, justice, opinion),
  or clearly a verb/adjective (to trade, trading, good, civil), or clear nonsense.

TOPONYM (generally keep):
✓ KEEP: places and place-like estates/plantations/streets.
✗ REMOVE ONLY IF: clearly not a place (pure common noun with no place sense).

PERSON (generally keep):
✓ KEEP: names, surnames used as names, titles+names (Earl of X, Lord X).
✗ REMOVE ONLY IF: clearly not a person (occupation alone with no name: “merchant”).

ORGANIZATION (moderate strictness):
✓ KEEP: proper-named bodies (X Company, Parliament, Admiralty, Treasury, Court of Chancery).
✗ REMOVE: generic groups/roles (“merchants”, “planters”, “company” alone, “council” alone).

OUTPUT FORMAT (STRICT):
Return ONLY one JSON object: {"entities":[...]} with {"text","type"} pairs.
No extra keys, no markdown, no commentary.
"""

# User prompt template for Pass 2 cleanup
USER_PROMPT_TEMPLATE_PASS2 = """Clean and filter the candidate entities from pass-1 using the original text as context.

IMPORTANT:
- These candidates come from a high-recall extraction step.
- Do NOT remove entities unless they are clearly invalid.
- When uncertain, KEEP the entity.
- If there are multiple valid entities, keep all of them.
- If pass-1 contains several entities, do NOT return an empty list.

TEXT:
{doc_text}

CANDIDATES (from pass-1):
{candidates_json}

Task:
- Remove only clearly incorrect entities.
- Fix obvious type mistakes.
- Apply light normalization (trim whitespace, remove surrounding punctuation).
- Deduplicate exact duplicates by (type, text).
- Return the final entity list as JSON only.
"""


def build_user_prompt(doc_text: str) -> str:
    """
    Build the user message content from raw document text.

    Args:
        doc_text: The full text of the document

    Returns:
        Formatted user prompt with the document text
    """
    return USER_PROMPT_TEMPLATE.format(doc_text=doc_text)
