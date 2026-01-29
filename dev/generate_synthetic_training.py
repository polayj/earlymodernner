"""Generate synthetic training examples to improve NER precision and recall.

Creates two types of synthetic training data:
1. HARD NEGATIVES: Short passages containing blocklisted terms where the correct
   output is empty entities. Teaches the model to NOT tag these terms.
2. HARD POSITIVES: Passages with properly tagged entities that the model
   currently misses (titled persons, specific named organizations).

Usage:
    python generate_synthetic_training.py

Output:
    data/ensemble_training_filtered/{entity_type}/synthetic_negatives.jsonl
    data/ensemble_training_filtered/{entity_type}/synthetic_positives.jsonl
    data/ensemble_training_filtered/{entity_type}/train_chat_augmented.jsonl  (combined)
"""
import json
import random
from pathlib import Path

BLOCKLIST_DIR = Path("data/blocklists")
FILTERED_DIR = Path("data/ensemble_training_filtered")

random.seed(42)

# ============================================================
# SYSTEM PROMPTS (matching training data format)
# ============================================================

SYSTEM_PROMPTS = {
    "PERSON": """You are EarlyModernNER specialized in extracting PERSON entities.

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
""",

    "ORGANIZATION": """You are EarlyModernNER specialized in extracting ORGANIZATION entities.

Your ONLY task is to extract all ORGANIZATION entities from the TEXT.

ORGANIZATION includes: institutions (companies, guilds, courts, parishes, governments)

CRITICAL RULES:
1. Extract the exact text of each entity as it appears in the document.
2. "type" MUST be exactly "ORGANIZATION".
3. If no ORGANIZATION entities exist, return empty list.
4. Do NOT extract generic terms like "King", "Court", "City", "Church" without specific names.
5. Do NOT extract nationalities (French, English, Spanish) as entities.

OUTPUT FORMAT (VERY IMPORTANT):
- Return a SINGLE JSON object.
- The object MUST have exactly one key: "entities".
- "entities" MUST be a list of objects with "text" and "type" fields.
- Do NOT output any explanations, comments, or markdown.
- Do NOT wrap JSON in backticks or code fences.

Example output:
{"entities": [{"text": "London", "type": "ORGANIZATION"}]}

If no ORGANIZATION entities:
{"entities": []}
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

# ============================================================
# HARD NEGATIVE TEMPLATES
# These are short passages containing common FP terms in natural
# historical context. The correct output is NO entities.
# ============================================================

PERSON_NEGATIVE_PASSAGES = [
    # Nationalities/groups (not persons)
    "The French were known to have traded extensively with the natives of those parts, and the English likewise sought to establish their presence in the region.",
    "The Portuguese merchants sailed along the coast, while the Spanish had already established settlements further to the south.",
    "The Dutch had long been rivals of the English in the East India trade, and their ships were frequently encountered in those waters.",
    "The Tartars made incursions into the northern provinces, causing great alarm among the Christian inhabitants of those lands.",
    "The Persians and Arabians carried on a considerable trade in spices and silks through the ports of the Levant.",
    "The Israelites wandered in the wilderness for forty years before entering the promised land.",
    "The Saracens had conquered much of the Holy Land, and the Christians sought to reclaim it by force of arms.",
    # Generic titles (not specific persons)
    "His Majesty was pleased to grant the petition, and the council agreed that the matter should be referred to the proper authorities.",
    "The King received the ambassador with great ceremony, and the Queen likewise attended the audience.",
    "The Governor issued a proclamation forbidding the export of provisions from the colony without special licence.",
    "The Lord Chancellor presided over the court, and the judges delivered their opinions in turn.",
    "The President of the council ordered that the prisoners should be examined the following day.",
    "The Captain commanded his men to advance, while the Lieutenant remained with the reserves.",
    "His Excellency the Governor General directed that the fortifications should be repaired without delay.",
    "The Emperor decreed that all foreign merchants should pay the customary duties upon their goods.",
    "The Bishop preached a sermon upon the duties of Christian charity, and the congregation was much moved thereby.",
    "The Duke commanded the army, and the Prince led the cavalry in the charge.",
    # Single common names that are ambiguous
    "The town of George was founded in the year sixteen hundred and fifty, upon the banks of the river.",
    "The parish of John was well-known for its fair, held upon the feast day of that saint.",
    "The ship Alexander sailed from Bristol with a cargo of cloth and iron goods.",
    "Mount David rises above the surrounding plain, and from its summit one may see the sea.",
    # Commodities/things that look like names
    "The apothecary prepared a quantity of aqua vitae and spirit of wine for the use of the sick.",
    "Take of saffron two drams, mercury one ounce, and mix them with the oil of vitriol.",
    "The calomel was administered in small doses, and the patient showed signs of improvement.",
    "Venus was visible in the evening sky, and the mariners took their bearings accordingly.",
    # Organizations mistaken for persons
    "The Parliament debated the question for many hours, but no resolution was reached.",
    "The South Sea Company declared a dividend of ten per cent upon its stock.",
    "The East India Company sent three ships to the coast of Malabar that season.",
    "The Royal African Company maintained several factories along the Guinea coast.",
    "The House of Commons ordered that the bill should be read a second time.",
    # Classical/mythological (not relevant persons)
    "Homer tells us that Odysseus wandered for ten years before returning to his native land.",
    "Sophocles wrote of the fate of Ajax, who fell upon his own sword in despair.",
    "Athena appeared to the hero in a dream, and counselled him to be patient.",
    "Herodotus records that the Persians crossed the Hellespont with a mighty army.",
    "Pliny describes many curious plants and animals found in the eastern lands.",
    "Alexander conquered the known world before the age of thirty, and his empire stretched from Greece to India.",
]

ORGANIZATION_NEGATIVE_PASSAGES = [
    # Generic institutional terms (not specific orgs)
    "The court met at the appointed hour, and the judges took their seats upon the bench.",
    "The council debated the matter at length, but could reach no agreement upon the question.",
    "The parliament was dissolved by the King, and new elections were ordered throughout the realm.",
    "The government directed that the customs should be collected with greater diligence than heretofore.",
    "The committee reported their findings to the assembly, which voted to adopt the recommendations.",
    "The corporation of the town granted permission for the market to be held on Saturdays.",
    "The colony prospered under the new governor, and the settlers planted large quantities of tobacco.",
    "The church stood at the corner of the high street, and its bells could be heard throughout the parish.",
    "The fleet sailed from Portsmouth upon the morning tide, bound for the West Indies.",
    "The army encamped upon the heath, and the soldiers lit their fires against the cold.",
    "The exchequer was much depleted by the costs of the war, and new taxes were proposed.",
    "The commissioners were appointed to examine the accounts and report upon the same.",
    "The militia was called out to suppress the disturbance, and order was soon restored.",
    # Nationalities (not orgs)
    "The French had established their settlements along the great river, trading furs with the natives.",
    "The Portuguese claimed dominion over those coasts by right of first discovery.",
    "The Spanish maintained a garrison in the fortress, which commanded the entrance to the harbour.",
    "The English merchants complained of the duties imposed upon their goods at that port.",
    # Guild terms without specific names
    "The grocers and drapers of the town petitioned for relief from the new duties.",
    "The goldsmiths were required to mark all plate with the proper hallmarks.",
    "The fishmongers sold their catch at the quay each morning when the boats came in.",
    "The coopers made barrels for the storage of ale and wine, and their trade was brisk.",
    # Religious groups (not orgs)
    "The Christians and the Moors fought for possession of the fortress throughout the summer.",
    "The Jesuits had established missions in the remote provinces of that country.",
    "The Presbyterians refused to conform to the established church, and many were fined.",
    "The Papists were suspected of plotting against the government, and several were arrested.",
    # Generic council/court/committee
    "The council of war determined that the town should be invested on all sides.",
    "The court of quarter sessions dealt with the case of the poachers.",
    "The lords of the treasury directed that the payment should be made from the civil list.",
    "The justices of the peace bound him over to keep the peace for twelve months.",
    "A counsellor of state advised that the treaty should be ratified without further delay.",
    # Ships/publications (not orgs)
    "The Royal Anne sailed from Deptford with a complement of three hundred men.",
    "The Philosophical Transactions contained several papers upon the subject of magnetism.",
]

COMMODITY_NEGATIVE_PASSAGES = [
    # Things that look like commodities but aren't in trade context
    "The stone walls of the castle were five feet thick, and no cannon could breach them.",
    "The paper was read before the assembled members, who received it with great interest.",
    "The ship sailed with the morning tide, and was not seen again for many months.",
    "The children played in the courtyard while their mothers attended to the household duties.",
    "The diamond ring was presented to the Queen upon the occasion of her birthday.",
    "The pounds sterling were counted out upon the table, and the receipt was signed.",
    "The hen roosted upon the beam above the stable door, and crowed each morning at dawn.",
    "The slavery of sin is worse than any bondage of the body, as the divines do teach us.",
    "The mercury in the glass rose steadily as the day grew warmer.",
    "The mint was established for the coinage of money, and all bullion was brought thither.",
    "The trade with Scotland was of great importance to the merchants of the northern counties.",
    "The manufacture of woollen cloth was the chief employment of the people in those parts.",
    # Currency and abstract terms (not commodities)
    "The sum of five hundred pounds was paid into the exchequer for the use of the navy.",
    "The guineas were weighed and found to be of full value according to the standard.",
    "The money was lent at six per cent interest, to be repaid at Michaelmas next.",
    "The slavery practised in the colonies was much debated in Parliament that session.",
    "The freedom of the seas was a principle much insisted upon by the English ambassadors.",
    # Places/buildings that contain commodity words
    "The sugar plantations extended for many miles along the coast of that island.",
    "The cotton fields were planted in the spring and harvested in the autumn months.",
    "The iron works at that place employed many hundreds of workmen in their furnaces.",
    # Ships and vessels (not commodities)
    "The ships lay at anchor in the harbour, waiting for a fair wind to sail.",
    "The vessel was laden and ready for sea, but contrary winds detained her for a week.",
    "The fleet consisted of twelve men of war and twenty merchant ships.",
]

COMMODITY_POSITIVE_EXAMPLES = [
    # Basic trade goods - sugar varieties
    {
        "text": "The cargo consisted of three hundred hogsheads of sugar, fifty barrels of rum, and twenty bales of cotton, all consigned to merchants in London.",
        "entities": [
            {"text": "sugar", "type": "COMMODITY"},
            {"text": "rum", "type": "COMMODITY"},
            {"text": "cotton", "type": "COMMODITY"}
        ]
    },
    {
        "text": "The planters exported their sugars to England, where they were refined and sold at a considerable profit.",
        "entities": [
            {"text": "sugars", "type": "COMMODITY"}
        ]
    },
    {
        "text": "The white sugar fetched a higher price than the brown sugar, but both were in great demand.",
        "entities": [
            {"text": "white sugar", "type": "COMMODITY"},
            {"text": "brown sugar", "type": "COMMODITY"}
        ]
    },
    # Metals
    {
        "text": "The mines yielded great quantities of silver and gold, which were shipped to Spain in the annual fleet.",
        "entities": [
            {"text": "silver", "type": "COMMODITY"},
            {"text": "gold", "type": "COMMODITY"}
        ]
    },
    {
        "text": "The merchants imported iron, lead, and copper from Sweden, which metals were much wanted for the manufactures.",
        "entities": [
            {"text": "iron", "type": "COMMODITY"},
            {"text": "lead", "type": "COMMODITY"},
            {"text": "copper", "type": "COMMODITY"}
        ]
    },
    # Enslaved people as commodities (historical trade context)
    {
        "text": "The Royal African Company delivered five hundred slaves to the island that year, for which the planters paid in sugar and tobacco.",
        "entities": [
            {"text": "slaves", "type": "COMMODITY"},
            {"text": "sugar", "type": "COMMODITY"},
            {"text": "tobacco", "type": "COMMODITY"}
        ]
    },
    {
        "text": "The price of a healthy slave had risen to thirty pounds sterling, owing to the great demand from the plantations.",
        "entities": [
            {"text": "slave", "type": "COMMODITY"}
        ]
    },
    {
        "text": "The negroes were employed in the cultivation of sugar and cotton, labouring from dawn until dusk.",
        "entities": [
            {"text": "negroes", "type": "COMMODITY"}
        ]
    },
    # Alcohol and beverages
    {
        "text": "The island produced great quantities of rum, which was traded for slaves on the coast of Guinea.",
        "entities": [
            {"text": "rum", "type": "COMMODITY"},
            {"text": "slaves", "type": "COMMODITY"}
        ]
    },
    {
        "text": "The wine was imported from Madeira and the Canaries, and sold to the planters at a good profit.",
        "entities": [
            {"text": "wine", "type": "COMMODITY"}
        ]
    },
    {
        "text": "The merchants traded beer, ale, and strong waters with the colonists for their tobacco.",
        "entities": [
            {"text": "beer", "type": "COMMODITY"},
            {"text": "ale", "type": "COMMODITY"},
            {"text": "strong waters", "type": "COMMODITY"},
            {"text": "tobacco", "type": "COMMODITY"}
        ]
    },
    # Spices and exotic goods
    {
        "text": "The ships returned laden with pepper, ginger, and nutmeg from the East Indies.",
        "entities": [
            {"text": "pepper", "type": "COMMODITY"},
            {"text": "ginger", "type": "COMMODITY"},
            {"text": "nutmeg", "type": "COMMODITY"}
        ]
    },
    {
        "text": "The apothecary required saffron, myrrh, and cinnamon for the preparation of his medicines.",
        "entities": [
            {"text": "saffron", "type": "COMMODITY"},
            {"text": "myrrh", "type": "COMMODITY"},
            {"text": "cinnamon", "type": "COMMODITY"}
        ]
    },
    # Raw materials
    {
        "text": "The colony exported indigo, rice, and wood to England, receiving in return cloth and iron goods.",
        "entities": [
            {"text": "indigo", "type": "COMMODITY"},
            {"text": "rice", "type": "COMMODITY"},
            {"text": "wood", "type": "COMMODITY"},
            {"text": "cloth", "type": "COMMODITY"},
            {"text": "iron", "type": "COMMODITY"}
        ]
    },
    {
        "text": "The trade in salt was of great importance to the fisheries, for without it the fish could not be preserved.",
        "entities": [
            {"text": "salt", "type": "COMMODITY"}
        ]
    },
    # Servants (indentured labor as commodity)
    {
        "text": "The servants were bound for four years, after which time they were to receive their freedom and fifty acres of land.",
        "entities": [
            {"text": "servants", "type": "COMMODITY"}
        ]
    },
    # Mixed cargo examples
    {
        "text": "The manifest listed sugar, molasses, rum, cotton, and ginger as the principal items of the cargo.",
        "entities": [
            {"text": "sugar", "type": "COMMODITY"},
            {"text": "molasses", "type": "COMMODITY"},
            {"text": "rum", "type": "COMMODITY"},
            {"text": "cotton", "type": "COMMODITY"},
            {"text": "ginger", "type": "COMMODITY"}
        ]
    },
    {
        "text": "The warehouse contained hogsheads of tobacco, barrels of tar, and bales of deerskins ready for shipment.",
        "entities": [
            {"text": "tobacco", "type": "COMMODITY"},
            {"text": "tar", "type": "COMMODITY"},
            {"text": "deerskins", "type": "COMMODITY"}
        ]
    },
]

# ============================================================
# HARD POSITIVE TEMPLATES
# Passages with entities the model commonly misses.
# ============================================================

PERSON_POSITIVE_EXAMPLES = [
    # Titled persons (Sir X Y, Mr. Z, Col. A B)
    {
        "text": "Sir William Penn commanded the fleet in those waters, and Sir Thomas Modyford was appointed Governor of Jamaica in the year sixteen hundred and sixty four.",
        "entities": [
            {"text": "Sir William Penn", "type": "PERSON"},
            {"text": "Sir Thomas Modyford", "type": "PERSON"}
        ]
    },
    {
        "text": "Mr. Brand reported that the trade was much decayed, and Mr. Strode concurred in that opinion, adding that the planters were in great distress.",
        "entities": [
            {"text": "Mr. Brand", "type": "PERSON"},
            {"text": "Mr. Strode", "type": "PERSON"}
        ]
    },
    {
        "text": "Col. D'Oyley maintained the garrison at Port Royal with great difficulty, for provisions were scarce and the soldiers mutinous.",
        "entities": [
            {"text": "Col. D'Oyley", "type": "PERSON"}
        ]
    },
    {
        "text": "Sir Paul Ricaut, who had long resided at Constantinople, wrote a most excellent account of the Ottoman Empire.",
        "entities": [
            {"text": "Sir Paul Ricaut", "type": "PERSON"}
        ]
    },
    {
        "text": "Mr. John Taylor gave evidence before the committee, stating that he had been a planter in Barbados for twenty years.",
        "entities": [
            {"text": "Mr. John Taylor", "type": "PERSON"}
        ]
    },
    {
        "text": "Sir William Godolphin negotiated the treaty with Spain, and Sir Thomas Lynch succeeded him in that employment.",
        "entities": [
            {"text": "Sir William Godolphin", "type": "PERSON"},
            {"text": "Sir Thomas Lynch", "type": "PERSON"}
        ]
    },
    {
        "text": "The Lord of Windsore arrived in the colony with instructions from his Majesty, and was received by the council with all due ceremony.",
        "entities": [
            {"text": "Lord of Windsore", "type": "PERSON"}
        ]
    },
    {
        "text": "Robert Yeomans and George Butcher were taken prisoner by the parliamentarian forces and confined in the castle.",
        "entities": [
            {"text": "Robert Yeomans", "type": "PERSON"},
            {"text": "George Butcher", "type": "PERSON"}
        ]
    },
    {
        "text": "Master George Butcher, a merchant of good repute, was chosen alderman of the ward in that year.",
        "entities": [
            {"text": "Master George Butcher", "type": "PERSON"}
        ]
    },
    {
        "text": "John Gadbury published his almanack yearly, and was much consulted upon matters of astrology.",
        "entities": [
            {"text": "John Gadbury", "type": "PERSON"}
        ]
    },
    {
        "text": "Joseph Defia de Carana, a Spanish merchant residing in London, petitioned the council for redress of his losses.",
        "entities": [
            {"text": "Joseph Defia de Carana", "type": "PERSON"}
        ]
    },
    {
        "text": "Cromwell directed that the expedition should proceed to Hispaniola, and thence to Jamaica if the first attempt should fail.",
        "entities": [
            {"text": "Cromwell", "type": "PERSON"}
        ]
    },
    {
        "text": "Madam G. kept a shop in the high street where she sold imported silks and fine linen to the gentry.",
        "entities": [
            {"text": "Madam G.", "type": "PERSON"}
        ]
    },
    {
        "text": "Blianor Holder, a widow of the parish, brought suit against her neighbour for the recovery of the debt.",
        "entities": [
            {"text": "Blianor Holder", "type": "PERSON"}
        ]
    },
    # Mixed: persons present alongside non-person terms
    {
        "text": "Sir Edward Hederset informed the council that the French had taken several English ships, and that the King of Spain had refused to intervene.",
        "entities": [
            {"text": "Sir Edward Hederset", "type": "PERSON"}
        ]
    },
    {
        "text": "Richard Randall, a factor employed by the company, reported that the sugar harvest had been poor that season.",
        "entities": [
            {"text": "Richard Randall", "type": "PERSON"}
        ]
    },
]

ORGANIZATION_POSITIVE_EXAMPLES = [
    # Specific named organizations
    {
        "text": "The East India Company maintained factories at Surat, Madras, and Calcutta, employing many hundreds of persons in their service.",
        "entities": [
            {"text": "East India Company", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "The South Sea Company was incorporated by Act of Parliament in the year seventeen hundred and eleven, for the purpose of trading with the Spanish colonies.",
        "entities": [
            {"text": "South Sea Company", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "The Royal African Company held a monopoly upon the trade in negroes from the coast of Guinea.",
        "entities": [
            {"text": "Royal African Company", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "The Levant Company traded with the ports of the Ottoman Empire, exporting cloth and importing silks and spices.",
        "entities": [
            {"text": "Levant Company", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "The House of Commons resolved that the petition should be received, and ordered it to be printed for the consideration of the members.",
        "entities": [
            {"text": "House of Commons", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "The House of Lords debated the bill at length, and several peers spoke against it with great warmth.",
        "entities": [
            {"text": "House of Lords", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "The Lords of Trade and Plantations directed that the governor should send a full account of the colony's revenues.",
        "entities": [
            {"text": "Lords of Trade and Plantations", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "The Company of Merchants trading to Africa petitioned for the renewal of their charter.",
        "entities": [
            {"text": "Company of Merchants trading to Africa", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "The Merchants of Bristol fitted out several ships for the Guinea trade that season.",
        "entities": [
            {"text": "Merchants of Bristol", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "Trinity College had long been a seat of learning, and many eminent scholars received their education there.",
        "entities": [
            {"text": "Trinity College", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "The General Post Office in London received letters from all parts of the kingdom for dispatch to the colonies.",
        "entities": [
            {"text": "General Post Office in London", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "The Houses of Parliament were much divided upon the question of the sugar duties.",
        "entities": [
            {"text": "Houses of Parliament", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "The Association Board met quarterly to consider the state of the trade and the grievances of the planters.",
        "entities": [
            {"text": "Association Board", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "The Crown Court at Westminster heard the case and pronounced judgment in favour of the defendant.",
        "entities": [
            {"text": "Crown Court", "type": "ORGANIZATION"}
        ]
    },
    # Mixed: specific orgs alongside generic terms the model should NOT tag
    {
        "text": "The council debated whether the East India Company should be permitted to export bullion, and the court heard arguments from both sides.",
        "entities": [
            {"text": "East India Company", "type": "ORGANIZATION"}
        ]
    },
    {
        "text": "The parliament resolved that the Royal African Company should surrender its monopoly, and that the trade should be open to all merchants.",
        "entities": [
            {"text": "Royal African Company", "type": "ORGANIZATION"}
        ]
    },
]


def make_training_example(text, entities, entity_type):
    """Create a training example in chat JSONL format."""
    system_prompt = SYSTEM_PROMPTS[entity_type]
    user_content = f"Please extract {entity_type} entities from the following text.\n\nTEXT:\n{text}"
    assistant_content = json.dumps({"entities": entities}, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def generate_negatives(entity_type):
    """Generate hard negative training examples for an entity type."""
    if entity_type == "PERSON":
        passages = PERSON_NEGATIVE_PASSAGES
    elif entity_type == "ORGANIZATION":
        passages = ORGANIZATION_NEGATIVE_PASSAGES
    elif entity_type == "COMMODITY":
        passages = COMMODITY_NEGATIVE_PASSAGES
    else:
        return []

    examples = []
    for passage in passages:
        example = make_training_example(passage, [], entity_type)
        examples.append(example)

    return examples


def generate_positives(entity_type):
    """Generate hard positive training examples for an entity type."""
    if entity_type == "PERSON":
        items = PERSON_POSITIVE_EXAMPLES
    elif entity_type == "ORGANIZATION":
        items = ORGANIZATION_POSITIVE_EXAMPLES
    elif entity_type == "COMMODITY":
        items = COMMODITY_POSITIVE_EXAMPLES
    else:
        return []

    examples = []
    for item in items:
        example = make_training_example(item["text"], item["entities"], entity_type)
        examples.append(example)

    return examples


def main():
    print("=" * 70)
    print("GENERATING SYNTHETIC TRAINING DATA")
    print("=" * 70)

    for entity_type in ["PERSON", "ORGANIZATION", "COMMODITY"]:
        print(f"\n{'='*50}")
        print(f"  {entity_type}")
        print(f"{'='*50}")

        output_dir = FILTERED_DIR / entity_type.lower()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate negatives
        negatives = generate_negatives(entity_type)
        neg_file = output_dir / "synthetic_negatives.jsonl"
        with open(neg_file, 'w', encoding='utf-8') as f:
            for ex in negatives:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        print(f"  Negatives: {len(negatives)} examples -> {neg_file}")

        # Generate positives
        positives = generate_positives(entity_type)
        pos_file = output_dir / "synthetic_positives.jsonl"
        with open(pos_file, 'w', encoding='utf-8') as f:
            for ex in positives:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        print(f"  Positives: {len(positives)} examples -> {pos_file}")

        # Combine with existing filtered training data
        train_file = output_dir / "train_chat.jsonl"
        augmented_file = output_dir / "train_chat_augmented.jsonl"

        if train_file.exists():
            # Read existing training data
            existing = []
            with open(train_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        existing.append(json.loads(line))

            # Combine: existing + negatives (1x) + positives (1x)
            # Kept to 1x to avoid overfitting on synthetic data
            combined = existing.copy()
            combined.extend(negatives)
            combined.extend(positives)

            # Shuffle
            random.shuffle(combined)

            with open(augmented_file, 'w', encoding='utf-8') as f:
                for ex in combined:
                    f.write(json.dumps(ex, ensure_ascii=False) + '\n')

            print(f"  Combined: {len(existing)} existing + {len(negatives)*3} neg + {len(positives)*2} pos = {len(combined)} total")
            print(f"  Output: {augmented_file}")
        else:
            print(f"  WARNING: No existing training file found at {train_file}")

    print(f"\n{'='*70}")
    print("DONE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review synthetic examples in data/ensemble_training_filtered/*/synthetic_*.jsonl")
    print("  2. Update training configs to point to train_chat_augmented.jsonl")
    print("  3. Retrain models")


if __name__ == "__main__":
    main()
