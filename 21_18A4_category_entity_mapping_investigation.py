import json
from pathlib import Path
from collections import Counter

base = Path("data/dataset/redial/nltk")

with open(base / "dbpedia_subkg.json", encoding="utf-8") as f:
    kg = json.load(f)

with open(base / "entity2id.json", encoding="utf-8") as f:
    entity2id = json.load(f)

id2entity = {int(v): k for k, v in entity2id.items()}

print("=" * 75)
print("EXPERIMENT 21.18A.4: CATEGORY ENTITY MAPPING INVESTIGATION")
print("=" * 75)

print("\n[1] CATEGORY URI PATTERN SEARCH")

patterns = [
    "category:",
    "category",
    "genre",
    "film_genre",
    "subject",
    "skos/core#concept",
]

for pattern in patterns:

    matches = [
        (uri, eid)
        for uri, eid in entity2id.items()
        if pattern.lower() in uri.lower()
    ]

    print(f"{pattern}: {len(matches)}")

    for uri, eid in matches[:5]:
        print(" ", eid, uri)

print("\n[2] ENTITY ID COVERAGE")

kg_ids = set(map(int, kg.keys()))
mapping_ids = set(id2entity.keys())

print("KG entity IDs:", len(kg_ids))
print("Mapping entity IDs:", len(mapping_ids))
print("KG IDs absent from mapping:", len(kg_ids - mapping_ids))
print("Mapping IDs absent from KG:", len(mapping_ids - kg_ids))

print("\n[3] ORIGINAL RELATION TAIL INSPECTION")

for rid in [69, 105, 138, 114, 125, 122, 183, 135]:

    tails = []

    for head, triples in kg.items():

        for relation, tail in triples:

            if int(relation) == rid:

                tail = int(tail)
                tails.append(tail)

                if len(tails) >= 10:
                    break

        if len(tails) >= 10:
            break

    print(f"\nRELATION {rid}")

    for tail in tails:
        print(tail, id2entity.get(tail, "MISSING_URI"))

print("\n[4] CATEGORY-LIKE URI EXAMPLES")

category_like = [
    (uri, eid)
    for uri, eid in entity2id.items()
    if any(x in uri.lower() for x in [
        "category",
        "genre",
        "film_classification",
        "film_genre"
    ])
]

print("Total:", len(category_like))

for uri, eid in category_like[:50]:
    print(eid, uri)

print("\n[5] CATEGORY INVENTORY SOURCE")

inventory = Path(
    "results_category_audit/dbpedia_pilot/"
    "canonical_category_inventory.csv"
)

print("Existing inventory:", inventory.exists())

print("\nINVESTIGATION COMPLETE")
