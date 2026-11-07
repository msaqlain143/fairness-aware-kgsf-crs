import json
import csv
from pathlib import Path
from collections import Counter, defaultdict

BASE = Path("data/dataset/redial/nltk")
OUT = Path("results_category_audit/experiment_21_18A3")

OUT.mkdir(parents=True, exist_ok=True)

KG_FILE = BASE / "dbpedia_subkg.json"
ENTITY_FILE = BASE / "entity2id.json"

print("=" * 75)
print("EXPERIMENT 21.18A.3: COMPLETE ORIGINAL KG RELATION RECONSTRUCTION")
print("=" * 75)

with open(KG_FILE, encoding="utf-8") as f:
    kg = json.load(f)

with open(ENTITY_FILE, encoding="utf-8") as f:
    entity2id = json.load(f)

id2entity = {int(v): k for k, v in entity2id.items()}

relation_counts = Counter()
relation_head_types = defaultdict(Counter)
relation_tail_types = defaultdict(Counter)
relation_examples = defaultdict(list)

total_triples = 0
missing_heads = 0
missing_tails = 0
self_loops = 0

for head, triples in kg.items():

    head_id = int(head)

    if head_id not in id2entity:
        missing_heads += 1

    for triple in triples:

        if len(triple) != 2:
            continue

        relation, tail = map(int, triple)

        total_triples += 1
        relation_counts[relation] += 1

        if head_id == tail:
            self_loops += 1

        head_uri = id2entity.get(head_id, "")
        tail_uri = id2entity.get(tail, "")

        if not tail_uri:
            missing_tails += 1

        relation_head_types[relation][
            "movie" if "_film" in head_uri.lower()
            or "_movie" in head_uri.lower()
            else "other"
        ] += 1

        relation_tail_types[relation][
            "category" if "category:" in tail_uri.lower()
            else "other"
        ] += 1

        if len(relation_examples[relation]) < 5:
            relation_examples[relation].append({
                "head_id": head_id,
                "head_uri": head_uri,
                "relation_id": relation,
                "tail_id": tail,
                "tail_uri": tail_uri,
                "self_loop": head_id == tail
            })

print("\n[1] POPULATION")
print("KG entities:", len(kg))
print("Total triples:", total_triples)
print("Unique relations:", len(relation_counts))
print("Entity mapping size:", len(entity2id))
print("Missing head mappings:", missing_heads)
print("Missing tail mappings:", missing_tails)
print("Self loops:", self_loops)

print("\n[2] RELATION INVENTORY")

relation_file = OUT / "original_relation_inventory.csv"

with open(relation_file, "w", newline="", encoding="utf-8-sig") as f:

    writer = csv.writer(f)

    writer.writerow([
        "relation_id",
        "triple_count",
        "movie_head_count",
        "other_head_count",
        "category_tail_count",
        "other_tail_count",
        "sample_examples"
    ])

    for rid, count in sorted(
        relation_counts.items(),
        key=lambda x: (-x[1], x[0])
    ):

        writer.writerow([
            rid,
            count,
            relation_head_types[rid]["movie"],
            relation_head_types[rid]["other"],
            relation_tail_types[rid]["category"],
            relation_tail_types[rid]["other"],
            json.dumps(
                relation_examples[rid],
                ensure_ascii=False
            )
        ])

print("Exported:", relation_file)

print("\n[3] TOP RELATIONS")

for rid, count in relation_counts.most_common(20):
    print(
        f"Relation {rid}: {count} triples | "
        f"Category tails: {relation_tail_types[rid]['category']}"
    )

print("\n[4] CATEGORY URI RELATION INVENTORY")

category_records = []

for head, triples in kg.items():

    head_id = int(head)

    for relation, tail in triples:

        tail = int(tail)
        tail_uri = id2entity.get(tail, "")

        if "category:" in tail_uri.lower():

            category_records.append({
                "movie_entity_id": head_id,
                "movie_uri": id2entity.get(head_id, ""),
                "relation_id": int(relation),
                "category_entity_id": tail,
                "category_uri": tail_uri
            })

category_file = OUT / "all_category_uri_edges.csv"

with open(category_file, "w", newline="", encoding="utf-8-sig") as f:

    fields = [
        "movie_entity_id",
        "movie_uri",
        "relation_id",
        "category_entity_id",
        "category_uri"
    ]

    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(category_records)

print("Category URI edges:", len(category_records))
print("Unique movies:", len(set(
    x["movie_entity_id"] for x in category_records
)))
print("Unique category entities:", len(set(
    x["category_entity_id"] for x in category_records
)))
print("Exported:", category_file)

print("\n[5] RELATION-SPECIFIC CATEGORY COUNTS")

category_relation_counts = Counter(
    x["relation_id"] for x in category_records
)

for rid, count in category_relation_counts.most_common():
    print(f"Relation {rid}: {count} category edges")

print("\n" + "=" * 75)
print("EXPERIMENT 21.18A.3 EXTRACTION COMPLETED")
print("No original KG modifications.")
print("No automatic genre classification.")
print("=" * 75)
