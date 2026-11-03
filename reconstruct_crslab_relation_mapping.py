
import json
import csv
from pathlib import Path
from collections import Counter

BASE = Path("data/dataset/redial/nltk")
OUTPUT = Path("results_category_audit")

with (BASE / "entity2id.json").open(encoding="utf-8") as f:
    entity2id = json.load(f)

with (BASE / "dbpedia_subkg.json").open(encoding="utf-8") as f:
    entity_kg = json.load(f)

# Reproduce CRSLab's entity indexing.
n_entity = max(entity2id.values()) + 1

# Reproduce CRSLab's exact edge-list construction.
edge_list = []

for entity in range(n_entity):
    if str(entity) not in entity_kg:
        continue

    # Original CRSLab self-loop.
    edge_list.append((entity, entity, 185))

    for relation, tail in entity_kg[str(entity)]:
        relation = int(relation)
        tail = int(tail)

        if entity != tail and relation != 185:
            edge_list.append((entity, tail, relation))
            edge_list.append((tail, entity, relation))

# Count relations exactly as CRSLab does.
relation_counts = Counter(r for _, _, r in edge_list)

# Reconstruct sequential relation IDs.
relation2id = {}

for _, _, relation in edge_list:
    if relation_counts[relation] > 1000:
        if relation not in relation2id:
            relation2id[relation] = len(relation2id)

# Export original-to-processed mapping.
OUTPUT.mkdir(parents=True, exist_ok=True)

mapping_path = OUTPUT / "crslab_relation_mapping.csv"

with mapping_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "original_relation_id",
        "processed_relation_id",
        "edge_record_count",
        "passes_threshold"
    ])

    for original, processed in relation2id.items():
        writer.writerow([
            original,
            processed,
            relation_counts[original],
            True
        ])

# Export complete relation frequency audit.
frequency_path = OUTPUT / "crslab_relation_frequency.csv"

with frequency_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow([
        "original_relation_id",
        "edge_record_count",
        "included_in_processed_graph",
        "processed_relation_id"
    ])

    for relation, count in relation_counts.most_common():
        writer.writerow([
            relation,
            count,
            relation in relation2id,
            relation2id.get(relation, "")
        ])

print("\n=== CRSLAB RELATION RECONSTRUCTION ===")
print("Original entities:", len(entity2id))
print("Graph entities:", len(entity_kg))
print("Total constructed edge records:", len(edge_list))
print("Unique original relations:", len(relation_counts))
print("Relations passing threshold:", len(relation2id))
print("Self-loop relation 185:", relation2id.get(185))
print("Threshold: strictly greater than 1000")

print("\n=== FIRST 20 MAPPINGS ===")

for original, processed in list(relation2id.items())[:20]:
    print(
        f"Original: {original:>4} -> "
        f"Processed: {processed:>3} | "
        f"Edges: {relation_counts[original]}"
    )

print("\nSaved:")
print(mapping_path)
print(frequency_path)
