
import json
import pickle
from pathlib import Path
from collections import Counter

BASE = Path("data/dataset/redial/nltk")
CACHE = BASE / "all_data.pkl"

# Load original DBpedia data.
with (BASE / "entity2id.json").open(encoding="utf-8") as f:
    entity2id = json.load(f)

with (BASE / "dbpedia_subkg.json").open(encoding="utf-8") as f:
    entity_kg = json.load(f)

# Load actual cached CRSLAB graph.
with CACHE.open("rb") as f:
    cached = pickle.load(f)

actual_graph = cached[3]["entity_kg"]

actual_edges = {
    tuple(map(int, edge))
    for edge in actual_graph["edge"]
}

actual_relations = actual_graph["n_relation"]

# Reproduce CRSLAB preprocessing exactly.
n_entity = max(entity2id.values()) + 1
edge_list = []

for entity in range(n_entity):
    if str(entity) not in entity_kg:
        continue

    edge_list.append((entity, entity, 185))

    for relation, tail in entity_kg[str(entity)]:
        relation = int(relation)
        tail = int(tail)

        if entity != tail and relation != 185:
            edge_list.append((entity, tail, relation))
            edge_list.append((tail, entity, relation))

relation_counts = Counter(
    relation for _, _, relation in edge_list
)

relation2id = {}

for _, _, relation in edge_list:
    if relation_counts[relation] > 1000:
        if relation not in relation2id:
            relation2id[relation] = len(relation2id)

# Reproduce the original set-based edge construction.
reconstructed_edges = set()

for h, t, r in edge_list:
    if relation_counts[r] > 1000:
        reconstructed_edges.add(
            (h, t, relation2id[r])
        )

missing = actual_edges - reconstructed_edges
extra = reconstructed_edges - actual_edges

print("\n=== EXACT KGSF GRAPH VERIFICATION ===")

print("Actual relation count:", actual_relations)
print("Reconstructed relation count:", len(relation2id))

print("Actual cached edges:", len(actual_edges))
print("Reconstructed unique edges:", len(reconstructed_edges))

print("Missing actual edges:", len(missing))
print("Extra reconstructed edges:", len(extra))

print("\n=== RELATION DISTRIBUTION CHECK ===")

actual_counts = Counter(r for _, _, r in actual_edges)
reconstructed_counts = Counter(
    r for _, _, r in reconstructed_edges
)

print("Relation distributions identical:",
      actual_counts == reconstructed_counts)

print("\n=== FINAL VERDICT ===")

if (
    actual_relations == len(relation2id)
    and actual_edges == reconstructed_edges
):
    print("SUCCESS: Exact graph reconstruction confirmed.")
else:
    print("WARNING: Graph mismatch requires investigation.")

    print("First missing edges:", list(missing)[:10])
    print("First extra edges:", list(extra)[:10])
