import pickle
from pathlib import Path
from collections import Counter

PICKLE = Path("data/dataset/redial/nltk/all_data.pkl")

with PICKLE.open("rb") as f:
    data = pickle.load(f)

kg = data[3]["entity_kg"]
vocab = data[4]

entity2id = vocab["entity2id"]
id2entity = vocab["id2entity"]

edges = kg["edge"]

numeric = {
    name: eid
    for name, eid in entity2id.items()
    if str(name).isdigit()
}

# Processed relation 0 corresponds to original
# self-loop relation 185, as verified previously.
SELF_LOOP_ID = 0

numeric_ids = set(numeric.values())

stats = Counter()
relation_counts = Counter()
examples = []

for name, eid in numeric.items():

    incident = [
        edge for edge in edges
        if edge[0] == eid or edge[1] == eid
    ]

    self_loops = [
        edge for edge in incident
        if edge[0] == eid
        and edge[1] == eid
    ]

    semantic_edges = [
        edge for edge in incident
        if edge[0] != edge[1]
    ]

    stats["total"] += 1
    stats["incident_edges"] += len(incident)
    stats["self_loops"] += len(self_loops)
    stats["semantic_edges"] += len(semantic_edges)

    if semantic_edges:
        stats["with_semantic_neighbors"] += 1
    else:
        stats["self_loop_only"] += 1

    for edge in incident:
        relation_counts[edge[2]] += 1

    if len(examples) < 30:
        examples.append({
            "numeric_name": name,
            "entity_id": eid,
            "incident": incident,
            "self_loops": self_loops,
            "semantic_edges": semantic_edges
        })

print("\n========== NUMERIC ENTITY NEIGHBOR AUDIT ==========")

print("Numeric entities:", stats["total"])
print("Total incident edge records:", stats["incident_edges"])
print("Self-loop records:", stats["self_loops"])
print("Non-self-loop edge records:", stats["semantic_edges"])

print("Entities with semantic neighbors:",
      stats["with_semantic_neighbors"])

print("Self-loop-only entities:",
      stats["self_loop_only"])

print("\n========== RELATION DISTRIBUTION ==========")

print(relation_counts.most_common(20))

print("\n========== FIRST 30 ENTITY EXAMPLES ==========")

for item in examples:

    print("\nNumeric name:", item["numeric_name"])
    print("Entity ID:", item["entity_id"])
    print("Incident edges:", item["incident"])
    print("Self-loops:", item["self_loops"])
    print("Semantic edges:", item["semantic_edges"])

print("\n========== OTHER ENTITY NAMES ==========")

other = {
    name: eid
    for name, eid in entity2id.items()
    if not str(name).isdigit()
    and "dbpedia.org/resource/" not in str(name)
}

print("Other entity names:", len(other))

for name, eid in list(other.items())[:30]:
    print(repr(name), "->", eid)

print("\n========== AUDIT COMPLETE ==========")