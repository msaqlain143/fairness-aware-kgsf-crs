import json
import pickle
from pathlib import Path
from collections import Counter

BASE = Path("data/dataset/redial/nltk")

with open(BASE / "all_data.pkl", "rb") as f:
    cached = pickle.load(f)

with open(BASE / "dbpedia_subkg.json", encoding="utf-8") as f:
    raw_graph = json.load(f)

vocab = cached[4]
entity2id = vocab["entity2id"]

numeric = {
    name: eid
    for name, eid in entity2id.items()
    if str(name).isdigit()
}

# Count raw relation occurrences.
relation_frequency = Counter()

for head, edges in raw_graph.items():
    for relation, tail in edges:
        relation_frequency[relation] += 2

# CRSLab uses a strict >1000 threshold.
retained_relations = {
    r for r, count in relation_frequency.items()
    if count > 1000
}

results = []

for name, eid in numeric.items():

    outgoing = raw_graph.get(str(eid), [])

    incoming = []

    for head, edges in raw_graph.items():
        for relation, tail in edges:
            if tail == eid:
                incoming.append((int(head), relation))

    all_edges = [
        (r, t, "outgoing")
        for r, t in outgoing
    ] + [
        (r, h, "incoming")
        for h, r in incoming
    ]

    retained = [
        edge for edge in all_edges
        if edge[0] in retained_relations
    ]

    results.append({
        "numeric_identifier": name,
        "entity_id": eid,
        "raw_outgoing": len(outgoing),
        "raw_incoming": len(incoming),
        "raw_semantic_edges": len(all_edges),
        "retained_edges": len(retained),
        "filtered_edges": len(all_edges) - len(retained),
        "outgoing_examples": outgoing[:5],
    })

print("\n========== RAW KG NUMERIC ENTITY AUDIT ==========")

print("Numeric entities:", len(results))

print("Entities with raw semantic edges:",
      sum(r["raw_semantic_edges"] > 0 for r in results))

print("Entities without raw semantic edges:",
      sum(r["raw_semantic_edges"] == 0 for r in results))

print("Entities with retained semantic edges:",
      sum(r["retained_edges"] > 0 for r in results))

print("Entities with filtered semantic edges:",
      sum(r["filtered_edges"] > 0 for r in results))

print("\n========== EDGE SUMMARY ==========")

print("Raw semantic edge records:",
      sum(r["raw_semantic_edges"] for r in results))

print("Retained semantic edge records:",
      sum(r["retained_edges"] for r in results))

print("Filtered semantic edge records:",
      sum(r["filtered_edges"] for r in results))

print("\n========== FIRST 30 EXAMPLES ==========")

for r in results[:30]:
    print(
        r["numeric_identifier"],
        "| ID:", r["entity_id"],
        "| Raw:", r["raw_semantic_edges"],
        "| Retained:", r["retained_edges"],
        "| Filtered:", r["filtered_edges"],
        "| Outgoing:", r["outgoing_examples"]
    )

print("\n========== AUDIT COMPLETE ==========")