
#!/usr/bin/env python
"""
Experiment 22.6C:
Movie-to-KG entity alignment and graph connectivity audit.

No model training, ranking changes, or test-set access.
"""

import json
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path("data/dataset/redial/nltk")

ENTITY_FILE = ROOT / "entity2id.json"
MOVIE_FILE = ROOT / "movie_ids.json"
DBPEDIA_FILE = ROOT / "dbpedia_subkg.json"

print("=" * 75)
print("EXPERIMENT 22.6C — MOVIE/KG ENTITY ALIGNMENT")
print("=" * 75)

with ENTITY_FILE.open(encoding="utf-8") as f:
    entity2id = json.load(f)

with MOVIE_FILE.open(encoding="utf-8") as f:
    movie_ids = json.load(f)

with DBPEDIA_FILE.open(encoding="utf-8") as f:
    dbpedia = json.load(f)

# --------------------------------------------------
# 1. Construct ID mappings
# --------------------------------------------------

entity_id_to_uri = {
    int(entity_id): uri
    for uri, entity_id in entity2id.items()
}

movie_set = set(map(int, movie_ids))

graph_keys = set(map(int, dbpedia.keys()))

print("\n========== ID ALIGNMENT ==========")

print("Movie IDs:", len(movie_set))
print("Entity IDs:", len(entity_id_to_uri))
print("DBpedia graph nodes:", len(graph_keys))

movie_entity_overlap = movie_set & set(entity_id_to_uri)
movie_graph_overlap = movie_set & graph_keys
movie_both_overlap = movie_set & graph_keys & set(entity_id_to_uri)

print("\nMovies matching entity2id:", len(movie_entity_overlap))
print("Movies with DBpedia graph records:", len(movie_graph_overlap))
print("Movies matching both:", len(movie_both_overlap))

print("Movie entity coverage:",
      len(movie_entity_overlap) / len(movie_set))

print("Movie graph coverage:",
      len(movie_graph_overlap) / len(movie_set))

# --------------------------------------------------
# 2. Inspect matched movie entities
# --------------------------------------------------

print("\n========== MATCHED MOVIE EXAMPLES ==========")

for mid in sorted(movie_both_overlap)[:15]:
    print({
        "movie_id": mid,
        "entity_uri": entity_id_to_uri[mid],
        "outgoing_relations": len(dbpedia[str(mid)])
    })

# --------------------------------------------------
# 3. Graph connectivity
# --------------------------------------------------

degree = Counter()
relation_counts = Counter()
endpoint_ids = set()

for head, triples in dbpedia.items():

    head = int(head)

    for triple in triples:

        if len(triple) != 2:
            continue

        relation_id, tail_id = map(int, triple)

        degree[head] += 1
        relation_counts[relation_id] += 1
        endpoint_ids.add(tail_id)

print("\n========== GRAPH STRUCTURE ==========")

print("Graph records:", len(dbpedia))
print("Total outgoing triples:", sum(degree.values()))
print("Distinct relation IDs:", len(relation_counts))
print("Distinct tail entity IDs:", len(endpoint_ids))

print("Tail IDs matching entity2id:",
      len(endpoint_ids & set(entity_id_to_uri)))

print("Graph nodes with no outgoing records:",
      len(endpoint_ids - graph_keys))

# --------------------------------------------------
# 4. Movie-level graph coverage
# --------------------------------------------------

movie_degrees = [
    degree.get(mid, 0)
    for mid in movie_set
]

print("\n========== MOVIE GRAPH DEGREE ==========")

print("Movies with zero outgoing edges:",
      sum(d == 0 for d in movie_degrees))

print("Movies with at least one edge:",
      sum(d > 0 for d in movie_degrees))

print("Mean outgoing degree:",
      float(np.mean(movie_degrees)))

print("Median outgoing degree:",
      float(np.median(movie_degrees)))

print("95th percentile:",
      float(np.percentile(movie_degrees, 95)))

# --------------------------------------------------
# 5. Save detailed movie mapping audit
# --------------------------------------------------

rows = []

for mid in sorted(movie_set):

    rows.append({
        "movie_id": mid,
        "entity_uri": entity_id_to_uri.get(mid, ""),
        "in_entity2id": mid in entity_id_to_uri,
        "in_dbpedia_graph": mid in graph_keys,
        "outgoing_degree": degree.get(mid, 0),
    })

import pandas as pd

output = Path("results_corrected/experiment_22_06C")
output.mkdir(parents=True, exist_ok=True)

pd.DataFrame(rows).to_csv(
    output / "movie_kg_entity_alignment.csv",
    index=False
)

summary = {
    "movies": len(movie_set),
    "entity_overlap": len(movie_entity_overlap),
    "graph_overlap": len(movie_graph_overlap),
    "both_overlap": len(movie_both_overlap),
    "total_graph_triples": sum(degree.values()),
    "distinct_relations": len(relation_counts),
    "mean_movie_degree": float(np.mean(movie_degrees)),
    "test_split_evaluated": False,
}

with (output / "audit_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n========== FINAL STATUS ==========")
print(json.dumps(summary, indent=2))
print("\nOutputs:", output.resolve())
print("\nExperiment 22.6C completed.")
