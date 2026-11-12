
import pickle
import json
import pandas as pd
from pathlib import Path

BASE = Path("data/dataset/redial/nltk")

# ==================================================
# 1. LOAD ORIGINAL DATA
# ==================================================

with open(BASE / "entity2id.json", encoding="utf-8") as f:
    entity2id = json.load(f)

with open(BASE / "dbpedia_subkg.json", encoding="utf-8") as f:
    graph = json.load(f)

with open(BASE / "movie_ids.json", encoding="utf-8") as f:
    movie_ids = json.load(f)

with open("catalog_audit_corrected.pkl", "rb") as f:
    catalog = pickle.load(f)

# ==================================================
# 2. BUILD INTERNAL GRAPH NODE SET
# ==================================================

graph_nodes = set()
connected_nodes = set()

for head, edges in graph.items():

    head_id = int(head)
    graph_nodes.add(head_id)

    for relation, tail in edges:

        tail_id = int(tail)
        graph_nodes.add(tail_id)

        # Exclude self-loops
        if head_id != tail_id:
            connected_nodes.add(head_id)
            connected_nodes.add(tail_id)

# ==================================================
# 3. VALIDATE CATALOG ID CONVENTION
# ==================================================

catalog_ids = set(map(int, catalog["sorted_items"]))
movie_ids_set = set(map(int, movie_ids))

print("\n========== ID VALIDATION ==========")

print("Entity dictionary entries:", len(entity2id))
print("Movie IDs in original dataset:", len(movie_ids_set))
print("Catalog items:", len(catalog_ids))
print("Graph nodes:", len(graph_nodes))

print("Catalog/movie ID intersection:",
      len(catalog_ids & movie_ids_set))

print("Catalog IDs missing from graph:",
      len(catalog_ids - graph_nodes))

print("Movie IDs missing from catalog:",
      len(movie_ids_set - catalog_ids))

# ==================================================
# 4. GROUP DATA
# ==================================================

item2group = {
    int(item): int(group)
    for item, group in catalog["item2group"].items()
}

zero_ids = set(map(int, catalog["zero_shot_ids"]))
one_ids = set(map(int, catalog["one_shot_ids"]))

# ==================================================
# 5. SEMANTIC COVERAGE FUNCTION
# ==================================================

def audit_group(name, items):

    items = set(items)

    present = items & graph_nodes
    connected = items & connected_nodes

    isolated = present - connected
    missing = items - graph_nodes

    total = len(items)

    def pct(value):
        return round(100 * value / total, 2) if total else 0

    print(f"\n========== {name} ==========")

    print("Total items:", total)

    print("Graph presence:", len(present))
    print("Graph presence coverage:", pct(len(present)), "%")

    print("Semantic connectivity:", len(connected))
    print("Semantic coverage:", pct(len(connected)), "%")

    print("Present but without external connections:",
          len(isolated))

    print("Missing from graph:", len(missing))

    print("Connectivity among graph-present items:",
          round(100 * len(connected) / len(present), 2)
          if present else 0, "%")

# ==================================================
# 6. ZERO-SHOT AND ONE-SHOT AUDIT
# ==================================================

audit_group("ZERO-SHOT", zero_ids)
audit_group("ONE-SHOT", one_ids)

# ==================================================
# 7. FREQUENCY GROUPS G0-G4
# ==================================================

for group in range(5):

    group_items = {
        item
        for item, assigned_group in item2group.items()
        if assigned_group == group
    }

    audit_group(f"FREQUENCY GROUP G{group}", group_items)

# ==================================================
# 8. FINAL DIAGNOSTICS
# ==================================================

print("\n========== FINAL DIAGNOSTICS ==========")

print("Total catalog items:", len(catalog_ids))
print("Total graph nodes:", len(graph_nodes))
print("Total connected graph nodes:", len(connected_nodes))

print("Catalog items with graph connections:",
      len(catalog_ids & connected_nodes))

print("Catalog items without graph connections:",
      len(catalog_ids - connected_nodes))

print("\nAUDIT COMPLETE")
