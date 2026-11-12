
import pickle
import json
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE = Path("data/dataset/redial/nltk")
OUT = Path("results_category_audit")
OUT.mkdir(parents=True, exist_ok=True)

# ==================================================
# 1. LOAD DATA
# ==================================================

with open(BASE / "dbpedia_subkg.json", encoding="utf-8") as f:
    raw_graph = json.load(f)

with open(BASE / "all_data.pkl", "rb") as f:
    all_data = pickle.load(f)

with open("catalog_audit_corrected.pkl", "rb") as f:
    catalog = pickle.load(f)

processed_kg = all_data[3]["entity_kg"]

# ==================================================
# 2. BUILD RAW GRAPH ADJACENCY
# ==================================================

raw_adj = defaultdict(set)
raw_nodes = set()

for head, edges in raw_graph.items():

    h = int(head)
    raw_nodes.add(h)

    for relation, tail in edges:

        t = int(tail)
        raw_nodes.add(t)

        if h != t:
            raw_adj[h].add(t)
            raw_adj[t].add(h)

raw_connected = set(raw_adj.keys())

# ==================================================
# 3. BUILD KGSF-PROCESSED GRAPH ADJACENCY
# ==================================================

processed_adj = defaultdict(set)
processed_nodes = set()

for h, t, relation in processed_kg["edge"]:

    h = int(h)
    t = int(t)

    processed_nodes.add(h)
    processed_nodes.add(t)

    if h != t:
        processed_adj[h].add(t)
        processed_adj[t].add(h)

processed_connected = set(processed_adj.keys())

# ==================================================
# 4. LOAD CATALOG AND GROUPS
# ==================================================

catalog_ids = set(map(int, catalog["sorted_items"]))

item2group = {
    int(item): int(group)
    for item, group in catalog["item2group"].items()
}

zero_ids = set(map(int, catalog["zero_shot_ids"]))
one_ids = set(map(int, catalog["one_shot_ids"]))

# ==================================================
# 5. ITEM-LEVEL COMPARISON
# ==================================================

rows = []

for item in sorted(catalog_ids):

    raw_degree = len(raw_adj[item])
    processed_degree = len(processed_adj[item])

    rows.append({

        "movie_entity_id": item,

        "frequency_group": item2group.get(item),

        "cold_start_type": (
            "zero_shot" if item in zero_ids
            else "one_shot" if item in one_ids
            else "other"
        ),

        "raw_graph_present": item in raw_nodes,

        "raw_semantic_connected": raw_degree > 0,

        "raw_semantic_degree": raw_degree,

        "kgsf_graph_present": item in processed_nodes,

        "kgsf_semantic_connected": processed_degree > 0,

        "kgsf_semantic_degree": processed_degree,

        "lost_semantic_connectivity": (
            raw_degree > 0 and processed_degree == 0
        )
    })

detail = pd.DataFrame(rows)

detail.to_csv(
    OUT / "raw_vs_kgsf_item_coverage.csv",
    index=False,
    encoding="utf-8-sig"
)

# ==================================================
# 6. GROUP-LEVEL SUMMARY
# ==================================================

def summarize(name, subset):

    n = len(subset)

    if n == 0:
        return

    print(f"\n========== {name} ==========")
    print("Items:", n)

    for label, column in [
        ("Raw graph presence", "raw_graph_present"),
        ("Raw semantic coverage", "raw_semantic_connected"),
        ("KGSF graph presence", "kgsf_graph_present"),
        ("KGSF semantic coverage", "kgsf_semantic_connected"),
        ("Lost semantic connectivity", "lost_semantic_connectivity")
    ]:

        count = int(subset[column].sum())
        pct = 100 * count / n

        print(f"{label}: {count} ({pct:.2f}%)")

    print(
        "Mean raw degree:",
        round(subset["raw_semantic_degree"].mean(), 2)
    )

    print(
        "Mean KGSF degree:",
        round(subset["kgsf_semantic_degree"].mean(), 2)
    )


print("\n========== GLOBAL GRAPH SUMMARY ==========")

print("Raw graph nodes:", len(raw_nodes))
print("Raw connected nodes:", len(raw_connected))

print("KGSF graph nodes:", len(processed_nodes))
print("KGSF connected nodes:", len(processed_connected))

print("KGSF retained relations:", processed_kg["n_relation"])
print("KGSF retained edges:", len(processed_kg["edge"]))

print("\n========== ZERO-SHOT / ONE-SHOT ==========")

summarize("ZERO-SHOT", detail[
    detail["cold_start_type"] == "zero_shot"
])

summarize("ONE-SHOT", detail[
    detail["cold_start_type"] == "one_shot"
])

print("\n========== FREQUENCY GROUPS ==========")

for group in range(5):

    summarize(
        f"G{group}",
        detail[detail["frequency_group"] == group]
    )

# ==================================================
# 7. SAVE GROUP SUMMARY
# ==================================================

summary_rows = []

groups = [
    ("ZERO-SHOT", detail[
        detail["cold_start_type"] == "zero_shot"
    ]),
    ("ONE-SHOT", detail[
        detail["cold_start_type"] == "one_shot"
    ])
]

for group in range(5):
    groups.append((
        f"G{group}",
        detail[detail["frequency_group"] == group]
    ))

for name, subset in groups:

    n = len(subset)

    if n == 0:
        continue

    summary_rows.append({

        "group": name,
        "item_count": n,

        "raw_coverage_pct":
            100 * subset["raw_semantic_connected"].mean(),

        "kgsf_coverage_pct":
            100 * subset["kgsf_semantic_connected"].mean(),

        "mean_raw_degree":
            subset["raw_semantic_degree"].mean(),

        "mean_kgsf_degree":
            subset["kgsf_semantic_degree"].mean(),

        "lost_connectivity_count":
            int(subset["lost_semantic_connectivity"].sum())
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUT / "raw_vs_kgsf_group_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n========== OUTPUT FILES ==========")

print("Item-level:",
      (OUT / "raw_vs_kgsf_item_coverage.csv").resolve())

print("Group-level:",
      (OUT / "raw_vs_kgsf_group_summary.csv").resolve())

print("\nEXPERIMENT 09 COMPLETE")
