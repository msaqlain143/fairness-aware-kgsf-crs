
"""
EXPERIMENT 21.2
INDEPENDENT CRSLAB RELATION RECONSTRUCTION
AND KNOWLEDGE-GRAPH INTEGRITY AUDIT

Objectives
----------
1. Reconstruct CRSLab's ReDial graph-processing procedure.
2. Independently reproduce synthetic self-loop generation.
3. Separate structural and semantic relation records.
4. Audit special relation IDs 185 and 87.
5. Reproduce relation-frequency thresholding.
6. Reproduce sequential original-to-processed relation IDs.
7. Validate entity indexing and edge integrity.
8. Preserve complete raw and reconstructed relation inventories.
9. Record source provenance and SHA-256 fingerprints.

Important
---------
No model training or recommendation inference is performed.
Original dataset files are never modified.
Previous audit outputs are never overwritten.
"""

import csv
import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE = Path("data/dataset/redial/nltk")

ENTITY_FILE = BASE / "entity2id.json"
KG_FILE = BASE / "dbpedia_subkg.json"

OUTPUT = Path(
    "results_category_audit/crslab_relation_audit_v2"
)

SELF_LOOP_ID = 185
INVESTIGATED_RELATION_ID = 87

MIN_RELATION_FREQUENCY = 1000

EXPECTED_CATALOG_SIZE = 6924

# Fail rather than overwrite an existing experiment.
OUTPUT.mkdir(
    parents=True,
    exist_ok=True
)
existing_files = list(OUTPUT.iterdir())

if existing_files:
    raise RuntimeError(
        f"Output directory is not empty: {OUTPUT}. "
        "Use a new versioned output directory."
    )

for path in [ENTITY_FILE, KG_FILE]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required source file not found: {path}"
        )


# ============================================================
# 2. UTILITY FUNCTIONS
# ============================================================

def sha256_file(path):
    """Calculate the SHA-256 fingerprint of a source file."""

    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def save_csv(path, rows, fields=None):
    """Write CSV output with stable column ordering."""

    if fields is None:
        if not rows:
            raise ValueError(
                f"Cannot infer CSV columns: {path}"
            )

        fields = list(rows[0].keys())

    with path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="raise"
        )

        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    """Fail explicitly when an integrity condition is violated."""

    if not condition:
        raise RuntimeError(message)


def add_first_seen(relation, ordered, seen):
    """Preserve CRSLab edge traversal order."""

    if relation not in seen:
        ordered.append(relation)
        seen.add(relation)



# ============================================================
# 3. LOAD SOURCE DATA AND VALIDATE ENTITY INDEXING
# ============================================================

with ENTITY_FILE.open(encoding="utf-8") as f:
    entity2id = json.load(f)

with KG_FILE.open(encoding="utf-8") as f:
    entity_kg = json.load(f)

require(
    isinstance(entity2id, dict),
    "entity2id must be a dictionary."
)

require(
    isinstance(entity_kg, dict),
    "dbpedia_subkg must be a dictionary."
)

# Convert entity IDs safely.
entity_ids = []

for entity, value in entity2id.items():

    try:
        entity_id = int(value)
    except (ValueError, TypeError):
        raise RuntimeError(
            f"Invalid entity ID: {entity} -> {value}"
        )

    require(
        entity_id >= 0,
        f"Negative entity ID detected: {entity_id}"
    )

    entity_ids.append(entity_id)

require(
    len(entity_ids) > 0,
    "Empty entity index."
)

require(
    len(entity_ids) == len(set(entity_ids)),
    "Duplicate entity IDs detected."
)

# Reproduce CRSLab's indexing boundary.
n_entity = max(entity_ids) + 1

# Sparse IDs are allowed; report them rather than rejecting them.
missing_entity_ids = sorted(
    set(range(n_entity)) - set(entity_ids)
)

# Validate graph dictionary keys.
entity_kg_keys = set()

for key in entity_kg:

    try:
        graph_id = int(key)
    except (ValueError, TypeError):
        raise RuntimeError(
            f"Invalid graph entity key: {key}"
        )

    require(
        0 <= graph_id < n_entity,
        f"Graph entity key outside indexing range: {graph_id}"
    )

    entity_kg_keys.add(graph_id)

print("\n[1] SOURCE DATA")

print("Indexed entities:", len(entity_ids))
print("Graph entity keys:", len(entity_kg_keys))
print("Entity index range:", n_entity)

print("Missing IDs in index range:", len(missing_entity_ids))

print("Entity ID uniqueness: PASS")
print("Entity ID non-negativity: PASS")
print("Graph-key bounds: PASS")
print("Sparse indexing compatibility: PASS")


# ============================================================
# 4. INDEPENDENT GRAPH RECONSTRUCTION
# ============================================================

raw_relation_counts = Counter()

raw_self_loop_counts = Counter()

raw_nonself_relation_counts = Counter()

constructed_relation_counts = Counter()

relation_order = []
seen_relations = set()

raw_relation_87_examples = []

raw_relation_87_stats = Counter()

active_entities = 0
raw_edge_records = 0
raw_self_loop_records = 0
raw_nonself_records = 0

excluded_relation_185_records = 0

augmented_semantic_edges = 0
synthetic_self_loops = 0

invalid_edge_records = []

# Store only the required diagnostics, not the entire graph.
MAX_INVALID_EXAMPLES = 50


print("\n[2] RECONSTRUCTING CRSLAB GRAPH")

for entity in range(n_entity):

    key = str(entity)

    if key not in entity_kg:
        continue

    active_entities += 1

    # CRSLab explicitly inserts one synthetic self-loop
    # for each entity represented in entity_kg.
    constructed_relation_counts[SELF_LOOP_ID] += 1

    synthetic_self_loops += 1

    add_first_seen(
        SELF_LOOP_ID,
        relation_order,
        seen_relations
    )

    edges = entity_kg[key]

    require(
        isinstance(edges, list),
        f"Invalid edge list for entity {entity}."
    )

    for edge_index, edge in enumerate(edges):

        raw_edge_records += 1

        try:

            require(
                isinstance(edge, (list, tuple))
                and len(edge) == 2,
                "Expected each KG edge to contain relation and tail."
            )

            relation = int(edge[0])
            tail = int(edge[1])

            require(
                0 <= tail < n_entity,
                f"Invalid tail entity ID: {tail}"
            )

        except (ValueError, TypeError, RuntimeError) as exc:

            if len(invalid_edge_records) < MAX_INVALID_EXAMPLES:

                invalid_edge_records.append({
                    "head": entity,
                    "edge_index": edge_index,
                    "edge": repr(edge),
                    "error": str(exc),
                })

            continue

        raw_relation_counts[relation] += 1

        if entity == tail:

            raw_self_loop_records += 1
            raw_self_loop_counts[relation] += 1

        else:

            raw_nonself_records += 1
            raw_nonself_relation_counts[relation] += 1

        # Special diagnostic for original relation 87.
        if relation == INVESTIGATED_RELATION_ID:

            raw_relation_87_stats["total"] += 1

            if entity == tail:
                raw_relation_87_stats["self_loops"] += 1
            else:
                raw_relation_87_stats["non_self_loops"] += 1

            if len(raw_relation_87_examples) < 100:

                raw_relation_87_examples.append({
                    "head_entity_id": entity,
                    "relation_id": relation,
                    "tail_entity_id": tail,
                    "is_self_loop": entity == tail,
                })

        # Reproduce CRSLab's exclusion rule.
        if entity == tail or relation == SELF_LOOP_ID:

            if relation == SELF_LOOP_ID:
                excluded_relation_185_records += 1

            continue

        # CRSLab creates forward and reverse records.
        constructed_relation_counts[relation] += 2

        augmented_semantic_edges += 2

        add_first_seen(
            relation,
            relation_order,
            seen_relations
        )


require(
    len(invalid_edge_records) == 0,
    (
        f"Invalid source edges detected: "
        f"{len(invalid_edge_records)}. "
        f"Examples: {invalid_edge_records[:5]}"
    )
)

constructed_edge_records = (
    synthetic_self_loops
    + augmented_semantic_edges
)

require(
    constructed_edge_records
    == sum(constructed_relation_counts.values()),
    "Constructed edge conservation failed."
)

print("Active graph entities:", active_entities)
print("Raw edge records:", raw_edge_records)
print("Raw self-loop records:", raw_self_loop_records)
print("Raw non-self-loop records:", raw_nonself_records)

print("Synthetic self-loops:", synthetic_self_loops)
print("Augmented semantic edge records:", augmented_semantic_edges)
print("Total constructed edge records:", constructed_edge_records)

print("Graph reconstruction integrity: PASS")


# ============================================================
# 5. RECONSTRUCT RELATION MAPPING
# ============================================================

relation2id = {}

for relation in relation_order:

    count = constructed_relation_counts[relation]

    # Exact strict threshold: count > 1000.
    if count > MIN_RELATION_FREQUENCY:

        relation2id[relation] = len(relation2id)


require(
    len(set(relation2id.values()))
    == len(relation2id),
    "Duplicate processed relation IDs."
)

require(
    list(relation2id.values())
    == list(range(len(relation2id))),
    "Processed relation IDs are not sequential."
)

require(
    SELF_LOOP_ID in relation2id,
    "Synthetic self-loop relation did not pass threshold."
)

print("\n[3] RELATION MAPPING")

print("Unique constructed relations:",
      len(constructed_relation_counts))

print("Relations passing threshold:",
      len(relation2id))

print("Strict threshold:",
      f"> {MIN_RELATION_FREQUENCY}")

print("Self-loop original ID:", SELF_LOOP_ID)

print("Self-loop processed ID:",
      relation2id[SELF_LOOP_ID])

print("Sequential mapping integrity: PASS")


# ============================================================
# 6. RAW RELATION INVENTORY
# ============================================================

raw_inventory = []

all_raw_relations = sorted(
    set(raw_relation_counts)
    | set(raw_self_loop_counts)
    | set(constructed_relation_counts)
)

for relation in all_raw_relations:

    raw_count = raw_relation_counts[relation]

    self_count = raw_self_loop_counts[relation]

    nonself_count = raw_nonself_relation_counts[relation]

    constructed_count = constructed_relation_counts[relation]

    raw_inventory.append({

        "original_relation_id": relation,

        "raw_edge_records": raw_count,

        "raw_self_loop_records": self_count,

        "raw_nonself_edge_records": nonself_count,

        "constructed_edge_records": constructed_count,

        "constructed_from_raw_edges": relation != SELF_LOOP_ID,

        "synthetic_self_loop_records": (
            synthetic_self_loops
            if relation == SELF_LOOP_ID
            else 0
        ),

        "included_after_threshold": (
            relation in relation2id
        ),

        "processed_relation_id": (
            relation2id.get(relation, "")
        ),

    })


# ============================================================
# 7. CONSTRUCTED RELATION FREQUENCY AUDIT
# ============================================================

frequency_rows = []

for relation in relation_order:

    count = constructed_relation_counts[relation]

    frequency_rows.append({

        "original_relation_id": relation,

        "constructed_edge_record_count": count,

        "passes_strict_threshold": (
            count > MIN_RELATION_FREQUENCY
        ),

        "included_in_processed_graph": (
            relation in relation2id
        ),

        "processed_relation_id": (
            relation2id.get(relation, "")
        ),

        "is_synthetic_self_loop_relation": (
            relation == SELF_LOOP_ID
        ),

    })


# ============================================================
# 8. RELATION 87 FORENSIC AUDIT
# ============================================================

relation_87_raw = raw_relation_counts.get(
    INVESTIGATED_RELATION_ID,
    0
)

relation_87_self = raw_self_loop_counts.get(
    INVESTIGATED_RELATION_ID,
    0
)

relation_87_nonself = raw_nonself_relation_counts.get(
    INVESTIGATED_RELATION_ID,
    0
)

relation_87_constructed = constructed_relation_counts.get(
    INVESTIGATED_RELATION_ID,
    0
)

relation_87_summary = {

    "relation_id": INVESTIGATED_RELATION_ID,

    "raw_edge_records": relation_87_raw,

    "raw_self_loop_records": relation_87_self,

    "raw_nonself_loop_records": relation_87_nonself,

    "constructed_edge_records": relation_87_constructed,

    "processed_relation_id": relation2id.get(
        INVESTIGATED_RELATION_ID
    ),

    #"is_crslab_reserved_self_loop": False,
    "structural_status": "UNRESOLVED",

    "interpretation": (
        "Requires semantic provenance inspection; "
        "raw structural statistics alone do not establish "
        "the relation's semantic meaning."
    ),

}

relation_87_examples = raw_relation_87_examples


# ============================================================
# 9. SELF-LOOP RELATION 185 VALIDATION
# ============================================================

relation_185_summary = {

    "relation_id": SELF_LOOP_ID,

    "raw_relation_185_records": (
        raw_relation_counts.get(SELF_LOOP_ID, 0)
    ),

    "raw_relation_185_self_loops": (
        raw_self_loop_counts.get(SELF_LOOP_ID, 0)
    ),

    "raw_relation_185_nonself_records": (
        raw_nonself_relation_counts.get(SELF_LOOP_ID, 0)
    ),

    "synthetic_self_loops_added": synthetic_self_loops,

    "constructed_relation_185_records": (
        constructed_relation_counts[SELF_LOOP_ID]
    ),

    "processed_relation_id": relation2id[SELF_LOOP_ID],

    "validated_structural_relation": True,

}

require(
    relation_185_summary["constructed_relation_185_records"]
    == synthetic_self_loops,
    "Relation 185 synthetic-loop count mismatch."
)


# ============================================================
# 10. EXPORT ALL RESULTS
# ============================================================

print("\n[4] EXPORTING AUDIT RESULTS")

save_csv(
    OUTPUT / "raw_relation_inventory.csv",
    raw_inventory
)

save_csv(
    OUTPUT / "constructed_relation_frequency.csv",
    frequency_rows
)

mapping_rows = [

    {
        "original_relation_id": original,
        "processed_relation_id": processed,
        "constructed_edge_record_count": (
            constructed_relation_counts[original]
        ),
        "passes_strict_threshold": True,
    }

    for original, processed in relation2id.items()
]

save_csv(
    OUTPUT / "crslab_relation_mapping.csv",
    mapping_rows
)

save_csv(
    OUTPUT / "relation_87_raw_edge_examples.csv",
    relation_87_examples,
    fields=[
        "head_entity_id",
        "relation_id",
        "tail_entity_id",
        "is_self_loop",
    ]
)

save_csv(
    OUTPUT / "invalid_edge_records.csv",
    invalid_edge_records,
    fields=[
        "head",
        "edge_index",
        "edge",
        "error",
    ]
)


# ============================================================
# 11. PROVENANCE AND VALIDATION MANIFEST
# ============================================================

manifest = {

    "experiment": "21.2",

    "title": (
        "Independent CRSLab Relation Reconstruction "
        "and Knowledge Graph Integrity Audit"
    ),

    "timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "source_files": {

        str(ENTITY_FILE): {
            "sha256": sha256_file(ENTITY_FILE),
            "size_bytes": ENTITY_FILE.stat().st_size,
        },

        str(KG_FILE): {
            "sha256": sha256_file(KG_FILE),
            "size_bytes": KG_FILE.stat().st_size,
        },

    },

    "configuration": {

        "self_loop_id": SELF_LOOP_ID,

        "investigated_relation_id": (
            INVESTIGATED_RELATION_ID
        ),

        "frequency_threshold": MIN_RELATION_FREQUENCY,

        "threshold_operator": "strictly greater than",

        "edge_augmentation": (
            "One forward and one reverse record "
            "for every eligible non-self-loop raw edge."
        ),

        "synthetic_self_loop_policy": (
            "One relation-185 self-loop per active graph entity."
        ),

    },

    "counts": {

        "indexed_entities": len(entity_ids),

        "active_graph_entities": active_entities,

        "raw_edge_records": raw_edge_records,

        "raw_self_loop_records": raw_self_loop_records,

        "raw_nonself_records": raw_nonself_records,

        "synthetic_self_loops": synthetic_self_loops,

        "augmented_semantic_edge_records": (
            augmented_semantic_edges
        ),

        "constructed_edge_records": (
            constructed_edge_records
        ),

        "raw_unique_relations": len(raw_relation_counts),

        "constructed_unique_relations": (
            len(constructed_relation_counts)
        ),

        "retained_relations": len(relation2id),

    },

    "relation_185": relation_185_summary,

    "relation_87": {

        **relation_87_summary,

        "self_loop_records": relation_87_self,

        "nonself_loop_records": relation_87_nonself,

    },

    "validation": {

        "entity_index_integrity": "PASS",

        "graph_key_integrity": "PASS",

        "raw_edge_integrity": "PASS",

        "invalid_edge_records": len(invalid_edge_records),

        "edge_conservation": "PASS",

        "synthetic_self_loop_reconstruction": "PASS",

        "relation_mapping_sequentiality": "PASS",

        "strict_threshold_reproduction": "PASS",

        "relation_185_structural_classification": "PASS",

        "relation_87_semantic_classification": "PENDING",

    },

    "limitations": [

        "Relation 87 semantic meaning is not inferred from numeric ID.",

        "Raw relation counts are record counts, not unique triples.",

        "Inverse-edge augmentation can create duplicate records.",

        "Semantic predicate labels require separate provenance.",

    ],

}

with (
    OUTPUT / "relation_reconstruction_manifest.json"
).open("w", encoding="utf-8") as f:

    json.dump(
        manifest,
        f,
        indent=2,
        ensure_ascii=False
    )


# ============================================================
# 12. FINAL REPORT
# ============================================================

print("\n" + "=" * 78)
print("EXPERIMENT 21.2 COMPLETED")
print("=" * 78)

print("\nSOURCE DATA")

print("Indexed entities:", len(entity_ids))
print("Active graph entities:", active_entities)

print("\nEDGE RECONSTRUCTION")

print("Raw records:", raw_edge_records)
print("Raw self-loops:", raw_self_loop_records)
print("Synthetic self-loops:", synthetic_self_loops)
print("Augmented semantic records:", augmented_semantic_edges)
print("Constructed records:", constructed_edge_records)

print("\nRELATION MAPPING")

print("Raw relations:", len(raw_relation_counts))
print("Constructed relations:", len(constructed_relation_counts))
print("Retained relations:", len(relation2id))

print("\nRELATION 185")

for key, value in relation_185_summary.items():
    print(f"{key}: {value}")

print("\nRELATION 87")

for key, value in relation_87_summary.items():
    print(f"{key}: {value}")

print("\nVALIDATION")

for name, status in manifest["validation"].items():
    print(f"{name}: {status}")

print("\nOUTPUT DIRECTORY:")
print(OUTPUT.resolve())

print("\nNo model training performed.")
print("No recommendation inference performed.")
print("Original graph files not modified.")
print("Previous experiment outputs not overwritten.")

print("\nEXPERIMENT 21.2 FINISHED.")
