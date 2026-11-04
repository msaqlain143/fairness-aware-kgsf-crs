
"""
EXPERIMENT 14: INDEPENDENT EXPOSURE METRIC INTEGRITY AUDIT

Purpose:
1. Independently load and validate the frozen catalog.
2. Validate all 4,003 held-out test events.
3. Validate all six recommendation methods.
4. Check top-10 recommendation integrity.
5. Reconstruct full-catalog exposure counts.
6. Independently calculate Coverage@10 and Item Gini.
7. Validate popularity-group exposure shares.
8. Compare reconstructed metrics with frozen test results.
9. Save a reproducible audit report.

No model training or inference is performed.
No existing experiment outputs are modified.
"""

import csv
import json
import pickle
from pathlib import Path

import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path("results_final_test_m2")

CATALOG_FILE = Path("catalog_audit_corrected.pkl")

EVENTS_FILE = (
    ROOT / "adaptive_m2_with_kgsf_baseline_events.csv"
)

MAPPING_FILE = (
    ROOT / "m2_test_event_conversation_mapping.csv"
)

REFERENCE_FILE = (
    ROOT / "adaptive_m2_with_kgsf_baseline.json"
)

OUT = Path("results_exposure_integrity")

METHODS = [
    "KGSF_only",
    "A0_fixed_025",
    "A1_frequency_1",
    "A2_frequency_context_1",
    "A1_frequency_3",
    "A2_frequency_context_3",
]

EXPECTED_CATALOG = 6924
EXPECTED_EVENTS = 4003
EXPECTED_CONVERSATIONS = 1283
TOP_K = 10
N_GROUPS = 5

ATOL = 1e-9


# ============================================================
# 2. UTILITIES
# ============================================================

def read_csv(path):

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        encoding="utf-8-sig",
        newline=""
    ) as f:

        return list(csv.DictReader(f))


def normalized_value(value):

    if value is None or str(value).strip() == "":
        return None

    try:
        number = float(value)

        if not np.isfinite(number):
            return None

        return number

    except (ValueError, TypeError):

        return str(value).strip()


def calculate_gini(exposure):

    """
    Standard Gini coefficient over the complete catalog,
    including all zero-exposure items.
    """

    exposure = np.asarray(
        exposure,
        dtype=np.float64
    )

    if exposure.ndim != 1:
        raise ValueError("Exposure must be one-dimensional.")

    if np.any(exposure < 0):
        raise ValueError("Negative exposure detected.")

    total = float(exposure.sum())

    if total <= 0:
        raise ValueError("Total exposure must be positive.")

    ordered = np.sort(exposure)

    n = len(ordered)

    coefficients = (
        2 * np.arange(1, n + 1)
        - n - 1
    )

    return float(
        np.dot(coefficients, ordered)
        / (n * total)
    )


def assert_close(name, actual, expected):

    if not np.isclose(
        actual,
        expected,
        rtol=0,
        atol=ATOL
    ):

        raise RuntimeError(
            f"{name} mismatch: "
            f"actual={actual}, expected={expected}"
        )


def save_csv(path, rows):

    if not rows:
        raise RuntimeError("Cannot save an empty CSV.")

    with path.open(
        "x",
        encoding="utf-8-sig",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# 3. INITIAL VALIDATION
# ============================================================

print("=" * 72)
print("EXPERIMENT 14: EXPOSURE METRIC INTEGRITY AUDIT")
print("=" * 72)

for path in (
    CATALOG_FILE,
    EVENTS_FILE,
    MAPPING_FILE,
    REFERENCE_FILE
):

    if not path.exists():
        raise FileNotFoundError(
            f"Required input file missing: {path}"
        )

if OUT.exists():

    raise FileExistsError(
        f"Output directory already exists: {OUT}. "
        "Archive it or select a new output directory."
    )


# ============================================================
# 4. CATALOG VALIDATION
# ============================================================

print("\n========== 1. CATALOG VALIDATION ==========")

with CATALOG_FILE.open("rb") as f:

    catalog = pickle.load(f)

if "movie_ids" not in catalog:

    raise RuntimeError(
        "movie_ids missing from catalog."
    )

catalog_ids = list(
    map(int, catalog["movie_ids"])
)

catalog_set = set(catalog_ids)

if len(catalog_ids) != EXPECTED_CATALOG:

    raise RuntimeError(
        f"Expected {EXPECTED_CATALOG} catalog items, "
        f"found {len(catalog_ids)}."
    )

if len(catalog_set) != EXPECTED_CATALOG:

    raise RuntimeError(
        "Duplicate movie IDs in catalog."
    )

N_CATALOG = len(catalog_ids)

movie_index = {
    movie: index
    for index, movie in enumerate(catalog_ids)
}

print("Catalog items:", N_CATALOG)
print("Unique catalog IDs:", len(catalog_set))
print("Catalog integrity: PASS")


# ============================================================
# 5. LOAD FROZEN REFERENCE RESULTS
# ============================================================

print("\n========== 2. REFERENCE VALIDATION ==========")

with REFERENCE_FILE.open(
    encoding="utf-8"
) as f:

    reference = json.load(f)

if int(reference["events"]) != EXPECTED_EVENTS:

    raise RuntimeError(
        "Reference event count mismatch."
    )

if int(reference["candidate_movies"]) != EXPECTED_CATALOG:

    raise RuntimeError(
        "Reference catalog size mismatch."
    )

reference_methods = {
    item["method"]: item
    for item in reference["results"]
}

if set(reference_methods) != set(METHODS):

    raise RuntimeError(
        "Reference method set does not match."
    )

print("Reference events:", reference["events"])
print("Reference catalog:", reference["candidate_movies"])
print("Reference methods:", len(reference_methods))
print("Reference integrity: PASS")


# ============================================================
# 6. EVENT-CONVERSATION MAPPING
# ============================================================

print("\n========== 3. EVENT MAPPING VALIDATION ==========")

mapping_rows = read_csv(MAPPING_FILE)

event_to_conv = {}
mapping_targets = {}

for row in mapping_rows:

    eid = int(row["event_id"])
    cid = int(row["conv_id"])
    target = int(row["target_movie_id"])

    if eid in event_to_conv:

        raise RuntimeError(
            f"Duplicate event mapping: {eid}"
        )

    if target not in catalog_set:

        raise RuntimeError(
            f"Target outside catalog: {target}"
        )

    event_to_conv[eid] = cid
    mapping_targets[eid] = target


event_ids = sorted(event_to_conv)

conv_ids = sorted(
    set(event_to_conv.values())
)

if len(event_ids) != EXPECTED_EVENTS:

    raise RuntimeError("Unexpected mapping event count.")

if len(conv_ids) != EXPECTED_CONVERSATIONS:

    raise RuntimeError("Unexpected conversation count.")

if event_ids != list(
    range(1, EXPECTED_EVENTS + 1)
):

    raise RuntimeError("Event IDs are not contiguous.")

print("Mapping rows:", len(mapping_rows))
print("Unique events:", len(event_ids))
print("Unique conversations:", len(conv_ids))
print("Mapping integrity: PASS")


# ============================================================
# 7. LOAD AND VALIDATE EVENT DATA
# ============================================================

print("\n========== 4. EVENT DATA VALIDATION ==========")

rows = read_csv(EVENTS_FILE)

METADATA_FIELDS = [
    "target_movie_id",
    "target_frequency",
    "frequency_group",
    "target_context_overlap",
    "context_nonempty",
]

records = {}
event_metadata = {}

for row in rows:

    eid = int(row["event_id"])
    method = row["method"]

    if method not in METHODS:

        raise RuntimeError(
            f"Unexpected method: {method}"
        )

    if eid not in event_to_conv:

        raise RuntimeError(
            f"Unmapped event: {eid}"
        )

    key = (eid, method)

    if key in records:

        raise RuntimeError(
            f"Duplicate event-method record: {key}"
        )

    target = int(row["target_movie_id"])

    if target != mapping_targets[eid]:

        raise RuntimeError(
            f"Target mismatch for event {eid}."
        )

    metadata = tuple(
        normalized_value(row[field])
        for field in METADATA_FIELDS
    )

    if eid in event_metadata:

        if event_metadata[eid] != metadata:

            raise RuntimeError(
                f"Metadata mismatch for event {eid}."
            )

    else:

        event_metadata[eid] = metadata

    movies = list(
        map(
            int,
            json.loads(row["top10_movie_ids"])
        )
    )

    groups = list(
        map(
            int,
            json.loads(row["top10_group_ids"])
        )
    )

    if len(movies) != TOP_K:

        raise RuntimeError(
            f"Invalid top-10 length: {key}"
        )

    if len(groups) != TOP_K:

        raise RuntimeError(
            f"Invalid group-list length: {key}"
        )

    if len(set(movies)) != TOP_K:

        raise RuntimeError(
            f"Duplicate recommendation: {key}"
        )

    if not set(movies).issubset(catalog_set):

        raise RuntimeError(
            f"Out-of-catalog recommendation: {key}"
        )

    if any(
        group not in range(N_GROUPS)
        for group in groups
    ):

        raise RuntimeError(
            f"Invalid group assignment: {key}"
        )

    records[key] = {
        "movies": movies,
        "groups": groups,
    }


expected_keys = {
    (eid, method)
    for eid in event_ids
    for method in METHODS
}

if set(records) != expected_keys:

    missing = expected_keys - set(records)
    extra = set(records) - expected_keys

    raise RuntimeError(
        f"Event-method mismatch. "
        f"Missing={len(missing)}, Extra={len(extra)}"
    )


# ============================================================
# 8. MOVIE-TO-GROUP CONSISTENCY
# ============================================================

print("\n========== 5. MOVIE-GROUP VALIDATION ==========")

movie_groups = {}

for record in records.values():

    for movie, group in zip(
        record["movies"],
        record["groups"]
    ):

        if movie in movie_groups:

            if movie_groups[movie] != group:

                raise RuntimeError(
                    f"Inconsistent group for movie {movie}."
                )

        else:

            movie_groups[movie] = group

print("Event-method records:", len(records))
print("Expected records:", len(expected_keys))
print("Movies exposed:", len(movie_groups))
print("Movie-group consistency: PASS")


# ============================================================
# 9. INDEPENDENT EXPOSURE RECONSTRUCTION
# ============================================================

print("\n========== 6. EXPOSURE RECONSTRUCTION ==========")

results = []
group_results = []

for method in METHODS:

    print(f"\nProcessing: {method}")

    exposure = np.zeros(
        N_CATALOG,
        dtype=np.int64
    )

    group_exposure = np.zeros(
        N_GROUPS,
        dtype=np.int64
    )

    total_events = 0
    duplicate_rows = 0

    for eid in event_ids:

        record = records[(eid, method)]

        movies = record["movies"]
        groups = record["groups"]

        if len(set(movies)) != TOP_K:

            duplicate_rows += 1

            raise RuntimeError(
                f"Duplicate recommendations in {method}, event {eid}."
            )

        indices = [
            movie_index[movie]
            for movie in movies
        ]

        np.add.at(exposure, indices, 1)

        np.add.at(
            group_exposure,
            groups,
            1
        )

        total_events += 1

    # Exposure conservation
    total_slots = int(exposure.sum())

    expected_slots = total_events * TOP_K

    if total_slots != expected_slots:

        raise RuntimeError(
            f"Exposure conservation failed: {method}"
        )

    if int(group_exposure.sum()) != total_slots:

        raise RuntimeError(
            f"Group exposure conservation failed: {method}"
        )

    # Full-catalog metrics, including zero-exposure items.
    unique_exposed = int(
        np.count_nonzero(exposure)
    )

    coverage = unique_exposed / N_CATALOG

    gini = calculate_gini(exposure)

    group_shares = (
        group_exposure / total_slots
    )

    if not np.isclose(
        group_shares.sum(),
        1.0,
        atol=ATOL
    ):

        raise RuntimeError(
            f"Group shares do not sum to one: {method}"
        )

    if not 0 <= coverage <= 1:

        raise RuntimeError(
            f"Invalid coverage: {method}"
        )

    if not 0 <= gini < 1:

        raise RuntimeError(
            f"Invalid Gini: {method}"
        )

    saved = reference_methods[method]

    # Compare with frozen final-test results.
    assert_close(
        f"{method} Coverage@10",
        coverage,
        float(saved["catalog_coverage10"])
    )

    assert_close(
        f"{method} Item Gini",
        gini,
        float(saved["item_gini"])
    )

    saved_groups = np.asarray(
        saved["group_exposure_shares"],
        dtype=float
    )

    if not np.allclose(
        group_shares,
        saved_groups,
        atol=ATOL,
        rtol=0
    ):

        raise RuntimeError(
            f"Group exposure mismatch: {method}"
        )

    results.append({

        "method": method,
        "events": total_events,
        "catalog_size": N_CATALOG,
        "total_exposure_slots": total_slots,
        "unique_exposed_items": unique_exposed,
        "coverage10_recomputed": float(coverage),
        "coverage10_reference": float(
            saved["catalog_coverage10"]
        ),
        "coverage_difference": float(
            coverage - saved["catalog_coverage10"]
        ),
        "item_gini_recomputed": float(gini),
        "item_gini_reference": float(
            saved["item_gini"]
        ),
        "gini_difference": float(
            gini - saved["item_gini"]
        ),
        "duplicate_rows": duplicate_rows,
        "exposure_conservation": "PASS",
        "coverage_validation": "PASS",
        "gini_validation": "PASS",
        "group_validation": "PASS",
    })

    for group in range(N_GROUPS):

        group_results.append({

            "method": method,
            "group": group,
            "exposure_slots": int(
                group_exposure[group]
            ),
            "exposure_share": float(
                group_shares[group]
            ),
        })

    print(
        f"Events: {total_events} | "
        f"Slots: {total_slots} | "
        f"Unique items: {unique_exposed}"
    )

    print(
        f"Coverage@10: {coverage:.9f} | "
        f"Gini: {gini:.9f}"
    )

    print(
        "Reference comparison: PASS"
    )


# ============================================================
# 10. FINAL AUDIT REPORT
# ============================================================

print("\n" + "=" * 72)
print("EXPERIMENT 14: FINAL VALIDATION")
print("=" * 72)

if len(results) != len(METHODS):

    raise RuntimeError(
        "Not all methods passed validation."
    )

for result in results:

    print(
        f"{result['method']:25s} "
        f"Coverage={result['coverage10_recomputed']:.6f} "
        f"Gini={result['item_gini_recomputed']:.6f} "
        f"PASS"
    )


# ============================================================
# 11. SAVE RESULTS
# ============================================================

OUT.mkdir(parents=True, exist_ok=False)

save_csv(
    OUT / "exposure_integrity_results.csv",
    results
)

save_csv(
    OUT / "group_exposure_integrity.csv",
    group_results
)

summary = {

    "experiment": 14,

    "title": (
        "Independent Exposure Metric Integrity Audit"
    ),

    "split": "held_out_test",

    "catalog_size": N_CATALOG,

    "events": EXPECTED_EVENTS,

    "conversations": EXPECTED_CONVERSATIONS,

    "methods": METHODS,

    "top_k": TOP_K,

    "gini_definition": (
        "Standard population Gini over the complete "
        "eligible catalog, including zero-exposure items."
    ),

    "reference_file": str(REFERENCE_FILE),

    "validation": {

        "catalog_integrity": "PASS",

        "event_mapping": "PASS",

        "event_method_completeness": "PASS",

        "target_consistency": "PASS",

        "top10_uniqueness": "PASS",

        "catalog_membership": "PASS",

        "movie_group_consistency": "PASS",

        "exposure_conservation": "PASS",

        "coverage_reproduction": "PASS",

        "gini_reproduction": "PASS",

        "group_exposure_reproduction": "PASS",
    },

    "method_results": results,

    "group_results": group_results,

    "model_training_performed": False,

    "model_inference_performed": False,

    "existing_results_modified": False,
}

with (
    OUT / "exposure_integrity_summary.json"
).open("x", encoding="utf-8") as f:

    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# 12. COMPLETION
# ============================================================

print("\n" + "=" * 72)
print("EXPERIMENT 14 COMPLETED SUCCESSFULLY")
print("=" * 72)

print("Catalog:", N_CATALOG)
print("Events:", EXPECTED_EVENTS)
print("Conversations:", EXPECTED_CONVERSATIONS)
print("Methods:", len(METHODS))

print("\nValidation:")
print("Catalog integrity: PASS")
print("Event integrity: PASS")
print("Recommendation integrity: PASS")
print("Movie-group consistency: PASS")
print("Exposure conservation: PASS")
print("Coverage reproduction: PASS")
print("Gini reproduction: PASS")
print("Group exposure reproduction: PASS")

print("\nOutput directory:", OUT.resolve())

for path in sorted(OUT.iterdir()):

    print(
        f"{path.name}: {path.stat().st_size:,} bytes"
    )

print("\nNo model training or inference performed.")
print("No previous experiment outputs modified.")
print("EXPERIMENT 14 COMPLETE.")
