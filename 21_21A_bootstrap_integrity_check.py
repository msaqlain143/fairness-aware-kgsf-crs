
import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


# ============================================================
# 1. PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent

RES = ROOT / "results_corrected"

EVENTS = RES / "adaptive_m2_validation_events_clustered.csv"
REPORT = RES / "adaptive_m2_validation.json"

MOVIE_IDS_FILE = (
    ROOT / "data" / "dataset" / "redial" /
    "nltk" / "movie_ids.json"
)

OUTPUT = RES / "m2_bootstrap_integrity.json"

BASE = "A0_fixed_025"
TOP_K = 10

print("=" * 70)
print("M2.1A: BOOTSTRAP INTEGRITY CHECK")
print("=" * 70)


# ============================================================
# 2. LOAD ORIGINAL MOVIE ID POPULATION
# ============================================================

if not MOVIE_IDS_FILE.exists():
    raise FileNotFoundError(MOVIE_IDS_FILE)

with MOVIE_IDS_FILE.open(encoding="utf-8") as f:
    original_movie_ids = json.load(f)

if not isinstance(original_movie_ids, list):
    raise ValueError("Original movie_ids.json must contain a list.")

VALID_MOVIE_IDS = set(map(int, original_movie_ids))

if len(VALID_MOVIE_IDS) != len(original_movie_ids):
    raise ValueError("Duplicate IDs in original movie catalog.")

catalog_size = len(VALID_MOVIE_IDS)

print("\n[1] ORIGINAL CATALOG")
print("Original catalog records:", len(original_movie_ids))
print("Unique movie IDs:", catalog_size)


# ============================================================
# 3. LOAD EXPERIMENT DATA
# ============================================================

if not EVENTS.exists():
    raise FileNotFoundError(EVENTS)

if not REPORT.exists():
    raise FileNotFoundError(REPORT)

with EVENTS.open(
    encoding="utf-8-sig",
    newline=""
) as f:
    rows = list(csv.DictReader(f))

report = json.loads(
    REPORT.read_text(encoding="utf-8")
)

if not rows:
    raise ValueError("Recommendation event dataset is empty.")

required_columns = {
    "method",
    "conv_id",
    "top10_movie_ids"
}

missing_columns = required_columns - set(rows[0].keys())

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

methods = sorted({r["method"] for r in rows})

conversations = sorted({r["conv_id"] for r in rows})

print("\n[2] DATASET")
print("Rows:", len(rows))
print("Methods:", methods)
print("Conversations:", len(conversations))
print("Catalog:", catalog_size)

if len(methods) != 7:
    raise ValueError(
        f"Expected seven experimental methods, found {len(methods)}"
    )

if BASE not in methods:
    raise ValueError(f"Baseline method missing: {BASE}")


# ============================================================
# 4. VALIDATE CATALOG SIZE AGAINST EXPERIMENT REPORT
# ============================================================

reported_catalog_size = int(report["candidate_movies"])

if reported_catalog_size != catalog_size:
    raise ValueError(
        "Catalog population mismatch: "
        f"report={reported_catalog_size}, "
        f"original={catalog_size}"
    )

print("\nCatalog population reconciliation: PASS")


# ============================================================
# 5. RECOMMENDATION EVENT INTEGRITY
# ============================================================

results = {}

method_opportunities = {}

method_event_counts = {}

global_invalid_movies = set()

global_invalid_lists = []

total_slots_all_methods = 0


for method in methods:

    selected = [
        r for r in rows
        if r["method"] == method
    ]

    counts = defaultdict(int)

    opportunities = defaultdict(int)

    invalid_movies = set()

    invalid_lists = []

    event_ids = set()

    method_slots = 0

    for row in selected:

        cid = row["conv_id"]

        opportunities[cid] += 1

        try:
            movies = json.loads(row["top10_movie_ids"])
        except Exception as exc:
            raise ValueError(
                f"Invalid JSON: {method}/{cid}: {exc}"
            )

        if not isinstance(movies, list):
            raise ValueError(
                f"Recommendation list is not a list: {method}/{cid}"
            )

        try:
            movies = [int(x) for x in movies]
        except (ValueError, TypeError):
            raise ValueError(
                f"Non-integer movie ID: {method}/{cid}"
            )

        if len(movies) != TOP_K:
            invalid_lists.append((cid, "INCORRECT_LENGTH"))
            continue

        if len(set(movies)) != TOP_K:
            invalid_lists.append((cid, "DUPLICATE_MOVIES"))
            continue

        event_invalid = [
            movie for movie in movies
            if movie not in VALID_MOVIE_IDS
        ]

        if event_invalid:

            invalid_movies.update(event_invalid)

            continue

        for movie in movies:
            counts[movie] += 1

        method_slots += len(movies)

    if invalid_lists:
        raise ValueError(
            f"Invalid recommendation lists in {method}: "
            f"{invalid_lists[:10]}"
        )

    if invalid_movies:
        raise ValueError(
            f"Movies absent from original catalog in {method}: "
            f"{sorted(invalid_movies)[:20]}"
        )

    unique_exposed = len(counts)

    # Include all catalog items, including zero-exposure items.
    exposure_array = np.array(
        [counts.get(movie, 0) for movie in VALID_MOVIE_IDS],
        dtype=np.int64
    )

    sorted_exposure = np.sort(exposure_array)

    total_exposure = int(exposure_array.sum())

    if total_exposure != method_slots:
        raise ValueError(
            f"Exposure conservation failed: {method}"
        )

    # Standard Gini coefficient.
    n = len(sorted_exposure)

    if total_exposure == 0:
        gini = 0.0
    else:
        gini = float(
            (
                2 * np.dot(
                    np.arange(1, n + 1),
                    sorted_exposure
                )
            ) / (n * total_exposure)
            - (n + 1) / n
        )

    coverage = unique_exposed / catalog_size

    results[method] = {
        "events": len(selected),
        "conversations": len(opportunities),
        "recommendation_slots": method_slots,
        "unique_exposed_movies": unique_exposed,
        "zero_exposure_movies": catalog_size - unique_exposed,
        "catalog_coverage": float(coverage),
        "item_exposure_gini": gini
    }

    method_opportunities[method] = dict(opportunities)

    method_event_counts[method] = len(selected)

    total_slots_all_methods += method_slots

    print(f"\n[{method}]")

    for key, value in results[method].items():
        print(f"{key}: {value}")


# ============================================================
# 6. VERIFY IDENTICAL EVALUATION OPPORTUNITIES
# ============================================================

print("\n[3] OPPORTUNITY CONSISTENCY")

baseline_opportunities = method_opportunities[BASE]

for method in methods:

    if method_opportunities[method] != baseline_opportunities:
        raise ValueError(
            f"Recommendation opportunities differ: {method}"
        )

    if method_event_counts[method] != method_event_counts[BASE]:
        raise ValueError(
            f"Event counts differ: {method}"
        )

print("Identical conversation opportunities: PASS")
print("Identical event counts: PASS")


# ============================================================
# 7. FINAL CONSERVATION CHECKS
# ============================================================

expected_slots_per_method = len(rows) // len(methods)

if len(rows) % len(methods) != 0:
    raise ValueError(
        "Event population is not evenly divisible by methods."
    )

for method in methods:

    if results[method]["recommendation_slots"] != (
        results[method]["events"] * TOP_K
    ):
        raise ValueError(
            f"Recommendation slot conservation failed: {method}"
        )

    if results[method]["events"] != expected_slots_per_method:
        raise ValueError(
            f"Unexpected event count: {method}"
        )

print("\n[4] FINAL VERIFICATION")
print("Original catalog membership: PASS")
print("Valid top-10 lists: PASS")
print("No duplicate recommendations: PASS")
print("Identical conversation opportunities: PASS")
print("Identical event counts: PASS")
print("Recommendation slot conservation: PASS")
print("Zero-exposure items included: PASS")


# ============================================================
# 8. EXPORT VALIDATION RESULTS
# ============================================================

output = {
    "experiment": "M2.1A",
    "split": "validation_only",
    "catalog_size": catalog_size,
    "catalog_source": str(MOVIE_IDS_FILE.relative_to(ROOT)),
    "conversation_count": len(conversations),
    "method_count": len(methods),
    "top_k": TOP_K,
    "total_events": len(rows),
    "total_recommendation_slots_all_methods": (
        total_slots_all_methods
    ),
    "methods": results,
    "checks": {
        "original_catalog_membership": True,
        "valid_top10_lists": True,
        "no_duplicate_recommendations": True,
        "identical_opportunities": True,
        "identical_event_counts": True,
        "slot_conservation": True,
        "zero_exposure_items_included": True
    },
    "identical_opportunities": True,
    "test_split_evaluated": False
}

RES.mkdir(parents=True, exist_ok=True)

OUTPUT.write_text(
    json.dumps(output, indent=4),
    encoding="utf-8"
)

print("\n[5] OUTPUT")
print("Saved:", OUTPUT)
print("DIAGNOSTIC COMPLETED")
