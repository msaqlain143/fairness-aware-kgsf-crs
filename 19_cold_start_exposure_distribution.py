"""
EXPERIMENT 19: COLD-START ITEM EXPOSURE DISTRIBUTION AND INEQUALITY

Descriptive audit of frozen held-out-test recommendations.
No model training or recommendation inference is performed.

Run from the repository root:
    python -m py_compile .\\19_cold_start_exposure_distribution.py
    python .\\19_cold_start_exposure_distribution.py

The script intentionally refuses to overwrite an existing output directory.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import pickle
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


# ============================================================================
# 1. CONFIGURATION
# ============================================================================

ROOT = Path("results_final_test_m2")
CATALOG_FILE = Path("catalog_audit_corrected.pkl")
EVENTS_FILE = ROOT / "adaptive_m2_with_kgsf_baseline_events.csv"

# New output version: preserves v1/v2 and avoids accidental overwrites.
OUT = Path("results_cold_start_exposure_v3")

METHODS = [
    "KGSF_only",
    "A0_fixed_025",
    "A1_frequency_1",
    "A2_frequency_context_1",
    "A1_frequency_3",
    "A2_frequency_context_3",
]

EXPECTED_EVENTS = 4003
EXPECTED_CATALOG = 6924
TOP_K = 10
ATOL = 1e-12

# The project event export has these target metadata fields. Keep this True
# unless you intentionally use a different, documented event-file schema.
VALIDATE_TARGET_METADATA = True

REQUIRED_EVENT_COLUMNS = {
    "event_id", "method", "top10_movie_ids"
}
TARGET_METADATA_COLUMNS = {
    "target_movie_id", "target_frequency", "frequency_group"
}


# ============================================================================
# 2. UTILITIES
# ============================================================================

def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(path: Path, records: list[dict[str, Any]]) -> None:
    require(bool(records), f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(records[0].keys()),
            extrasaction="raise",
        )
        writer.writeheader()
        writer.writerows(records)


def parse_integer_list(value: str, context: str) -> list[int]:
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(
            f"Invalid JSON recommendation list at {context}: {exc}"
        ) from exc

    require(isinstance(parsed, list),
            f"Recommendation field is not a list at {context}.")

    result = []
    for item in parsed:
        require(
            isinstance(item, int) and not isinstance(item, bool),
            f"Non-integer recommendation ID {item!r} at {context}.",
        )
        result.append(item)
    return result


def calculate_gini(values: np.ndarray) -> float:
    """
    Gini over the complete item population, including zero-exposure items.
    """
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and values.size > 0,
            "Gini input must be a non-empty one-dimensional array.")
    require(np.all(np.isfinite(values)) and np.all(values >= 0),
            "Invalid values supplied to Gini calculation.")

    total = float(values.sum())
    if total == 0:
        return 0.0

    ordered = np.sort(values)
    n = ordered.size
    coefficients = 2.0 * np.arange(1, n + 1) - n - 1.0
    result = float(np.dot(coefficients, ordered) / (n * total))

    require(-ATOL <= result <= 1.0 + ATOL,
            f"Gini outside valid range: {result}")
    return min(1.0, max(0.0, result))


def concentration_metrics(counts: np.ndarray) -> dict[str, float]:
    """
    HHI and top-share concentration are computed over the complete group,
    including items with zero exposure. Top-p shares use ceil(p * N) items.
    """
    counts = np.asarray(counts, dtype=np.float64)
    require(counts.ndim == 1 and counts.size > 0,
            "Concentration input must be a non-empty 1-D array.")
    require(np.all(np.isfinite(counts)) and np.all(counts >= 0),
            "Invalid values supplied to concentration calculation.")

    total = float(counts.sum())
    n_items = counts.size

    if total == 0:
        return {
            "hhi": 0.0,
            "effective_items": 0.0,
            "top_1pct_exposure_share": 0.0,
            "top_10pct_exposure_share": 0.0,
        }

    shares = counts / total
    descending = np.sort(counts)[::-1]
    top_1_n = max(1, math.ceil(n_items * 0.01))
    top_10_n = max(1, math.ceil(n_items * 0.10))
    hhi = float(np.dot(shares, shares))

    return {
        "hhi": hhi,
        "effective_items": float(1.0 / hhi) if hhi > 0 else 0.0,
        "top_1pct_exposure_share": float(
            descending[:top_1_n].sum() / total
        ),
        "top_10pct_exposure_share": float(
            descending[:top_10_n].sum() / total
        ),
    }


# ============================================================================
# 3. INPUT FILE AND OUTPUT PATH CHECKS
# ============================================================================

print("=" * 78)
print("EXPERIMENT 19: COLD-START EXPOSURE DISTRIBUTION AND INEQUALITY")
print("=" * 78)

require(CATALOG_FILE.is_file(), f"Catalog file not found: {CATALOG_FILE}")
require(EVENTS_FILE.is_file(), f"Event file not found: {EVENTS_FILE}")
require(
    not OUT.exists(),
    f"Output directory already exists: {OUT}. Change OUT to a new version.",
)

input_hashes = {
    "catalog_file_sha256": sha256_file(CATALOG_FILE),
    "events_file_sha256": sha256_file(EVENTS_FILE),
}


# ============================================================================
# 4. LOAD AND VALIDATE CANONICAL CATALOG
# ============================================================================

# Only unpickle project artifacts from trusted sources.
with CATALOG_FILE.open("rb") as handle:
    catalog = pickle.load(handle)

require(isinstance(catalog, dict), "Catalog artifact must be a dictionary.")

required_catalog_keys = {
    "movie_ids", "freq_dict", "zero_shot_ids", "one_shot_ids",
    "item2group", "tau",
}
missing = required_catalog_keys - set(catalog)
require(not missing, f"Catalog is missing keys: {sorted(missing)}")

catalog_id_list = [int(item) for item in catalog["movie_ids"]]
movie_ids = set(catalog_id_list)

require(len(catalog_id_list) == len(movie_ids),
        "Duplicate IDs in canonical catalog.")
require(len(movie_ids) == EXPECTED_CATALOG,
        f"Expected {EXPECTED_CATALOG} catalog items; found {len(movie_ids)}.")

freq_dict = {
    int(item): int(freq)
    for item, freq in catalog["freq_dict"].items()
}
zero_shot = set(map(int, catalog["zero_shot_ids"]))
one_shot = set(map(int, catalog["one_shot_ids"]))
item2group = {
    int(item): int(group)
    for item, group in catalog["item2group"].items()
}
tau = int(catalog["tau"])

require(set(freq_dict) == movie_ids,
        "Frequency dictionary keys do not match catalog IDs.")
require(all(freq >= 0 for freq in freq_dict.values()),
        "Negative training frequency found.")
require(zero_shot.isdisjoint(one_shot),
        "Zero-shot and one-shot populations overlap.")
require((zero_shot | one_shot) <= movie_ids,
        "Cold-start set contains IDs outside the catalog.")
require(
    zero_shot == {item for item, freq in freq_dict.items() if freq == 0},
    "Zero-shot IDs do not match frequency==0.",
)
require(
    one_shot == {item for item, freq in freq_dict.items() if freq == 1},
    "One-shot IDs do not match frequency==1.",
)
require(set(item2group) == movie_ids,
        "Canonical group-map keys do not match catalog IDs.")
require(set(item2group.values()) == {0, 1, 2, 3, 4},
        "Canonical group map must contain groups 0 through 4.")
require(tau == 1, f"Unexpected tau={tau}; expected tau=1.")

print("\n[1] CANONICAL CATALOG")
print(f"Catalog items: {len(movie_ids)}")
print(f"Zero-shot items: {len(zero_shot)}")
print(f"One-shot items: {len(one_shot)}")
print(f"Cold-start threshold: {tau}")
print("Catalog, frequency, cold-start and group-map integrity: PASS")


# ============================================================================
# 5. LOAD AND VALIDATE EVENT / RECOMMENDATION DATA
# ============================================================================

with EVENTS_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
    reader = csv.DictReader(handle)
    columns = set(reader.fieldnames or [])

    missing_columns = REQUIRED_EVENT_COLUMNS - columns
    require(not missing_columns,
            f"Event CSV missing columns: {sorted(missing_columns)}")

    if VALIDATE_TARGET_METADATA:
        missing_target_columns = TARGET_METADATA_COLUMNS - columns
        require(
            not missing_target_columns,
            "Target metadata validation enabled but columns are missing: "
            f"{sorted(missing_target_columns)}",
        )

    rows = list(reader)

expected_rows = EXPECTED_EVENTS * len(METHODS)
require(len(rows) == expected_rows,
        f"Expected {expected_rows} rows; found {len(rows)}.")

method_set = set(METHODS)
method_counts: Counter[str] = Counter()
event_methods: set[tuple[int, str]] = set()
event_target_metadata: dict[int, tuple[int, int, str]] = {}

# Key names here intentionally match the later analysis loop.
exposure: dict[str, dict[str, Counter[int]]] = {
    method: {
        "zero_shot": Counter(),
        "one_shot": Counter(),
    }
    for method in METHODS
}

for row_number, row in enumerate(rows, start=2):
    context = f"CSV row {row_number}"

    try:
        event_id = int(row["event_id"])
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"Invalid event_id at {context}.") from exc

    method = str(row["method"]).strip()
    require(method in method_set,
            f"Unknown method {method!r} at {context}.")
    require(1 <= event_id <= EXPECTED_EVENTS,
            f"Event ID out of range at {context}: {event_id}")

    key = (event_id, method)
    require(key not in event_methods,
            f"Duplicate event-method record: {key}")
    event_methods.add(key)
    method_counts[method] += 1

    if VALIDATE_TARGET_METADATA:
        try:
            target_id = int(row["target_movie_id"])
            target_frequency = int(row["target_frequency"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Invalid target metadata at {context}.") from exc

        require(target_id in movie_ids,
                f"Target ID {target_id} outside catalog at {context}.")

        canonical_frequency = freq_dict[target_id]
        require(
            target_frequency == canonical_frequency,
            f"Target frequency mismatch at {context}: "
            f"event={target_frequency}, catalog={canonical_frequency}",
        )

        if canonical_frequency == 0:
            expected_group = "zero_shot"
        elif canonical_frequency == 1:
            expected_group = "one_shot"
        else:
            expected_group = "other"

        actual_group = str(row["frequency_group"]).strip().lower()
        require(
            actual_group == expected_group,
            f"Target frequency_group mismatch at {context}: "
            f"expected={expected_group}, actual={actual_group}",
        )

        metadata = (target_id, target_frequency, actual_group)
        if event_id in event_target_metadata:
            require(
                event_target_metadata[event_id] == metadata,
                f"Target metadata differs across methods for event {event_id}.",
            )
        else:
            event_target_metadata[event_id] = metadata

    movies = parse_integer_list(
        row["top10_movie_ids"],
        context=f"{context}, method={method}",
    )
    require(len(movies) == TOP_K,
            f"Expected top-{TOP_K} list at {context}; got {len(movies)}.")
    require(len(set(movies)) == TOP_K,
            f"Duplicate top-{TOP_K} item at {context}.")

    unknown = set(movies) - movie_ids
    require(not unknown,
            f"Unknown recommended item(s) at {context}: {sorted(unknown)[:10]}")

    for item_id in movies:
        if item_id in zero_shot:
            exposure[method]["zero_shot"][item_id] += 1
        elif item_id in one_shot:
            exposure[method]["one_shot"][item_id] += 1

expected_keys = {
    (event_id, method)
    for event_id in range(1, EXPECTED_EVENTS + 1)
    for method in METHODS
}
require(
    event_methods == expected_keys,
    "Event-method pairing mismatch: "
    f"missing={len(expected_keys - event_methods)}, "
    f"unexpected={len(event_methods - expected_keys)}",
)

for method in METHODS:
    require(
        method_counts[method] == EXPECTED_EVENTS,
        f"Wrong event count for {method}: {method_counts[method]}",
    )

if VALIDATE_TARGET_METADATA:
    require(
        set(event_target_metadata) == set(range(1, EXPECTED_EVENTS + 1)),
        "Target metadata does not cover all expected events.",
    )

TOTAL_SLOTS = EXPECTED_EVENTS * TOP_K

for method in METHODS:
    cold_slots = (
        sum(exposure[method]["zero_shot"].values())
        + sum(exposure[method]["one_shot"].values())
    )
    require(cold_slots <= TOTAL_SLOTS,
            f"Cold-start exposure exceeds all slots for {method}.")

print("\n[2] EVENT AND RECOMMENDATION VALIDATION")
print(f"Event-method records: {len(event_methods)}")
print(f"Events per method: {EXPECTED_EVENTS}")
print(f"Methods: {len(METHODS)}")
print(f"Top-{TOP_K} slots per method: {TOTAL_SLOTS}")
print("Event-method completeness: PASS")
print("Target metadata consistency: "
      + ("PASS" if VALIDATE_TARGET_METADATA else "NOT RUN"))
print("Top-K length and uniqueness: PASS")
print("Catalog membership: PASS")


# ============================================================================
# 6. COMPUTE COLD-START EXPOSURE METRICS
# ============================================================================

summary_rows: list[dict[str, Any]] = []
item_rows: list[dict[str, Any]] = []
multiplicity_rows: list[dict[str, Any]] = []
distribution_rows: list[dict[str, Any]] = []

cold_groups = [
    ("zero_shot", zero_shot),
    ("one_shot", one_shot),
]

for method in METHODS:
    print(f"\nProcessing: {method}")

    method_summary: dict[str, Any] = {
        "method": method,
        "events": EXPECTED_EVENTS,
        "top_k": TOP_K,
        "total_recommendation_slots": TOTAL_SLOTS,
    }

    for group_name, group_items in cold_groups:
        ordered_items = sorted(group_items)

        # Zero-exposure items are explicitly retained in the vector.
        counts = np.asarray(
            [exposure[method][group_name].get(item, 0)
             for item in ordered_items],
            dtype=np.int64,
        )

        require(counts.size == len(group_items),
                f"Population denominator mismatch: {method}/{group_name}")

        total_exposure = int(counts.sum())
        exposed_items = int(np.count_nonzero(counts))
        unexposed_items = int(counts.size - exposed_items)

        coverage = exposed_items / len(group_items)
        exposure_share_all_slots = total_exposure / TOTAL_SLOTS

        multiplicity = {
            "zero_exposure_items": int(np.sum(counts == 0)),
            "one_exposure_items": int(np.sum(counts == 1)),
            "two_or_more_exposure_items": int(np.sum(counts >= 2)),
        }
        require(
            sum(multiplicity.values()) == len(group_items),
            f"Multiplicity categories do not sum to catalog group size: "
            f"{method}/{group_name}",
        )

        gini = calculate_gini(counts)
        concentration = concentration_metrics(counts)

        mean_all = float(counts.mean())
        mean_exposed = (
            float(counts[counts > 0].mean()) if exposed_items else 0.0
        )
        median_all = float(np.median(counts))
        max_exposure = int(counts.max())

        prefix = group_name
        method_summary.update({
            f"{prefix}_catalog_items": int(len(group_items)),
            f"{prefix}_unique_exposed_items": exposed_items,
            f"{prefix}_unexposed_items": unexposed_items,
            f"{prefix}_catalog_coverage": float(coverage),
            f"{prefix}_total_exposure_slots": total_exposure,
            f"{prefix}_share_of_all_recommendation_slots": float(
                exposure_share_all_slots
            ),
            f"{prefix}_exposure_gini": float(gini),
            f"{prefix}_mean_exposure_per_catalog_item": mean_all,
            f"{prefix}_mean_exposure_per_exposed_item": mean_exposed,
            f"{prefix}_median_exposure_per_catalog_item": median_all,
            f"{prefix}_maximum_item_exposure": max_exposure,
            f"{prefix}_zero_exposure_items": multiplicity["zero_exposure_items"],
            f"{prefix}_one_exposure_items": multiplicity["one_exposure_items"],
            f"{prefix}_two_or_more_exposure_items": (
                multiplicity["two_or_more_exposure_items"]
            ),
            f"{prefix}_hhi": concentration["hhi"],
            f"{prefix}_effective_items": concentration["effective_items"],
            f"{prefix}_top_1pct_exposure_share": (
                concentration["top_1pct_exposure_share"]
            ),
            f"{prefix}_top_10pct_exposure_share": (
                concentration["top_10pct_exposure_share"]
            ),
        })

        multiplicity_rows.append({
            "method": method,
            "group": group_name,
            "catalog_items": len(group_items),
            **multiplicity,
            "total_exposure_slots": total_exposure,
            "catalog_coverage": coverage,
        })

        distribution_rows.append({
            "method": method,
            "group": group_name,
            "catalog_items": len(group_items),
            "total_exposure_slots": total_exposure,
            "minimum_exposure": int(counts.min()),
            "q25_exposure": float(np.quantile(counts, 0.25)),
            "median_exposure": median_all,
            "q75_exposure": float(np.quantile(counts, 0.75)),
            "q90_exposure": float(np.quantile(counts, 0.90)),
            "q95_exposure": float(np.quantile(counts, 0.95)),
            "q99_exposure": float(np.quantile(counts, 0.99)),
            "maximum_exposure": max_exposure,
        })

        for item_id, count in zip(ordered_items, counts):
            item_rows.append({
                "method": method,
                "group": group_name,
                "movie_id": int(item_id),
                "training_frequency": int(freq_dict[item_id]),
                "canonical_group": int(item2group[item_id]),
                "exposure_count": int(count),
                "was_exposed": int(count > 0),
            })

        print(
            f"  {group_name}: catalog={len(group_items)}, "
            f"exposed={exposed_items}, coverage={coverage:.4%}, "
            f"slots={total_exposure}, all-slot-share={exposure_share_all_slots:.4%}, "
            f"Gini={gini:.6f}, HHI={concentration['hhi']:.6f}"
        )

    summary_rows.append(method_summary)


# ============================================================================
# 7. FINAL ARITHMETIC AND CROSS-METHOD CHECKS
# ============================================================================

for group_name, group_items in cold_groups:
    for result in summary_rows:
        require(
            result[f"{group_name}_catalog_items"] == len(group_items),
            f"Inconsistent catalog denominator for {group_name}.",
        )
        require(
            result[f"{group_name}_unique_exposed_items"]
            + result[f"{group_name}_unexposed_items"] == len(group_items),
            f"Coverage count accounting failed for {result['method']}/{group_name}.",
        )
        require(
            0.0 <= result[f"{group_name}_catalog_coverage"] <= 1.0,
            f"Coverage outside [0,1] for {result['method']}/{group_name}.",
        )
        require(
            0.0 <= result[f"{group_name}_share_of_all_recommendation_slots"]
            <= 1.0,
            f"Exposure share outside [0,1] for {result['method']}/{group_name}.",
        )

print("\n[3] METRIC INTEGRITY")
print("Zero-exposure items retained in denominators: PASS")
print("Multiplicity accounting: PASS")
print("Coverage denominator consistency: PASS")
print("Exposure-share bounds: PASS")


# ============================================================================
# 8. REPRODUCIBILITY MANIFEST
# ============================================================================

created_utc = datetime.now(timezone.utc).isoformat()

manifest = {
    "experiment": 19,
    "title": "Cold-start item exposure distribution and inequality",
    "created_utc": created_utc,
    "analysis_split": "held_out_test",
    "analysis_type": "descriptive",
    "catalog_size": len(movie_ids),
    "events": EXPECTED_EVENTS,
    "methods": METHODS,
    "top_k": TOP_K,
    "recommendation_slots_per_method": TOTAL_SLOTS,
    "cold_start_definition": {
        "zero_shot": "training target frequency == 0",
        "one_shot": "training target frequency == 1",
        "tau": tau,
    },
    "metric_definitions": {
        "catalog_coverage": (
            "Distinct exposed items divided by the full cold-start group "
            "catalog population."
        ),
        "share_of_all_recommendation_slots": (
            "Cold-start group exposure slots divided by events * top_k."
        ),
        "exposure_gini": (
            "Gini coefficient over item exposure counts for the entire "
            "cold-start group, including zero-exposure items."
        ),
        "hhi": "Sum of squared item exposure shares within the group.",
        "effective_items": "Inverse HHI (1 / HHI).",
        "top_1pct_exposure_share": (
            "Exposure share received by the ceil(1% of group size) most "
            "exposed items."
        ),
        "top_10pct_exposure_share": (
            "Exposure share received by the ceil(10% of group size) most "
            "exposed items."
        ),
    },
    "input_files": {
        "catalog": str(CATALOG_FILE),
        "events": str(EVENTS_FILE),
    },
    "input_sha256": input_hashes,
    "software": {
        "python": platform.python_version(),
        "numpy": np.__version__,
    },
    "validation": {
        "catalog_integrity": "PASS",
        "frequency_integrity": "PASS",
        "cold_start_population_integrity": "PASS",
        "canonical_group_map_integrity": "PASS",
        "event_method_completeness": "PASS",
        "target_metadata_consistency": (
            "PASS" if VALIDATE_TARGET_METADATA else "NOT_RUN"
        ),
        "top_k_length": "PASS",
        "top_k_uniqueness": "PASS",
        "catalog_membership": "PASS",
        "zero_exposure_denominator": "PASS",
        "multiplicity_accounting": "PASS",
        "metric_bounds": "PASS",
    },
    "interpretation_limitations": [
        "Descriptive analysis of frozen held-out-test recommendations.",
        "No confidence intervals or significance tests are computed here.",
        "Exposure share and catalog coverage are distinct quantities.",
        "Gini and HHI include all items in the relevant cold-start population, "
        "including items with zero exposure.",
        "This analysis alone does not establish causal effects.",
    ],
    "method_results": summary_rows,
}


# ============================================================================
# 9. SAVE OUTPUTS ONLY AFTER ALL VALIDATION PASSES
# ============================================================================

OUT.mkdir(parents=True, exist_ok=False)

save_csv(OUT / "cold_start_exposure_summary.csv", summary_rows)
save_csv(OUT / "cold_start_item_exposure.csv", item_rows)
save_csv(OUT / "cold_start_exposure_multiplicity.csv", multiplicity_rows)
save_csv(OUT / "cold_start_exposure_distribution.csv", distribution_rows)

with (OUT / "cold_start_exposure_manifest.json").open(
    "w", encoding="utf-8"
) as handle:
    json.dump(manifest, handle, indent=2, ensure_ascii=False)

with (OUT / "cold_start_exposure_summary.json").open(
    "w", encoding="utf-8"
) as handle:
    json.dump(
        {
            "experiment": 19,
            "created_utc": created_utc,
            "method_results": summary_rows,
            "validation": manifest["validation"],
            "input_sha256": input_hashes,
        },
        handle,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================================
# 10. FINAL REPORT
# ============================================================================

print("\n" + "=" * 78)
print("EXPERIMENT 19 COMPLETED SUCCESSFULLY")
print("=" * 78)
print(f"Catalog: {len(movie_ids)}")
print(f"Zero-shot items: {len(zero_shot)}")
print(f"One-shot items: {len(one_shot)}")
print(f"Events: {EXPECTED_EVENTS}")
print(f"Methods: {len(METHODS)}")
print(f"Top-K: {TOP_K}")
print(f"Slots per method: {TOTAL_SLOTS}")

print("\nValidation:")
for name, status in manifest["validation"].items():
    print(f"{name}: {status}")

print("\nOutput files:")
for output_file in sorted(OUT.iterdir()):
    if output_file.is_file():
        print(f"{output_file.name}: {output_file.stat().st_size:,} bytes")

print(f"\nOutput directory: {OUT.resolve()}")
print("No model training performed.")
print("No recommendation inference performed.")
print("Frozen recommendations were not modified.")
print("No statistical significance or causal claim is made.")
