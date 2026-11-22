#!/usr/bin/env python
"""Experiment 22.15: Cold-start exposure statistical validation.

Uses frozen independent test-set recommendation exports only.
No model training, retrieval, coefficient search, or test-set tuning.

Inference unit: conversation_id. All methods share identical paired
conversation-cluster bootstrap resamples.

Exposure definition: un-discounted recommendation count within Top-10.
Each appearance of an item in a Top-10 contributes one exposure unit.
"""

from __future__ import annotations

import argparse
import ast
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

SEED = 22115
N_BOOTSTRAP = 10_000
TOP_K = 10
EPS = 1e-12

METHODS = ["KGSF_only", "A0_fixed_025", "A1_frequency_3"]
GROUPS = ("zero_shot", "one_shot")
BASE = Path("results_final_test_m2")
EVENT_FILE = BASE / "adaptive_m2_with_kgsf_baseline_events.csv"
CLUSTER_FILE = BASE / "adaptive_m2_final_test_events_clustered.csv"
MAPPING_FILE = BASE / "m2_test_event_conversation_mapping.csv"
CATALOG_FILE = Path("catalog_audit_corrected.pkl")
OUT = Path("results_statistical_validation/experiment_22_15")
METRICS = ("catalog_coverage", "exposure_gini", "hhi", "exposure_share")


def parse_ids(value) -> list[int]:
    """Parse a CSV-serialized list of integer movie IDs safely."""
    if isinstance(value, (list, tuple, np.ndarray)):
        parsed = value
    elif value is None or (np.isscalar(value) and pd.isna(value)):
        return []
    else:
        raw = str(value).strip()
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(raw)
    if not isinstance(parsed, (list, tuple, np.ndarray)):
        raise ValueError(f"Expected a list of movie IDs, got {type(parsed).__name__}")
    return [int(x) for x in parsed]


def find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Cannot identify {label}; available columns: {df.columns.tolist()}")


def gini(values: np.ndarray) -> float:
    """Finite-sample Gini; zero-exposure catalog members remain included."""
    x = np.sort(np.asarray(values, dtype=float))
    if x.size == 0 or float(x.sum()) <= EPS:
        return float("nan")  # inequality is undefined with no exposure
    n = x.size
    return float((2.0 * np.dot(np.arange(1, n + 1), x) / (n * x.sum())) - (n + 1) / n)


def calculate_metrics(
    group_exposure: np.ndarray,
    all_exposure_total: float,
) -> dict[str, float | int]:
    counts = np.asarray(group_exposure, dtype=float)
    total = float(counts.sum())
    catalog_size = int(counts.size)
    exposed = int(np.count_nonzero(counts > EPS))
    shares = counts / total if total > EPS else np.zeros_like(counts)
    return {
        "catalog_size": catalog_size,
        "exposed_unique_items": exposed,
        "catalog_coverage": exposed / catalog_size if catalog_size else float("nan"),
        "exposure_gini": gini(counts),
        "hhi": float(np.square(shares).sum()) if total > EPS else float("nan"),
        "exposure_share": total / all_exposure_total if all_exposure_total > EPS else float("nan"),
        "total_group_exposure": total,
    }


def _normalise_ids(series: pd.Series, label: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise")
    if numeric.isna().any():
        raise ValueError(f"Missing values in {label}.")
    if not np.all(np.equal(numeric.to_numpy(), np.floor(numeric.to_numpy()))):
        raise ValueError(f"Non-integer values in {label}.")
    return numeric.astype("int64")


def prepare_data():
    for path in (EVENT_FILE, CLUSTER_FILE, MAPPING_FILE, CATALOG_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)

    # This pickle is a trusted, locally generated project artifact.
    with CATALOG_FILE.open("rb") as f:
        catalog_data = pickle.load(f)
    if "freq_dict" not in catalog_data:
        raise ValueError("Catalog pickle has no freq_dict.")

    freq = {int(movie): int(value) for movie, value in catalog_data["freq_dict"].items()}
    zero_ids = {int(x) for x in catalog_data["zero_shot_ids"]}
    one_ids = {int(x) for x in catalog_data["one_shot_ids"]}
    if zero_ids != {m for m, value in freq.items() if value == 0}:
        raise ValueError("Zero-shot IDs do not match freq_dict.")
    if one_ids != {m for m, value in freq.items() if value == 1}:
        raise ValueError("One-shot IDs do not match freq_dict.")
    if zero_ids & one_ids:
        raise ValueError("Zero-shot and one-shot catalog groups overlap.")

    events = pd.read_csv(EVENT_FILE)
    clustered = pd.read_csv(CLUSTER_FILE)
    mapping = pd.read_csv(MAPPING_FILE)

    required = {"event_id", "method", "top10_movie_ids"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"Event export is missing columns: {sorted(missing)}")

    events = events.copy()
    events["event_id"] = _normalise_ids(events["event_id"], "event_id")
    events["method"] = events["method"].astype(str)

    # Keep only the predeclared arms; other frozen export methods are not analysed.
    available = set(events["method"])
    missing_methods = set(METHODS) - available
    if missing_methods:
        raise ValueError(f"Required methods absent from event export: {sorted(missing_methods)}")
    events = events[events["method"].isin(METHODS)].copy()

    # Independently map every event to its conversation, even if the event
    # export already contains conv_id. This also covers KGSF_only, which may
    # not be present in the separate clustered export.
    map_event = find_column(mapping, ["event_id"], "mapping event ID")
    map_conv = find_column(mapping, ["conv_id", "conversation_id", "conversation_idx"], "mapping conversation ID")
    mapping = mapping[[map_event, map_conv]].copy()
    mapping[map_event] = _normalise_ids(mapping[map_event], "mapping event_id")
    if mapping[map_event].duplicated().any():
        raise ValueError("Conversation mapping contains duplicate event IDs.")
    mapping = mapping.rename(columns={map_event: "event_id", map_conv: "mapped_conversation_id"})

    events_conv_col = None
    if "conv_id" in events.columns:
        events_conv_col = "conv_id"
    elif "conversation_id" in events.columns:
        events_conv_col = "conversation_id"

    events = events.merge(mapping, on="event_id", how="left", validate="many_to_one")
    if events["mapped_conversation_id"].isna().any():
        raise ValueError("Some selected event rows have no conversation mapping.")
    if events_conv_col:
        left = events[events_conv_col].astype(str).str.strip()
        right = events["mapped_conversation_id"].astype(str).str.strip()
        if not left.equals(right):
            raise ValueError("Event export conversation IDs disagree with mapping file.")
    events["conversation_id"] = events["mapped_conversation_id"]

    # The clustered export is an independent cross-check only for methods it
    # actually contains. Do not require KGSF_only to appear in that file.
    if not {"event_id", "method"}.issubset(clustered.columns):
        raise ValueError("Clustered export must contain event_id and method.")
    cluster_col = find_column(clustered, ["conv_id", "conversation_id", "conversation", "conversation_idx"], "cluster conversation ID")
    cluster_check = clustered[["event_id", "method", cluster_col]].copy()
    cluster_check["event_id"] = _normalise_ids(cluster_check["event_id"], "cluster event_id")
    cluster_check["method"] = cluster_check["method"].astype(str)
    cluster_check = cluster_check[cluster_check["method"].isin(METHODS)]
    cluster_check = cluster_check.rename(columns={cluster_col: "cluster_conversation_id"})
    if cluster_check.duplicated(["event_id", "method"]).any():
        raise ValueError("Clustered export contains duplicate event-method rows.")

    overlap_methods = sorted(set(events["method"]) & set(cluster_check["method"]))
    if not overlap_methods:
        raise ValueError("No selected methods overlap the clustered export.")
    left = events[events["method"].isin(overlap_methods)][["event_id", "method", "conversation_id"]]
    right = cluster_check[["event_id", "method", "cluster_conversation_id"]]
    crosscheck = left.merge(right, on=["event_id", "method"], how="outer", validate="one_to_one", indicator=True)
    if not (crosscheck["_merge"] == "both").all():
        raise ValueError("Selected event-method rows do not align with clustered export.")
    if not (crosscheck["conversation_id"].astype(str) == crosscheck["cluster_conversation_id"].astype(str)).all():
        raise ValueError("Conversation IDs disagree with clustered export.")

    if events.duplicated(["event_id", "method"]).any():
        raise ValueError("Duplicate selected event-method records.")
    method_event_sets = {
        method: set(rows["event_id"].tolist())
        for method, rows in events.groupby("method")
    }
    reference_events = method_event_sets[METHODS[0]]
    if any(ids != reference_events for ids in method_event_sets.values()):
        raise ValueError("Selected methods do not contain identical event IDs.")

    events["top10"] = events["top10_movie_ids"].apply(parse_ids)
    if not events["top10"].map(lambda ids: len(ids) == TOP_K).all():
        raise ValueError("Every selected recommendation must contain exactly 10 items.")
    if not events["top10"].map(lambda ids: len(set(ids)) == TOP_K).all():
        raise ValueError("A Top-10 list contains duplicate movie IDs.")

    catalog_ids = set(freq)
    recommended = {movie for row in events["top10"] for movie in row}
    unknown = recommended - catalog_ids
    if unknown:
        raise ValueError(f"{len(unknown)} recommended movie IDs are absent from freq_dict.")
    if not zero_ids or not one_ids:
        raise ValueError("Zero-shot or one-shot catalog group is empty.")

    # Stable ordering is essential for the sparse matrix column mapping.
    catalog_order = np.asarray(sorted(freq), dtype=np.int64)
    movie_to_col = {int(movie): idx for idx, movie in enumerate(catalog_order)}
    conversations = sorted(events["conversation_id"].unique().tolist(), key=str)
    conversation_to_row = {value: idx for idx, value in enumerate(conversations)}

    matrices = {}
    for method in METHODS:
        rows, cols, data = [], [], []
        subset = events[events["method"] == method]
        for row in subset.itertuples(index=False):
            cluster_row = conversation_to_row[row.conversation_id]
            for movie in row.top10:
                rows.append(cluster_row)
                cols.append(movie_to_col[movie])
                data.append(1.0)
        matrices[method] = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(len(conversations), len(catalog_order)),
            dtype=np.float64,
        ).tocsr()
        matrices[method].sum_duplicates()

    checks = {
        "required_methods_present": set(METHODS).issubset(set(events["method"])),
        "selected_methods_have_identical_event_ids": True,
        "conversation_mapping_complete": not events["conversation_id"].isna().any(),
        "clustered_export_crosscheck_passed": True,
        "event_method_pairs_unique": not events.duplicated(["event_id", "method"]).any(),
        "catalog_frequency_mapping_validated": True,
        "zero_shot_mapping_validated": True,
        "one_shot_mapping_validated": True,
        "all_recommended_movies_in_catalog": True,
        "all_top10_lists_have_ten_unique_items": True,
    }
    metadata = {
        "events": int(events["event_id"].nunique()),
        "event_method_rows": int(len(events)),
        "conversation_clusters": int(len(conversations)),
        "cluster_crosscheck_methods": overlap_methods,
        "selected_methods": METHODS,
        "catalog_size": int(len(catalog_order)),
        "zero_shot_catalog_size": int(len(zero_ids)),
        "one_shot_catalog_size": int(len(one_ids)),
    }
    return events, conversations, catalog_order, zero_ids, one_ids, matrices, checks, metadata


def evaluate_weights(weights: np.ndarray, matrices: dict, catalog_order: np.ndarray,
                     zero_ids: set[int], one_ids: set[int], movie_to_col: dict[int, int]) -> dict:
    """Aggregate one paired cluster resample and calculate group metrics."""
    results = {}
    group_cols = {
        "zero_shot": np.asarray([movie_to_col[m] for m in sorted(zero_ids)], dtype=np.int64),
        "one_shot": np.asarray([movie_to_col[m] for m in sorted(one_ids)], dtype=np.int64),
    }
    all_total_by_method = {
        method: float(np.asarray(weights @ matrix).ravel().sum())
        for method, matrix in matrices.items()
    }
    for method, matrix in matrices.items():
        total_vector = np.asarray(weights @ matrix).ravel()
        results[method] = {}
        for group, cols in group_cols.items():
            results[method][group] = calculate_metrics(
                total_vector[cols], all_total_by_method[method]
            )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.bootstrap < 1000:
        raise ValueError("Use at least 1,000 bootstrap replicates.")

    print("Loading frozen independent test exports...")
    (events, conversations, catalog_order, zero_ids, one_ids,
     matrices, checks, metadata) = prepare_data()
    movie_to_col = {int(movie): idx for idx, movie in enumerate(catalog_order)}
    print(f"Events: {metadata['events']:,} | conversations: {metadata['conversation_clusters']:,}")
    print(f"Selected methods: {', '.join(METHODS)}")
    print(f"Clustered-export cross-check methods: {', '.join(metadata['cluster_crosscheck_methods'])}")
    print(f"Zero-shot catalog: {len(zero_ids):,} | one-shot catalog: {len(one_ids):,}")

    args.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    n_clusters = len(conversations)
    metric_samples = {
        (method, group, metric): np.empty(args.bootstrap, dtype=np.float64)
        for method in METHODS for group in GROUPS for metric in METRICS
    }

    # Full-sample point estimates correspond to one inclusion of every cluster.
    point = evaluate_weights(
        np.ones(n_clusters, dtype=np.float64), matrices, catalog_order,
        zero_ids, one_ids, movie_to_col,
    )

    # One draw of cluster multiplicities is shared by every method and metric.
    for replicate in range(args.bootstrap):
        sampled_indices = rng.integers(0, n_clusters, size=n_clusters)
        weights = np.bincount(sampled_indices, minlength=n_clusters).astype(np.float64)
        result = evaluate_weights(
            weights, matrices, catalog_order, zero_ids, one_ids, movie_to_col
        )
        for method in METHODS:
            for group in GROUPS:
                for metric in METRICS:
                    metric_samples[(method, group, metric)][replicate] = result[method][group][metric]
        if (replicate + 1) % 1000 == 0:
            print(f"Bootstrap: {replicate + 1:,}/{args.bootstrap:,}")

    metric_rows = []
    for (method, group, metric), values in metric_samples.items():
        finite = values[np.isfinite(values)]
        metric_rows.append({
            "method": method,
            "group": group,
            "metric": metric,
            "point_estimate": point[method][group][metric],
            "bootstrap_mean": float(np.mean(finite)) if finite.size else float("nan"),
            "ci95_low": float(np.percentile(finite, 2.5)) if finite.size else float("nan"),
            "ci95_high": float(np.percentile(finite, 97.5)) if finite.size else float("nan"),
            "finite_bootstrap_replicates": int(finite.size),
            "bootstrap_replicates": int(args.bootstrap),
        })
    metric_df = pd.DataFrame(metric_rows)
    metric_df.to_csv(args.out / "cold_start_cluster_bootstrap.csv", index=False)

    comparisons = [("A1_frequency_3", "KGSF_only"), ("A1_frequency_3", "A0_fixed_025")]
    paired_rows = []
    for group in GROUPS:
        for metric in METRICS:
            for treatment, control in comparisons:
                a = metric_samples[(treatment, group, metric)]
                b = metric_samples[(control, group, metric)]
                valid = np.isfinite(a) & np.isfinite(b)
                delta = a[valid] - b[valid]
                point_delta = point[treatment][group][metric] - point[control][group][metric]
                paired_rows.append({
                    "group": group,
                    "metric": metric,
                    "treatment": treatment,
                    "control": control,
                    "point_difference": point_delta,
                    "bootstrap_mean_difference": float(delta.mean()) if delta.size else float("nan"),
                    "ci95_low": float(np.percentile(delta, 2.5)) if delta.size else float("nan"),
                    "ci95_high": float(np.percentile(delta, 97.5)) if delta.size else float("nan"),
                    "probability_treatment_higher": float(np.mean(delta > 0)) if delta.size else float("nan"),
                    "probability_treatment_lower": float(np.mean(delta < 0)) if delta.size else float("nan"),
                    "finite_paired_replicates": int(delta.size),
                    "bootstrap_replicates": int(args.bootstrap),
                })
    paired_df = pd.DataFrame(paired_rows)
    paired_df.to_csv(args.out / "cold_start_paired_effects.csv", index=False)

    # Sparse groups can receive zero exposure in some cluster resamples.
    # Gini and HHI are undefined for those resamples; they are counted as
    # non-finite rather than being silently converted to zero.
    min_finite_replicates = min(1000, args.bootstrap)
    finite_replicates_sufficient = bool(
        (metric_df["finite_bootstrap_replicates"] >= min_finite_replicates).all()
        and (paired_df["finite_paired_replicates"] >= min_finite_replicates).all()
    )
    paired_counts_reconciled = bool(
        (paired_df["finite_paired_replicates"] <= paired_df["bootstrap_replicates"]).all()
        and (paired_df["finite_paired_replicates"] > 0).all()
    )

    checks.update({
        "paired_cluster_bootstrap_used": True,
        "same_resamples_used_for_all_methods": True,
        "bootstrap_replicate_count_valid": args.bootstrap >= 1000,
        "undefined_zero_exposure_replicates_counted": bool(
            (metric_df["finite_bootstrap_replicates"] <= args.bootstrap).all()
            and (paired_df["finite_paired_replicates"] <= args.bootstrap).all()
        ),
        "finite_bootstrap_replicates_sufficient": finite_replicates_sufficient,
        "paired_finite_replicate_counts_valid": paired_counts_reconciled,
        # These are PASS assertions (positive polarity), not activity flags.
        # The corresponding activity fields remain False in the manifest.
        "training_not_performed": True,
        "retrieval_not_rerun": True,
        "coefficient_search_not_performed": True,
        "test_set_used_only_for_frozen_evaluation": True,
    })
    # Only validation assertions belong in this dictionary. All entries
    # must therefore be True for the experiment to pass.
    checks["overall_pass"] = all(bool(value) for value in checks.values())

    summary = {
        "experiment": "22.15",
        "name": "Cold-start exposure cluster bootstrap",
        **metadata,
        "bootstrap_replicates": int(args.bootstrap),
        "seed": int(args.seed),
        "sampling_unit": "conversation_id",
        "sampling": "paired_cluster_bootstrap_with_replacement",
        "exposure_definition": "un-discounted Top-10 recommendation counts",
        "zero_exposure_resample_policy": "Gini and HHI are undefined for all-zero group exposure; non-finite resamples are excluded from percentile intervals and counted explicitly.",
        "metrics": list(METRICS),
        "comparisons": [{"treatment": a, "control": b} for a, b in comparisons],
        "independent_test_set": True,
        "training_performed": False,
        "retrieval_rerun": False,
        "coefficient_search": False,
    }
    (args.out / "cold_start_statistical_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (args.out / "validation_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    manifest = {
        **summary,
        "inputs": {
            "events": str(EVENT_FILE),
            "clustered_events": str(CLUSTER_FILE),
            "conversation_mapping": str(MAPPING_FILE),
            "catalog_frequency_mapping": str(CATALOG_FILE),
        },
        "outputs": [
            "cold_start_cluster_bootstrap.csv",
            "cold_start_paired_effects.csv",
            "cold_start_statistical_summary.json",
            "validation_checks.json",
            "experiment_manifest.json",
        ],
        "interpretation": [
            "Zero-exposure catalog items remain in Gini and HHI denominators.",
            "Gini and HHI are undefined when a resampled group receives no exposure; such replicates are reported as non-finite.",
            "Percentile intervals for Gini and HHI are conditional on resamples with positive group exposure; finite replicate counts must be considered when interpreting sparse zero-shot results.",
            "Paired differences use identical conversation-cluster resamples.",
            "Exposure share is descriptive, not an equality metric.",
            "Intervals quantify conversation-cluster sampling uncertainty, not catalog-label uncertainty.",
            "This is a frozen test-set evaluation; no model or policy is tuned here.",
            "Exposure is an un-discounted count per Top-10 appearance, not position-discounted attention.",
        ],
    }
    (args.out / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n========== EXPERIMENT 22.15 COMPLETE ==========")
    print("\nPaired effects:")
    print(paired_df.to_string(index=False))
    print("\nValidation:")
    print(json.dumps(checks, indent=2))
    print("\nOutput:", args.out.resolve())
    if not checks["overall_pass"]:
        raise RuntimeError("Experiment 22.15 validation failed.")


if __name__ == "__main__":
    main()
