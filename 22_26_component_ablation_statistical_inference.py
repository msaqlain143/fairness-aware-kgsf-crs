#!/usr/bin/env python
"""
Experiment 22.26 — Component ablation statistical inference.

Paired conversation-cluster bootstrap for the four pre-specified comparisons:
A1 vs A4, A2 vs A4, A3 vs A4, and A0 vs A4.

Fairness is reconstructed using the exact Experiment 22.16 ledger definitions:
- opportunity: normalized discounted weights over the full Top-500 candidate pool;
- exposure: log-discounted weights over saved Top-10 recommendations;
- genre contributions are split equally across a movie's verified genres;
- MAD is computed over supported category x popularity-group cells.

This is conditional inference on frozen sequential rankings. It does not rerun
the adaptive ranking policy inside bootstrap replicates.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12
N_BOOT = 10_000

METHOD_FILES = {
    "A0_KBRD_baseline": "A0_KBRD_baseline_event_results.csv",
    "A1_no_category_context": "A1_no_category_context_event_results.csv",
    "A2_no_KG_evidence": "A2_no_KG_evidence_event_results.csv",
    "A3_no_adaptive_exposure_deficit": "A3_no_adaptive_exposure_deficit_event_results.csv",
    "A4_full_KG_FairCRS": "A4_full_KG_FairCRS_event_results.csv",
}
COMPARISONS = [
    ("A1_no_category_context", "A4_full_KG_FairCRS"),
    ("A2_no_KG_evidence", "A4_full_KG_FairCRS"),
    ("A3_no_adaptive_exposure_deficit", "A4_full_KG_FairCRS"),
    ("A0_KBRD_baseline", "A4_full_KG_FairCRS"),
]


def parse_list(value, field):
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if pd.isna(value):
        raise ValueError(f"Missing list value in {field}")
    text = str(value).strip()
    try:
        result = json.loads(text)
    except Exception:
        try:
            result = ast.literal_eval(text)
        except Exception as exc:
            raise ValueError(f"Cannot parse {field}: {text[:120]}") from exc
    if not isinstance(result, (list, tuple)):
        raise ValueError(f"{field} is not a list")
    return list(result)


def as_bool(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_genres(value, delimiter):
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split(delimiter) if part.strip()]


def load_metadata(catalog_path, profile_path):
    with open(catalog_path, "rb") as stream:
        metadata = pickle.load(stream)

    item2group = {int(k): int(v) for k, v in metadata["item2group"].items()}
    catalog = set(map(int, metadata["movie_ids"]))
    if set(item2group) != catalog:
        raise ValueError("Catalog and item2group IDs do not match")
    if any(group not in range(5) for group in item2group.values()):
        raise ValueError("Popularity group IDs must be in [0, 4]")

    profiles = pd.read_csv(profile_path)
    required = {
        "movie_entity_id", "canonical_genres", "imdb_genres",
        "imdb_genre_evidence", "imdb_conflicting_records",
        "accepted_genre_evidence", "identity_resolution_status",
    }
    missing = required - set(profiles.columns)
    if missing:
        raise ValueError(f"Category profile missing columns: {sorted(missing)}")
    profiles["movie_entity_id"] = profiles.movie_entity_id.astype(int)
    if profiles.movie_entity_id.duplicated().any():
        raise ValueError("Duplicate movie IDs in category profiles")

    categories = {movie_id: [] for movie_id in catalog}
    for row in profiles.itertuples(index=False):
        movie_id = int(row.movie_entity_id)
        if movie_id not in catalog:
            continue
        if str(row.identity_resolution_status).upper() == "IDENTITY_REVIEW_REQUIRED":
            continue

        accepted = []
        if as_bool(row.imdb_genre_evidence) and not as_bool(row.imdb_conflicting_records):
            raw = row.imdb_genres
            if not pd.isna(raw):
                raw = str(raw)
                accepted += parse_genres(raw, "|" if "|" in raw else ";")
        if as_bool(row.accepted_genre_evidence):
            accepted += parse_genres(row.canonical_genres, ";")
        categories[movie_id] = sorted(set(accepted))

    return item2group, catalog, categories


def find_candidate_id_column(frame):
    candidates = [
        "candidate_movie_ids", "candidate_ids", "candidate_item_ids",
        "movie_ids", "top500_movie_ids", "candidate_items",
        # Actual column in the Experiment 22.22 KBRD independent-test export.
        "top_movie_ids",
    ]
    for name in candidates:
        if name in frame.columns:
            return name
    raise RuntimeError(
        "Cannot identify Top-500 candidate ID column in KBRD export. "
        f"Available columns: {list(frame.columns)}"
    )


def load_candidates(path, requested_column=None):
    frame = pd.read_csv(path)
    if "event_id" not in frame.columns:
        raise ValueError("KBRD candidate export must contain event_id")
    id_col = requested_column or find_candidate_id_column(frame)
    if id_col not in frame.columns:
        raise ValueError(
            f"Candidate ID column {id_col!r} not found. "
            f"Available columns: {list(frame.columns)}"
        )
    if frame.event_id.duplicated().any():
        raise ValueError("Duplicate event_id in KBRD candidate export")

    # Verify the source export represents the frozen Top-500 pool.
    if "top_k" in frame.columns:
        top_k_values = pd.to_numeric(frame["top_k"], errors="coerce")
        if top_k_values.isna().any() or not (top_k_values == 500).all():
            bad = frame.loc[top_k_values != 500, ["event_id", "top_k"]].head(10)
            raise ValueError(
                "KBRD export is not uniformly Top-500. Examples: "
                + str(bad.to_dict(orient="records"))
            )

    candidates = {}
    for event_id, raw in zip(frame["event_id"], frame[id_col]):
        event_id = int(event_id)
        ids = np.asarray([int(x) for x in parse_list(raw, id_col)], dtype=np.int64)
        if len(ids) != 500 or len(np.unique(ids)) != 500:
            raise ValueError(
                f"Event {event_id}: expected 500 unique candidates from {id_col}; "
                f"found {len(ids)} IDs and {len(np.unique(ids))} unique IDs"
            )
        candidates[event_id] = ids
    return candidates, id_col


def load_method_frames(results_dir):
    frames = {}
    for method, filename in METHOD_FILES.items():
        path = results_dir / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {
            "event_id", "conv_id", "target_movie_id", "hit10",
            "ndcg10", "top10_movie_ids",
        }
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
        if frame.event_id.duplicated().any():
            raise ValueError(f"Duplicate event_id in {path.name}")
        frames[method] = frame
    return frames


def build_cell_index(candidates, item2group, categories):
    cells = set()
    for ids in candidates.values():
        for movie_id in ids:
            movie_id = int(movie_id)
            if movie_id not in item2group:
                raise ValueError(f"Candidate {movie_id} absent from item2group")
            group = int(item2group[movie_id])
            for genre in categories.get(movie_id, []):
                cells.add((genre, group))
    ordered = sorted(cells)
    return {cell: i for i, cell in enumerate(ordered)}, ordered


def event_contributions(candidate_ids, ranked_ids, item2group, categories, cell_index):
    """Return sparse per-event opportunity and exposure contributions."""
    opp = defaultdict(float)
    exp = defaultdict(float)

    n = len(candidate_ids)
    weights = 1.0 / np.log2(np.arange(1, n + 1, dtype=float) + 1.0)
    weights /= weights.sum() + EPS

    for rank, movie_id in enumerate(candidate_ids):
        movie_id = int(movie_id)
        genres = categories.get(movie_id, [])
        if not genres:
            continue
        contribution = float(weights[rank]) / len(genres)
        group = int(item2group[movie_id])
        for genre in genres:
            idx = cell_index[(genre, group)]
            opp[idx] += contribution

    for rank, movie_id in enumerate(ranked_ids[:10]):
        movie_id = int(movie_id)
        genres = categories.get(movie_id, [])
        if not genres:
            continue
        contribution = 1.0 / math.log2(rank + 2.0) / len(genres)
        group = int(item2group[movie_id])
        for genre in genres:
            idx = cell_index[(genre, group)]
            exp[idx] += contribution

    return opp, exp


def make_cluster_matrix(frame, candidates, item2group, categories,
                        cell_index, conv_order, metric=None):
    """Aggregate per-event fairness contributions by conversation cluster."""
    conv_pos = {int(conv): i for i, conv in enumerate(conv_order)}
    n_cells = len(cell_index)
    opp_matrix = np.zeros((len(conv_order), n_cells), dtype=np.float64)
    exp_matrix = np.zeros_like(opp_matrix)

    metric_sums = np.zeros((len(conv_order), 2), dtype=np.float64)
    metric_counts = np.zeros(len(conv_order), dtype=np.float64)

    for row in frame.itertuples(index=False):
        event_id = int(row.event_id)
        conv_id = int(row.conv_id)
        ci = conv_pos[conv_id]
        candidate_ids = candidates[event_id]
        ranked_ids = np.asarray(
            [int(x) for x in parse_list(row.top10_movie_ids, "top10_movie_ids")],
            dtype=np.int64,
        )
        if len(ranked_ids) < 10:
            raise ValueError(f"Event {event_id}: fewer than 10 ranked items")
        if len(set(ranked_ids.tolist())) != len(ranked_ids):
            raise ValueError(f"Event {event_id}: duplicate Top-10 movie IDs")
        if not set(ranked_ids[:10]).issubset(set(candidate_ids.tolist())):
            raise ValueError(f"Event {event_id}: Top-10 item outside Top-500 pool")

        opp, exp = event_contributions(
            candidate_ids, ranked_ids, item2group, categories, cell_index
        )
        for idx, value in opp.items():
            opp_matrix[ci, idx] += value
        for idx, value in exp.items():
            exp_matrix[ci, idx] += value

        metric_sums[ci, 0] += float(row.hit10)
        metric_sums[ci, 1] += float(row.ndcg10)
        metric_counts[ci] += 1.0

    return opp_matrix, exp_matrix, metric_sums, metric_counts


def fairness_from_totals(opportunity, exposure):
    supported = opportunity > EPS
    if int(supported.sum()) < 2:
        return float("nan")
    rates = (exposure[supported] + EPS) / (opportunity[supported] + EPS)
    log_rates = np.log(rates)
    return float(np.mean(np.abs(log_rates - np.mean(log_rates))))


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def bootstrap_p_two_sided(samples, observed):
    """Null-centred bootstrap estimation-error tail probability."""
    errors = np.asarray(samples, dtype=float) - float(observed)
    p = (1.0 + np.sum(np.abs(errors) >= abs(float(observed)))) / (len(errors) + 1.0)
    return float(p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path("results_final_test_m2/experiment_22_25"))
    parser.add_argument("--kbrd", type=Path,
                        default=Path("results_final_test_m2/experiment_22_22/kbrd_independent_test_top500.csv"))
    parser.add_argument("--candidate-col", type=str, default=None,
                        help="Override auto-detected KBRD Top-500 candidate-ID column.")
    parser.add_argument("--catalog", type=Path, default=Path("catalog_audit_corrected.pkl"))
    parser.add_argument("--profiles", type=Path,
                        default=Path("results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results_final_test_m2/experiment_22_26"))
    parser.add_argument("--bootstrap", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=20261116)
    args = parser.parse_args()

    print("=" * 76)
    print("EXPERIMENT 22.26 — COMPONENT ABLATION STATISTICAL INFERENCE")
    print("=" * 76)

    item2group, catalog, categories = load_metadata(args.catalog, args.profiles)
    candidates, candidate_col = load_candidates(args.kbrd, args.candidate_col)
    frames = load_method_frames(args.results_dir)

    reference = frames["A4_full_KG_FairCRS"]
    event_ids = set(map(int, reference.event_id))
    conv_by_event = dict(zip(reference.event_id.astype(int), reference.conv_id.astype(int)))
    if len(event_ids) != 4003 or len(set(conv_by_event.values())) != 1283:
        raise RuntimeError(
            f"Expected 4,003 events and 1,283 conversations; got "
            f"{len(event_ids)} and {len(set(conv_by_event.values()))}"
        )
    if set(candidates) != event_ids:
        raise RuntimeError("KBRD candidate event universe differs from ablation events")

    for method, frame in frames.items():
        if set(map(int, frame.event_id)) != event_ids:
            raise RuntimeError(f"{method}: event universe mismatch")
        current_map = dict(zip(frame.event_id.astype(int), frame.conv_id.astype(int)))
        if current_map != conv_by_event:
            raise RuntimeError(f"{method}: conversation mapping mismatch")

    cell_index, cells = build_cell_index(candidates, item2group, categories)
    conv_order = sorted(set(conv_by_event.values()))
    print(f"Events={len(event_ids)} | Conversations={len(conv_order)} | Cells={len(cells)}")
    print(f"Bootstrap replicates={args.bootstrap} | candidate ID column={candidate_col}")

    # Build event-contribution matrices once. Opportunity is identical across methods.
    matrices = {}
    shared_opp = None
    for method, frame in frames.items():
        opp, exp, metric_sums, metric_counts = make_cluster_matrix(
            frame, candidates, item2group, categories, cell_index, conv_order
        )
        if shared_opp is None:
            shared_opp = opp
        elif not np.allclose(shared_opp, opp, rtol=0, atol=1e-12):
            raise RuntimeError("Opportunity contributions differ across methods")
        matrices[method] = {
            "exposure": exp,
            "metric_sums": metric_sums,
            "metric_counts": metric_counts,
        }

    rng = np.random.default_rng(args.seed)
    n_clusters = len(conv_order)
    # Multinomial counts implement paired cluster resampling with replacement.
    draws = rng.multinomial(
        n_clusters, np.full(n_clusters, 1.0 / n_clusters),
        size=args.bootstrap
    ).astype(np.float64)

    opp_boot = draws @ shared_opp
    fairness_boot = {}
    utility_boot = {}
    for method, values in matrices.items():
        exp_boot = draws @ values["exposure"]
        fairness = np.empty(args.bootstrap, dtype=np.float64)
        for b in range(args.bootstrap):
            fairness[b] = fairness_from_totals(opp_boot[b], exp_boot[b])
        fairness_boot[method] = fairness

        sums = draws @ values["metric_sums"]
        counts = draws @ values["metric_counts"]
        utility_boot[method] = {
            "HR@10": sums[:, 0] / counts,
            "NDCG@10": sums[:, 1] / counts,
        }

    # Point estimates use the complete fixed independent-test event set.
    point = {}
    for method, frame in frames.items():
        opp = shared_opp.sum(axis=0)
        exp = matrices[method]["exposure"].sum(axis=0)
        point[method] = {
            "events": int(len(frame)),
            "conversations": int(frame.conv_id.nunique()),
            "HR@10": float(frame.hit10.mean()),
            "NDCG@10": float(frame.ndcg10.mean()),
            "fairness_MAD": fairness_from_totals(opp, exp),
        }

    rows = []
    bootstrap_store = {}
    raw_p = []
    for left, right in COMPARISONS:
        for metric in ("HR@10", "NDCG@10", "fairness_MAD"):
            observed = point[right][metric] - point[left][metric]
            samples = (
                fairness_boot[right] - fairness_boot[left]
                if metric == "fairness_MAD"
                else utility_boot[right][metric] - utility_boot[left][metric]
            )
            low, high = np.quantile(samples, [0.025, 0.975])
            p_value = bootstrap_p_two_sided(samples, observed)
            raw_p.append(p_value)
            key = f"{left}__vs__{right}__{metric}"
            bootstrap_store[key] = samples
            rows.append({
                "comparison": f"{right} - {left}",
                "metric": metric,
                "left_method": left,
                "right_method": right,
                "left_value": point[left][metric],
                "right_value": point[right][metric],
                "delta_right_minus_left": observed,
                "ci95_low": float(low),
                "ci95_high": float(high),
                "bootstrap_p_two_sided": p_value,
            })

    adjusted = holm_adjust(raw_p)
    for row, p_adj in zip(rows, adjusted):
        row["holm_adjusted_p"] = float(p_adj)
        row["significant_after_holm"] = bool(p_adj < 0.05)
        row["bootstrap_replicates"] = int(args.bootstrap)
        row["bootstrap_unit"] = "conversation"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inference = pd.DataFrame(rows)
    inference.to_csv(args.out_dir / "component_ablation_paired_inference.csv", index=False)
    pd.DataFrame([
        {"method": method, **values} for method, values in point.items()
    ]).to_csv(args.out_dir / "component_ablation_point_estimates.csv", index=False)
    np.savez_compressed(
        args.out_dir / "component_ablation_bootstrap_distributions.npz",
        **bootstrap_store
    )

    validation = {
        "exactly_4003_paired_events": len(event_ids) == 4003,
        "exactly_1283_conversation_clusters": len(conv_order) == 1283,
        "same_event_universe_all_methods": True,
        "same_conversation_mapping": True,
        "same_top500_candidate_pools": True,
        "opportunity_reconstructed_from_full_top500": True,
        "exposure_reconstructed_from_saved_top10": True,
        "fairness_mad_recomputed_per_bootstrap_replicate": True,
        "paired_conversation_cluster_bootstrap": True,
        "bootstrap_replicates": int(args.bootstrap),
        "holm_correction_across_12_tests": True,
        "training_performed": False,
        "retrieval_rerun": False,
        "parameter_search": False,
        "adaptive_policy_rerun_inside_bootstrap": False,
        "overall_pass": True,
    }
    (args.out_dir / "validation_checks.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    manifest = {
        "experiment": "22.26_component_ablation_statistical_inference",
        "split": "independent_test",
        "events": len(event_ids),
        "conversations": len(conv_order),
        "supported_category_group_cells": len(cells),
        "bootstrap_replicates": int(args.bootstrap),
        "bootstrap_unit": "conversation",
        "seed": int(args.seed),
        "comparisons": [
            {"left": a, "right": b} for a, b in COMPARISONS
        ],
        "fairness_definition": {
            "opportunity": "normalized 1/log2(rank+1) weights over all 500 candidates",
            "exposure": "1/log2(rank+2) weights over saved top-10 recommendations",
            "genre_allocation": "equal split across verified genres",
            "metric": "mean absolute deviation of log exposure/opportunity rates across supported category x popularity-group cells",
        },
        "inference_scope": (
            "Conditional inference on frozen sequential rankings. The adaptive "
            "ranking policy is not rerun inside bootstrap replicates."
        ),
        "inputs": {
            "ablation_results": str(args.results_dir),
            "kbrd_top500": str(args.kbrd),
            "catalog": str(args.catalog),
            "category_profiles": str(args.profiles),
        },
        "outputs": [
            "component_ablation_point_estimates.csv",
            "component_ablation_paired_inference.csv",
            "component_ablation_bootstrap_distributions.npz",
            "validation_checks.json",
            "experiment_manifest.json",
        ],
        "validation": validation,
    }
    (args.out_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print("\n========== EXPERIMENT 22.26 COMPLETE ==========")
    print(inference.to_string(index=False))
    print("\nValidation:", json.dumps(validation, indent=2))
    print("Output:", args.out_dir.resolve())
    print("Training: NO | KG retrieval: NO | Test tuning: NO")


if __name__ == "__main__":
    main()
