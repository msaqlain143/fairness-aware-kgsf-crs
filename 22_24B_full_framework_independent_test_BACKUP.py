#!/usr/bin/env python
"""
Experiment 22.24B — Full framework on the independent ReDial test split.

Uses:
- Frozen KBRD Top-500 predictions from Experiment 22.22 as the common candidate pool.
- R1_raw KG ranks exported by Experiment 22.24A (no retrieval or training here).
- Verified category profiles and popularity groups from Experiment 21.22A / catalog audit.
- The exact reranking and exposure-ledger implementation from Experiment 22.16.
- Frozen lambda/tau selected on validation by Experiment 22.18.

Important:
"Contextual relevance" means category prevalence in the frozen candidate pool,
not semantic dialogue understanding. KG rank is retrieval support, not semantic
relevance. The independent test is evaluated once; no parameter tuning occurs.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(".")
DEFAULT_KBRD = Path(
    "results_final_test_m2/experiment_22_22/kbrd_independent_test_top500.csv"
)
DEFAULT_KG_DIR = Path("results_final_test_m2/experiment_22_24A")
DEFAULT_VALIDATION_MANIFEST = Path(
    "results_corrected/experiment_22_18/experiment_manifest.json"
)
DEFAULT_CATALOG = Path("catalog_audit_corrected.pkl")
DEFAULT_PROFILES = Path(
    "results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv"
)
DEFAULT_REFERENCE = Path("22_16_category_context_utility_calibration.py")
DEFAULT_GATE_REFERENCE = Path("22_18_contextual_relevance_gate_validation.py")
DEFAULT_OUT = Path("results_final_test_m2/experiment_22_24B")


def parse_array(value: Any, field: str) -> list:
    if isinstance(value, (list, tuple, np.ndarray)):
        result = list(value)
    elif value is None or (isinstance(value, float) and np.isnan(value)):
        raise ValueError(f"{field} is empty.")
    else:
        text = str(value).strip()
        try:
            result = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            # Also accept JSON arrays.
            result = json.loads(text)
    if not isinstance(result, (list, tuple, np.ndarray)):
        raise ValueError(f"{field} is not an array.")
    return list(result)


def import_reference(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Experiment 22.16 script not found: {path}")
    spec = importlib.util.spec_from_file_location("experiment_22_16_reference", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import reference script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    required = [
        "load_metadata", "ExposureLedger", "rerank",
        "ranking_metrics", "fairness_mad",
    ]
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Reference script missing required API: {missing}")
    return module


def find_r1_raw_export(
    kg_dir: Path,
    explicit_path: Path | None,
) -> tuple[pd.DataFrame, Path]:
    """
    Load the actual wide-format Experiment 22.24A export.

    22.24A stores one row per event and stores the R1_raw ranks in
    R1_raw_candidate_kg_ranks. The rank array is positionally aligned with
    the frozen KBRD Top-500 candidate order used when 22.24A was generated.
    There is no candidate_movie_ids column in this export.
    """
    if explicit_path is not None:
        path = explicit_path
        if not path.is_file():
            raise FileNotFoundError(f"22.24A KG CSV not found: {path}")
    else:
        path = kg_dir / "independent_test_kg_candidate_ranks.csv"
        if not path.is_file():
            raise FileNotFoundError(
                "Expected the verified 22.24A export at "
                f"{path}. Pass --kg-csv to specify another file."
            )

    frame = pd.read_csv(path)

    required = {
        "event_id",
        "conv_id",
        "target_movie_id",
        "candidate_count",
        "R1_raw_candidate_kg_ranks",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError(
            f"22.24A export missing required columns: {sorted(missing)}"
        )

    frame["event_id"] = pd.to_numeric(
        frame["event_id"], errors="raise"
    ).astype(np.int64)
    frame["conv_id"] = pd.to_numeric(
        frame["conv_id"], errors="raise"
    ).astype(np.int64)
    frame["target_movie_id"] = pd.to_numeric(
        frame["target_movie_id"], errors="raise"
    ).astype(np.int64)
    frame["candidate_count"] = pd.to_numeric(
        frame["candidate_count"], errors="raise"
    ).astype(np.int64)

    if len(frame) != 4003:
        raise RuntimeError(
            f"Expected 4,003 22.24A events; found {len(frame)}."
        )
    if frame["event_id"].duplicated().any():
        raise RuntimeError("Duplicate event_id values in 22.24A export.")
    if not frame["candidate_count"].eq(500).all():
        raise RuntimeError("Not all 22.24A events have exactly 500 candidates.")

    def decode_ranks(value: Any) -> np.ndarray:
        values = parse_array(value, "R1_raw_candidate_kg_ranks")
        ranks = np.asarray(values, dtype=np.int64)
        if ranks.shape != (500,):
            raise ValueError(
                "R1_raw KG rank array must contain exactly 500 values; "
                f"observed shape {ranks.shape}."
            )
        if np.any(ranks < 0):
            raise ValueError("Negative KG rank encountered.")
        return ranks

    frame["candidate_kg_ranks"] = frame[
        "R1_raw_candidate_kg_ranks"
    ].map(decode_ranks)

    if "R1_raw_retrieved_count" in frame.columns:
        retrieved = pd.to_numeric(
            frame["R1_raw_retrieved_count"], errors="raise"
        ).astype(np.int64)
        positive_counts = frame["candidate_kg_ranks"].map(
            lambda x: int(np.count_nonzero(x > 0))
        )
        # A retrieved movie can be outside the KBRD Top-500, so this is a
        # consistency bound, not an equality requirement.
        if (positive_counts > retrieved).any():
            raise RuntimeError(
                "More positive candidate ranks than total R1_raw retrieved movies."
            )

    return frame, path


def load_validation_parameters(path: Path) -> tuple[float, float, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen validation manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    lam = manifest.get("selected_lambda")
    tau = manifest.get("selected_tau")
    if lam is None or tau is None:
        raise ValueError("Validation manifest lacks selected_lambda / selected_tau.")
    lam, tau = float(lam), float(tau)
    if not (math.isfinite(lam) and math.isfinite(tau) and tau > 0):
        raise ValueError(f"Invalid frozen parameters: lambda={lam}, tau={tau}")
    return lam, tau, manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kbrd", type=Path, default=DEFAULT_KBRD)
    parser.add_argument("--kg-dir", type=Path, default=DEFAULT_KG_DIR)
    parser.add_argument("--kg-csv", type=Path, default=None)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--catalog-pickle", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--category-profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--gate-reference", type=Path, default=DEFAULT_GATE_REFERENCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    print("=" * 72)
    print("EXPERIMENT 22.24B — FULL FRAMEWORK INDEPENDENT-TEST EVALUATION")
    print("=" * 72)

    for path in (args.kbrd, args.validation_manifest, args.catalog_pickle,
                 args.category_profiles, args.reference, args.gate_reference):
        if not path.exists():
            raise FileNotFoundError(path)

    ref = import_reference(args.reference)
    gate_spec = importlib.util.spec_from_file_location(
        "experiment_22_18_gate_reference", args.gate_reference
    )
    if gate_spec is None or gate_spec.loader is None:
        raise RuntimeError(f"Cannot import gate reference: {args.gate_reference}")
    gate_ref = importlib.util.module_from_spec(gate_spec)
    gate_spec.loader.exec_module(gate_ref)
    for required_name in ("contextual_gate_rerank", "baseline_rank", "exposure_update"):
        if not hasattr(gate_ref, required_name):
            raise RuntimeError(f"22.18 reference lacks required API: {required_name}")

    lam, tau, validation_manifest = load_validation_parameters(args.validation_manifest)
    print(f"Frozen validation parameters: lambda={lam}, tau={tau}")
    print("Validation selection status:", validation_manifest.get("selection_status"))

    # Metadata/category construction is delegated to the validated 22.16 code.
    item2group, catalog, categories = ref.load_metadata(
        args.catalog_pickle, args.category_profiles
    )

    kbrd = pd.read_csv(args.kbrd)
    required_kbrd = {
        "event_id", "conv_id", "target_movie_id", "top_movie_ids", "top_scores"
    }
    missing = required_kbrd - set(kbrd.columns)
    if missing:
        raise ValueError(f"KBRD independent-test export missing columns: {sorted(missing)}")
    if kbrd.event_id.duplicated().any():
        raise ValueError("Duplicate event_id values in KBRD export.")

    kg, kg_source = find_r1_raw_export(args.kg_dir, args.kg_csv)

    if kg["event_id"].duplicated().any():
        raise ValueError("Duplicate event_id values in R1_raw KG export.")

    kbrd["event_id"] = pd.to_numeric(
        kbrd["event_id"], errors="raise"
    ).astype(np.int64)
    kg["event_id"] = pd.to_numeric(
        kg["event_id"], errors="raise"
    ).astype(np.int64)

    kbrd = kbrd.sort_values("event_id").reset_index(drop=True)
    kg = kg.sort_values("event_id").reset_index(drop=True)

    if set(kbrd["event_id"]) != set(kg["event_id"]):
        missing_kg = sorted(
            set(kbrd["event_id"]) - set(kg["event_id"])
        )[:10]
        extra_kg = sorted(
            set(kg["event_id"]) - set(kbrd["event_id"])
        )[:10]
        raise ValueError(
            f"KBRD/KG event mismatch. Missing KG={missing_kg}; "
            f"extra KG={extra_kg}"
        )

    # The 22.24A schema does not export candidate IDs as a separate column.
    # Its 500-element rank arrays were created in the original KBRD Top-500
    # candidate order. Therefore join by event_id and preserve array position.
    kg_by_event = kg.set_index("event_id")

    for row in kbrd.itertuples(index=False):
        event_id = int(row.event_id)
        kg_row = kg_by_event.loc[event_id]

        if int(row.conv_id) != int(kg_row["conv_id"]):
            raise ValueError(
                f"Conversation alignment mismatch at event {event_id}: "
                f"KBRD={row.conv_id}, KG={kg_row['conv_id']}"
            )
        if int(row.target_movie_id) != int(kg_row["target_movie_id"]):
            raise ValueError(
                f"Target alignment mismatch at event {event_id}: "
                f"KBRD={row.target_movie_id}, KG={kg_row['target_movie_id']}"
            )

    events, candidates = {}, {}
    candidate_sizes = []
    for row in kbrd.itertuples(index=False):
        event_id = int(row.event_id)
        ids = np.asarray(
            [int(x) for x in parse_array(row.top_movie_ids, "top_movie_ids")],
            dtype=np.int64,
        )
        scores = np.asarray(
            [float(x) for x in parse_array(row.top_scores, "top_scores")],
            dtype=np.float64,
        )
        if len(ids) != len(scores) or len(np.unique(ids)) != len(ids):
            raise ValueError(f"Invalid KBRD candidate arrays at event {event_id}")
        if len(ids) != 500:
            raise ValueError(f"Expected Top-500 candidates at event {event_id}; got {len(ids)}")
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite KBRD scores at event {event_id}")
        if not set(map(int, ids)).issubset(catalog):
            raise ValueError(f"Out-of-catalog candidate at event {event_id}")
        absent_groups = [int(x) for x in ids if int(x) not in item2group]
        if absent_groups:
            raise ValueError(f"Missing popularity groups at event {event_id}: {absent_groups[:5]}")

        kg_row = kg_by_event.loc[event_id]
        ranks = np.asarray(kg_row["candidate_kg_ranks"], dtype=np.int64)

        if len(ranks) != len(ids):
            raise ValueError(
                f"KBRD/R1_raw positional alignment failure at event {event_id}: "
                f"{len(ids)} candidate IDs vs {len(ranks)} KG ranks."
            )
        events[event_id] = {"conv": int(row.conv_id), "target": int(row.target_movie_id)}
        candidates[event_id] = {"ids": ids, "scores": scores, "kg": ranks}
        candidate_sizes.append(len(ids))

    if len(events) != 4003 or len(set(e["conv"] for e in events.values())) != 1283:
        raise ValueError(
            "Independent-test identity check failed: expected 4,003 events / "
            f"1,283 conversations; got {len(events)} / {len(set(e['conv'] for e in events.values()))}."
        )

    # Baseline and intervention are evaluated on identical frozen candidate pools.
    # Each method receives an independent ledger, with the exact 22.18 gate and
    # exposure-update functions used during validation.
    baseline_ledger = ref.ExposureLedger()
    proposed_ledger = ref.ExposureLedger()
    baseline_rows, proposed_rows = [], []
    top_k = int(getattr(gate_ref, "TOP_K", 10))

    for event_id in sorted(events):
        event = events[event_id]
        candidate = candidates[event_id]
        ids = candidate["ids"]

        baseline_ranked = gate_ref.baseline_rank(candidate)
        baseline_hit, baseline_ndcg = gate_ref.ranking_metrics(
            baseline_ranked, event["target"]
        )
        baseline_rank_pos = np.flatnonzero(
            baseline_ranked[:top_k] == int(event["target"])
        )
        baseline_rank = int(baseline_rank_pos[0]) + 1 if len(baseline_rank_pos) else 0
        gate_ref.exposure_update(
            baseline_ledger, candidate, baseline_ranked, item2group, categories
        )

        proposed_ranked, info = gate_ref.contextual_gate_rerank(
            candidate, item2group, categories, proposed_ledger, lam, tau
        )
        proposed_hit, proposed_ndcg = gate_ref.ranking_metrics(
            proposed_ranked, event["target"]
        )
        proposed_rank_pos = np.flatnonzero(
            proposed_ranked[:top_k] == int(event["target"])
        )
        proposed_rank = int(proposed_rank_pos[0]) + 1 if len(proposed_rank_pos) else 0
        gate_ref.exposure_update(
            proposed_ledger, candidate, proposed_ranked, item2group, categories
        )

        baseline_order = np.lexsort((ids, -candidate["scores"]))
        baseline_groups = np.asarray(
            [item2group[int(movie)] for movie in baseline_ranked], dtype=int
        )
        proposed_groups = np.asarray(
            [item2group[int(movie)] for movie in proposed_ranked], dtype=int
        )

        baseline_rows.append({
            "event_id": event_id, "conv_id": event["conv"],
            "target_movie_id": event["target"],
            "hit10": int(baseline_hit), "ndcg10": float(baseline_ndcg),
            "target_rank_top500": baseline_rank,
            "top10_movie_ids": json.dumps(list(map(int, baseline_ranked[:top_k]))),
            "top10_group_ids": json.dumps(list(map(int, baseline_groups[:top_k]))),
        })
        proposed_rows.append({
            "event_id": event_id, "conv_id": event["conv"],
            "target_movie_id": event["target"],
            "hit10": int(proposed_hit), "ndcg10": float(proposed_ndcg),
            "target_rank_top500": proposed_rank,
            "top10_movie_ids": json.dumps(list(map(int, proposed_ranked[:top_k]))),
            "top10_group_ids": json.dumps(list(map(int, proposed_groups[:top_k]))),
            **info,
        })

    baseline_df = pd.DataFrame(baseline_rows)
    proposed_df = pd.DataFrame(proposed_rows)

    def summarize(name: str, frame: pd.DataFrame, ledger) -> dict:
        return {
            "method": name,
            "events": int(len(frame)),
            "HR@10": float(frame.hit10.mean()),
            "NDCG@10": float(frame.ndcg10.mean()),
            "category_group_log_rate_MAD": float(ref.fairness_mad(ledger)),
            "mean_target_rank_top500": float(
                frame.loc[frame.target_rank_top500 > 0, "target_rank_top500"].mean()
            ) if (frame.target_rank_top500 > 0).any() else None,
        }

    summary = pd.DataFrame([
        summarize("A0_KBRD_baseline", baseline_df, baseline_ledger),
        summarize("proposed_context_kg_gated", proposed_df, proposed_ledger),
    ])
    baseline_fairness = float(summary.loc[0, "category_group_log_rate_MAD"])
    proposed_fairness = float(summary.loc[1, "category_group_log_rate_MAD"])
    baseline_ndcg = float(summary.loc[0, "NDCG@10"])
    proposed_ndcg = float(summary.loc[1, "NDCG@10"])
    retention = proposed_ndcg / baseline_ndcg if baseline_ndcg > 0 else None

    comparison = pd.DataFrame([{
        "baseline_HR@10": float(summary.loc[0, "HR@10"]),
        "proposed_HR@10": float(summary.loc[1, "HR@10"]),
        "delta_HR@10": float(summary.loc[1, "HR@10"] - summary.loc[0, "HR@10"]),
        "baseline_NDCG@10": baseline_ndcg,
        "proposed_NDCG@10": proposed_ndcg,
        "delta_NDCG@10": proposed_ndcg - baseline_ndcg,
        "NDCG_retention": retention,
        "baseline_category_group_log_rate_MAD": baseline_fairness,
        "proposed_category_group_log_rate_MAD": proposed_fairness,
        "delta_category_group_log_rate_MAD": proposed_fairness - baseline_fairness,
        "fairness_improved": bool(proposed_fairness < baseline_fairness),
        "utility_retained_95pct": bool(retention is not None and retention >= 0.95),
        "independent_test_success": bool(
            proposed_fairness < baseline_fairness
            and retention is not None and retention >= 0.95
        ),
    }])

    args.out.mkdir(parents=True, exist_ok=True)
    baseline_df.to_csv(args.out / "baseline_event_results.csv", index=False)
    proposed_df.to_csv(args.out / "proposed_event_results.csv", index=False)
    summary.to_csv(args.out / "aggregate_metrics.csv", index=False)
    comparison.to_csv(args.out / "final_comparison.csv", index=False)

    manifest = {
        "experiment": "22.24B_full_framework_independent_test",
        "dataset": "ReDial",
        "split": "independent_test",
        "events": len(events),
        "unique_conversations": len(set(e["conv"] for e in events.values())),
        "candidate_pool": "Frozen KBRD Top-500; identical for both methods",
        "kg_graph": "R1_raw",
        "kg_source_file": str(kg_source),
        "kg_rank_transfer": "(event_id, movie_id); missing support rank=0",
        "selected_lambda": lam,
        "selected_tau": tau,
        "parameters_source": str(args.validation_manifest),
        "selection_status": validation_manifest.get("selection_status"),
        "reranking_mode": "context_kg_gated",
        "context_definition": "normalized category prevalence in frozen Top-500 candidate pool; not semantic dialogue relevance",
        "kg_signal_interpretation": "retrieval support/rank confidence; not semantic relevance",
        "training_performed": False,
        "parameter_search_performed": False,
        "kg_retrieval_performed": False,
        "frozen_kbrd_predictions_reused": True,
        "frozen_kgsf_kg_export_reused": True,
        "category_profiles_reused": str(args.category_profiles),
        "reranking_implementation": str(args.gate_reference),
        "exposure_ledger_implementation": str(args.reference),
        "catalog_metadata_reused": str(args.catalog_pickle),
        "results": comparison.iloc[0].to_dict(),
        "outputs": [
            "baseline_event_results.csv", "proposed_event_results.csv",
            "aggregate_metrics.csv", "final_comparison.csv",
            "experiment_manifest.json",
        ],
    }
    (args.out / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=lambda x: bool(x) if isinstance(x, np.bool_) else float(x)),
        encoding="utf-8",
    )

    print("\n========== EXPERIMENT 22.24B COMPLETE ==========")
    print(summary.to_string(index=False))
    print("\nFinal comparison:")
    print(comparison.to_string(index=False))
    print("\nOutput:", args.out.resolve())
    print("Training: NO | Parameter search: NO | Retrieval rerun: NO")


if __name__ == "__main__":
    main()
