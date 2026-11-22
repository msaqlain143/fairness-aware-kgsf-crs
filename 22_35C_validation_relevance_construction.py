#!/usr/bin/env python
"""
Experiment 22.35C — Validation-only operational relevance construction.

Reuses the exact 22.31 construct_q formulation with validation exports from
22.35A/22.35B. Produces separate q_ei files for R0_filtered and R1_raw so
the graph variant is not silently selected or conflated.

No training, retrieval, target use in q, parameter tuning, or ranking changes.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
BASE = ROOT / "results_corrected" / "experiment_22_35"
TOP500_PATH = BASE / "kgsf_validation" / "kgsf_validation_top500.csv"
SCORES_PATH = BASE / "kgsf_validation" / "kgsf_validation_full_scores.npz"
KG_PATH = BASE / "kg_validation" / "kgsf_validation_kg_ranks.csv"
MANIFEST_PATH = BASE / "kgsf_validation" / "kgsf_validation_event_manifest.csv"
OUT_DIR = BASE / "relevance_validation"

EXPECTED_EVENTS = 3010
EXPECTED_CANDIDATES = 500
EXPECTED_CATALOG = 6924
KG_TAU = 50.0
EPS = 1e-12

KG_VARIANTS = {
    "R0_filtered": "R0_filtered_candidate_kg_ranks",
    "R1_raw": "R1_raw_candidate_kg_ranks",
}


def parse_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [int(x) for x in value]
    if pd.isna(value):
        return []
    s = str(value).strip()
    try:
        return [int(x) for x in json.loads(s)]
    except Exception:
        return [int(x) for x in ast.literal_eval(s)]


def stable_softmax(values):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("KGSF candidate scores must be a nonempty finite vector.")
    exp_x = np.exp(x - np.max(x))
    denom = float(exp_x.sum())
    if not np.isfinite(denom) or denom <= 0:
        raise RuntimeError("Invalid softmax denominator.")
    return exp_x / denom


def load_and_validate():
    for path in (TOP500_PATH, SCORES_PATH, KG_PATH, MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required 22.35 input is missing: {path}")

    top = pd.read_csv(TOP500_PATH).sort_values("event_id").reset_index(drop=True)
    kg = pd.read_csv(KG_PATH).sort_values("event_id").reset_index(drop=True)
    manifest = pd.read_csv(MANIFEST_PATH).sort_values("event_id").reset_index(drop=True)

    required_top = {"event_id", "conv_id", "top_movie_ids", "top_scores"}
    required_kg = {
        "event_id", "conv_id", "candidate_movie_ids",
        *KG_VARIANTS.values(),
    }
    required_manifest = {"event_id", "conv_id", "score_row"}
    for label, frame, required in (
        ("Top-500", top, required_top),
        ("KG", kg, required_kg),
        ("event manifest", manifest, required_manifest),
    ):
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"{label} export missing columns: {sorted(missing)}")

    if not (len(top) == len(kg) == len(manifest) == EXPECTED_EVENTS):
        raise RuntimeError("Expected exactly 3,010 rows in all event-level inputs.")

    for label, frame in (("Top-500", top), ("KG", kg), ("manifest", manifest)):
        if frame.event_id.duplicated().any():
            raise RuntimeError(f"Duplicate event_id in {label} input.")

    for col in ("event_id", "conv_id"):
        if not np.array_equal(top[col].to_numpy(), kg[col].to_numpy()):
            raise RuntimeError(f"Top-500 and KG {col} alignment failed.")
        if not np.array_equal(top[col].to_numpy(), manifest[col].to_numpy()):
            raise RuntimeError(f"Top-500 and manifest {col} alignment failed.")

    npz = np.load(SCORES_PATH, allow_pickle=False)
    required_npz = {"scores", "movie_ids", "event_ids"}
    if not required_npz.issubset(npz.files):
        raise RuntimeError(f"NPZ missing keys: {sorted(required_npz - set(npz.files))}")

    score_matrix = np.asarray(npz["scores"], dtype=np.float64)
    movie_ids = np.asarray(npz["movie_ids"], dtype=np.int64)
    score_event_ids = np.asarray(npz["event_ids"], dtype=np.int64)

    if score_matrix.shape != (EXPECTED_EVENTS, EXPECTED_CATALOG):
        raise RuntimeError(f"Unexpected score matrix shape: {score_matrix.shape}")
    if len(movie_ids) != EXPECTED_CATALOG or len(set(movie_ids.tolist())) != EXPECTED_CATALOG:
        raise RuntimeError("Invalid movie_ids catalog vector.")
    if not np.isfinite(score_matrix).all():
        raise RuntimeError("Non-finite values in frozen KGSF score matrix.")
    if not np.array_equal(score_event_ids, top.event_id.to_numpy(dtype=np.int64)):
        raise RuntimeError("NPZ event order does not match sorted Top-500 events.")

    movie_to_col = {int(mid): j for j, mid in enumerate(movie_ids)}
    return top, kg, manifest, score_matrix, movie_ids, movie_to_col


def construct_variant(top, kg, score_matrix, movie_to_col, variant, rank_column):
    records = []
    event_sums = []
    total_supported = 0
    total_unsupported = 0
    rank_values = []

    for row_idx in range(len(top)):
        t = top.iloc[row_idx]
        k = kg.iloc[row_idx]
        event_id = int(t.event_id)
        conv_id = int(t.conv_id)

        candidate_ids = parse_list(t.top_movie_ids)
        exported_scores = np.asarray(parse_numeric_list(t.top_scores), dtype=np.float64)
        kg_candidate_ids = parse_list(k.candidate_movie_ids)
        ranks = np.asarray(parse_list(k[rank_column]), dtype=np.int64)

        if not (len(candidate_ids) == len(exported_scores) == len(ranks) == EXPECTED_CANDIDATES):
            raise RuntimeError(f"Event {event_id}: candidate/score/KG-rank lengths are not 500.")
        if len(set(candidate_ids)) != EXPECTED_CANDIDATES:
            raise RuntimeError(f"Event {event_id}: duplicate candidate movie IDs.")
        if candidate_ids != kg_candidate_ids:
            raise RuntimeError(f"Event {event_id}: KG candidate order differs from frozen KGSF order.")
        if np.any(ranks < 0):
            raise RuntimeError(f"Event {event_id}: negative KG rank.")
        if any(mid not in movie_to_col for mid in candidate_ids):
            raise RuntimeError(f"Event {event_id}: candidate absent from frozen catalog.")

        cols = np.asarray([movie_to_col[mid] for mid in candidate_ids], dtype=np.int64)
        frozen_scores = score_matrix[row_idx, cols]
        if not np.allclose(frozen_scores, exported_scores, rtol=1e-6, atol=1e-7):
            delta = float(np.max(np.abs(frozen_scores - exported_scores)))
            raise RuntimeError(f"Event {event_id}: Top-500/NPZ score mismatch; max diff={delta}")

        p_kgsf = stable_softmax(frozen_scores)

        kg_conf = np.zeros(EXPECTED_CANDIDATES, dtype=np.float64)
        supported = ranks > 0
        kg_conf[supported] = np.exp(-ranks[supported].astype(np.float64) / KG_TAU)

        # Exact 22.31 operational construction:
        # q_raw = softmax(KGSF score) * (1 + KG confidence), then event normalize.
        q_raw = p_kgsf * (1.0 + kg_conf)
        denom = float(q_raw.sum())
        if not np.isfinite(denom) or denom <= 0:
            raise RuntimeError(f"Event {event_id}: invalid q normalization denominator.")
        q = q_raw / denom

        if not np.isclose(q.sum(), 1.0, rtol=0.0, atol=1e-10):
            raise RuntimeError(f"Event {event_id}: q does not sum to one.")

        event_sums.append(float(q.sum()))
        total_supported += int(supported.sum())
        total_unsupported += int((~supported).sum())
        rank_values.extend(ranks.tolist())

        for j, movie_id in enumerate(candidate_ids):
            records.append({
                "event_id": event_id,
                "conv_id": conv_id,
                "candidate_rank": j + 1,
                "movie_id": int(movie_id),
                "k_gsf_score": float(frozen_scores[j]),
                "k_gsf_candidate_probability": float(p_kgsf[j]),
                "kg_rank": int(ranks[j]),
                "kg_confidence": float(kg_conf[j]),
                "q_ei": float(q[j]),
            })

    out = pd.DataFrame.from_records(records)
    if len(out) != EXPECTED_EVENTS * EXPECTED_CANDIDATES:
        raise RuntimeError("Unexpected candidate-level output row count.")
    if out.duplicated(["event_id", "movie_id"]).any():
        raise RuntimeError("Duplicate event-candidate pairs in q output.")
    if not np.isfinite(out[["k_gsf_score", "k_gsf_candidate_probability",
                            "kg_confidence", "q_ei"]].to_numpy()).all():
        raise RuntimeError("Non-finite relevance output.")
    if (out.q_ei < 0).any() or (out.kg_confidence < 0).any():
        raise RuntimeError("Negative relevance/confidence value.")
    if (out.loc[out.kg_rank == 0, "kg_confidence"] != 0).any():
        raise RuntimeError("Unsupported candidates must have zero KG confidence.")

    grouped = out.groupby("event_id").q_ei.sum()
    stats = {
        "variant": variant,
        "kg_rank_column": rank_column,
        "events": int(out.event_id.nunique()),
        "candidate_instances": int(len(out)),
        "kg_tau": KG_TAU,
        "q_sum_min": float(grouped.min()),
        "q_sum_max": float(grouped.max()),
        "q_sum_mean": float(grouped.mean()),
        "q_min": float(out.q_ei.min()),
        "q_max": float(out.q_ei.max()),
        "q_mean": float(out.q_ei.mean()),
        "positive_q_instances": int((out.q_ei > 0).sum()),
        "kg_supported_candidates": total_supported,
        "kg_unsupported_candidates": total_unsupported,
        "kg_coverage": float(total_supported / len(out)),
        "kg_confidence_min": float(out.kg_confidence.min()),
        "kg_confidence_median": float(out.kg_confidence.median()),
        "kg_confidence_max": float(out.kg_confidence.max()),
        "event_normalization_pass": bool(np.allclose(grouped.to_numpy(), 1.0, atol=1e-10)),
        "unsupported_confidence_zero_pass": True,
        "candidate_alignment_pass": True,
        "score_matrix_alignment_pass": True,
    }
    return out, stats


def parse_numeric_list(value):
    if isinstance(value, (list, tuple, np.ndarray)):
        return [float(x) for x in value]
    if pd.isna(value):
        return []
    s = str(value).strip()
    try:
        return [float(x) for x in json.loads(s)]
    except Exception:
        return [float(x) for x in ast.literal_eval(s)]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 76)
    print("EXPERIMENT 22.35C — VALIDATION RELEVANCE CONSTRUCTION")
    print("=" * 76)

    top, kg, manifest, score_matrix, movie_ids, movie_to_col = load_and_validate()
    print(f"Events: {len(top)} | conversations: {top.conv_id.nunique()}")
    print(f"Score matrix: {score_matrix.shape} | candidates/event: {EXPECTED_CANDIDATES}")
    print("Event, conversation, candidate and frozen-score alignment: PASS")

    all_stats = {}
    for variant, rank_column in KG_VARIANTS.items():
        q_df, stats = construct_variant(
            top, kg, score_matrix, movie_to_col, variant, rank_column
        )
        out_path = OUT_DIR / f"event_candidate_q_ei_{variant}.csv"
        q_df.to_csv(out_path, index=False)
        (OUT_DIR / f"q_statistics_{variant}.json").write_text(
            json.dumps(stats, indent=2), encoding="utf-8"
        )
        all_stats[variant] = stats
        print(f"\n[{variant}]")
        print(f"KG coverage: {stats['kg_coverage']:.4%}")
        print(f"q range: {stats['q_min']:.10g} to {stats['q_max']:.10g}")
        print(f"Event q-sum range: {stats['q_sum_min']:.12f} to {stats['q_sum_max']:.12f}")
        print(f"Output: {out_path}")

    manifest_out = {
        "experiment": "22.35C_validation_relevance_construction",
        "split": "validation",
        "input_sources": {
            "kgsf_top500": str(TOP500_PATH),
            "kgsf_full_scores": str(SCORES_PATH),
            "kg_validation_evidence": str(KG_PATH),
            "event_manifest": str(MANIFEST_PATH),
        },
        "events": EXPECTED_EVENTS,
        "unique_conversations": int(top.conv_id.nunique()),
        "catalog_movies": EXPECTED_CATALOG,
        "candidates_per_event": EXPECTED_CANDIDATES,
        "kg_tau": KG_TAU,
        "q_definition": (
            "p_KGSF=softmax(frozen KGSF Top-500 scores); "
            "KG_conf=exp(-rank/tau) for rank>0 and 0 otherwise; "
            "q=normalize(p_KGSF*(1+KG_conf)) within each event."
        ),
        "target_used_to_construct_q": False,
        "future_dialogue_used": False,
        "training_performed": False,
        "retrieval_performed": False,
        "parameter_search_performed": False,
        "ranking_changed": False,
        "independent_test_evaluated": False,
        "validated_contextual_relevance": False,
        "scientific_limitation": (
            "q is an operational relevance signal derived from frozen KGSF "
            "scores and KG evidence; it is not a calibrated probability of "
            "user acceptance or independently validated contextual relevance."
        ),
        "graph_variants_kept_separate": True,
        "variant_statistics": all_stats,
    }
    (OUT_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest_out, indent=2), encoding="utf-8"
    )

    checks = {
        "expected_events": len(top) == EXPECTED_EVENTS,
        "expected_unique_conversations": int(top.conv_id.nunique()) == 883,
        "expected_candidate_rows_per_variant": EXPECTED_EVENTS * EXPECTED_CANDIDATES,
        "both_graph_variants_exported": set(all_stats) == set(KG_VARIANTS),
        "event_normalization_all_variants": all(
            s["event_normalization_pass"] for s in all_stats.values()
        ),
        "unsupported_kg_confidence_zero_all_variants": all(
            s["unsupported_confidence_zero_pass"] for s in all_stats.values()
        ),
        "candidate_alignment_all_variants": all(
            s["candidate_alignment_pass"] for s in all_stats.values()
        ),
        "score_alignment_all_variants": all(
            s["score_matrix_alignment_pass"] for s in all_stats.values()
        ),
        "training_performed": False,
        "retrieval_performed": False,
        "target_used_to_construct_q": False,
        "independent_test_evaluated": False,
    }
    checks["overall_pass"] = all(
        v for k, v in checks.items()
        if isinstance(v, bool) and k not in {
            "training_performed", "retrieval_performed",
            "target_used_to_construct_q", "independent_test_evaluated"
        }
    )
    (OUT_DIR / "validation_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8"
    )
    print("\n" + "=" * 76)
    print("EXPERIMENT 22.35C " + ("COMPLETE — PASS" if checks["overall_pass"] else "FAILED"))
    print("Training: NO | Retrieval: NO | Target used for q: NO | Test evaluated: NO")
    print(f"Output: {OUT_DIR.resolve()}")
    if not checks["overall_pass"]:
        raise RuntimeError(f"22.35C validation checks failed: {checks}")


if __name__ == "__main__":
    main()


