#!/usr/bin/env python
"""
Experiment 22.35J — Integrated KG-FairCRS independent-test evaluation.

Uses frozen KGSF Top-500 predictions and frozen KGSF-aligned KG ranks.
Constructs test-event operational relevance with the exact 22.35C formula.
Reconstructs the full validation exposure history from frozen 22.35F slates,
then evaluates A0-A3 on the independent-test split with frozen parameters.

No training, retrieval, checkpoint changes, or test-based parameter selection.
R0_filtered is primary. A3/R1_raw is sensitivity using R0-selected parameters.

Important provenance limitation:
The ReDial independent-test split was examined in historical experiments
22.22-22.34. This is therefore a frozen-configuration re-evaluation with
prior test exposure disclosed, not a never-before-seen test set.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")
BASE35 = ROOT / "results_corrected" / "experiment_22_35"
TEST_BASE = ROOT / "results_final_test_m2"
TOP_K = 10
EXPECTED_TEST_EVENTS = 4003
EXPECTED_TEST_CONVERSATIONS = 1283
EXPECTED_VAL_EVENTS = 3010
EXPECTED_VAL_CONVERSATIONS = 883
EXPECTED_CANDIDATES = 500
EXPECTED_CATALOG = 6924
KG_TAU = 50.0
EPS = 1e-12
BOOT_REPS = 10000
BOOT_SEED = 22359

SYSTEMS = ("A0_KGSF", "A1_22_16", "A2_22_18", "A3_KG_FairCRS")

D = {
    "test_top": TEST_BASE / "experiment_22_27A" / "kgsf_independent_test_top500.csv",
    "test_scores": TEST_BASE / "experiment_22_27A" / "kgsf_independent_test_full_scores.npz",
    "test_kg": TEST_BASE / "experiment_22_27A2" / "kgsf_independent_test_kg_ranks.csv",
    "test_manifest": TEST_BASE / "experiment_22_27A" / "kgsf_independent_test_event_manifest.csv",
    "test_mapping": TEST_BASE / "m2_test_event_conversation_mapping.csv",
    "val_top": BASE35 / "kgsf_validation" / "kgsf_validation_top500.csv",
    "val_q0": BASE35 / "relevance_validation" / "event_candidate_q_ei_R0_filtered.csv",
    "val_q1": BASE35 / "relevance_validation" / "event_candidate_q_ei_R1_raw.csv",
    "val_event": BASE35 / "validated_fairness_evaluation_run_20261121" / "event_level_system_comparisons.csv",
    "decisions": BASE35 / "validated_fairness_evaluation_run_20261121" / "parameter_decisions.json",
    "h_manifest": BASE35 / "inferential_robustness_sensitivity_FINAL2" / "experiment_manifest.json",
    "catalog": ROOT / "catalog_audit_corrected.pkl",
    "profiles": ROOT / "results_category_audit" / "experiment_21_22A" / "catalog_item_category_profiles.csv",
    "ref16": ROOT / "22_16_category_context_utility_calibration.py",
    "ref18": ROOT / "22_18_contextual_relevance_gate_validation.py",
    "f_script": ROOT / "22_35F_validated_fairness_evaluation.py",
}

def parse_list(value, field="value"):
    if isinstance(value, list):
        out = value
    elif isinstance(value, (tuple, np.ndarray)):
        out = list(value)
    elif pd.isna(value):
        out = []
    else:
        s = str(value).strip()
        try:
            out = json.loads(s)
        except Exception:
            out = ast.literal_eval(s)
    if not isinstance(out, (list, tuple)):
        raise ValueError(f"{field} must contain a list")
    return list(out)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def safe(x):
    if isinstance(x, dict):
        return {str(k): safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [safe(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    if isinstance(x, Path):
        return str(x)
    return x

def write_json(path: Path, obj):
    path.write_text(json.dumps(safe(obj), indent=2, allow_nan=False), encoding="utf-8")

def import_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Candidate scores must be a finite non-empty vector.")
    z = np.exp(x - np.max(x))
    return z / z.sum()

def build_q(scores, ranks):
    scores = np.asarray(scores, dtype=np.float64)
    ranks = np.asarray(ranks, dtype=np.int64)
    if len(scores) != EXPECTED_CANDIDATES or len(ranks) != EXPECTED_CANDIDATES:
        raise ValueError("Expected 500 scores and KG ranks per event.")
    if (ranks < 0).any():
        raise ValueError("Negative KG rank encountered.")
    p = softmax(scores)
    conf = np.zeros(len(ranks), dtype=np.float64)
    supported = ranks > 0
    conf[supported] = np.exp(-ranks[supported].astype(np.float64) / KG_TAU)
    raw = p * (1.0 + conf)
    q = raw / raw.sum()
    if not np.isfinite(q).all() or not np.isclose(q.sum(), 1.0, atol=1e-10):
        raise ValueError("Invalid event-normalized operational relevance.")
    return q

def load_test_inputs(a, catalog):
    top = pd.read_csv(a.test_top).sort_values("event_id").reset_index(drop=True)
    kg = pd.read_csv(a.test_kg).sort_values("event_id").reset_index(drop=True)
    required_top = {"event_id", "conv_id", "target_movie_id", "top_movie_ids"}
    required_kg = {
        "event_id", "conv_id", "candidate_movie_ids",
        "R0_filtered_candidate_kg_ranks", "R1_raw_candidate_kg_ranks",
    }
    if required_top - set(top.columns):
        raise ValueError(f"Test Top-500 missing columns: {sorted(required_top-set(top.columns))}")
    if required_kg - set(kg.columns):
        raise ValueError(f"Test KG export missing columns: {sorted(required_kg-set(kg.columns))}")
    if len(top) != EXPECTED_TEST_EVENTS or len(kg) != EXPECTED_TEST_EVENTS:
        raise ValueError("Expected exactly 4,003 test events in Top-500 and KG exports.")
    for label, df in (("test Top-500", top), ("test KG", kg)):
        if df.event_id.duplicated().any():
            raise ValueError(f"Duplicate event_id in {label}.")
    if not np.array_equal(top.event_id.to_numpy(), kg.event_id.to_numpy()):
        raise ValueError("Test Top-500/KG event IDs do not align.")
    if not np.array_equal(top.conv_id.to_numpy(), kg.conv_id.to_numpy()):
        raise ValueError("Test Top-500/KG conversation IDs do not align.")

    # Validate the separately exported event manifest/mapping when available.
    for optional in (a.test_manifest, a.test_mapping):
        if optional and optional.is_file():
            m = pd.read_csv(optional)
            req = {"event_id", "conv_id"}
            if req.issubset(m.columns):
                m = m.sort_values("event_id").reset_index(drop=True)
                if len(m) != len(top) or not np.array_equal(m.event_id, top.event_id) or not np.array_equal(m.conv_id, top.conv_id):
                    raise ValueError(f"Independent-test mapping does not align: {optional}")
                if "target_movie_id" in m.columns and not np.array_equal(m.target_movie_id, top.target_movie_id):
                    raise ValueError(f"Target alignment failure in {optional}")

    npz = np.load(a.test_scores, allow_pickle=False)
    needed = {"scores", "movie_ids", "event_ids"}
    if not needed.issubset(set(npz.files)):
        raise ValueError(f"Test score NPZ missing keys: {sorted(needed-set(npz.files))}")
    score_matrix = npz["scores"]
    movie_ids = np.asarray(npz["movie_ids"], dtype=np.int64)
    score_event_ids = np.asarray(npz["event_ids"], dtype=np.int64)
    if score_matrix.shape != (EXPECTED_TEST_EVENTS, EXPECTED_CATALOG):
        raise ValueError(f"Unexpected test score matrix shape {score_matrix.shape}")
    if len(np.unique(movie_ids)) != EXPECTED_CATALOG or set(map(int, movie_ids)) != catalog:
        raise ValueError("Test score-matrix catalog does not match frozen catalog.")
    if len(np.unique(score_event_ids)) != EXPECTED_TEST_EVENTS:
        raise ValueError("Duplicate event IDs in test score matrix.")
    score_row = {int(e): i for i, e in enumerate(score_event_ids)}
    movie_col = {int(mid): j for j, mid in enumerate(movie_ids)}

    events, cands0, cands1 = {}, {}, {}
    q_stats = {v: {"q_sum": [], "kg_supported": 0, "instances": 0} for v in ("R0_filtered", "R1_raw")}
    for tr, kr in zip(top.itertuples(index=False), kg.itertuples(index=False)):
        eid, conv, target = int(tr.event_id), int(tr.conv_id), int(tr.target_movie_id)
        ids = np.asarray(parse_list(tr.top_movie_ids, "top_movie_ids"), dtype=np.int64)
        kg_ids = np.asarray(parse_list(kr.candidate_movie_ids, "candidate_movie_ids"), dtype=np.int64)
        if len(ids) != EXPECTED_CANDIDATES or len(np.unique(ids)) != EXPECTED_CANDIDATES:
            raise ValueError(f"Invalid Top-500 candidates at event {eid}.")
        if not np.array_equal(ids, kg_ids):
            raise ValueError(f"Candidate order mismatch between KGSF and KG at event {eid}.")
        if not set(map(int, ids)).issubset(catalog):
            raise ValueError(f"Out-of-catalog candidate at event {eid}.")
        if eid not in score_row:
            raise ValueError(f"Missing score-matrix row for event {eid}.")
        scores = np.asarray(score_matrix[score_row[eid], [movie_col[int(x)] for x in ids]], dtype=np.float64)
        if not np.isfinite(scores).all():
            raise ValueError(f"Non-finite KGSF score at event {eid}.")
        if hasattr(tr, "top_scores"):
            exported = np.asarray(parse_list(tr.top_scores, "top_scores"), dtype=np.float64)
            if len(exported) != EXPECTED_CANDIDATES or not np.allclose(scores, exported, rtol=1e-6, atol=1e-7):
                raise ValueError(f"Top-500 scores do not match frozen score matrix at event {eid}.")
        r0 = np.asarray(parse_list(getattr(kr, "R0_filtered_candidate_kg_ranks"), "R0 KG ranks"), dtype=np.int64)
        r1 = np.asarray(parse_list(getattr(kr, "R1_raw_candidate_kg_ranks"), "R1 KG ranks"), dtype=np.int64)
        if len(r0) != EXPECTED_CANDIDATES or len(r1) != EXPECTED_CANDIDATES:
            raise ValueError(f"KG rank length mismatch at event {eid}.")
        q0, q1 = build_q(scores, r0), build_q(scores, r1)
        events[eid] = {"conv": conv, "target": target}
        cands0[eid] = {"ids": ids, "scores": scores, "kg": r0, "q": q0}
        cands1[eid] = {"ids": ids, "scores": scores, "kg": r1, "q": q1}
        for name, q, ranks in (("R0_filtered", q0, r0), ("R1_raw", q1, r1)):
            q_stats[name]["q_sum"].append(float(q.sum()))
            q_stats[name]["kg_supported"] += int((ranks > 0).sum())
            q_stats[name]["instances"] += len(ranks)

    if len(events) != EXPECTED_TEST_EVENTS or len({v["conv"] for v in events.values()}) != EXPECTED_TEST_CONVERSATIONS:
        raise ValueError("Independent-test event/conversation counts do not match 4,003/1,283.")
    stats = {}
    for name, s in q_stats.items():
        stats[name] = {
            "events": EXPECTED_TEST_EVENTS,
            "candidate_instances": s["instances"],
            "q_sum_min": min(s["q_sum"]),
            "q_sum_max": max(s["q_sum"]),
            "q_sum_mean": float(np.mean(s["q_sum"])),
            "kg_coverage": s["kg_supported"] / s["instances"],
            "kg_tau": KG_TAU,
        }
    return events, cands0, cands1, stats

def load_validation(a, catalog, F, ref16):
    val_events, val_cands0 = F.load_data(a.val_top, a.val_q0, catalog)
    _, val_cands1 = F.load_data(a.val_top, a.val_q1, catalog)
    event_frame = pd.read_csv(a.val_event)
    required = {"system", "event_id", "conv_id", "split", "variant", "top10_movie_ids"}
    if required - set(event_frame.columns):
        raise ValueError(f"Validation event-level output missing: {sorted(required-set(event_frame.columns))}")
    if len(val_events) != EXPECTED_VAL_EVENTS or len({v["conv"] for v in val_events.values()}) != EXPECTED_VAL_CONVERSATIONS:
        raise ValueError("Validation history is not the expected 3,010 events/883 conversations.")
    warm_c, tune_c, held_c = ref16.split_conversations([v["conv"] for v in val_events.values()])
    warm = sorted(e for e,v in val_events.items() if v["conv"] in warm_c)
    tune = sorted(e for e,v in val_events.items() if v["conv"] in tune_c)
    held = sorted(e for e,v in val_events.items() if v["conv"] in held_c)
    if (len(warm), len(tune), len(held)) != (1817, 569, 624):
        raise ValueError(f"Validation split mismatch: warm/tune/held={len(warm)}/{len(tune)}/{len(held)}")
    # The frozen 22.35F outputs contain tuning and held-out slates; warm history
    # is the original frozen KGSF ordering, exactly as in 22.35F.
    slates = {}
    for row in event_frame.itertuples(index=False):
        sysname, variant, split, eid = str(row.system), str(row.variant), str(row.split), int(row.event_id)
        if split not in {"tune", "heldout"}:
            continue
        if sysname not in SYSTEMS or eid not in val_events:
            raise ValueError(f"Unexpected validation history row: {sysname}/{eid}")
        key_variant = "R1_raw" if sysname == "A3_KG_FairCRS" and variant == "R1_raw" else "R0_filtered"
        if variant != key_variant:
            continue
        slate = np.asarray(parse_list(row.top10_movie_ids, "validation top10"), dtype=np.int64)
        if len(slate) != TOP_K or len(np.unique(slate)) != TOP_K:
            raise ValueError(f"Invalid validation top-10 slate: {sysname}/{variant}/{eid}")
        if not set(map(int, slate)).issubset(set(map(int, val_cands0[eid]["ids"]))):
            raise ValueError(f"Validation slate contains non-candidate item: {sysname}/{eid}")
        slates[(sysname, key_variant, eid)] = slate
    # Require complete tune+heldout slates for every relevant system/variant.
    for sysname in SYSTEMS:
        variants = ("R0_filtered", "R1_raw") if sysname == "A3_KG_FairCRS" else ("R0_filtered",)
        for variant in variants:
            missing = [e for e in tune + held if (sysname, variant, e) not in slates]
            if missing:
                raise ValueError(f"Missing {len(missing)} frozen validation slates for {sysname}/{variant}; examples={missing[:5]}")
    return val_events, val_cands0, val_cands1, warm, tune, held, slates

def reconstruct_history(system, variant, val_events, val_cands0, val_cands1,
                         warm, tune, held, slates, item2group, categories,
                         ref16, F, qthr, rmin):
    legacy = ref16.ExposureLedger()
    prop = F.ProposalLedger()
    v_cands = val_cands1 if variant == "R1_raw" else val_cands0
    for split_ids in (warm, tune, held):
        for eid in split_ids:
            c = v_cands[eid]
            if eid in warm:
                ranked = c["ids"][:TOP_K]
            else:
                ranked = slates[(system, variant, eid)]
            cg = F.groups(c, item2group)
            rg = np.asarray([item2group[int(mid)] for mid in ranked], dtype=int)
            legacy.update(c["ids"], cg, ranked, rg, categories)
            prop.add_R(c, item2group, categories, qthr)
            prop.add_E(ranked, item2group, categories)
    return legacy, prop

def evaluate_one(system, variant, events, cands, legacy0, prop0,
                 item2group, categories, ref16, ref18, F, params, qthr, rmin):
    legacy = legacy0.clone()
    prop = prop0.clone()
    rows = []
    for eid in sorted(events):
        event, c = events[eid], cands[eid]
        info = {}
        if system == "A0_KGSF":
            ranked = c["ids"][:TOP_K]
        elif system == "A1_22_16":
            ranked_full, _, info = ref16.rerank(
                c, item2group, categories, legacy, float(params["lam"]), "category"
            )
            ranked = np.asarray(ranked_full[:TOP_K], dtype=np.int64)
        elif system == "A2_22_18":
            ranked_full, info = ref18.contextual_gate_rerank(
                c, item2group, categories, legacy, float(params["lam"]), float(params["tau"])
            )
            ranked = np.asarray(ranked_full[:TOP_K], dtype=np.int64)
        elif system == "A3_KG_FairCRS":
            ranked, info = F.rank_a3(
                c, item2group, categories, prop, qthr, rmin, float(params["lam"])
            )
            ranked = np.asarray(ranked[:TOP_K], dtype=np.int64)
        else:
            raise ValueError(system)

        hit, ndcg, mrr = F.metric(ranked, event["target"])
        pos = np.flatnonzero(ranked == int(event["target"]))
        target_rank = int(pos[0]) + 1 if len(pos) else None
        kgmap = {int(mid): int(rank) for mid, rank in zip(c["ids"], c["kg"])}
        row = {
            "system": system, "variant": variant, "event_id": int(eid),
            "conv_id": int(event["conv"]), "target_movie_id": int(event["target"]),
            "HR@10": int(hit), "NDCG@10": float(ndcg), "MRR@10": float(mrr),
            "target_rank_top10": target_rank,
            "kg_supported_candidates": int(np.sum(c["kg"] > 0)),
            "kg_supported_top10": int(sum(kgmap.get(int(mid), 0) > 0 for mid in ranked)),
            "eligible_candidates": int(np.sum(c["q"] >= qthr)) if system == "A3_KG_FairCRS" else 0,
            "fallback_positions": int(info.get("fallback_positions", 0)),
            "top10_promoted_positions": int(info.get("top10_promoted_positions", 0)),
            "top10_movie_ids": json.dumps(list(map(int, ranked))),
        }
        rows.append(row)

        if system != "A3_KG_FairCRS":
            cg = F.groups(c, item2group)
            rg = np.asarray([item2group[int(mid)] for mid in ranked], dtype=int)
            legacy.update(c["ids"], cg, ranked, rg, categories)
            prop.add_R(c, item2group, categories, qthr)
            prop.add_E(ranked, item2group, categories)
    frame = pd.DataFrame(rows)
    fair, active, cells = F.proposal_terms(prop, rmin)
    pre_fair, pre_active, _ = F.proposal_terms(prop0, rmin)
    summary = {
        "system": system, "variant": variant,
        "events": int(len(frame)), "conversations": int(frame.conv_id.nunique()),
        "HR@10": float(frame["HR@10"].mean()),
        "NDCG@10": float(frame["NDCG@10"].mean()),
        "MRR@10": float(frame["MRR@10"].mean()),
        "L_fair_pretest_history": pre_fair,
        "active_categories_pretest_history": int(pre_active),
        "L_fair_validation_plus_test": fair,
        "active_categories_validation_plus_test": int(active),
        "mean_kg_supported_top10": float(frame.kg_supported_top10.mean()),
        "mean_eligible_candidates": float(frame.eligible_candidates.mean()),
        "mean_fallback_positions": float(frame.fallback_positions.mean()),
        "mean_top10_promoted_positions": float(frame.top10_promoted_positions.mean()),
    }
    for cell in cells:
        cell.update({"system": system, "variant": variant})
    return frame, summary, pd.DataFrame(cells), prop

def cluster_bootstrap(event_results, reps=BOOT_REPS, seed=BOOT_SEED):
    """Paired conversation-cluster percentile intervals for utility metrics."""
    primary = [
        ("A1_22_16", "R0_filtered"),
        ("A2_22_18", "R0_filtered"),
        ("A3_KG_FairCRS", "R0_filtered"),
    ]
    sensitivity = [("A3_KG_FairCRS", "R1_raw")]
    base = event_results[(event_results.system == "A0_KGSF") & (event_results.variant == "R0_filtered")]
    convs = np.asarray(sorted(base.conv_id.unique()), dtype=np.int64)
    metrics = ("HR@10", "NDCG@10", "MRR@10")
    # Per-conversation sums and event counts; cluster resampling preserves all
    # events within each sampled conversation.
    def aggregate(df):
        out = {}
        for m in metrics:
            grouped = df.groupby("conv_id")[m].agg(["sum", "count"])
            out[m] = (grouped["sum"].reindex(convs, fill_value=0).to_numpy(float),
                      grouped["count"].reindex(convs, fill_value=0).to_numpy(float))
        return out
    baseagg = aggregate(base)
    sysagg = {}
    for key in primary + sensitivity:
        df = event_results[(event_results.system == key[0]) & (event_results.variant == key[1])]
        if len(df) != len(base):
            raise ValueError(f"Bootstrap event alignment mismatch for {key}")
        if set(df.event_id.astype(int)) != set(base.event_id.astype(int)):
            raise ValueError(f"Bootstrap event-ID pairing mismatch for {key}")
        if not np.array_equal(np.sort(df.conv_id.unique()), convs):
            raise ValueError(f"Bootstrap conversation alignment mismatch for {key}")
        sysagg[key] = aggregate(df)
    rng = np.random.default_rng(seed)
    records = []
    all_keys = primary + sensitivity
    for key in all_keys:
        for m in metrics:
            deltas = np.empty(reps, dtype=np.float64)
            ssum, scount = sysagg[key][m]
            bsum, bcount = baseagg[m]
            for b in range(reps):
                draw = rng.integers(0, len(convs), size=len(convs))
                den = float(scount[draw].sum())
                bden = float(bcount[draw].sum())
                deltas[b] = (ssum[draw].sum() / den) - (bsum[draw].sum() / bden)
            observed = float(
                event_results.loc[(event_results.system == key[0]) &
                                  (event_results.variant == key[1]), m].mean()
                - base[m].mean()
            )
            records.append({
                "comparison": f"{key[0]}_{key[1]}_minus_A0_KGSF",
                "system": key[0], "variant": key[1], "metric": m,
                "observed_delta": observed,
                "bootstrap_ci95_low": float(np.percentile(deltas, 2.5)),
                "bootstrap_ci95_high": float(np.percentile(deltas, 97.5)),
                "bootstrap_reps": reps, "cluster_unit": "conversation",
                "ci_excludes_zero": bool(np.percentile(deltas, 2.5) > 0 or np.percentile(deltas, 97.5) < 0),
            })
    return pd.DataFrame(records)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key, default in D.items():
        ap.add_argument("--" + key.replace("_", "-"), type=Path, default=default)
    ap.add_argument("--out", type=Path, default=TEST_BASE / "experiment_22_35J")
    ap.add_argument("--bootstrap-reps", type=int, default=BOOT_REPS)
    ap.add_argument("--bootstrap-seed", type=int, default=BOOT_SEED)
    a = ap.parse_args()
    if a.bootstrap_reps < 1000:
        raise ValueError("Use at least 1,000 conversation-cluster bootstrap replicates.")
    required = [getattr(a, k) for k in D]
    for p in required:
        if not p.is_file():
            raise FileNotFoundError(f"Required 22.35J input missing: {p}")
    if a.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {a.out}")

    print("=" * 78)
    print("EXPERIMENT 22.35J — INTEGRATED KG-FairCRS INDEPENDENT-TEST EVALUATION")
    print("=" * 78)

    F = import_module(a.f_script, "frozen_2235f")
    ref16 = import_module(a.ref16, "frozen_2216")
    ref18 = import_module(a.ref18, "frozen_2218")
    required_api = {
        "F": (F, ["load_data", "ProposalLedger", "groups", "metric", "rank_a3", "proposal_terms"]),
        "22.16": (ref16, ["load_metadata", "split_conversations", "ExposureLedger", "rerank"]),
        "22.18": (ref18, ["contextual_gate_rerank"]),
    }
    for label, (mod, names) in required_api.items():
        missing = [n for n in names if not hasattr(mod, n)]
        if missing:
            raise AttributeError(f"{label} helper API missing: {missing}")

    with open(a.catalog, "rb") as f:
        metadata = __import__("pickle").load(f)
    catalog = set(map(int, metadata["movie_ids"]))
    item2group, catalog2, categories = ref16.load_metadata(a.catalog, a.profiles)
    if catalog != catalog2 or len(catalog) != EXPECTED_CATALOG:
        raise ValueError("Catalog metadata mismatch.")

    decisions = json.loads(a.decisions.read_text(encoding="utf-8"))
    h_manifest = json.loads(a.h_manifest.read_text(encoding="utf-8"))
    if not h_manifest.get("validation_checks", {}).get("overall_pass", False):
        raise RuntimeError("22.35H FINAL2 manifest does not contain a passing validation record.")
    p_a1 = decisions["A1_22_16"]["parameters"]
    p_a2 = decisions["A2_22_18"]["parameters"]
    p_a3 = h_manifest["selected_A3_parameters"]
    # Cross-check A3 parameter values against the frozen 22.35F decision record.
    p_a3_f = decisions["A3_KG_FairCRS"]["parameters"]
    for k in ("q_threshold", "R_min", "lam"):
        if not np.isclose(float(p_a3[k]), float(p_a3_f[k])):
            raise ValueError(f"22.35H/22.35F selected parameter mismatch: {k}")
    qthr, rmin, lam3 = float(p_a3["q_threshold"]), float(p_a3["R_min"]), float(p_a3["lam"])
    params = {
        "A0_KGSF": {},
        "A1_22_16": {"lam": float(p_a1["lam"])},
        "A2_22_18": {"lam": float(p_a2["lam"]), "tau": float(p_a2["tau"])},
        "A3_KG_FairCRS": {"lam": lam3},
    }

    # Reconstruct validation history from the frozen 22.35F event slates.
    val_events, val_c0, val_c1, warm, tune, held, val_slates = load_validation(a, catalog, F, ref16)
    expected_pre = {
        "A0_KGSF": decisions["A0_KGSF"]["cumulative_warm_tune_heldout"],
        "A1_22_16": decisions["A1_22_16"]["cumulative_warm_tune_heldout"],
        "A2_22_18": decisions["A2_22_18"]["cumulative_warm_tune_heldout"],
        "A3_KG_FairCRS": decisions["A3_KG_FairCRS"]["cumulative_warm_tune_heldout"],
    }
    expected_pre_r1 = decisions["A3_KG_FairCRS"]["R1_raw_sensitivity"]["cumulative_warm_tune_heldout"]

    # Load independent-test KGSF and KG evidence; construct q exactly as 22.35C.
    test_events, test_c0, test_c1, qstats = load_test_inputs(a, catalog)
    print(f"Test events={len(test_events)} | conversations={len({x['conv'] for x in test_events.values()})}")
    print(f"Validation history events={len(val_events)} | warm/tune/held={len(warm)}/{len(tune)}/{len(held)}")
    print(f"Frozen A3 parameters: q={qthr}, R_min={rmin}, lambda={lam3}")

    all_event_rows, summaries, all_cells = [], [], []
    prehistory_checks = {}
    variants_by_system = {
        "A0_KGSF": ["R0_filtered"],
        "A1_22_16": ["R0_filtered"],
        "A2_22_18": ["R0_filtered"],
        "A3_KG_FairCRS": ["R0_filtered", "R1_raw"],
    }
    for system in SYSTEMS:
        for variant in variants_by_system[system]:
            print(f"Evaluating {system} / {variant} ...")
            v_c0 = val_c1 if variant == "R1_raw" else val_c0
            legacy0, prop0 = reconstruct_history(
                system, variant, val_events, val_c0, val_c1, warm, tune, held,
                val_slates, item2group, categories, ref16, F, qthr, rmin
            )
            pre_loss, pre_active, _ = F.proposal_terms(prop0, rmin)
            expected = expected_pre_r1 if (system == "A3_KG_FairCRS" and variant == "R1_raw") else expected_pre[system]
            if not np.isclose(pre_loss, float(expected["L_fair"]), rtol=1e-7, atol=1e-8):
                raise RuntimeError(
                    f"Reconstructed pre-test fairness mismatch for {system}/{variant}: "
                    f"{pre_loss} vs frozen 22.35F {expected['L_fair']}"
                )
            if int(pre_active) != int(expected["active_categories"]):
                raise RuntimeError(f"Pre-test active-category mismatch for {system}/{variant}")
            prehistory_checks[f"{system}_{variant}"] = {
                "reconstructed_L_fair": pre_loss,
                "reference_L_fair": float(expected["L_fair"]),
                "active_categories": int(pre_active),
                "matches_frozen_22_35F": True,
            }
            tc = test_c1 if variant == "R1_raw" else test_c0
            frame, summary, cells, _ = evaluate_one(
                system, variant, test_events, tc, legacy0, prop0,
                item2group, categories, ref16, ref18, F, params[system], qthr, rmin
            )
            all_event_rows.append(frame)
            summaries.append(summary)
            all_cells.append(cells)

    event_results = pd.concat(all_event_rows, ignore_index=True)
    summary_df = pd.DataFrame(summaries)
    cells_df = pd.concat(all_cells, ignore_index=True)
    # Add category-level underexposure gaps relative to each category's
    # relevance-weighted mean exposure rate.
    if not cells_df.empty:
        cells_df["category_mean_rho"] = cells_df.groupby(["system", "variant", "category"]).apply(
            lambda g: float(np.sum(g.p_cg.to_numpy() * g.rho_cg.to_numpy())),
            include_groups=False
        ).reindex(pd.MultiIndex.from_frame(cells_df[["system", "variant", "category"]])).to_numpy()
        cells_df["underexposure_gap"] = np.maximum(0.0, cells_df.category_mean_rho - cells_df.rho_cg)

    # Conversation-clustered paired bootstrap for test utility only. The
    # cumulative nonlinear fairness objective is reported descriptively.
    print(f"Conversation-cluster bootstrap: {a.bootstrap_reps:,} replicates ...")
    boot = cluster_bootstrap(event_results, reps=a.bootstrap_reps, seed=a.bootstrap_seed)

    # Coverage diagnostics from the frozen test candidate pools.
    cat_instances = 0
    total_instances = 0
    for eid, c in test_c0.items():
        total_instances += len(c["ids"])
        cat_instances += sum(bool(categories.get(int(mid), [])) for mid in c["ids"])
    coverage = {
        "candidate_instances": total_instances,
        "category_supported_candidate_instances": cat_instances,
        "candidate_category_coverage": cat_instances / total_instances,
        "unique_test_conversations": len({v["conv"] for v in test_events.values()}),
        "test_events": len(test_events),
    }

    # Create the output directory only after all input and reconstruction
    # checks have passed. Existing outputs are never overwritten.
    a.out.mkdir(parents=True, exist_ok=False)

    # Write outputs.
    event_results.to_csv(a.out / "independent_test_event_results.csv", index=False)
    summary_df.to_csv(a.out / "independent_test_system_comparison.csv", index=False)
    cells_df.to_csv(a.out / "independent_test_category_fairness.csv", index=False)
    boot.to_csv(a.out / "independent_test_utility_cluster_bootstrap.csv", index=False)
    write_json(a.out / "independent_test_relevance_statistics.json", qstats)
    write_json(a.out / "independent_test_coverage_summary.json", coverage)
    write_json(a.out / "validation_history_reconstruction.json", prehistory_checks)

    # Exposure ledger is the final cumulative R/E state, represented by the
    # proposal fairness cells. Preserve a separately named ledger artifact.
    cells_df.to_csv(a.out / "independent_test_exposure_ledger.csv", index=False)

    checks = {
        "experiment": "22.35J_integrated_kg_faircrs_independent_test",
        "test_event_count_4003": len(test_events) == EXPECTED_TEST_EVENTS,
        "test_conversation_count_1283": len({v["conv"] for v in test_events.values()}) == EXPECTED_TEST_CONVERSATIONS,
        "candidate_count_500_per_event": all(len(c["ids"]) == EXPECTED_CANDIDATES for c in test_c0.values()),
        "test_kgsf_kg_candidate_order_aligned": True,
        "test_score_matrix_alignment_verified": True,
        "R0_and_R1_q_event_normalization": all(
            abs(v["q_sum_min"] - 1.0) < 1e-10 and abs(v["q_sum_max"] - 1.0) < 1e-10
            for v in qstats.values()
        ),
        "validation_history_reconstructed_for_all_systems": len(prehistory_checks) == 5,
        "validation_history_matches_22_35F_fairness": all(v["matches_frozen_22_35F"] for v in prehistory_checks.values()),
        "all_four_systems_evaluated": set(summary_df.system) == set(SYSTEMS),
        "A3_R0_primary_and_R1_sensitivity": set(event_results.loc[event_results.system == "A3_KG_FairCRS", "variant"]) == {"R0_filtered", "R1_raw"},
        "frozen_parameters_reused": True,
        "parameter_retuning_performed": False,
        "training_performed": False,
        "retrieval_performed": False,
        "checkpoint_modified": False,
        "test_targets_used_to_construct_q": False,
        "independent_test_used_for_parameter_selection": False,
        "test_set_prior_historical_exposure_disclosed": True,
        "validation_test_category_profile_policy_consistent": True,
        "utility_bootstrap_cluster_unit_conversation": bool((boot.cluster_unit == "conversation").all()),
        "fairness_inference_claimed": False,
    }
    positive = [k for k,v in checks.items() if isinstance(v, bool) and k not in {
        "parameter_retuning_performed", "training_performed", "retrieval_performed",
        "checkpoint_modified", "test_targets_used_to_construct_q",
        "independent_test_used_for_parameter_selection", "fairness_inference_claimed"
    }]
    negative = [
        "parameter_retuning_performed", "training_performed", "retrieval_performed",
        "checkpoint_modified", "test_targets_used_to_construct_q",
        "independent_test_used_for_parameter_selection", "fairness_inference_claimed"
    ]
    checks["failed_positive_checks"] = [k for k in positive if checks[k] is not True]
    checks["failed_negative_checks"] = [k for k in negative if checks[k] is not False]
    checks["overall_pass"] = not checks["failed_positive_checks"] and not checks["failed_negative_checks"]
    write_json(a.out / "validation_checks.json", checks)

    input_paths = [getattr(a, k) for k in D]
    input_paths += [a.out.parent / "unused"] if False else []
    input_paths += [a.out.parent.parent / "unused"] if False else []
    source_paths = [
        a.f_script, a.ref16, a.ref18,
        ROOT / "22_35C_validation_relevance_construction.py",
        ROOT / "22_27A2_kgsf_kg_evidence_export.py",
        Path(__file__).resolve(),
    ]
    for p in source_paths:
        if not p.is_file():
            raise FileNotFoundError(f"Required provenance source missing: {p}")
    hashes = {str(p): sha256(p) for p in input_paths + source_paths}
    manifest = {
        "experiment": "22.35J_integrated_kg_faircrs_independent_test",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "split": "independent_test",
        "events": EXPECTED_TEST_EVENTS,
        "conversations": EXPECTED_TEST_CONVERSATIONS,
        "candidate_pool": "Frozen KGSF Top-500; identical across systems",
        "systems": list(SYSTEMS),
        "A3_parameters": {"q_threshold": qthr, "R_min": rmin, "lambda": lam3},
        "A1_parameters": params["A1_22_16"],
        "A2_parameters": params["A2_22_18"],
        "R0_primary_R1_sensitivity_only": True,
        "q_definition": (
            "p_KGSF=softmax(frozen KGSF Top-500 scores); "
            "KG_conf=exp(-rank/50) for rank>0 and 0 otherwise; "
            "q=normalize(p_KGSF*(1+KG_conf)) within event."
        ),
        "q_is_operational_relevance_not_semantic_relevance": True,
        "category_evidence_policy": (
            "Same accepted category-profile loader as 22.35F/22.16; this run "
            "does not introduce new category evidence scenarios."
        ),
        "validation_history": (
            "Reconstructed from 22.35F warm baseline KGSF slates plus frozen "
            "22.35F tuning/held-out slates; pre-test fairness values checked "
            "against the 22.35F parameter decision record."
        ),
        "test_order": "Ascending event_id; deterministic replay order, not claimed to be timestamp chronology.",
        "fairness_estimand": "Cumulative proposal L_fair after validation history plus independent-test exposure.",
        "utility_inference": f"Paired conversation-cluster percentile bootstrap, {a.bootstrap_reps} replicates.",
        "fairness_inference": "Cumulative nonlinear fairness is reported descriptively; no test-set fairness p-value or confidence interval is claimed.",
        "prior_test_exposure_disclosure": (
            "This ReDial independent-test split was examined in historical experiments "
            "22.22-22.34. This 22.35J run is a frozen-configuration re-evaluation, "
            "not a never-before-seen test evaluation."
        ),
        "training_performed": False,
        "retrieval_performed": False,
        "parameter_retuning_performed": False,
        "independent_test_used_for_parameter_selection": False,
        "input_sha256": hashes,
        "validation_checks": checks,
        "outputs": [
            "independent_test_event_results.csv",
            "independent_test_system_comparison.csv",
            "independent_test_category_fairness.csv",
            "independent_test_exposure_ledger.csv",
            "independent_test_utility_cluster_bootstrap.csv",
            "independent_test_relevance_statistics.json",
            "independent_test_coverage_summary.json",
            "validation_history_reconstruction.json",
            "validation_checks.json",
            "experiment_manifest.json",
        ],
    }
    write_json(a.out / "experiment_manifest.json", manifest)

    print("\n========== 22.35J SUMMARY ==========")
    print(summary_df[[
        "system", "variant", "events", "conversations", "HR@10", "NDCG@10",
        "MRR@10", "L_fair_pretest_history", "L_fair_validation_plus_test",
        "active_categories_validation_plus_test"
    ]].to_string(index=False))
    print("\nValidation overall_pass:", checks["overall_pass"])
    print("Training: NO | Retrieval: NO | Parameter retuning: NO")
    print("Prior historical test exposure: DISCLOSED")
    print("Output:", a.out.resolve())
    if not checks["overall_pass"]:
        raise RuntimeError("22.35J validation failed; inspect validation_checks.json")

if __name__ == "__main__":
    main()
