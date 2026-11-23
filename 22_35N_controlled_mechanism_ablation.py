#!/usr/bin/env python
"""
Experiment 22.35N — controlled mechanism attribution for KG-FairCRS.

Validation-only replay on the frozen 22.35F candidate population and conversation
splits. No training, retrieval, checkpoint changes, test-set access, or retuning.

Configurations:
  N0_KGSF             Original frozen KGSF Top-10.
  N_FULL_KG_FAIRCRS   Frozen full A3 mechanism (22.35F rank_a3).
  N1_ELIGIBILITY_ONLY Eligibility-first selection, lambda=0.
  N2_NO_CATEGORY      Global popularity-group opportunity/exposure calibration.
  N3_NO_KG            q=softmax(KGSF scores), otherwise full A3.
  N4_NO_HISTORY       Event-local ledger reset; expected to equal N1.

N2 is an explicitly defined adaptation, not an exact reproduction of a published
baseline. N4 is expected to be mechanically equivalent to N1 because the
current-event opportunity is added before ranking while no prior exposure exists.
The script checks this equivalence rather than treating it as independent evidence.

Run from the repository root:
  python 22_35N_controlled_mechanism_ablation.py --out results_corrected/experiment_22_35N_RELEASE

Required inputs follow the 22.35F repository layout. Existing outputs are never
overwritten. The script uses frozen A3 parameters from 22.35F parameter_decisions.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import platform
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

TOP_K = 10
EPS = 1e-12
BOOT_REPS = 2000
BOOT_SEED = 2235

ROOT = Path(".")
BASE = ROOT / "results_corrected" / "experiment_22_35"
DEFAULTS = {
    "kgsf": BASE / "kgsf_validation" / "kgsf_validation_top500.csv",
    "q_r0": BASE / "relevance_validation" / "event_candidate_q_ei_R0_filtered.csv",
    "q_r1": BASE / "relevance_validation" / "event_candidate_q_ei_R1_raw.csv",
    "catalog": ROOT / "catalog_audit_corrected.pkl",
    "profiles": ROOT / "results_category_audit" / "experiment_21_22A" / "catalog_item_category_profiles.csv",
    "fairness_script": ROOT / "22_35F_validated_fairness_evaluation.py",
    "category_helper": ROOT / "22_16_category_context_utility_calibration.py",
    "decisions": BASE / "validated_fairness_evaluation_run_20261121" / "parameter_decisions.json",
}

CONFIGS = (
    "N0_KGSF",
    "N_FULL_KG_FAIRCRS",
    "N1_ELIGIBILITY_ONLY",
    "N2_NO_CATEGORY",
    "N3_NO_KG",
    "N4_NO_HISTORY",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if math.isfinite(x) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, obj):
    path.write_text(json.dumps(safe(obj), indent=2, allow_nan=False), encoding="utf-8")


def import_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helper: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or len(scores) == 0 or not np.isfinite(scores).all():
        raise ValueError("KGSF scores must be a finite non-empty vector")
    z = np.exp(scores - np.max(scores))
    return z / z.sum()


def metric(ranked: np.ndarray, target: int) -> Tuple[int, float, float]:
    pos = np.flatnonzero(np.asarray(ranked[:TOP_K]) == int(target))
    if not len(pos):
        return 0, 0.0, 0.0
    rank = int(pos[0]) + 1
    return 1, 1.0 / math.log2(rank + 1), 1.0 / rank


def load_args():
    ap = argparse.ArgumentParser(description=__doc__)
    for key, default in DEFAULTS.items():
        ap.add_argument("--" + key.replace("_", "-"), type=Path, default=default)
    ap.add_argument("--bootstrap-reps", type=int, default=BOOT_REPS)
    ap.add_argument("--bootstrap-seed", type=int, default=BOOT_SEED)
    ap.add_argument("--out", type=Path,
                    default=ROOT / "results_corrected" / "experiment_22_35N_RELEASE")
    args = ap.parse_args()
    if args.bootstrap_reps < 100:
        raise ValueError("--bootstrap-reps must be >=100")
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {args.out}")
    for key in DEFAULTS:
        path = getattr(args, key)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {key} input: {path}")
    return args


class GlobalLedger:
    """No-category comparator: one opportunity/exposure ledger per popularity group.

    Each candidate contributes its q mass once to its popularity group (no
    multi-label category duplication). Each displayed item contributes its
    rank-discounted exposure once to its popularity group.
    """
    def __init__(self):
        self.R = defaultdict(float)
        self.E = defaultdict(float)

    def clone(self):
        other = GlobalLedger()
        other.R.update(self.R)
        other.E.update(self.E)
        return other

    def add_R(self, c, item2group, qthr):
        for mid, q in zip(c["ids"], c["q"]):
            if q >= qthr:
                self.R[int(item2group[int(mid)])] += float(q)

    def add_E(self, ranked, item2group):
        for rank, mid in enumerate(ranked[:TOP_K], 1):
            self.E[int(item2group[int(mid)])] += 1.0 / math.log2(rank + 1)

    def objective_and_deficits(self, rmin):
        groups = sorted(g for g, r in self.R.items() if r > EPS and r >= rmin)
        if len(groups) < 2:
            return float("nan"), {}, 0
        r = np.asarray([self.R[g] for g in groups], dtype=float)
        rho = np.asarray([self.E[g] / (self.R[g] + EPS) for g in groups], dtype=float)
        p = r / r.sum()
        mean = float(np.sum(p * rho))
        loss = float(np.sum(p * (rho - mean) ** 2))
        deficits = {g: max(0.0, mean - float(v)) for g, v in zip(groups, rho)}
        return loss, deficits, len(groups)


def make_warm_prop(ids, cands, F, item2group, categories, qthr):
    ledger = F.ProposalLedger()
    for eid in sorted(ids):
        c = cands[eid]
        ledger.add_R(c, item2group, categories, qthr)
        ledger.add_E(c["ids"][:TOP_K], item2group, categories)
    return ledger


def make_warm_global(ids, cands, item2group, qthr):
    ledger = GlobalLedger()
    for eid in sorted(ids):
        c = cands[eid]
        ledger.add_R(c, item2group, qthr)
        ledger.add_E(c["ids"][:TOP_K], item2group)
    return ledger


def eligibility_rank(c, qthr):
    eligible = c["q"] >= qthr
    ids, scores = c["ids"], c["scores"]
    eligible_idx = np.flatnonzero(eligible)
    eligible_idx = eligible_idx[np.lexsort((ids[eligible_idx], -scores[eligible_idx]))]
    chosen = list(map(int, eligible_idx[:TOP_K]))
    fallback = max(0, TOP_K - len(chosen))
    if fallback:
        base = np.argsort(-scores, kind="stable")
        used = set(chosen)
        chosen.extend(int(j) for j in base if int(j) not in used and len(chosen) < TOP_K)
    chosen = chosen[:TOP_K]
    return ids[np.asarray(chosen, dtype=int)], {
        "eligible_candidates": int(eligible.sum()),
        "eligible_relevance_mass": float(c["q"][eligible].sum()),
        "fallback_positions": int(fallback),
        "top10_promoted_positions": int(np.sum(np.asarray(chosen) != np.arange(TOP_K))),
    }


def rank_no_category(c, item2group, state: GlobalLedger, qthr, rmin, lam):
    state.add_R(c, item2group, qthr)
    _, deficits, _ = state.objective_and_deficits(rmin)
    eligible = c["q"] >= qthr
    ids, scores = c["ids"], c["scores"]
    adjusted = scores.copy()
    for j, mid in enumerate(ids):
        if eligible[j]:
            adjusted[j] += float(lam) * deficits.get(int(item2group[int(mid)]), 0.0)
    order = np.lexsort((ids, -adjusted))
    chosen = [int(j) for j in order if eligible[j]][:TOP_K]
    fallback = max(0, TOP_K - len(chosen))
    if fallback:
        base = np.argsort(-scores, kind="stable")
        used = set(chosen)
        chosen.extend(int(j) for j in base if int(j) not in used and len(chosen) < TOP_K)
    chosen = chosen[:TOP_K]
    ranked = ids[np.asarray(chosen, dtype=int)]
    state.add_E(ranked, item2group)
    return ranked, {
        "eligible_candidates": int(eligible.sum()),
        "eligible_relevance_mass": float(c["q"][eligible].sum()),
        "fallback_positions": int(fallback),
        "top10_promoted_positions": int(np.sum(np.asarray(chosen) != np.arange(TOP_K))),
    }


def update_category_ledger(state, c, ranked, F, item2group, categories, qthr):
    state.add_R(c, item2group, categories, qthr)
    state.add_E(ranked, item2group, categories)


def run_one_segment(config, event_ids, events, cands, state0, global0,
                    F, item2group, categories, qthr, rmin, lam):
    rows = []
    state = state0.clone() if state0 is not None else None
    global_state = global0.clone() if global0 is not None else None

    for eid in sorted(event_ids):
        c = cands[eid]
        info = {}
        if config == "N0_KGSF":
            ranked = c["ids"][:TOP_K]
        elif config == "N_FULL_KG_FAIRCRS":
            ranked, info = F.rank_a3(c, item2group, categories, state, qthr, rmin, lam)
        elif config == "N1_ELIGIBILITY_ONLY":
            ranked, info = eligibility_rank(c, qthr)
            update_category_ledger(state, c, ranked, F, item2group, categories, qthr)
        elif config == "N2_NO_CATEGORY":
            ranked, info = rank_no_category(c, item2group, global_state, qthr, rmin, lam)
            # Parallel common-estimand ledger: descriptive only, never used by N2 ranking.
            update_category_ledger(state, c, ranked, F, item2group, categories, qthr)
        elif config == "N3_NO_KG":
            ranked, info = F.rank_a3(c, item2group, categories, state, qthr, rmin, lam)
        elif config == "N4_NO_HISTORY":
            # Ranking uses a fresh event-local ledger, so prior exposure is absent.
            # The separate `state` is an outcome-accounting ledger only; it never
            # influences the N4 ranking and permits cumulative descriptive reporting.
            local = F.ProposalLedger()
            ranked, info = F.rank_a3(c, item2group, categories, local, qthr, rmin, lam)
            update_category_ledger(state, c, ranked, F, item2group, categories, qthr)
        else:
            raise ValueError(config)

        hit, ndcg, mrr = metric(ranked, events[eid]["target"])
        eligible_count = int(np.sum(c["q"] >= qthr))
        kgmap = {int(mid): int(rank) for mid, rank in zip(c["ids"], c["kg"])}
        rows.append({
            "config": config, "event_id": int(eid),
            "conv_id": int(events[eid]["conv"]),
            "target_movie_id": int(events[eid]["target"]),
            "HR@10": hit, "NDCG@10": ndcg, "MRR@10": mrr,
            "eligible_candidates": int(info.get("eligible_candidates", eligible_count)),
            "eligible_relevance_mass": float(info.get("eligible_relevance_mass",
                                                        c["q"][c["q"] >= qthr].sum())),
            "kg_supported_candidates": int(np.sum(c["kg"] > 0)),
            "kg_supported_top10": int(sum(kgmap.get(int(mid), 0) > 0 for mid in ranked[:TOP_K])),
            "fallback_positions": int(info.get("fallback_positions", 0)),
            "top10_promoted_positions": int(info.get("top10_promoted_positions", 0)),
            "top10_movie_ids": json.dumps(list(map(int, ranked[:TOP_K]))),
        })

        # N0 still accrues exposure history for its descriptive fairness ledger.
        if config == "N0_KGSF":
            update_category_ledger(state, c, ranked, F, item2group, categories, qthr)
            global_state.add_R(c, item2group, qthr)
            global_state.add_E(ranked, item2group)

    frame = pd.DataFrame(rows)
    return frame, state, global_state


def summarize(frame, config, state, global_state, F, rmin):
    # Common category-conditioned objective is computed for every slate system,
    # including N2, solely as a shared evaluation metric.
    common_loss, active_categories, _ = F.proposal_terms(state, rmin)
    native_global_loss, _, active_groups = (global_state.objective_and_deficits(rmin)
                                             if config == "N2_NO_CATEGORY" else (None, {}, None))
    return {
        "config": config,
        "events": int(len(frame)),
        "conversations": int(frame.conv_id.nunique()),
        "HR@10": float(frame["HR@10"].mean()),
        "NDCG@10": float(frame["NDCG@10"].mean()),
        "MRR@10": float(frame["MRR@10"].mean()),
        "common_category_fairness_loss": common_loss,
        "common_active_categories": active_categories,
        "native_global_fairness_loss": native_global_loss,
        "native_active_global_popularity_groups": active_groups,
        "mean_eligible_candidates": float(frame.eligible_candidates.mean()),
        "mean_kg_supported_top10": float(frame.kg_supported_top10.mean()),
        "mean_fallback_positions": float(frame.fallback_positions.mean()),
        "mean_top10_promoted_positions": float(frame.top10_promoted_positions.mean()),
    }


def global_group_rows(state: GlobalLedger, config: str, split: str, rmin: float):
    rows = []
    groups = sorted(g for g, r in state.R.items() if r > EPS and r >= rmin)
    if not groups:
        return rows
    total = sum(state.R[g] for g in groups)
    rho = {g: state.E[g] / (state.R[g] + EPS) for g in groups}
    mean = sum((state.R[g] / total) * rho[g] for g in groups)
    for g in groups:
        rows.append({
            "config": config, "split": split, "category": "GLOBAL",
            "popularity_group": int(g), "R_cg": float(state.R[g]),
            "E_cg": float(state.E[g]), "rho_cg": float(rho[g]),
            "p_cg": float(state.R[g] / total),
            "underexposure_gap": float(max(0.0, mean - rho[g])),
            "estimand": "global_group_weighted_variance",
        })
    return rows


def bootstrap_utility(event_frames: Dict[str, pd.DataFrame], reps: int, seed: int):
    """Paired conversation-cluster percentile bootstrap on held-out utility."""
    base = event_frames["N0_KGSF"].set_index("event_id")
    convs = np.asarray(sorted(base.conv_id.unique()), dtype=np.int64)
    by_conv = {int(c): base.index[base.conv_id == c].tolist() for c in convs}
    rng = np.random.default_rng(seed)
    metrics = ("HR@10", "NDCG@10", "MRR@10")
    records = []
    for config, frame in event_frames.items():
        if config == "N0_KGSF":
            continue
        cur = frame.set_index("event_id")
        if set(cur.index) != set(base.index):
            raise ValueError(f"Event mismatch in bootstrap for {config}")
        for m in metrics:
            deltas = np.empty(reps, dtype=float)
            for b in range(reps):
                draw = rng.choice(convs, size=len(convs), replace=True)
                sample_ids = [eid for conv in draw for eid in by_conv[int(conv)]]
                deltas[b] = float(cur.loc[sample_ids, m].mean() -
                                  base.loc[sample_ids, m].mean())
            observed = float(cur[m].mean() - base[m].mean())
            records.append({
                "config": config, "metric": m, "observed_delta_vs_N0": observed,
                "ci95_low": float(np.percentile(deltas, 2.5)),
                "ci95_high": float(np.percentile(deltas, 97.5)),
                "bootstrap_reps": reps, "cluster_unit": "conversation",
            })
    return pd.DataFrame(records)


def main():
    a = load_args()
    F = import_module(a.fairness_script, "kgfaircrs_2235f")
    ref = import_module(a.category_helper, "kgfaircrs_2216")
    required_f = ("load_data", "ProposalLedger", "rank_a3", "proposal_terms", "groups")
    required_ref = ("load_metadata", "split_conversations")
    for module, names, label in ((F, required_f, "22.35F"), (ref, required_ref, "22.16")):
        missing = [name for name in names if not hasattr(module, name)]
        if missing:
            raise AttributeError(f"{label} helper API missing: {missing}")

    decisions = json.loads(a.decisions.read_text(encoding="utf-8"))
    p = decisions["A3_KG_FairCRS"]["parameters"]
    qthr, rmin, lam = float(p["q_threshold"]), float(p["R_min"]), float(p["lam"])
    if not np.isclose(qthr, 0.0005) or not np.isclose(rmin, 0.5) or not np.isclose(lam, 1.0):
        raise ValueError(f"Unexpected frozen A3 parameters: {p}")

    item2group, catalog, categories = ref.load_metadata(a.catalog, a.profiles)
    events, cands_r0 = F.load_data(a.kgsf, a.q_r0, catalog)
    events_r1, _ = F.load_data(a.kgsf, a.q_r1, catalog)
    if events != events_r1:
        raise RuntimeError("R0/R1 event manifests differ")

    # No-KG intervention: retain frozen scores and replace q with KGSF softmax.
    cands_nokg = {}
    for eid, c in cands_r0.items():
        z = dict(c)
        z["q"] = softmax(c["scores"])
        cands_nokg[eid] = z

    convs = [v["conv"] for v in events.values()]
    warm_c, tune_c, held_c = ref.split_conversations(convs)
    if warm_c & tune_c or warm_c & held_c or tune_c & held_c:
        raise RuntimeError("Conversation splits overlap")
    warm = sorted(e for e, v in events.items() if v["conv"] in warm_c)
    tune = sorted(e for e, v in events.items() if v["conv"] in tune_c)
    held = sorted(e for e, v in events.items() if v["conv"] in held_c)
    if set(warm) | set(tune) | set(held) != set(events):
        raise RuntimeError("Warm/tune/held-out split does not cover all events")
    if (len(events), len(set(convs))) != (3010, 883):
        raise RuntimeError(f"Unexpected validation population: {len(events)} events / {len(set(convs))} conversations")

    # All configurations begin from the same frozen KGSF warm-up exposure history.
    warm_prop = make_warm_prop(warm, cands_r0, F, item2group, categories, qthr)
    warm_global = make_warm_global(warm, cands_r0, item2group, qthr)

    all_rows, summaries, held_frames = [], [], {}
    final_states = {}
    fairness_cell_rows = []
    for config in CONFIGS:
        if config == "N2_NO_CATEGORY":
            state0, global0 = warm_prop.clone(), warm_global.clone()
            data = cands_r0
        elif config == "N3_NO_KG":
            warm_nokg = make_warm_prop(warm, cands_nokg, F, item2group, categories, qthr)
            state0, global0 = warm_nokg, None
            data = cands_nokg
        elif config == "N4_NO_HISTORY":
            # Warm history is retained only for outcome accounting, never supplied
            # to the event-local ranking ledger.
            state0, global0 = warm_prop.clone(), None
            data = cands_r0
        elif config == "N0_KGSF":
            state0, global0 = warm_prop.clone(), warm_global.clone()
            data = cands_r0
        else:
            state0, global0 = warm_prop.clone(), None
            data = cands_r0

        tune_frame, tune_state, tune_global = run_one_segment(
            config, tune, events, data, state0, global0,
            F, item2group, categories, qthr, rmin, lam
        )
        held_frame, held_state, held_global = run_one_segment(
            config, held, events, data, tune_state, tune_global,
            F, item2group, categories, qthr, rmin, lam
        )
        tune_frame["split"] = "tune"
        held_frame["split"] = "heldout"
        all_rows.extend([tune_frame, held_frame])
        held_frames[config] = held_frame
        final_states[config] = (held_state, held_global)
        summaries.append(summarize(tune_frame, config, tune_state, tune_global, F, rmin) | {"split": "tune"})
        summaries.append(summarize(held_frame, config, held_state, held_global, F, rmin) | {"split": "heldout"})
        for split_name, cat_state, glob_state in (("tune", tune_state, tune_global),
                                                   ("heldout", held_state, held_global)):
            _, _, cells = F.proposal_terms(cat_state, rmin)
            fairness_cell_rows.extend({"config": config, "split": split_name,
                                       "estimand": "common_category_mean_weighted_variance", **cell}
                                      for cell in cells)
            if config == "N2_NO_CATEGORY":
                fairness_cell_rows.extend(global_group_rows(glob_state, config, split_name, rmin))

    # N4 is intentionally a falsifiable equivalence check, not a separate claim.
    n1 = held_frames["N1_ELIGIBILITY_ONLY"].set_index("event_id")
    n4 = held_frames["N4_NO_HISTORY"].set_index("event_id")
    n4_equivalent_n1 = bool(
        np.array_equal(n1.loc[sorted(n1.index), "top10_movie_ids"].to_numpy(),
                       n4.loc[sorted(n4.index), "top10_movie_ids"].to_numpy())
    )
    if not n4_equivalent_n1:
        raise RuntimeError("Expected N4_NO_HISTORY to match N1_ELIGIBILITY_ONLY under event-local zero-exposure state")

    # Verify identical held-out event and conversation sets across all configurations.
    base_ids = set(held_frames["N0_KGSF"].event_id.astype(int))
    base_convs = set(held_frames["N0_KGSF"].conv_id.astype(int))
    same_events = all(set(f.event_id.astype(int)) == base_ids for f in held_frames.values())
    same_convs = all(set(f.conv_id.astype(int)) == base_convs for f in held_frames.values())
    if not same_events or not same_convs:
        raise RuntimeError("Held-out event/conversation alignment failed")

    a.out.mkdir(parents=True, exist_ok=False)
    out_all = pd.concat(all_rows, ignore_index=True)
    out_summary = pd.DataFrame(summaries)
    out_all.to_csv(a.out / "event_level_results.csv", index=False)
    out_summary.to_csv(a.out / "split_summary.csv", index=False)
    out_summary[out_summary.split == "heldout"].to_csv(a.out / "heldout_comparison.csv", index=False)
    pd.DataFrame(fairness_cell_rows).to_csv(a.out / "fairness_cell_comparison.csv", index=False)
    for config in CONFIGS:
        config_dir = a.out / config
        config_dir.mkdir(parents=True, exist_ok=False)
        out_all[out_all.config == config].to_csv(config_dir / "event_results.csv", index=False)
        out_summary[out_summary.config == config].to_csv(config_dir / "split_summary.csv", index=False)
        pd.DataFrame([r for r in fairness_cell_rows if r["config"] == config]).to_csv(
            config_dir / "fairness_cells.csv", index=False)

    boot = bootstrap_utility(held_frames, a.bootstrap_reps, a.bootstrap_seed)
    boot.to_csv(a.out / "paired_conversation_bootstrap_utility.csv", index=False)

    # Explicit mechanism contrasts; N4 is marked as an equivalence diagnostic.
    contrast_rows = []
    h = out_summary[out_summary.split == "heldout"].set_index("config")
    for config in CONFIGS:
        row = {"config": config, "NDCG_delta_vs_N0": float(h.loc[config, "NDCG@10"] - h.loc["N0_KGSF", "NDCG@10"]),
               "common_category_fairness_loss": h.loc[config, "common_category_fairness_loss"],
               "native_global_fairness_loss": h.loc[config, "native_global_fairness_loss"],
               "mean_eligible_candidates": h.loc[config, "mean_eligible_candidates"]}
        contrast_rows.append(row)
    pd.DataFrame(contrast_rows).to_csv(a.out / "mechanism_contrasts.csv", index=False)

    checks = {
        "experiment": "22.35N_controlled_mechanism_ablation",
        "validation_only": True,
        "independent_test_loaded": False,
        "training_performed": False,
        "retrieval_rerun": False,
        "checkpoint_modified": False,
        "parameter_retuning_performed": False,
        "expected_validation_events_3010": len(events) == 3010,
        "expected_validation_conversations_883": len(set(convs)) == 883,
        "frozen_top500_per_event": all(len(c["ids"]) == 500 and len(np.unique(c["ids"])) == 500 for c in cands_r0.values()),
        "R0_R1_event_manifests_identical": events == events_r1,
        "conversation_splits_disjoint": not (warm_c & tune_c or warm_c & held_c or tune_c & held_c),
        "all_events_assigned_once": len(set(warm) | set(tune) | set(held)) == len(events),
        "all_configurations_same_heldout_events": same_events,
        "all_configurations_same_heldout_conversations": same_convs,
        "N4_matches_N1_as_predicted": n4_equivalent_n1,
        "N2_definition": "Single global popularity-group R_g/E_g ledger; each item contributes once, without category multi-label duplication.",
        "N3_definition": "q=softmax(frozen KGSF Top-500 scores); fixed q_threshold=0.0005; no retuning.",
        "N4_definition": "Fresh per-event proposal ledger; current-event opportunity added before ranking and prior exposure absent.",
        "all_six_configurations_present": set(out_summary.config) == set(CONFIGS),
        "paired_bootstrap_cluster_unit_conversation": bool((boot.cluster_unit == "conversation").all()),
    }
    positive = [k for k, v in checks.items() if isinstance(v, bool) and k not in {
        "independent_test_loaded", "training_performed", "retrieval_rerun",
        "checkpoint_modified", "parameter_retuning_performed"
    }]
    negative = ["independent_test_loaded", "training_performed", "retrieval_rerun",
                "checkpoint_modified", "parameter_retuning_performed"]
    checks["failed_positive_checks"] = [k for k in positive if checks[k] is not True]
    checks["failed_negative_checks"] = [k for k in negative if checks[k] is not False]
    checks["overall_pass"] = not checks["failed_positive_checks"] and not checks["failed_negative_checks"]
    write_json(a.out / "validation_checks.json", checks)

    inputs = {
        "kgsf": a.kgsf, "q_r0": a.q_r0, "q_r1": a.q_r1,
        "catalog": a.catalog, "profiles": a.profiles,
        "fairness_script": a.fairness_script, "category_helper": a.category_helper,
        "decisions": a.decisions,
    }
    manifest = {
        "experiment": "22.35N_controlled_mechanism_ablation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
        "scope": "frozen_validation_only_mechanism_attribution",
        "events": len(events), "conversations": len(set(convs)),
        "warm_events": len(warm), "tuning_events": len(tune), "heldout_events": len(held),
        "warm_conversations": len(warm_c), "tuning_conversations": len(tune_c),
        "heldout_conversations": len(held_c),
        "frozen_parameters": {"q_threshold": qthr, "R_min": rmin, "lambda": lam},
        "configurations": list(CONFIGS),
        "N2_no_category_estimand": checks["N2_definition"],
        "N3_no_KG_estimand": checks["N3_definition"],
        "N4_no_history_estimand": checks["N4_definition"],
        "N4_equivalence_note": "N4 is expected to equal N1; no prior exposure means the event-local fairness bonus is zero after current-event opportunity is added.",
        "bootstrap": {"reps": a.bootstrap_reps, "seed": a.bootstrap_seed, "unit": "conversation"},
        "inputs_sha256": {key: sha256(path) for key, path in inputs.items()},
        "validation_checks": checks,
        "outputs": [
            "event_level_results.csv", "split_summary.csv", "heldout_comparison.csv",
            "paired_conversation_bootstrap_utility.csv", "mechanism_contrasts.csv",
            "fairness_cell_comparison.csv", "per_configuration_directories",
            "validation_checks.json", "experiment_manifest.json",
        ],
    }
    write_json(a.out / "experiment_manifest.json", manifest)
    print("\n========== 22.35N CONTROLLED MECHANISM ABLATION ==========")
    print(out_summary[out_summary.split == "heldout"].to_string(index=False))
    print("\nN4 matches N1:", n4_equivalent_n1)
    print("Validation overall_pass:", checks["overall_pass"])
    print("Training: NO | Retrieval: NO | Independent test: NOT READ | Retuning: NO")
    print("Output:", a.out.resolve())
    if not checks["overall_pass"]:
        raise RuntimeError("22.35N validation failed; inspect validation_checks.json")


if __name__ == "__main__":
    main()
