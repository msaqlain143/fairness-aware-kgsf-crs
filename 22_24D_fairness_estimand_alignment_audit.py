#!/usr/bin/env python
"""Experiment 22.24D: read-only fairness estimand alignment audit.

Run from the repository root:
  python 22_24D_fairness_estimand_alignment_audit.py

Optional proposal inputs:
  --relevance-csv event_item_relevance.csv
      Required columns: event_id,movie_id,q
  --category-weights-csv category_weights.csv
      Required columns: category,omega

The q column must contain contextual relevance q[e,i] for eligible items.
Do not substitute KG rank/confidence or candidate-pool category prevalence.
No training, retrieval, tuning, reranking, or test inference is performed.
"""
from __future__ import annotations
import argparse, ast, importlib.util, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

EPS = 1e-12
TOP_K = 10
EXPECTED_EVENTS = 4003
EXPECTED_CONVERSATIONS = 1283

DEFAULT_RESULTS = Path("results_final_test_m2/experiment_22_24B")
DEFAULT_KBRD = Path("results_final_test_m2/experiment_22_22/kbrd_independent_test_top500.csv")
DEFAULT_CATALOG = Path("catalog_audit_corrected.pkl")
DEFAULT_PROFILES = Path("results_category_audit/experiment_21_22A/catalog_item_category_profiles.csv")
DEFAULT_REFERENCE = Path("22_16_category_context_utility_calibration.py")
DEFAULT_OUT = Path("results_final_test_m2/experiment_22_24D")


def parse_array(value, field):
    if isinstance(value, (list, tuple, np.ndarray)):
        out = list(value)
    else:
        try:
            out = ast.literal_eval(str(value))
        except (ValueError, SyntaxError):
            out = json.loads(str(value))
    if not isinstance(out, (list, tuple, np.ndarray)):
        raise ValueError(f"{field} is not an array.")
    return list(out)


def load_reference(path):
    if not path.is_file():
        raise FileNotFoundError(f"22.16 reference not found: {path}")
    spec = importlib.util.spec_from_file_location("exp_22_16_for_22_24D", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import 22.16 reference: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("load_metadata", "ExposureLedger"):
        if not hasattr(mod, name):
            raise RuntimeError(f"22.16 reference missing {name}.")
    return mod


def load_inputs(results_dir, kbrd_path):
    bp, pp = results_dir / "baseline_event_results.csv", results_dir / "proposed_event_results.csv"
    for p in (bp, pp, kbrd_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    base, prop, kbrd = pd.read_csv(bp), pd.read_csv(pp), pd.read_csv(kbrd_path)
    result_cols = {"event_id", "conv_id", "target_movie_id", "top10_movie_ids", "top10_group_ids"}
    kbrd_cols = {"event_id", "conv_id", "target_movie_id", "top_movie_ids"}
    for name, df in (("baseline", base), ("proposed", prop)):
        missing = result_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} results missing columns: {sorted(missing)}")
        if df.event_id.duplicated().any():
            raise ValueError(f"Duplicate event IDs in {name}.")
    missing = kbrd_cols - set(kbrd.columns)
    if missing:
        raise ValueError(f"KBRD Top-500 missing columns: {sorted(missing)}")
    if kbrd.event_id.duplicated().any():
        raise ValueError("Duplicate KBRD event IDs.")
    for df in (base, prop, kbrd):
        for col in ("event_id", "conv_id", "target_movie_id"):
            df[col] = pd.to_numeric(df[col], errors="raise").astype("int64")
        df.sort_values("event_id", inplace=True)
        df.reset_index(drop=True, inplace=True)
    if not (base.event_id.equals(prop.event_id) and base.event_id.equals(kbrd.event_id)):
        raise ValueError("Baseline, proposed and Top-500 event universes differ.")
    for label, df in (("baseline", base), ("proposed", prop)):
        if not np.array_equal(df.conv_id, kbrd.conv_id) or not np.array_equal(df.target_movie_id, kbrd.target_movie_id):
            raise ValueError(f"{label} event identity does not match KBRD export.")
    if len(base) != EXPECTED_EVENTS or base.conv_id.nunique() != EXPECTED_CONVERSATIONS:
        raise ValueError(f"Expected 4,003 events/1,283 conversations; got {len(base)}/{base.conv_id.nunique()}.")
    return base, prop, kbrd


def make_cells(kbrd, item2group, categories):
    cells = set()
    for raw in kbrd.top_movie_ids:
        ids = parse_array(raw, "top_movie_ids")
        if len(ids) != 500 or len(set(ids)) != 500:
            raise ValueError("Each frozen candidate pool must contain 500 unique IDs.")
        for mid in ids:
            if int(mid) not in item2group:
                raise KeyError(f"Missing popularity group for item {mid}.")
            for cat in categories.get(int(mid), []):
                cells.add((str(cat), int(item2group[int(mid)])))
    cells = sorted(cells)
    if len(cells) < 2:
        raise RuntimeError("Fewer than two category-group cells.")
    return cells


def reconstruct(frame, kbrd, item2group, categories, cells, ref):
    """Per-event opportunity/exposure sufficient statistics, using exact 22.16 weights."""
    kbrd_idx = kbrd.set_index("event_id")
    cell_idx = {cell: j for j, cell in enumerate(cells)}
    opp = np.zeros((len(frame), len(cells)), dtype=float)
    exp = np.zeros_like(opp)
    event_ids, conv_ids = [], []
    for i, row in enumerate(frame.itertuples(index=False)):
        kr = kbrd_idx.loc[int(row.event_id)]
        candidates = [int(x) for x in parse_array(kr.top_movie_ids, "top_movie_ids")]
        ranked = [int(x) for x in parse_array(row.top10_movie_ids, "top10_movie_ids")]
        saved_groups = [int(x) for x in parse_array(row.top10_group_ids, "top10_group_ids")]
        if len(candidates) != 500 or len(ranked) != TOP_K or len(saved_groups) != TOP_K:
            raise ValueError(f"Unexpected candidate/Top-10 size at event {row.event_id}.")
        if len(set(ranked)) != TOP_K:
            raise ValueError(f"Duplicate Top-10 item at event {row.event_id}.")
        expected = [int(item2group[x]) for x in ranked]
        if expected != saved_groups:
            raise ValueError(f"Saved group IDs do not match catalog at event {row.event_id}.")
        # 22.16 opportunity: normalized DCG weights over all 500 candidates.
        weights = ref.ExposureLedger.opportunity_weights(len(candidates))
        for rank, mid in enumerate(candidates):
            cats = categories.get(mid, [])
            if cats:
                value = float(weights[rank]) / len(cats)
                group = int(item2group[mid])
                for cat in cats:
                    opp[i, cell_idx[(str(cat), group)]] += value
        # 22.16 exposure: DCG weights over the saved Top-10.
        for rank, mid in enumerate(ranked):
            cats = categories.get(mid, [])
            if cats:
                value = 1.0 / math.log2(rank + 2.0) / len(cats)
                group = int(item2group[mid])
                for cat in cats:
                    exp[i, cell_idx[(str(cat), group)]] += value
        event_ids.append(int(row.event_id))
        conv_ids.append(int(row.conv_id))
    return np.asarray(event_ids), np.asarray(conv_ids), opp, exp


def log_rate_mad(exposure, opportunity, cell_mask=None):
    mask = opportunity > EPS
    if cell_mask is not None:
        mask = mask & np.asarray(cell_mask, dtype=bool)
    if mask.sum() < 2:
        return float("nan")
    logs = np.log((exposure[mask] + EPS) / (opportunity[mask] + EPS))
    return float(np.mean(np.abs(logs - logs.mean())))


def read_q(path, event_ids, kbrd, item2group, categories):
    if path is None:
        return None, {"status": "UNAVAILABLE", "reason": "No event-item q[e,i] file supplied."}
    q = pd.read_csv(path)
    required = {"event_id", "movie_id", "q"}
    if required - set(q.columns):
        raise ValueError(f"Relevance CSV requires {sorted(required)}.")
    if q.duplicated(["event_id", "movie_id"]).any():
        raise ValueError("Duplicate (event_id,movie_id) q rows.")
    q["event_id"] = pd.to_numeric(q.event_id, errors="raise").astype("int64")
    q["movie_id"] = pd.to_numeric(q.movie_id, errors="raise").astype("int64")
    q["q"] = pd.to_numeric(q.q, errors="raise").astype(float)
    if not np.isfinite(q.q).all() or (q.q < 0).any():
        raise ValueError("q must be finite and non-negative.")
    valid_events = set(map(int, event_ids))
    if not set(q.event_id).issubset(valid_events):
        raise ValueError("q file contains events outside the audited split.")
    kidx = kbrd.set_index("event_id")
    candidate_sets = {int(e): set(parse_array(r.top_movie_ids, "top_movie_ids")) for e, r in kidx.iterrows()}
    if any(int(r.movie_id) not in candidate_sets[int(r.event_id)] for r in q.itertuples(index=False)):
        raise ValueError("q file includes an item outside its event's frozen Top-500.")
    q["categories"] = q.movie_id.map(lambda x: categories.get(int(x), []))
    q["group"] = q.movie_id.map(lambda x: int(item2group[int(x)]))
    return q, {
        "status": "AVAILABLE", "rows": int(len(q)),
        "events": int(q.event_id.nunique()), "items": int(q.movie_id.nunique()),
        "warning": "Numeric q values are not proof of semantic relevance; verify their provenance independently."
    }


def proposal_exposure(frame, item2group, categories, cells):
    """Proposal Eq. (22): full binary multi-label membership, no category split."""
    cell_idx = {cell: j for j, cell in enumerate(cells)}
    totals = np.zeros(len(cells), dtype=float)
    for row in frame.itertuples(index=False):
        ranked = [int(x) for x in parse_array(row.top10_movie_ids, "top10_movie_ids")]
        for rank, mid in enumerate(ranked):
            weight = 1.0 / math.log2(rank + 2.0)
            for cat in categories.get(mid, []):
                totals[cell_idx[(str(cat), int(item2group[mid]))]] += weight
    return totals


def proposal_loss(exposure_totals, cells, q, all_categories, omega, min_mass):
    if q is None or q.empty:
        return None, "UNAVAILABLE: event-item contextual relevance q[e,i] is missing."
    relevance = defaultdict(float)
    for row in q.itertuples(index=False):
        # Proposal Eq. (21) uses binary A_ic membership: q[e,i] contributes
        # fully to every verified category the item belongs to (no 1/|C_i| split).
        for cat in row.categories:
            relevance[(str(cat), int(row.group))] += float(row.q)
    exp_map = {cell: float(exposure_totals[j]) for j, cell in enumerate(cells)}
    cell_details, cat_details = [], []
    active_cats = []
    for cat in all_categories:
        groups = sorted(g for (c, g), mass in relevance.items() if c == cat and mass > min_mass)
        if len(groups) < 2:
            cat_details.append({"category": cat, "active_groups": len(groups),
                                "status": "EXCLUDED_INSUFFICIENT_RELEVANCE_SUPPORT",
                                "omega": float(omega[cat]), "weighted_variance": np.nan})
            continue
        masses = np.asarray([relevance[(cat, g)] for g in groups], dtype=float)
        exposures = np.asarray([exp_map.get((cat, g), 0.0) for g in groups], dtype=float)
        rho = exposures / (masses + EPS)
        p = masses / masses.sum()
        rho_bar = float(np.sum(p * rho))
        var = float(np.sum(p * (rho - rho_bar) ** 2))
        active_cats.append(cat)
        for g, mass, ex, rate, prob in zip(groups, masses, exposures, rho, p):
            cell_details.append({"category": cat, "popularity_group": g,
                                 "relevance_mass_R_cg": float(mass),
                                 "exposure_E_cg": float(ex), "rho_cg": float(rate),
                                 "p_cg": float(prob), "rho_bar_c": rho_bar})
        cat_details.append({"category": cat, "active_groups": len(groups),
                            "status": "CALCULATED", "omega": float(omega[cat]),
                            "relevance_mass_total": float(masses.sum()),
                            "category_mean_rate": rho_bar, "weighted_variance": var})
    weight_sum = sum(float(omega[c]) for c in active_cats)
    if weight_sum <= 0:
        return {"categories": pd.DataFrame(cat_details), "cells": pd.DataFrame(cell_details)}, \
               "UNAVAILABLE: no active categories with positive omega."
    loss = sum(float(omega[c]) * float(next(x["weighted_variance"] for x in cat_details if x["category"] == c))
               for c in active_cats) / weight_sum
    return {"categories": pd.DataFrame(cat_details), "cells": pd.DataFrame(cell_details)}, {
        "status": "CALCULATED", "L_fair": float(loss),
        "omega_normalization": "normalized over active categories",
        "support_threshold": float(min_mass),
        "active_categories": len(active_cats),
        "qualification": "Proposal does not prescribe numerical omega_c or a minimum support threshold; these are explicit audit settings."
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--kbrd", type=Path, default=DEFAULT_KBRD)
    ap.add_argument("--catalog-pickle", type=Path, default=DEFAULT_CATALOG)
    ap.add_argument("--category-profiles", type=Path, default=DEFAULT_PROFILES)
    ap.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--relevance-csv", type=Path, default=None)
    ap.add_argument("--category-weights-csv", type=Path, default=None)
    ap.add_argument("--min-relevance-mass", type=float, default=0.0)
    ap.add_argument("--minimum-cell-conversations", type=int, default=20,
                    help="Fixed 22.24C support threshold (distinct conversations per cell).")
    args = ap.parse_args()
    if args.min_relevance_mass < 0 or not math.isfinite(args.min_relevance_mass):
        raise ValueError("--min-relevance-mass must be finite and non-negative.")

    ref = load_reference(args.reference)
    item2group, catalog, categories = ref.load_metadata(args.catalog_pickle, args.category_profiles)
    base, prop, kbrd = load_inputs(args.results_dir, args.kbrd)
    cells = make_cells(kbrd, item2group, categories)
    all_categories = sorted({c for c, _ in cells})

    bi, bc, bo, be = reconstruct(base, kbrd, item2group, categories, cells, ref)
    pi, pc, po, pe = reconstruct(prop, kbrd, item2group, categories, cells, ref)
    if not np.array_equal(bi, pi) or not np.array_equal(bc, pc):
        raise RuntimeError("Baseline/proposed event alignment failed.")
    if not np.allclose(bo, po, rtol=0, atol=1e-12):
        raise RuntimeError("Baseline/proposed opportunity contributions differ.")

    total_opp = bo.sum(axis=0)
    base_exp_total, prop_exp_total = be.sum(axis=0), pe.sum(axis=0)

    # Reconstruct 22.24C's fixed cell universe: at least N distinct
    # conversations contribute positive Top-500 opportunity to a cell.
    unique_convs = np.unique(bc)
    conv_opp = np.zeros((len(unique_convs), len(cells)), dtype=float)
    conv_index = {int(cid): i for i, cid in enumerate(unique_convs)}
    for event_i, cid in enumerate(bc):
        conv_opp[conv_index[int(cid)]] += bo[event_i]
    cell_support = (conv_opp > EPS).sum(axis=0)
    fixed_22_24c_mask = cell_support >= args.minimum_cell_conversations
    positive_opp_mask = total_opp > EPS
    if int(fixed_22_24c_mask.sum()) < 2:
        raise RuntimeError("Fewer than two cells meet the 22.24C conversation-support threshold.")

    # Preserve both estimands explicitly: 22.24B all positive-opportunity cells,
    # and 22.24C fixed cells with >= minimum_cell_conversations.
    base_mad_22_24b = log_rate_mad(base_exp_total, total_opp)
    prop_mad_22_24b = log_rate_mad(prop_exp_total, total_opp)
    reduction_22_24b = 100.0 * (base_mad_22_24b - prop_mad_22_24b) / base_mad_22_24b
    base_mad, prop_mad = (
        log_rate_mad(base_exp_total, total_opp, fixed_22_24c_mask),
        log_rate_mad(prop_exp_total, total_opp, fixed_22_24c_mask)
    )
    reduction = 100.0 * (base_mad - prop_mad) / base_mad if base_mad > 0 else float("nan")

    # Event-level candidate opportunity accounting.
    opp_rows = []
    for i, (eid, cid) in enumerate(zip(bi, bc)):
        for j, (cat, group) in enumerate(cells):
            value = float(bo[i, j])
            if value > 0:
                opp_rows.append({"event_id": int(eid), "conv_id": int(cid), "category": cat,
                                 "popularity_group": int(group), "opportunity_contribution": value})
    event_opp_df = pd.DataFrame(opp_rows)

    # Conversation-level exposure contributions, aggregated over events in each dialogue.
    exp_rows = []
    for method, matrix in (("baseline", be), ("proposed", pe)):
        for i, (eid, cid) in enumerate(zip(bi, bc)):
            for j, (cat, group) in enumerate(cells):
                value = float(matrix[i, j])
                if value > 0:
                    exp_rows.append({"method": method, "event_id": int(eid), "conv_id": int(cid),
                                     "category": cat, "popularity_group": int(group),
                                     "exposure_contribution": value})
    event_exp_df = pd.DataFrame(exp_rows)
    if not event_exp_df.empty:
        conversation_exp_df = event_exp_df.groupby(
            ["method", "conv_id", "category", "popularity_group"], as_index=False
        ).exposure_contribution.sum()
    else:
        conversation_exp_df = pd.DataFrame(columns=[
            "method", "conv_id", "category", "popularity_group", "exposure_contribution"
        ])

    comp_rows = []
    for j, (cat, group) in enumerate(cells):
        opp = float(total_opp[j]); eb = float(be[:, j].sum()); ep = float(pe[:, j].sum())
        comp_rows.append({
            "category": cat, "popularity_group": group, "opportunity_total": opp,
            "baseline_exposure": eb, "proposed_exposure": ep,
            "baseline_exposure_opportunity_rate": eb / (opp + EPS),
            "proposed_exposure_opportunity_rate": ep / (opp + EPS),
            "baseline_log_rate": math.log((eb + EPS) / (opp + EPS)),
            "proposed_log_rate": math.log((ep + EPS) / (opp + EPS)),
        })
    comp = pd.DataFrame(comp_rows)
    comp["supporting_conversations"] = cell_support
    comp["included_22_24B_positive_opportunity"] = positive_opp_mask
    comp["included_22_24C_support_threshold"] = fixed_22_24c_mask

    q, q_status = read_q(args.relevance_csv, bi, kbrd, item2group, categories)
    if args.category_weights_csv is None:
        omega = {c: 1.0 for c in all_categories}
        omega_source = "equal-category sensitivity convention (not specified by proposal)"
    else:
        wdf = pd.read_csv(args.category_weights_csv)
        if not {"category", "omega"}.issubset(wdf.columns) or wdf.category.duplicated().any():
            raise ValueError("Weights CSV requires unique category,omega columns.")
        omega = dict(zip(wdf.category.astype(str), pd.to_numeric(wdf.omega, errors="raise").astype(float)))
        if set(omega) != set(all_categories) or any(not math.isfinite(v) or v < 0 for v in omega.values()) or sum(omega.values()) <= 0:
            raise ValueError("Weights must cover all categories and be finite, non-negative, with positive sum.")
        omega_source = "user-supplied category weights"

    # Proposal Eq. (22) uses A_ic directly; unlike Track A's legacy ledger,
    # multi-label exposure is NOT divided by the number of item categories.
    b_proposal_exp = proposal_exposure(base, item2group, categories, cells)
    p_proposal_exp = proposal_exposure(prop, item2group, categories, cells)
    b_diag, b_loss = proposal_loss(b_proposal_exp, cells, q, all_categories, omega, args.min_relevance_mass)
    p_diag, p_loss = proposal_loss(p_proposal_exp, cells, q, all_categories, omega, args.min_relevance_mass)

    # Add proposal-specific R_cg and rho_cg to the same comparison table.
    comp["proposal_relevance_mass_R_cg"] = np.nan
    comp["proposal_baseline_exposure_E_cg"] = np.nan
    comp["proposal_proposed_exposure_E_cg"] = np.nan
    comp["proposal_baseline_rho_cg"] = np.nan
    comp["proposal_proposed_rho_cg"] = np.nan
    if q is not None and not q.empty:
        r_mass = defaultdict(float)
        for qr in q.itertuples(index=False):
            for cat in qr.categories:
                # Eq. (21): full q[e,i] contribution for each binary A_ic membership.
                r_mass[(str(cat), int(qr.group))] += float(qr.q)
        for j, (cat, group) in enumerate(cells):
            mass = float(r_mass[(cat, group)])
            comp.loc[j, "proposal_relevance_mass_R_cg"] = mass
            comp.loc[j, "proposal_baseline_exposure_E_cg"] = b_proposal_exp[j]
            comp.loc[j, "proposal_proposed_exposure_E_cg"] = p_proposal_exp[j]
            if mass > args.min_relevance_mass:
                comp.loc[j, "proposal_baseline_rho_cg"] = b_proposal_exp[j] / (mass + EPS)
                comp.loc[j, "proposal_proposed_rho_cg"] = p_proposal_exp[j] / (mass + EPS)

    args.out.mkdir(parents=True, exist_ok=True)
    event_opp_df.to_csv(args.out / "event_opportunity_reconstruction.csv", index=False)
    conversation_exp_df.to_csv(args.out / "conversation_exposure_reconstruction.csv", index=False)
    comp.to_csv(args.out / "category_group_estimand_comparison.csv", index=False)
    pd.DataFrame({
        "category": [c for c, _ in cells],
        "popularity_group": [g for _, g in cells],
        "supporting_conversations": cell_support,
        "positive_opportunity": positive_opp_mask,
        "included_22_24C_support_ge_threshold": fixed_22_24c_mask,
        "minimum_conversation_support": args.minimum_cell_conversations,
    }).to_csv(args.out / "fairness_cell_support_reconstruction.csv", index=False)
    if b_diag is not None:
        b_diag["categories"].to_csv(args.out / "proposal_category_diagnostics_baseline.csv", index=False)
        b_diag["cells"].to_csv(args.out / "proposal_cell_diagnostics_baseline.csv", index=False)
    if p_diag is not None:
        p_diag["categories"].to_csv(args.out / "proposal_category_diagnostics_proposed.csv", index=False)
        p_diag["cells"].to_csv(args.out / "proposal_cell_diagnostics_proposed.csv", index=False)

    audit = {
        "experiment": "22.24D_fairness_estimand_alignment_audit",
        "read_only": True, "training_performed": False, "retrieval_performed": False,
        "ranking_changes": False, "independent_test_rerun": False,
        "sample": {"events": len(base), "conversations": int(base.conv_id.nunique()),
                   "category_group_cells": len(cells)},
        "track_A_22_16_22_24C": {
            "exposure": "Top-10 DCG position weight 1/log2(rank+2), split across verified genres (legacy 22.16 ledger)",
            "opportunity": "normalized Top-500 DCG weights, split across verified genres",
            "metric": "mean absolute deviation of log((exposure+eps)/(opportunity+eps))",
            "22_24B_all_positive_opportunity": {
                "cells": int(positive_opp_mask.sum()),
                "baseline_log_rate_MAD": base_mad_22_24b,
                "proposed_log_rate_MAD": prop_mad_22_24b,
                "reduction_percent": reduction_22_24b
            },
            "22_24C_fixed_support": {
                "minimum_distinct_conversations": args.minimum_cell_conversations,
                "cells": int(fixed_22_24c_mask.sum()),
                "baseline_log_rate_MAD": base_mad,
                "proposed_log_rate_MAD": prop_mad,
                "reduction_percent": reduction,
                "matches_38_26_percent_within_0_05pp": bool(math.isfinite(reduction) and abs(reduction - 38.26) <= 0.05)
            }
        },
        "track_B_proposal_eq_21_to_25": {
            "q_input": q_status, "category_weight_source": omega_source,
            "baseline": b_loss, "proposed": p_loss,
            "exact_proposal_estimand_identifiable": bool(
                q is not None and not q.empty and args.category_weights_csv is not None
            ),
            "limitations": [
                "Frozen 22.24B event outputs do not contain event-item contextual relevance q[e,i].",
                "KG rank/confidence and candidate-pool category prevalence are not substitutes for semantic dialogue relevance.",
                "Proposal Eq. (21)-(22) use binary multi-label membership without dividing contributions by the number of categories; this differs from the legacy 22.16 ledger.",
                "The proposal does not numerically specify omega_c or the minimum relevance-support threshold; these must be reported as analysis choices."
            ]
        },
        "interpretation": "Track A opportunity-normalized log-rate MAD and Track B relevance-normalized within-category weighted variance are different estimands.",
        "outputs": ["metric_definition_audit.json", "event_opportunity_reconstruction.csv",
                    "conversation_exposure_reconstruction.csv", "category_group_estimand_comparison.csv",
                    "estimand_alignment_report.txt"]
    }
    (args.out / "metric_definition_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")

    report = [
        "EXPERIMENT 22.24D — FAIRNESS ESTIMAND ALIGNMENT REPORT", "=" * 64, "",
        f"Events: {len(base)}", f"Conversations: {base.conv_id.nunique()}",
        f"Category-group cells: {len(cells)}",
        f"22.24B positive-opportunity cells: {int(positive_opp_mask.sum())}",
        f"22.24C cells with >= {args.minimum_cell_conversations} conversations: {int(fixed_22_24c_mask.sum())}", "",
        "TRACK A — EXISTING IMPLEMENTATION",
        "22.24B all-positive-opportunity estimand:",
        f"  Baseline MAD: {base_mad_22_24b:.12g}",
        f"  Proposed MAD: {prop_mad_22_24b:.12g}",
        f"  Reduction: {reduction_22_24b:.6f}%",
        "22.24C fixed-support estimand:",
        f"  Baseline MAD: {base_mad:.12g}",
        f"  Proposed MAD: {prop_mad:.12g}",
        f"  Reduction: {reduction:.6f}%",
        f"  Matches 38.26% within 0.05 pp: {audit['track_A_22_16_22_24C']['22_24C_fixed_support']['matches_38_26_percent_within_0_05pp']}", "",
        "TRACK B — PROPOSAL EQUATIONS (21)-(25)",
        f"q[e,i] status: {q_status['status']}",
        f"Baseline L_fair: {b_loss}", f"Proposed L_fair: {p_loss}",
        f"Category weights: {omega_source}", "",
        "SCIENTIFIC INTERPRETATION",
        "The existing metric is opportunity-normalized log-rate MAD.",
        "The proposal metric is relevance-normalized, within-category weighted variance.",
        "Track B follows the proposal's binary multi-label A_ic membership; unlike the legacy",
        "22.16 ledger, it does not divide an item's q or exposure contribution by its genre count.",
        "They are not interchangeable. Without q[e,i], the proposal objective cannot be",
        "identified from the frozen 22.24B outputs. Do not substitute KG rank/confidence",
        "or category prevalence for dialogue-semantic relevance.", "",
        "READ-ONLY GUARANTEES",
        "No training, retrieval, parameter search, ranking changes, or test rerun."
    ]
    (args.out / "estimand_alignment_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    print("22.24D audit complete.")
    print(f"22.24B (112-cell) MAD: baseline={base_mad_22_24b:.10f}, proposed={prop_mad_22_24b:.10f}, reduction={reduction_22_24b:.4f}%")
    print(f"22.24C ({int(fixed_22_24c_mask.sum())}-cell) MAD: baseline={base_mad:.10f}, proposed={prop_mad:.10f}, reduction={reduction:.4f}%")
    print("Track B q status:", q_status["status"])
    print("Output:", args.out.resolve())
    print("Read-only: YES | Training: NO | Retrieval: NO | Ranking changes: NO")


if __name__ == "__main__":
    main()
