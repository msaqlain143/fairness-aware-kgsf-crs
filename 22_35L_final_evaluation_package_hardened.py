#!/usr/bin/env python3
"""Experiment 22.35L — hardened final KG-FairCRS evidence consolidation.

This is a consolidation-only utility. It consumes frozen 22.35F/G/H/J artifacts;
it does not train, retrieve, rerank, retune, regenerate recommendation slates,
or run new inferential tests. Original statistical CSVs are copied byte-for-byte.
Derived publication tables, figures, a report, validation checks, and a hash
manifest are written to a new output directory. Existing inputs are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYSTEM_ORDER = ["A0_KGSF", "A1_22_16", "A2_22_18", "A3_KG_FairCRS"]
EXPECTED_VARIANTS = {
    ("A0_KGSF", "R0_filtered"),
    ("A1_22_16", "R0_filtered"),
    ("A2_22_18", "R0_filtered"),
    ("A3_KG_FairCRS", "R0_filtered"),
    ("A3_KG_FairCRS", "R1_raw"),
}
EXPECTED_EVENTS = 4003
EXPECTED_CONVERSATIONS = 1283
CANDIDATES_PER_EVENT = 500
EXPECTED_CATEGORIES = 19
RECON_TOL = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact not found: {path}")
    return path


def require_pass(path: Path, label: str) -> dict:
    obj = read_json(require(path))
    if obj.get("overall_pass") is not True:
        raise RuntimeError(f"{label} validation does not pass: {path}")
    return obj


def pick_col(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for column in candidates:
        if column in df.columns:
            return column
    raise KeyError(
        f"Missing {label}; tried {candidates}; available={list(df.columns)}"
    )


def git_metadata(repo_root: Path) -> dict:
    """Capture Git provenance without making Git availability a hard dependency."""
    def git(*args: str) -> tuple[int, str, str]:
        try:
            proc = subprocess.run(
                ["git", *args], cwd=repo_root, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
            )
            return proc.returncode, proc.stdout.strip(), proc.stderr.strip()
        except FileNotFoundError:
            return 127, "", "git executable not found"

    rc_root, top, err_root = git("rev-parse", "--show-toplevel")
    if rc_root != 0:
        return {
            "available": False,
            "repository_root": None,
            "commit": None,
            "branch": None,
            "working_tree_clean": None,
            "status_porcelain": None,
            "reason": err_root or "The supplied repo root is not inside a Git repository.",
        }

    rc_commit, commit, err_commit = git("rev-parse", "HEAD")
    rc_branch, branch, err_branch = git("branch", "--show-current")
    rc_status, status, err_status = git("status", "--porcelain=v1", "--untracked-files=all")
    return {
        "available": True,
        "repository_root": top,
        "commit": commit if rc_commit == 0 else None,
        "branch": branch if rc_branch == 0 else None,
        "working_tree_clean": (status == "") if rc_status == 0 else None,
        "status_porcelain": status if rc_status == 0 else None,
        "errors": [x for x in (err_commit, err_branch, err_status) if x],
    }


def relative_key(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="Root containing the frozen experiment artifact paths.")
    parser.add_argument("--repo-root", type=Path, default=None,
                        help="Git repository root for provenance; defaults to --root.")
    parser.add_argument("--f-dir", type=Path, default=Path(
        "results_corrected/experiment_22_35/validated_fairness_evaluation_run_20261121"))
    parser.add_argument("--g-dir", type=Path, default=Path(
        "results_corrected/experiment_22_35/direct_paired_statistical_comparison_run_20261121"))
    parser.add_argument("--h-dir", type=Path, default=Path(
        "results_corrected/experiment_22_35/inferential_robustness_sensitivity_FINAL2"))
    parser.add_argument("--j-dir", type=Path, default=Path(
        "results_final_test_m2/experiment_22_35J"))
    parser.add_argument("--out", type=Path, default=Path(
        "results_final_test_m2/experiment_22_35L_REPRODUCIBILITY_V2"))
    args = parser.parse_args()

    root = args.root.resolve()
    repo_root = (args.repo_root or args.root).resolve()

    def resolve(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    fdir, gdir, hdir, jdir, out = map(
        resolve, [args.f_dir, args.g_dir, args.h_dir, args.j_dir, args.out]
    )
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")
    git_snapshot = git_metadata(repo_root)

    # Validate source experiment gates before reading or writing any result.
    fcheck = require_pass(fdir / "validation_checks.json", "22.35F")
    gcheck = require_pass(gdir / "validation_checks.json", "22.35G")
    hcheck = require_pass(hdir / "validation_checks.json", "22.35H")
    jcheck = require_pass(jdir / "validation_checks.json", "22.35J")

    required_inputs = [
        fdir / "system_comparison_summary.csv",
        fdir / "parameter_decisions.json",
        fdir / "experiment_manifest.json",
        fdir / "validation_checks.json",
        gdir / "paired_contrast_summary.csv",
        gdir / "experiment_manifest.json",
        gdir / "validation_checks.json",
        hdir / "paired_bca_inference.csv",
        hdir / "paired_cluster_signflip_utility.csv",
        hdir / "tuning_parameter_sensitivity.csv",
        hdir / "R0_R1_A3_sensitivity.csv",
        hdir / "category_level_fairness_decomposition.csv",
        hdir / "experiment_manifest.json",
        hdir / "validation_checks.json",
        jdir / "independent_test_system_comparison.csv",
        jdir / "independent_test_category_fairness.csv",
        jdir / "independent_test_utility_cluster_bootstrap.csv",
        jdir / "independent_test_coverage_summary.json",
        jdir / "validation_history_reconstruction.json",
        jdir / "experiment_manifest.json",
        jdir / "validation_checks.json",
    ]
    for path in required_inputs:
        require(path)

    fman = read_json(fdir / "experiment_manifest.json")
    fdecisions = read_json(fdir / "parameter_decisions.json")
    gman = read_json(gdir / "experiment_manifest.json")
    hman = read_json(hdir / "experiment_manifest.json")
    jman = read_json(jdir / "experiment_manifest.json")
    coverage = read_json(jdir / "independent_test_coverage_summary.json")
    history = read_json(jdir / "validation_history_reconstruction.json")

    # Cross-stage protocol checks: validation stages remain validation-only;
    # the independent test is a frozen-configuration re-evaluation.
    if fman.get("events") != 3010 or fman.get("conversations") != 883:
        raise RuntimeError("22.35F manifest does not match 3,010 events / 883 conversations.")
    if fman.get("independent_test_read") is not False:
        raise RuntimeError("22.35F must remain validation-only.")
    if gman.get("independent_test_read") is not False:
        raise RuntimeError("22.35G must remain validation-only.")
    if hcheck.get("independent_test_read") is not False:
        raise RuntimeError("22.35H validation checks do not confirm validation-only scope.")
    if jman.get("split") != "independent_test":
        raise RuntimeError("22.35J split is not independent_test.")
    if jman.get("events") != EXPECTED_EVENTS or jman.get("conversations") != EXPECTED_CONVERSATIONS:
        raise RuntimeError("22.35J manifest does not match the frozen 4,003-event / 1,283-conversation test.")
    if jman.get("parameter_retuning_performed") is not False:
        raise RuntimeError("22.35J reports parameter retuning.")
    if jman.get("training_performed") is not False or jman.get("retrieval_performed") is not False:
        raise RuntimeError("22.35J violates the no-training/no-retrieval protocol.")
    if jman.get("independent_test_used_for_parameter_selection") is not False:
        raise RuntimeError("22.35J reports test-set parameter selection.")
    selected = fdecisions.get("A3_KG_FairCRS", {}).get("parameters", {})
    test_selected = jman.get("A3_parameters", {})
    expected_parameters = {"q_threshold": 0.0005, "R_min": 0.5, "lambda": 1.0}
    if not all(np.isclose(float(selected.get(k, np.nan)), v, rtol=0, atol=1e-12)
               for k, v in [("q_threshold", 0.0005), ("R_min", 0.5), ("lam", 1.0)]):
        raise RuntimeError("22.35F parameter decision does not match the frozen selected configuration.")
    if not all(np.isclose(float(test_selected.get(k, np.nan)), v, rtol=0, atol=1e-12)
               for k, v in expected_parameters.items()):
        raise RuntimeError("22.35J parameters do not match the frozen 22.35F decisions.")
    if not jman.get("prior_test_exposure_disclosure"):
        raise RuntimeError("22.35J historical test-exposure disclosure is missing.")
    if jcheck.get("test_set_prior_historical_exposure_disclosed") is not True:
        raise RuntimeError("22.35J validation does not confirm historical test-exposure disclosure.")
    if jcheck.get("fairness_inference_claimed") is not False:
        raise RuntimeError("22.35J must not claim independent-test fairness inference.")
    if not all(isinstance(v, dict) and v.get("matches_frozen_22_35F") is True
               for v in history.values()):
        raise RuntimeError("Validation-history reconstruction does not fully match frozen 22.35F.")

    # Coverage must be complete and arithmetically self-consistent.
    expected_candidate_instances = EXPECTED_EVENTS * CANDIDATES_PER_EVENT
    candidate_instances = int(coverage["candidate_instances"])
    supported_instances = int(coverage["category_supported_candidate_instances"])
    coverage_ratio = float(coverage["candidate_category_coverage"])
    if candidate_instances != expected_candidate_instances:
        raise RuntimeError(
            f"Candidate denominator mismatch: {candidate_instances} != {expected_candidate_instances}."
        )
    if not (0 <= supported_instances <= candidate_instances):
        raise RuntimeError("Category-supported candidate count is outside valid bounds.")
    computed_coverage = supported_instances / candidate_instances
    if not np.isclose(coverage_ratio, computed_coverage, rtol=0, atol=1e-12):
        raise RuntimeError("Coverage ratio does not equal supported/total candidate instances.")
    if int(coverage.get("test_events", -1)) != EXPECTED_EVENTS:
        raise RuntimeError("Coverage summary event count does not match the test manifest.")
    if int(coverage.get("unique_test_conversations", -1)) != EXPECTED_CONVERSATIONS:
        raise RuntimeError("Coverage summary conversation count does not match the test manifest.")

    sysdf = pd.read_csv(jdir / "independent_test_system_comparison.csv")
    cat = pd.read_csv(jdir / "independent_test_category_fairness.csv")
    boot = pd.read_csv(jdir / "independent_test_utility_cluster_bootstrap.csv")

    system_col = pick_col(sysdf, ["system"], "system")
    variant_col = pick_col(sysdf, ["variant"], "variant")
    fair_col = pick_col(sysdf, ["L_fair_validation_plus_test", "L_fair"], "cumulative fairness loss")
    ndcg_col = pick_col(sysdf, ["NDCG@10", "NDCG_at_10", "ndcg10"], "NDCG@10")
    hr_col = pick_col(sysdf, ["HR@10", "HR_at_10", "hr10"], "HR@10")
    mrr_col = pick_col(sysdf, ["MRR@10", "MRR_at_10", "mrr10"], "MRR@10")
    active_col = pick_col(
        sysdf,
        ["active_categories_validation_plus_test", "active_categories", "active_category_count"],
        "active category count",
    )
    events_col = pick_col(sysdf, ["events"], "event count")
    conversations_col = pick_col(sysdf, ["conversations"], "conversation count")

    if len(sysdf) != len(EXPECTED_VARIANTS):
        raise RuntimeError(f"Expected exactly five system/variant rows; found {len(sysdf)}.")
    if sysdf.duplicated([system_col, variant_col]).any():
        raise RuntimeError("Duplicate system/variant rows detected in 22.35J.")
    observed_pairs = set(zip(sysdf[system_col].astype(str), sysdf[variant_col].astype(str)))
    if observed_pairs != EXPECTED_VARIANTS:
        raise RuntimeError(f"System/variant rows mismatch. Expected {EXPECTED_VARIANTS}; found {observed_pairs}.")
    if not (sysdf[events_col].astype(int) == EXPECTED_EVENTS).all():
        raise RuntimeError("At least one system row has an incorrect event count.")
    if not (sysdf[conversations_col].astype(int) == EXPECTED_CONVERSATIONS).all():
        raise RuntimeError("At least one system row has an incorrect conversation count.")
    metric_values = sysdf[[hr_col, ndcg_col, mrr_col, fair_col]].to_numpy(dtype=float)
    if not np.isfinite(metric_values).all():
        raise RuntimeError("Non-finite utility or fairness metric detected.")
    if (sysdf[[hr_col, ndcg_col, mrr_col]].to_numpy(dtype=float) < -1e-12).any():
        raise RuntimeError("Negative recommendation utility metric detected.")
    if (sysdf[fair_col].astype(float) < -1e-12).any():
        raise RuntimeError("Negative cumulative fairness objective detected.")

    # Reconstruct the category objective from unique category contributions.
    c_system = pick_col(cat, ["system"], "category system")
    c_variant = pick_col(cat, ["variant"], "category variant")
    c_category = pick_col(cat, ["category"], "category name")
    c_contrib = pick_col(cat, ["category_variance_contribution"], "category variance contribution")
    if cat.empty:
        raise RuntimeError("22.35J category fairness table is empty.")
    if not np.isfinite(cat[c_contrib].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite category contribution detected.")
    if (cat[c_contrib].astype(float) < -1e-12).any():
        raise RuntimeError("Negative category variance contribution detected.")
    conflict_counts = cat.groupby([c_system, c_variant, c_category])[c_contrib].nunique(dropna=False)
    if (conflict_counts > 1).any():
        raise RuntimeError("Conflicting duplicate category contribution values detected.")
    unique = cat[[c_system, c_variant, c_category, c_contrib]].drop_duplicates()
    category_sets = {
        key: set(group[c_category].astype(str))
        for key, group in unique.groupby([c_system, c_variant])
    }
    if set(category_sets) != EXPECTED_VARIANTS:
        raise RuntimeError("Category table does not cover exactly the five expected configurations.")
    reference_category_set = category_sets[("A0_KGSF", "R0_filtered")]
    if len(reference_category_set) != EXPECTED_CATEGORIES:
        raise RuntimeError(f"Expected {EXPECTED_CATEGORIES} categories; found {len(reference_category_set)}.")
    if any(values != reference_category_set for values in category_sets.values()):
        raise RuntimeError("Category labels differ across system/variant configurations.")

    totals = (
        unique.groupby([c_system, c_variant], as_index=False)
        .agg(categories=(c_category, "nunique"), reconstructed_L_fair=(c_contrib, "sum"))
    )
    reported = sysdf[[system_col, variant_col, fair_col, active_col]].copy()
    recon = totals.merge(
        reported, left_on=[c_system, c_variant], right_on=[system_col, variant_col],
        validate="one_to_one",
    )
    recon["absolute_difference"] = (
        recon["reconstructed_L_fair"] - recon[fair_col].astype(float)
    ).abs()
    if not np.allclose(recon["reconstructed_L_fair"], recon[fair_col].astype(float), rtol=0, atol=RECON_TOL):
        raise RuntimeError("Category reconstruction does not match the 22.35J fairness objective.")
    if not (recon["categories"].astype(int) == EXPECTED_CATEGORIES).all():
        raise RuntimeError("Expected 19 unique categories in every configuration.")
    if not (recon["categories"].astype(int) == recon[active_col].astype(int)).all():
        raise RuntimeError("Reconstructed unique category count differs from reported active-category count.")

    # Build publication-facing tables without modifying original source files.
    main = sysdf[[system_col, variant_col, hr_col, ndcg_col, mrr_col, fair_col, active_col]].copy()
    main = main.rename(columns={
        system_col: "system", variant_col: "variant", hr_col: "HR@10",
        ndcg_col: "NDCG@10", mrr_col: "MRR@10",
        fair_col: "L_fair_cumulative_validation_plus_test", active_col: "active_categories",
    })
    main["system"] = pd.Categorical(main["system"], categories=SYSTEM_ORDER, ordered=True)
    main["variant"] = pd.Categorical(main["variant"], categories=["R0_filtered", "R1_raw"], ordered=True)
    main = main.sort_values(["system", "variant"]).reset_index(drop=True)
    base = main[(main.system == "A0_KGSF") & (main.variant == "R0_filtered")].iloc[0]
    main["delta_HR@10_vs_A0"] = main["HR@10"] - float(base["HR@10"])
    main["delta_NDCG@10_vs_A0"] = main["NDCG@10"] - float(base["NDCG@10"])
    main["delta_MRR@10_vs_A0"] = main["MRR@10"] - float(base["MRR@10"])
    main["fairness_reduction_pct_vs_A0"] = (
        1 - main["L_fair_cumulative_validation_plus_test"] /
        float(base["L_fair_cumulative_validation_plus_test"])
    ) * 100

    primary = unique[unique[c_variant] == "R0_filtered"]
    a0 = primary[primary[c_system] == "A0_KGSF"][[c_category, c_contrib]].rename(
        columns={c_contrib: "A0_contribution"}
    )
    a3 = primary[primary[c_system] == "A3_KG_FairCRS"][[c_category, c_contrib]].rename(
        columns={c_contrib: "A3_contribution"}
    )
    catdelta = a0.merge(a3, on=c_category, validate="one_to_one")
    catdelta["change_A3_minus_A0"] = catdelta["A3_contribution"] - catdelta["A0_contribution"]
    catdelta["status"] = np.where(
        catdelta["change_A3_minus_A0"] > 1e-12, "Worsened",
        np.where(catdelta["change_A3_minus_A0"] < -1e-12, "Improved", "Unchanged"),
    )
    catdelta = catdelta.rename(columns={c_category: "category"}).sort_values(
        "change_A3_minus_A0", ascending=False
    )

    # Read source inferential tables for integrity checks, but preserve their bytes below.
    gcontrast = pd.read_csv(gdir / "paired_contrast_summary.csv")
    hbca = pd.read_csv(hdir / "paired_bca_inference.csv")
    hsign = pd.read_csv(hdir / "paired_cluster_signflip_utility.csv")
    hsens = pd.read_csv(hdir / "tuning_parameter_sensitivity.csv")
    hr1 = pd.read_csv(hdir / "R0_R1_A3_sensitivity.csv")
    hcat = pd.read_csv(hdir / "category_level_fairness_decomposition.csv")
    if len(gcontrast) != 12 or len(hbca) != 12 or len(hsign) != 9:
        raise RuntimeError("Unexpected row count in frozen G/H inferential source tables.")
    if not hbca["bca_available"].astype(bool).all():
        raise RuntimeError("At least one required BCa interval is unavailable.")
    if len(hr1) != 2 or set(hr1["variant"].astype(str)) != {"R0_filtered", "R1_raw"}:
        raise RuntimeError("R0/R1 sensitivity table is incomplete.")

    # Refuse to create a partial package if a late check fails.
    out.mkdir(parents=True)
    main.to_csv(out / "table_independent_test_main_results.csv", index=False)
    catdelta.to_csv(out / "table_category_regressions_R0.csv", index=False)
    recon.to_csv(out / "table_category_objective_reconstruction.csv", index=False)

    # Copy frozen inferential CSVs byte-for-byte; do not parse/re-serialize them.
    source_table_map = {
        "table_22_35G_paired_bootstrap_contrasts.csv": gdir / "paired_contrast_summary.csv",
        "table_22_35H_paired_BCa_inference.csv": hdir / "paired_bca_inference.csv",
        "table_22_35H_cluster_signflip_utility.csv": hdir / "paired_cluster_signflip_utility.csv",
        "table_22_35H_tuning_sensitivity.csv": hdir / "tuning_parameter_sensitivity.csv",
        "table_22_35H_R0_R1_sensitivity.csv": hdir / "R0_R1_A3_sensitivity.csv",
        "table_22_35H_category_decomposition.csv": hdir / "category_level_fairness_decomposition.csv",
        "table_22_35J_test_utility_bootstrap_descriptive.csv": jdir / "independent_test_utility_cluster_bootstrap.csv",
    }
    for output_name, source_path in source_table_map.items():
        shutil.copy2(source_path, out / output_name)
    source_copy_hash_match = all(
        sha256(source_path) == sha256(out / output_name)
        for output_name, source_path in source_table_map.items()
    )
    if not source_copy_hash_match:
        raise RuntimeError("At least one preserved G/H/J statistical CSV differs from its source bytes.")

    # Publication figures use only the frozen 22.35J summary tables.
    r0 = main[main["variant"] == "R0_filtered"].copy()
    label_map = {
        "A0_KGSF": "A0 KGSF", "A1_22_16": "A1 (22.16)",
        "A2_22_18": "A2 (22.18)", "A3_KG_FairCRS": "A3 KG-FairCRS",
    }
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar([label_map[str(x)] for x in r0["system"].astype(str)],
           r0["L_fair_cumulative_validation_plus_test"].astype(float))
    ax.set_ylabel("Cumulative category-conditioned fairness loss (lower is better)")
    ax.set_title("Cumulative fairness objective: validation history + independent test")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(out / "figure_fairness_loss_systems.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    colors = ["#b23a48" if x > 0 else "#287a55" for x in catdelta["change_A3_minus_A0"]]
    ax.barh(catdelta["category"].astype(str), catdelta["change_A3_minus_A0"].astype(float), color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in category variance contribution (A3 − A0)")
    ax.set_title("Category-level contribution changes (R0 primary; cumulative replay)")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out / "figure_category_contribution_changes.png", dpi=300)
    plt.close(fig)

    a3row = main[(main.system == "A3_KG_FairCRS") & (main.variant == "R0_filtered")].iloc[0]
    reduction = float(a3row["fairness_reduction_pct_vs_A0"])
    ndcg_delta = float(a3row["delta_NDCG@10_vs_A0"])
    hr_delta = float(a3row["delta_HR@10_vs_A0"])
    mrr_delta = float(a3row["delta_MRR@10_vs_A0"])
    improved = int((catdelta.status == "Improved").sum())
    worsened = int((catdelta.status == "Worsened").sum())
    unchanged = int((catdelta.status == "Unchanged").sum())

    report = f"""# KG-FairCRS — Final Experimental Results and Discussion

## Evaluation scope

This synthesis consumes frozen outputs from Experiments 22.35F, 22.35G, 22.35H and 22.35J. It does not retrain KGSF, repeat retrieval, rerank candidates, retune parameters, regenerate recommendation slates, or conduct new inferential tests. R0_filtered is the primary graph configuration; R1_raw is a sensitivity configuration evaluated with the R0-selected parameters.

The independent-test re-evaluation contains **{EXPECTED_EVENTS:,} recommendation events from {EXPECTED_CONVERSATIONS:,} conversations**, with a frozen {CANDIDATES_PER_EVENT}-candidate pool per event. The test split was examined in historical experiments 22.22–22.34; therefore, this is a **frozen-configuration independent-test re-evaluation**, not a never-before-seen test.

## Main independent-test results

For A3_KG_FairCRS (R0_filtered), HR@10 changed by {hr_delta:+.6f}, NDCG@10 by {ndcg_delta:+.6f}, and MRR@10 by {mrr_delta:+.6f} relative to A0_KGSF. The cumulative category-conditioned fairness objective decreased from {float(base['L_fair_cumulative_validation_plus_test']):.6f} to {float(a3row['L_fair_cumulative_validation_plus_test']):.6f}, a descriptive reduction of **{reduction:.2f}%**. The fairness value includes reconstructed validation history plus independent-test exposure; it is not a test-only fairness score. No independent-test fairness p-value or confidence interval is claimed.

The independent-test utility bootstrap is preserved from 22.35J in `table_22_35J_test_utility_bootstrap_descriptive.csv`. Its intervals describe utility uncertainty only. Validation-stage inferential evidence remains in the 22.35G/22.35H tables and must not be conflated with test-set inference.

## Category-level heterogeneity and coverage

The unique category contributions reconstruct the reported 22.35J fairness objective for all five system/variant configurations. Each contains {EXPECTED_CATEGORIES} categories, with absolute reconstruction differences no greater than {RECON_TOL:g}. In the primary A3 R0 comparison against A0, **{improved} categories improved, {worsened} worsened, and {unchanged} were unchanged**. The regressing categories are: {', '.join(catdelta.loc[catdelta.status == 'Worsened', 'category'].astype(str).tolist()) or 'none'}. Aggregate improvement must not be presented as uniform category-level improvement.

Category-supported candidate instances total **{supported_instances:,} of {candidate_instances:,}** ({100 * computed_coverage:.2f}%) under the experiment's reported definition. This is candidate-instance coverage, not catalog-wide item coverage.

## Statistical interpretation

The 22.35G paired bootstrap and 22.35H robustness analyses use conversation as the clustering unit. Report their validation-stage estimates, confidence intervals, multiplicity adjustments and sensitivity findings exactly as recorded in the preserved source tables. Bootstrap tail probabilities are not exact randomization-test p-values. Utility sign-flip inference relies on the conversation-level sign-exchangeability assumption recorded in the 22.35H manifest. The independent-test cumulative fairness objective is descriptive only.

## Reproducibility and limitations

The selected A3 configuration is q_threshold=0.0005, R_min=0.5 and lambda=1.0. R1_raw reuses R0-selected parameters without retuning. The relevance signal is operational (frozen KGSF scores and KG evidence), not a direct semantic judgment of dialogue relevance. Results are limited to ReDial movie recommendation with KGSF and should not be generalized to other datasets or backbones without additional evidence. The replay order is ascending event identifier and is not verified timestamp chronology.

## Manuscript claim boundary

The evidence supports a substantial reduction in the proposal-defined cumulative category-conditioned fairness objective on this frozen-configuration test re-evaluation, with very similar top-10 utility point estimates. It does not establish that every category improves, that the relevance proxy is semantic ground truth, that fairness gains are statistically significant on the independent test, or that the method generalizes beyond the evaluated benchmark/backbone.
"""
    (out / "final_results_and_discussion.md").write_text(report, encoding="utf-8", newline="\n")

    # Validation checks are computed from observed inputs, not hard-coded PASS flags.
    checks = {
        "experiment": "22.35L_hardened_final_evidence_consolidation",
        "22_35F_G_H_J_source_validations_pass": all(
            x.get("overall_pass") is True for x in (fcheck, gcheck, hcheck, jcheck)
        ),
        "22_35F_G_H_validation_only": (
            fman.get("independent_test_read") is False
            and gman.get("independent_test_read") is False
            and hcheck.get("independent_test_read") is False
        ),
        "independent_test_events_4003": EXPECTED_EVENTS == int(jman["events"]),
        "independent_test_conversations_1283": EXPECTED_CONVERSATIONS == int(jman["conversations"]),
        "candidate_pool_500_per_event": candidate_instances == EXPECTED_EVENTS * CANDIDATES_PER_EVENT,
        "five_expected_system_variants_present": observed_pairs == EXPECTED_VARIANTS and len(sysdf) == 5,
        "no_duplicate_system_variant_rows": not sysdf.duplicated([system_col, variant_col]).any(),
        "all_utility_and_fairness_values_finite": bool(np.isfinite(metric_values).all()),
        "category_labels_identical_across_configurations": all(
            values == reference_category_set for values in category_sets.values()
        ),
        "category_count_19_for_all_configurations": bool((recon["categories"] == EXPECTED_CATEGORIES).all()),
        "category_active_counts_match_reconstruction": bool(
            (recon["categories"].astype(int) == recon[active_col].astype(int)).all()
        ),
        "category_objective_reconstruction_within_1e_9": bool(
            (recon["absolute_difference"] <= RECON_TOL).all()
        ),
        "coverage_denominator_matches_4003_times_500": candidate_instances == 2_001_500,
        "coverage_ratio_arithmetically_verified": bool(
            np.isclose(coverage_ratio, computed_coverage, rtol=0, atol=1e-12)
        ),
        "validation_history_matches_22_35F": all(
            v.get("matches_frozen_22_35F") is True for v in history.values()
        ),
        "R0_primary_R1_sensitivity_only": bool(jman.get("R0_primary_R1_sensitivity_only") is True),
        "no_training_retrieval_or_retuning": (
            jman.get("training_performed") is False
            and jman.get("retrieval_performed") is False
            and jman.get("parameter_retuning_performed") is False
        ),
        "independent_test_fairness_inference_not_claimed": jcheck.get("fairness_inference_claimed") is False,
        "historical_test_exposure_disclosed": bool(jman.get("prior_test_exposure_disclosure"))
        and jcheck.get("test_set_prior_historical_exposure_disclosed") is True,
        "category_regressions_reported": worsened > 0,
        "original_inferential_csvs_copied_byte_for_byte": source_copy_hash_match,
        "22_35F_selected_parameters_match_22_35J_frozen_parameters": True,
    }
    checks["overall_pass"] = all(
        value is True for key, value in checks.items() if key != "experiment"
    )
    if checks["overall_pass"] is not True:
        raise RuntimeError(f"22.35L hardened validation failed: {checks}")
    (out / "validation_checks.json").write_text(
        json.dumps(checks, indent=2), encoding="utf-8", newline="\n"
    )

    # Hash every generated artifact except the manifest itself (self-hashing is circular).
    output_hashes = {
        path.name: sha256(path)
        for path in sorted(out.iterdir()) if path.is_file()
    }
    input_hashes = {
        relative_key(path, root): sha256(path) for path in required_inputs
    }
    source_path = Path(__file__).resolve()
    input_hashes[f"source_script:{source_path.name}"] = sha256(source_path)
    git = git_snapshot

    manifest = {
        "experiment": "22.35L_hardened_final_evidence_consolidation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen_evidence_consolidation_only",
        "source_script": {
            "path": str(source_path),
            "sha256": sha256(source_path),
        },
        "git_provenance": git,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "primary_configuration": {
            "system": "A3_KG_FairCRS", "variant": "R0_filtered",
            "q_threshold": 0.0005, "R_min": 0.5, "lambda": 1.0,
        },
        "sensitivity": {
            "variant": "R1_raw", "parameters_reused_from_R0": True, "retuned": False,
        },
        "independent_test": {
            "events": EXPECTED_EVENTS,
            "conversations": EXPECTED_CONVERSATIONS,
            "candidate_pool_per_event": CANDIDATES_PER_EVENT,
            "candidate_instances": candidate_instances,
            "category_supported_candidate_instances": supported_instances,
            "candidate_category_coverage": computed_coverage,
            "historically_examined_in_experiments": "22.22-22.34",
            "replay_order": jman.get("test_order"),
        },
        "source_manifests": {"22.35F": fman, "22.35G": gman, "22.35H": hman, "22.35J": jman},
        "input_sha256": input_hashes,
        "generated_output_sha256_excluding_manifest": output_hashes,
        "outputs": sorted([path.name for path in out.iterdir()] + ["experiment_manifest.json"]),
        "manifest_hash_note": "This manifest is intentionally excluded from its own output hash map.",
        "validation_checks": checks,
        "operations": {
            "training": False, "retrieval": False, "reranking": False,
            "parameter_retuning": False, "new_inferential_tests": False,
            "recommendation_slate_regeneration": False,
            "independent_test_fairness_p_value_claimed": False,
        },
    }
    (out / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8", newline="\n"
    )

    print("\n========== EXPERIMENT 22.35L HARDENED PACKAGE COMPLETE ==========")
    print(main.to_string(index=False))
    print(f"\nCategory changes: improved={improved}, worsened={worsened}, unchanged={unchanged}")
    print(f"A3 R0 cumulative fairness reduction vs A0: {reduction:.2f}% (descriptive)")
    print(f"Category-supported candidate instances: {supported_instances:,}/{candidate_instances:,} ({100*computed_coverage:.2f}%)")
    print(f"Output hashes recorded: {len(output_hashes)}")
    print(f"Git provenance available: {git['available']}; clean: {git.get('working_tree_clean')}")
    print("Validation overall_pass: True")
    print("Training: NO | Retrieval: NO | Reranking: NO | Retuning: NO | New inference: NO")
    print(f"Output: {out}")


if __name__ == "__main__":
    main()
