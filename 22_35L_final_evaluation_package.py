#!/usr/bin/env python3
"""Experiment 22.35L — final KG-FairCRS evidence consolidation.

Consumes frozen 22.35F/G/H/J artifacts only. Does not train, retrieve, rerank,
retune, regenerate predictions, or calculate new inferential tests. Produces
publication tables, category diagnostics, figures, a research synthesis, and a
hash-based reproducibility manifest. Existing source artifacts are read-only.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd

SYSTEM_ORDER = ["A0_KGSF", "A1_22_16", "A2_22_18", "A3_KG_FairCRS"]
VARIANT_ORDER = ["R0_filtered", "R1_raw"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen artifact not found: {path}")
    return path


def require_pass(path: Path, label: str):
    obj = read_json(require(path))
    if obj.get("overall_pass") is not True:
        raise RuntimeError(f"{label} validation does not pass: {path}")
    return obj


def pick_col(df: pd.DataFrame, candidates, label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"Missing {label}; tried {candidates}; available={list(df.columns)}")


def fmt(x, n=6):
    return f"{float(x):.{n}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--f-dir", type=Path, default=Path("results_corrected/experiment_22_35/validated_fairness_evaluation_run_20261121"))
    ap.add_argument("--g-dir", type=Path, default=Path("results_corrected/experiment_22_35/direct_paired_statistical_comparison_run_20261121"))
    ap.add_argument("--h-dir", type=Path, default=Path("results_corrected/experiment_22_35/inferential_robustness_sensitivity_FINAL2"))
    ap.add_argument("--j-dir", type=Path, default=Path("results_final_test_m2/experiment_22_35J"))
    ap.add_argument("--out", type=Path, default=Path("results_final_test_m2/experiment_22_35L_final_package"))
    a = ap.parse_args()
    root = a.root.resolve()
    def p(x): return x if x.is_absolute() else root / x
    fdir, gdir, hdir, jdir, out = map(p, [a.f_dir, a.g_dir, a.h_dir, a.j_dir, a.out])
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {out}")

    # Frozen validation gates; no analysis input is modified.
    fcheck = require_pass(fdir / "validation_checks.json", "22.35F")
    gcheck = require_pass(gdir / "validation_checks.json", "22.35G")
    hcheck = require_pass(hdir / "validation_checks.json", "22.35H")
    jcheck = require_pass(jdir / "validation_checks.json", "22.35J")

    paths = [
        fdir / "system_comparison_summary.csv", fdir / "parameter_decisions.json",
        fdir / "experiment_manifest.json", fdir / "validation_checks.json",
        gdir / "paired_contrast_summary.csv", gdir / "experiment_manifest.json",
        gdir / "validation_checks.json", hdir / "paired_bca_inference.csv",
        hdir / "paired_cluster_signflip_utility.csv", hdir / "tuning_parameter_sensitivity.csv",
        hdir / "R0_R1_A3_sensitivity.csv", hdir / "category_level_fairness_decomposition.csv",
        hdir / "experiment_manifest.json", hdir / "validation_checks.json",
        jdir / "independent_test_system_comparison.csv",
        jdir / "independent_test_category_fairness.csv",
        jdir / "independent_test_utility_cluster_bootstrap.csv",
        jdir / "independent_test_coverage_summary.json",
        jdir / "experiment_manifest.json", jdir / "validation_checks.json",
    ]
    for x in paths: require(x)

    jman = read_json(jdir / "experiment_manifest.json")
    if jman.get("split") != "independent_test" or jman.get("events") != 4003 or jman.get("conversations") != 1283:
        raise RuntimeError("22.35J manifest does not match the frozen 4,003-event / 1,283-conversation test.")
    if jman.get("parameter_retuning_performed") is not False or jman.get("training_performed") is not False or jman.get("retrieval_performed") is not False:
        raise RuntimeError("22.35J manifest does not preserve the frozen no-training/no-retrieval/no-retuning protocol.")
    if jman.get("independent_test_used_for_parameter_selection") is not False:
        raise RuntimeError("22.35J reports test-set parameter selection; cannot freeze this package.")
    if jman.get("prior_test_exposure_disclosure") is None or jcheck.get("test_set_prior_historical_exposure_disclosed") is not True:
        raise RuntimeError("22.35J historical test-exposure disclosure is missing.")

    sysdf = pd.read_csv(jdir / "independent_test_system_comparison.csv")
    cat = pd.read_csv(jdir / "independent_test_category_fairness.csv")
    boot = pd.read_csv(jdir / "independent_test_utility_cluster_bootstrap.csv")
    coverage = read_json(require(jdir / "independent_test_coverage_summary.json"))
    system_col = pick_col(sysdf, ["system"], "system")
    variant_col = pick_col(sysdf, ["variant"], "variant")
    fair_col = pick_col(sysdf, ["L_fair_validation_plus_test", "L_fair"], "cumulative fairness loss")
    ndcg_col = pick_col(sysdf, ["NDCG@10", "NDCG_at_10", "ndcg10"], "NDCG@10")
    hr_col = pick_col(sysdf, ["HR@10", "HR_at_10", "hr10"], "HR@10")
    mrr_col = pick_col(sysdf, ["MRR@10", "MRR_at_10", "mrr10"], "MRR@10")
    active_col = pick_col(sysdf, ["active_categories_validation_plus_test", "active_categories", "active_category_count"], "active category count")

    expected = {(s, "R0_filtered") for s in SYSTEM_ORDER} | {("A3_KG_FairCRS", "R1_raw")}
    got = set(zip(sysdf[system_col].astype(str), sysdf[variant_col].astype(str)))
    if got != expected:
        raise RuntimeError(f"22.35J system rows mismatch. Expected {expected}; found {got}")
    if len(cat) == 0:
        raise RuntimeError("22.35J category fairness table is empty.")
    c_system = pick_col(cat, ["system"], "category system")
    c_variant = pick_col(cat, ["variant"], "category variant")
    c_category = pick_col(cat, ["category"], "category name")
    c_contrib = pick_col(cat, ["category_variance_contribution"], "category variance contribution")
    # Exact 22.35K reconstruction: de-duplicate repeated event/category rows, then sum once.
    unique = cat[[c_system, c_variant, c_category, c_contrib]].drop_duplicates()
    if unique.duplicated([c_system, c_variant, c_category]).any():
        raise RuntimeError("Conflicting duplicate category contribution values in 22.35J table.")
    totals = (unique.groupby([c_system, c_variant], as_index=False)
              .agg(categories=(c_category, "nunique"), reconstructed_L_fair=(c_contrib, "sum")))
    reported = sysdf[[system_col, variant_col, fair_col, active_col]].copy()
    recon = totals.merge(reported, left_on=[c_system, c_variant], right_on=[system_col, variant_col], validate="one_to_one")
    recon["absolute_difference"] = (recon["reconstructed_L_fair"] - recon[fair_col]).abs()
    if not np.allclose(recon["reconstructed_L_fair"], recon[fair_col], rtol=0, atol=1e-9):
        raise RuntimeError("22.35K category reconstruction no longer matches 22.35J reported fairness objective.")
    if not (recon["categories"] == 19).all():
        raise RuntimeError("Expected 19 supported categories in every 22.35J system configuration.")

    # Publication-ready main table; utility remains descriptive on independent test.
    main = sysdf[[system_col, variant_col, hr_col, ndcg_col, mrr_col, fair_col, active_col]].copy()
    main = main.rename(columns={system_col:"system", variant_col:"variant", hr_col:"HR@10", ndcg_col:"NDCG@10", mrr_col:"MRR@10", fair_col:"L_fair_cumulative_validation_plus_test", active_col:"active_categories"})
    base = main[(main.system == "A0_KGSF") & (main.variant == "R0_filtered")].iloc[0]
    main["delta_HR@10_vs_A0"] = main["HR@10"] - float(base["HR@10"])
    main["delta_NDCG@10_vs_A0"] = main["NDCG@10"] - float(base["NDCG@10"])
    main["fairness_reduction_pct_vs_A0"] = (1 - main["L_fair_cumulative_validation_plus_test"] / float(base["L_fair_cumulative_validation_plus_test"])) * 100

    # Per-category regressions/improvements for primary R0 comparison only.
    primary = unique[unique[c_variant] == "R0_filtered"]
    a0 = primary[primary[c_system] == "A0_KGSF"][[c_category, c_contrib]].rename(columns={c_contrib:"A0_contribution"})
    a3 = primary[primary[c_system] == "A3_KG_FairCRS"][[c_category, c_contrib]].rename(columns={c_contrib:"A3_contribution"})
    catdelta = a0.merge(a3, on=c_category, validate="one_to_one")
    catdelta["change_A3_minus_A0"] = catdelta["A3_contribution"] - catdelta["A0_contribution"]
    catdelta["status"] = np.where(catdelta["change_A3_minus_A0"] > 1e-12, "Worsened", np.where(catdelta["change_A3_minus_A0"] < -1e-12, "Improved", "Unchanged"))
    catdelta = catdelta.rename(columns={c_category:"category"}).sort_values("change_A3_minus_A0", ascending=False)

    # Freeze inferential source tables without recalculating or upgrading their claims.
    gcontrast = pd.read_csv(gdir / "paired_contrast_summary.csv")
    hbca = pd.read_csv(hdir / "paired_bca_inference.csv")
    hsign = pd.read_csv(hdir / "paired_cluster_signflip_utility.csv")
    hsens = pd.read_csv(hdir / "tuning_parameter_sensitivity.csv")
    hr1 = pd.read_csv(hdir / "R0_R1_A3_sensitivity.csv")
    hcat = pd.read_csv(hdir / "category_level_fairness_decomposition.csv")

    out.mkdir(parents=True)
    main.to_csv(out / "table_independent_test_main_results.csv", index=False)
    catdelta.to_csv(out / "table_category_regressions_R0.csv", index=False)
    recon.to_csv(out / "table_category_objective_reconstruction.csv", index=False)
    # Include source statistical tables byte-for-byte as read/re-exported data, clearly labelled by provenance.
    for name, df in [("table_22_35G_paired_bootstrap_contrasts.csv", gcontrast),
                     ("table_22_35H_paired_BCa_inference.csv", hbca),
                     ("table_22_35H_cluster_signflip_utility.csv", hsign),
                     ("table_22_35H_tuning_sensitivity.csv", hsens),
                     ("table_22_35H_R0_R1_sensitivity.csv", hr1),
                     ("table_22_35H_category_decomposition.csv", hcat),
                     ("table_22_35J_test_utility_bootstrap_descriptive.csv", boot)]:
        df.to_csv(out / name, index=False)

    # Publication plots use only the frozen summary tables.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r0 = main[main.variant == "R0_filtered"].copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.bar(r0.system, r0["L_fair_cumulative_validation_plus_test"])
    ax.set_ylabel("Cumulative category-conditioned fairness loss (lower is better)")
    ax.set_title("Independent-test cumulative fairness objective")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout(); fig.savefig(out / "figure_fairness_loss_systems.png", dpi=300); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    colors = ["#b23a48" if x > 0 else "#287a55" for x in catdelta["change_A3_minus_A0"]]
    ax.barh(catdelta["category"], catdelta["change_A3_minus_A0"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Change in category variance contribution (A3 − A0)")
    ax.set_title("Category-level fairness contribution changes (R0 primary)")
    ax.invert_yaxis(); fig.tight_layout(); fig.savefig(out / "figure_category_contribution_changes.png", dpi=300); plt.close(fig)

    a3 = main[(main.system == "A3_KG_FairCRS") & (main.variant == "R0_filtered")].iloc[0]
    reduction = float(a3["fairness_reduction_pct_vs_A0"])
    ndcg_delta = float(a3["delta_NDCG@10_vs_A0"])
    hr_delta = float(a3["delta_HR@10_vs_A0"])
    improved = int((catdelta.status == "Improved").sum())
    worsened = int((catdelta.status == "Worsened").sum())
    unchanged = int((catdelta.status == "Unchanged").sum())
    report = f"""# KG-FairCRS — Final Experimental Results and Discussion

## Evaluation scope

This synthesis consumes the frozen outputs of Experiments 22.35F, 22.35G, 22.35H and 22.35J. It does not retrain KGSF, repeat retrieval, rerank candidates, retune parameters, or conduct new statistical tests. The primary graph configuration is R0_filtered; R1_raw is a prespecified sensitivity configuration using the R0-selected A3 parameters.

The independent-test re-evaluation contains **4,003 recommendation events from 1,283 conversations**, with a frozen 500-candidate pool per event. The test split had been examined in historical experiments 22.22–22.34; therefore, these results must be described as a **frozen-configuration independent-test re-evaluation**, not as a never-before-seen test.

## Main independent-test results

See `table_independent_test_main_results.csv`. For A3_KG_FairCRS (R0_filtered), HR@10 changed by {hr_delta:+.6f} relative to A0_KGSF, NDCG@10 changed by {ndcg_delta:+.6f}, and cumulative category-conditioned fairness loss decreased from {float(base['L_fair_cumulative_validation_plus_test']):.6f} to {float(a3['L_fair_cumulative_validation_plus_test']):.6f}, a reduction of **{reduction:.2f}%**. This is a descriptive independent-test fairness comparison; no independent-test fairness p-value or confidence interval is claimed.

The paired conversation-cluster bootstrap for independent-test utility is retained in `table_22_35J_test_utility_bootstrap_descriptive.csv`. Its intervals are used only to describe utility uncertainty. The validation-stage inferential evidence remains in the 22.35G/22.35H tables and must not be conflated with test-set inference.

## Category-level heterogeneity

The 22.35K reconstruction is reproduced here: unique category contributions sum to the reported 22.35J fairness objective for all five system configurations, with 19 categories and absolute reconstruction differences no greater than 1e-9. In the primary A3 R0 comparison against A0, **{improved} categories improved, {worsened} worsened, and {unchanged} were unchanged**. The largest regressions are shown first in `table_category_regressions_R0.csv`; the complete category profile is plotted in `figure_category_contribution_changes.png`. The regressing categories are: {', '.join(catdelta.loc[catdelta.status == 'Worsened', 'category'].astype(str).tolist()) or 'none'}.

The aggregate improvement must not be presented as uniform category-level improvement. These category regressions motivate a limitation and future-work discussion, not post-test parameter changes.

## Statistical interpretation

The 22.35G paired bootstrap and 22.35H robustness analyses use conversation as the clustering unit. Report their validation-stage estimates, confidence intervals, multiplicity adjustments and sensitivity findings exactly as recorded in the source tables. Do not claim an exact randomization p-value for the nonlinear cumulative fairness objective. Utility sign-flip inference depends on the exchangeability/symmetry assumption stated in the 22.35H manifest. The independent-test fairness objective is descriptive only.

## Reproducibility and limitations

The selected A3 configuration is q_threshold=0.0005, R_min=0.5 and lambda=1.0. R1_raw reuses the R0-selected parameters without retuning. The relevance signal is operational (KGSF score and KG support), not a direct semantic judgment of dialogue relevance. Category evidence covers only a subset of catalog/candidate instances; the 22.35J coverage summary reports {int(coverage.get('category_supported_candidate_instances', 0)):,} category-supported candidate instances out of {int(coverage.get('candidate_instances', 0)):,} ({100*float(coverage.get('candidate_category_coverage', 0)):.2f}%). Results are limited to ReDial movie recommendation with KGSF and should not be generalized to other datasets or backbones without additional evidence.

## Manuscript claim boundary

The evidence supports a substantial reduction in the proposal-defined cumulative category-conditioned fairness objective on this frozen-configuration test re-evaluation, with essentially preserved top-10 utility. It does not establish that every category improves, that the relevance score is semantic ground truth, that fairness gains are statistically significant on the independent test, or that the method generalizes beyond the evaluated benchmark/backbone.
"""
    (out / "final_results_and_discussion.md").write_text(report, encoding="utf-8")

    source_manifests = {"22.35F": read_json(fdir/"experiment_manifest.json"),
                        "22.35G": read_json(gdir/"experiment_manifest.json"),
                        "22.35H": read_json(hdir/"experiment_manifest.json"),
                        "22.35J": jman}
    input_hashes = {str(x.relative_to(root)) if x.is_relative_to(root) else str(x): sha256(x) for x in paths}
    checks = {
        "experiment":"22.35L_final_evaluation_and_reproducibility_freeze",
        "source_22_35F_G_H_J_validations_pass":True,
        "independent_test_events_4003":True,
        "independent_test_conversations_1283":True,
        "five_expected_system_variants_present":True,
        "category_objective_reconstruction_matches_22_35J":True,
        "category_count_19_for_all_configurations":True,
        "R0_primary_R1_sensitivity_only":True,
        "no_training_retrieval_reranking_or_retuning_performed":True,
        "independent_test_fairness_inference_claimed":False,
        "historical_test_exposure_disclosed":True,
        "category_regressions_reported":True,
        "overall_pass":True,
    }
    (out/"validation_checks.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    manifest = {
        "experiment":"22.35L_final_evaluation_and_reproducibility_freeze",
        "created_utc":datetime.now(timezone.utc).isoformat(),
        "python":sys.version,
        "platform":platform.platform(),
        "pandas":pd.__version__, "numpy":np.__version__,
        "scope":"frozen_evidence_consolidation_only",
        "primary_configuration":{"system":"A3_KG_FairCRS","variant":"R0_filtered","q_threshold":0.0005,"R_min":0.5,"lambda":1.0},
        "sensitivity":{"variant":"R1_raw","parameters_reused_from_R0":True,"retuned":False},
        "independent_test":{"events":4003,"conversations":1283,"candidate_pool_per_event":500,"historically_examined_in_experiments":"22.22-22.34"},
        "source_manifests":source_manifests,
        "input_sha256":input_hashes,
        "outputs":sorted(x.name for x in out.iterdir()),
        "validation_checks":checks,
        "operations":{"training":False,"retrieval":False,"reranking":False,"parameter_retuning":False,"new_inferential_tests":False,"test_fairness_p_value_claimed":False},
    }
    (out/"experiment_manifest.json").write_text(json.dumps(manifest,indent=2,default=str),encoding="utf-8")

    print("\n========== EXPERIMENT 22.35L COMPLETE ==========")
    print(main.to_string(index=False))
    print(f"\nCategory changes: improved={improved}, worsened={worsened}, unchanged={unchanged}")
    print(f"A3 R0 fairness reduction vs A0: {reduction:.2f}%")
    print("Validation overall_pass: True")
    print("Training: NO | Retrieval: NO | Reranking: NO | Retuning: NO | New inference: NO")
    print(f"Output: {out}")

if __name__ == "__main__":
    main()
