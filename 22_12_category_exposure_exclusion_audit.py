#!/usr/bin/env python
"""
Experiment 22.12: Category Exposure and Exclusion Audit.

Validation-only analysis of existing Experiment 22.10 outputs.
No training, retrieval, coefficient search, or test evaluation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-12

METHODS = [
    "B0_KBRD",
    "B_category_only_lambda_0p2",
    "B_KG_gated_lambda_0p2",
    "B_category_only_lambda_0p4",
    "B_KG_gated_lambda_0p4",
]


def category_metrics(group: pd.DataFrame) -> dict:
    """Calculate category-level exposure and exclusion diagnostics."""
    group = group.copy()

    exposure = group["exposure"].to_numpy(dtype=float)
    opportunity = group["opportunity_mass"].to_numpy(dtype=float)

    # Only cells with positive opportunity belong in the disparity calculation.
    eligible = opportunity > EPS
    exposure = exposure[eligible]
    opportunity = opportunity[eligible]

    if len(exposure) == 0:
        mad = float("nan")
        zero_rate = float("nan")
    else:
        rates = exposure / (opportunity + EPS)
        log_rates = np.log(np.maximum(rates, EPS))
        mad = float(np.mean(np.abs(log_rates - log_rates.mean())))
        zero_rate = float(np.mean(exposure <= EPS))

    total_exposure = float(group["exposure"].sum())
    total_opportunity = float(group["opportunity_mass"].sum())

    return {
        "groups": int(len(exposure)),
        "total_exposure": total_exposure,
        "total_opportunity": total_opportunity,
        "zero_exposure_groups": int(np.sum(exposure <= EPS)),
        "zero_exposure_group_rate": zero_rate,
        "completely_unexposed": total_exposure <= EPS,
        "category_MAD_recalculated": mad,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 22.12: category exposure and exclusion audit."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results_corrected/experiment_22_10/"
            "category_fairness_detail.csv"
        ),
    )
    parser.add_argument(
        "--original-summary",
        type=Path,
        default=Path(
            "results_corrected/experiment_22_10/"
            "category_fairness_results.csv"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results_corrected/experiment_22_12"),
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.original_summary.is_file():
        raise FileNotFoundError(args.original_summary)

    df = pd.read_csv(args.input)
    original = pd.read_csv(args.original_summary)

    required = {
        "method",
        "category",
        "popularity_group",
        "exposure",
        "opportunity_mass",
        "category_log_rate_mad",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")

    original_required = {"method", "category_group_log_rate_MAD"}
    original_missing = original_required - set(original.columns)
    if original_missing:
        raise ValueError(
            "Original summary is missing columns: "
            f"{sorted(original_missing)}"
        )

    df["exposure"] = pd.to_numeric(df["exposure"], errors="raise")
    df["opportunity_mass"] = pd.to_numeric(
        df["opportunity_mass"], errors="raise"
    )
    df["popularity_group"] = pd.to_numeric(
        df["popularity_group"], errors="raise"
    ).astype(int)

    if not np.isfinite(df[["exposure", "opportunity_mass"]].to_numpy()).all():
        raise ValueError("Exposure/opportunity contains non-finite values.")
    if (df["exposure"] < -EPS).any():
        raise ValueError("Negative exposure values found.")
    if (df["opportunity_mass"] < -EPS).any():
        raise ValueError("Negative opportunity values found.")

    duplicate_cells = df.duplicated(
        ["method", "category", "popularity_group"]
    ).any()
    if duplicate_cells:
        raise ValueError("Duplicate method/category/popularity-group cells.")

    observed_methods = set(df["method"].dropna().unique())
    if observed_methods != set(METHODS):
        raise ValueError(
            f"Unexpected method set. Expected {METHODS}; "
            f"found {sorted(observed_methods)}"
        )

    if original["method"].duplicated().any():
        raise ValueError("Original summary contains duplicate methods.")
    original_methods = set(original["method"].dropna().unique())
    if original_methods != set(METHODS):
        raise ValueError(
            "Original summary method set does not match the five expected methods."
        )

    args.out.mkdir(parents=True, exist_ok=True)

    # Category-by-method metrics.
    category_rows = []
    for (method, category), group in df.groupby(
        ["method", "category"], sort=True
    ):
        category_rows.append({
            "method": method,
            "category": category,
            **category_metrics(group),
        })
    category_df = pd.DataFrame(category_rows)

    method_rows = []
    group_rows = []

    for method in METHODS:
        subset = df[df["method"] == method].copy()
        categories = category_df[category_df["method"] == method]

        if categories.empty:
            raise ValueError(f"No category records found for {method}.")

        fully_excluded = int(categories["completely_unexposed"].sum())
        original_mad = float(
            categories["category_MAD_recalculated"].mean()
        )

        included = categories[~categories["completely_unexposed"]]
        sensitivity_mad = (
            float(included["category_MAD_recalculated"].mean())
            if not included.empty else float("nan")
        )

        opportunity_cells = len(subset)
        zero_cells = int((subset["exposure"] <= EPS).sum())

        method_rows.append({
            "method": method,
            "categories": int(len(categories)),
            "completely_unexposed_categories": fully_excluded,
            "category_exposure_coverage": (
                1.0 - fully_excluded / len(categories)
            ),
            "opportunity_cells": int(opportunity_cells),
            "zero_exposure_cells": zero_cells,
            "zero_exposure_cell_rate": (
                zero_cells / opportunity_cells
                if opportunity_cells else float("nan")
            ),
            "total_exposure": float(subset["exposure"].sum()),
            "total_opportunity": float(
                subset["opportunity_mass"].sum()
            ),
            "original_definition_MAD_recalculated": original_mad,
            "MAD_excluding_completely_unexposed_categories":
                sensitivity_mad,
        })

        for group_id, cells in subset.groupby(
            "popularity_group", sort=True
        ):
            eligible_cells = cells[cells["opportunity_mass"] > EPS]
            group_rows.append({
                "method": method,
                "popularity_group": int(group_id),
                "eligible_category_cells": int(len(eligible_cells)),
                "zero_exposure_categories": int(
                    (eligible_cells["exposure"] <= EPS).sum()
                ),
                "zero_exposure_rate": (
                    float(np.mean(eligible_cells["exposure"] <= EPS))
                    if len(eligible_cells) else float("nan")
                ),
                "total_exposure": float(cells["exposure"].sum()),
                "total_opportunity": float(
                    cells["opportunity_mass"].sum()
                ),
            })

    method_df = pd.DataFrame(method_rows)
    group_df = pd.DataFrame(group_rows)

    # Check that the same category/group opportunity definition is used
    # for every method, with identical cell coverage and values.
    opportunity_pivot = df.pivot(
        index=["category", "popularity_group"],
        columns="method",
        values="opportunity_mass",
    )
    opportunity_invariant = bool(
        set(opportunity_pivot.columns) == set(METHODS)
        and opportunity_pivot.notna().all().all()
        and np.allclose(
            opportunity_pivot.to_numpy(dtype=float),
            opportunity_pivot.iloc[:, 0].to_numpy(dtype=float)[:, None],
            rtol=1e-9,
            atol=1e-12,
        )
    )

    # Reconcile the original category-group MAD with Experiment 22.10.
    original_lookup = dict(zip(
        original["method"],
        pd.to_numeric(
            original["category_group_log_rate_MAD"], errors="raise"
        ),
    ))

    mad_checks = {}
    for row in method_rows:
        method = row["method"]
        expected = float(original_lookup[method])
        observed = float(row["original_definition_MAD_recalculated"])
        mad_checks[method] = {
            "original_MAD": expected,
            "recalculated_MAD": observed,
            "matches": bool(
                np.isclose(expected, observed, rtol=1e-8, atol=1e-8)
            ),
        }

    # These are positive assertions about prohibited operations:
    # True means the operation was not performed.
    checks = {
        "all_five_methods_present": (
            set(method_df["method"]) == set(METHODS)
            and len(method_df) == len(METHODS)
        ),
        "unique_category_group_cells": not duplicate_cells,
        "opportunity_mass_identical_across_methods": opportunity_invariant,
        "original_MAD_recalculation_matches": all(
            result["matches"] for result in mad_checks.values()
        ),
        "test_split_not_used": True,
        "training_not_performed": True,
        "retrieval_not_rerun": True,
        "new_coefficients_not_introduced": True,
    }
    checks["overall_pass"] = all(checks.values())

    category_df.to_csv(
        args.out / "category_exclusion_detail.csv", index=False
    )
    method_df.to_csv(
        args.out / "method_exclusion_summary.csv", index=False
    )
    group_df.to_csv(
        args.out / "popularity_group_exclusion_summary.csv", index=False
    )

    with open(
        args.out / "validation_checks.json", "w", encoding="utf-8"
    ) as f:
        json.dump(
            {"checks": checks, "mad_reconciliation": mad_checks},
            f,
            indent=2,
            allow_nan=False,
        )

    manifest = {
        "experiment": "22.12",
        "name": "Category Exposure and Exclusion Audit",
        "validation_only": True,
        "test_split_evaluated": False,
        "training_performed": False,
        "retrieval_rerun": False,
        "new_coefficients_introduced": False,
        "methods": METHODS,
        "input": str(args.input),
        "outputs": [
            "category_exclusion_detail.csv",
            "method_exclusion_summary.csv",
            "popularity_group_exclusion_summary.csv",
            "validation_checks.json",
            "experiment_manifest.json",
        ],
        "interpretation": [
            "Original category-group MAD is retained and reconciled.",
            "Completely unexposed categories are separately identified.",
            "Sensitivity MAD excludes completely unexposed categories.",
            "Sensitivity MAD is supplementary, not a replacement for original MAD.",
            "Exposure totals can overlap across multi-category items.",
            "KG gating and category reranking results are associations, not causal proof.",
        ],
    }

    with open(
        args.out / "experiment_manifest.json", "w", encoding="utf-8"
    ) as f:
        json.dump(manifest, f, indent=2)

    print("\n========== EXPERIMENT 22.12 ==========")
    print("\nMethod-level results:")
    print(method_df.to_string(index=False))
    print("\nPopularity-group exclusion:")
    print(group_df.to_string(index=False))
    print("\nValidation checks:")
    print(json.dumps(checks, indent=2))
    print("\nOutput:", args.out.resolve())

    if not checks["overall_pass"]:
        raise RuntimeError("Experiment validation failed.")


if __name__ == "__main__":
    main()
