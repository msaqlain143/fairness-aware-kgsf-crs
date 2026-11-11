
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


def category_metrics(group):
    group = group.copy()

    exposure = group["exposure"].to_numpy(float)
    opportunity = group["opportunity_mass"].to_numpy(float)

    rates = exposure / (opportunity + EPS)
    log_rates = np.log(np.maximum(rates, EPS))

    mad = float(np.mean(np.abs(log_rates - log_rates.mean())))

    total_exposure = float(exposure.sum())
    total_opportunity = float(opportunity.sum())

    return {
        "groups": len(group),
        "total_exposure": total_exposure,
        "total_opportunity": total_opportunity,
        "zero_exposure_groups": int(np.sum(exposure <= EPS)),
        "zero_exposure_group_rate": float(
            np.mean(exposure <= EPS)
        ),
        "completely_unexposed": total_exposure <= EPS,
        "category_MAD_recalculated": mad,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results_corrected/experiment_22_10"
            "/category_fairness_detail.csv"
        ),
    )
    parser.add_argument(
        "--original-summary",
        type=Path,
        default=Path(
            "results_corrected/experiment_22_10"
            "/category_fairness_results.csv"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "results_corrected/experiment_22_12"
        ),
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    if not args.original_summary.exists():
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
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["exposure"] = pd.to_numeric(df["exposure"])
    df["opportunity_mass"] = pd.to_numeric(
        df["opportunity_mass"]
    )

    if df.duplicated(
        ["method", "category", "popularity_group"]
    ).any():
        raise ValueError("Duplicate category-group cells.")

    if not set(df["method"]) == set(METHODS):
        raise ValueError("Unexpected method set.")

    args.out.mkdir(parents=True, exist_ok=True)

    category_rows = []

    for (method, category), group in df.groupby(
        ["method", "category"]
    ):
        result = category_metrics(group)

        category_rows.append({
            "method": method,
            "category": category,
            **result,
        })

    category_df = pd.DataFrame(category_rows)

    method_rows = []
    group_rows = []

    for method in METHODS:
        subset = df[df.method == method].copy()
        categories = category_df[
            category_df.method == method
        ]

        fully_excluded = int(
            categories.completely_unexposed.sum()
        )

        original_mad = float(
            categories["category_MAD_recalculated"].mean()
        )

        included = categories[
            ~categories.completely_unexposed
        ]

        sensitivity_mad = (
            float(included.category_MAD_recalculated.mean())
            if len(included)
            else np.nan
        )

        opportunity_cells = len(subset)

        zero_cells = int(
            (subset.exposure <= EPS).sum()
        )

        method_rows.append({
            "method": method,
            "categories": len(categories),
            "completely_unexposed_categories": fully_excluded,
            "category_exposure_coverage": (
                1 - fully_excluded / len(categories)
            ),
            "opportunity_cells": opportunity_cells,
            "zero_exposure_cells": zero_cells,
            "zero_exposure_cell_rate": (
                zero_cells / opportunity_cells
            ),
            "total_exposure": float(subset.exposure.sum()),
            "total_opportunity": float(
                subset.opportunity_mass.sum()
            ),
            "original_definition_MAD_recalculated": original_mad,
            "MAD_excluding_completely_unexposed_categories":
                sensitivity_mad,
        })

        for group_id, cells in subset.groupby(
            "popularity_group"
        ):
            group_rows.append({
                "method": method,
                "popularity_group": int(group_id),
                "eligible_category_cells": len(cells),
                "zero_exposure_categories": int(
                    (cells.exposure <= EPS).sum()
                ),
                "zero_exposure_rate": float(
                    np.mean(cells.exposure <= EPS)
                ),
                "total_exposure": float(cells.exposure.sum()),
                "total_opportunity": float(
                    cells.opportunity_mass.sum()
                ),
            })

    method_df = pd.DataFrame(method_rows)
    group_df = pd.DataFrame(group_rows)

    # Verify opportunity invariance across all methods.
    opportunity_pivot = df.pivot(
        index=["category", "popularity_group"],
        columns="method",
        values="opportunity_mass",
    )

    opportunity_invariant = bool(
        opportunity_pivot.notna().all().all()
        and np.allclose(
            opportunity_pivot.to_numpy(),
            opportunity_pivot.iloc[:, 0].to_numpy()[:, None],
            rtol=1e-9,
            atol=1e-12,
        )
    )

    # Verify recalculated original MAD against Experiment 22.10.
    original_lookup = dict(
        zip(original.method, original.category_group_log_rate_MAD)
    )

    mad_checks = {}

    for row in method_rows:
        method = row["method"]

        expected = float(original_lookup[method])
        observed = row["original_definition_MAD_recalculated"]

        mad_checks[method] = {
            "original_MAD": expected,
            "recalculated_MAD": observed,
            "matches": bool(
                np.isclose(expected, observed, rtol=1e-8, atol=1e-8)
            ),
        }

    checks = {
        "all_five_methods_present": len(method_df) == 5,
        "unique_category_group_cells": True,
        "opportunity_mass_identical_across_methods":
            opportunity_invariant,
        "original_MAD_recalculation_matches":
            all(x["matches"] for x in mad_checks.values()),
        "test_data_used": False,
        "training_performed": False,
        "retrieval_rerun": False,
        "new_coefficients": False,
    }

    checks["overall_pass"] = all(checks.values())

    category_df.to_csv(
        args.out / "category_exclusion_detail.csv",
        index=False,
    )

    method_df.to_csv(
        args.out / "method_exclusion_summary.csv",
        index=False,
    )

    group_df.to_csv(
        args.out / "popularity_group_exclusion_summary.csv",
        index=False,
    )

    with open(
        args.out / "validation_checks.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "checks": checks,
                "mad_reconciliation": mad_checks,
            },
            f,
            indent=2,
        )

    manifest = {
        "experiment": "22.12",
        "name": "Category Exposure and Exclusion Audit",
        "validation_only": True,
        "test_split_evaluated": False,
        "training_performed": False,
        "retrieval_rerun": False,
        "methods": METHODS,
        "input": str(args.input),
        "outputs": [
            "category_exclusion_detail.csv",
            "method_exclusion_summary.csv",
            "popularity_group_exclusion_summary.csv",
            "validation_checks.json",
        ],
        "interpretation": [
            "Original MAD is retained.",
            "Completely unexposed categories are separately identified.",
            "Sensitivity MAD excludes completely unexposed categories.",
            "Sensitivity MAD is not a replacement for original MAD.",
            "Exposure totals can overlap across multi-category items.",
        ],
    }

    with open(
        args.out / "experiment_manifest.json",
        "w",
        encoding="utf-8",
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
