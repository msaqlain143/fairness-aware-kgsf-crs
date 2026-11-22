"""
Experiment 22.33
Proposal Fairness Loss Completion

Purpose
-------
Complete the proposal-defined fairness objective using the already validated
22.31 q[e,i] and 22.32 R_cg / rho_cg reconstruction outputs.

Important
---------
- Read-only.
- No training.
- No retrieval.
- No ranking changes.
- No target-assisted relevance calculation.
- Does not invent a replacement fairness objective.
- Uses R_cg_contribution from 22.32 and aggregates it to R_cg.
- Uses the exact rho_cg values produced by 22.32 when available.
- omega_c remains explicitly unresolved if the proposal does not specify
  an operational weighting rule.

The purpose is to determine whether the proposal fairness loss can now be
computed mathematically, and if not, identify the single remaining input.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# PATHS
# ============================================================================

BASE = Path("results_final_test_m2")

Q_PATH = (
    BASE
    / "experiment_22_31"
    / "event_candidate_q_ei.csv"
)

RHO_PATH = (
    BASE
    / "experiment_22_32"
    / "R_cg_rho_cg_reconstruction.csv"
)

R_EVENT_PATH = (
    BASE
    / "experiment_22_32"
    / "R_cg_event_contributions.csv"
)

CATEGORY_PATH = (
    Path("results_category_audit")
    / "experiment_21_22A"
    / "catalog_item_category_profiles.csv"
)

OUT = BASE / "experiment_22_33"


# ============================================================================
# HELPERS
# ============================================================================

def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Required file not found: {path}")


def detect_column(
    df: pd.DataFrame,
    candidates,
    required=True,
):
    """
    Detect a column using exact and case-insensitive matching.
    """
    lower = {str(c).lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

        key = str(candidate).lower()

        if key in lower:
            return lower[key]

    if required:
        raise RuntimeError(
            "Required column not found.\n"
            f"Candidates: {candidates}\n"
            f"Available: {list(df.columns)}"
        )

    return None


def numeric_series(df, column, default=np.nan):
    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(default)


def load_q():
    require_file(Q_PATH)

    q = pd.read_csv(Q_PATH)

    event_col = detect_column(
        q,
        ["event_id", "event", "eventid"]
    )

    movie_col = detect_column(
        q,
        ["movie_id", "item_id", "movie"]
    )

    q_col = detect_column(
        q,
        ["q_ei", "q", "relevance", "q_e_i"]
    )

    out = q[
        [event_col, movie_col, q_col]
    ].copy()

    out.columns = [
        "event_id",
        "movie_id",
        "q_ei",
    ]

    out["event_id"] = numeric_series(
        out, "event_id"
    ).astype("int64")

    out["movie_id"] = numeric_series(
        out, "movie_id"
    ).astype("int64")

    out["q_ei"] = numeric_series(
        out, "q_ei"
    )

    return out


def load_event_contributions():
    """
    Load the actual 22.32 event-level R_cg contribution file.

    22.32 intentionally stores:

        R_cg_contribution

    rather than a pre-aggregated R_cg column.

    Therefore:

        R_cg = sum_e R_cg_contribution

    within each (category, popularity_group) cell.
    """

    require_file(R_EVENT_PATH)

    df = pd.read_csv(R_EVENT_PATH)

    print("\n22.32 EVENT CONTRIBUTION SCHEMA")
    print(list(df.columns))

    event_col = detect_column(
        df,
        ["event_id"]
    )

    movie_col = detect_column(
        df,
        ["movie_id", "item_id"]
    )

    category_col = detect_column(
        df,
        ["category", "category_id", "cat"]
    )

    group_col = detect_column(
        df,
        [
            "popularity_group",
            "pop_group",
            "group",
            "g",
        ]
    )

    contribution_col = detect_column(
        df,
        [
            "R_cg_contribution",
            "R_c_g_contribution",
            "relevance_mass_contribution",
            "relevance_contribution",
        ]
    )

    out = df[
        [
            event_col,
            movie_col,
            category_col,
            group_col,
            contribution_col,
        ]
    ].copy()

    out.columns = [
        "event_id",
        "movie_id",
        "category",
        "popularity_group",
        "R_cg_contribution",
    ]

    out["event_id"] = numeric_series(
        out, "event_id"
    ).astype("int64")

    out["movie_id"] = numeric_series(
        out, "movie_id"
    ).astype("int64")

    out["R_cg_contribution"] = numeric_series(
        out,
        "R_cg_contribution",
        default=0.0,
    )

    out["category"] = (
        out["category"]
        .astype(str)
        .str.strip()
    )

    out["popularity_group"] = (
        out["popularity_group"]
        .astype(str)
        .str.strip()
    )

    return out


def load_rho():
    """
    Load the exact 22.32 rho reconstruction.

    This file is used as an independent validation of the reconstructed
    category/group relevance quantities.
    """

    require_file(RHO_PATH)

    df = pd.read_csv(RHO_PATH)

    print("\n22.32 RHO SCHEMA")
    print(list(df.columns))

    category_col = detect_column(
        df,
        ["category", "category_id", "cat"]
    )

    group_col = detect_column(
        df,
        [
            "popularity_group",
            "pop_group",
            "group",
            "g",
        ]
    )

    rho_col = detect_column(
        df,
        [
            "rho_cg",
            "rho_c_g",
            "rho",
        ],
        required=False,
    )

    R_col = detect_column(
        df,
        [
            "R_cg",
            "R_c_g",
            "relevance_mass",
        ],
        required=False,
    )

    out = df.copy()

    out["category"] = (
        out[category_col]
        .astype(str)
        .str.strip()
    )

    out["popularity_group"] = (
        out[group_col]
        .astype(str)
        .str.strip()
    )

    if rho_col is not None:
        out["rho_cg"] = numeric_series(
            out,
            rho_col,
        )
    else:
        out["rho_cg"] = np.nan

    if R_col is not None:
        out["R_cg_2232"] = numeric_series(
            out,
            R_col,
        )
    else:
        out["R_cg_2232"] = np.nan

    return out[
        [
            "category",
            "popularity_group",
            "rho_cg",
            "R_cg_2232",
        ]
    ]


def load_categories():
    require_file(CATEGORY_PATH)

    df = pd.read_csv(CATEGORY_PATH)

    print("\nCATEGORY SCHEMA")
    print(list(df.columns))

    return df


# ============================================================================
# R_cg RECONSTRUCTION
# ============================================================================

def reconstruct_R_cg(event_contrib):
    """
    Proposal quantity:

        R_cg = sum over recommendation events of
               category/group relevance contribution.

    The 22.32 event-level file already contains the exact contribution
    produced from q[e,i].

    No target is introduced here.
    """

    grouped = (
        event_contrib
        .groupby(
            [
                "category",
                "popularity_group",
            ],
            as_index=False,
        )[
            "R_cg_contribution"
        ]
        .sum()
        .rename(
            columns={
                "R_cg_contribution": "R_cg"
            }
        )
    )

    return grouped


# ============================================================================
# SUPPORT
# ============================================================================

def calculate_support(event_contrib):
    support = (
        event_contrib
        .groupby(
            [
                "category",
                "popularity_group",
            ],
            as_index=False,
        )
        .agg(
            events=("event_id", "nunique"),
            candidate_instances=("movie_id", "size"),
            positive_contributions=(
                "R_cg_contribution",
                lambda x: int((x > 0).sum()),
            ),
        )
    )

    return support


# ============================================================================
# OMEGA
# ============================================================================

def determine_omega(r_cg):
    """
    The November proposal does not provide an operational numeric rule for
    omega_c in the current implementation.

    Therefore we DO NOT silently invent one.

    The output explicitly records that omega remains unresolved.

    For diagnostic purposes only, an equal category weighting is calculated
    separately. It is NOT declared the final proposal objective.
    """

    categories = sorted(
        r_cg["category"].unique()
    )

    omega = pd.DataFrame(
        {
            "category": categories,
            "omega_c": np.nan,
        }
    )

    return omega


# ============================================================================
# DIAGNOSTIC WEIGHTED VARIANCE
# ============================================================================

def calculate_equal_weight_diagnostic(
    rho_df,
):
    """
    Diagnostic only.

    If rho_cg is available, calculate an equal-category-weight variance
    to demonstrate that the final loss is mathematically computable once
    omega_c is specified.

    This is NOT reported as the final proposal L_fair.
    """

    valid = rho_df[
        np.isfinite(rho_df["rho_cg"])
    ].copy()

    if valid.empty:
        return None, "rho_cg unavailable"

    # category-level equal weighting
    category_means = (
        valid
        .groupby("category")["rho_cg"]
        .mean()
    )

    if len(category_means) == 0:
        return None, "no category-level rho values"

    global_mean = category_means.mean()

    category_variance = (
        (category_means - global_mean) ** 2
    ).mean()

    result = {
        "category_count": int(
            len(category_means)
        ),
        "global_mean_rho": float(
            global_mean
        ),
        "equal_category_weight_variance": float(
            category_variance
        ),
        "status": (
            "DIAGNOSTIC_ONLY_NOT_FINAL_PROPOSAL_LOSS"
        ),
    }

    return result, "available"


# ============================================================================
# MAIN
# ============================================================================

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("EXPERIMENT 22.33")
    print("PROPOSAL FAIRNESS LOSS COMPLETION")
    print("=" * 72)

    print("\nINPUTS")
    print(f"q: {Q_PATH}")
    print(f"R contributions: {R_EVENT_PATH}")
    print(f"rho reconstruction: {RHO_PATH}")
    print(f"categories: {CATEGORY_PATH}")

    # ------------------------------------------------------------------
    # 1. q
    # ------------------------------------------------------------------

    q = load_q()

    print("\nq[e,i]")
    print("Events:", q["event_id"].nunique())
    print("Candidate instances:", len(q))
    print(
        "q range:",
        f"{q.q_ei.min():.10f}",
        "to",
        f"{q.q_ei.max():.10f}",
    )

    # ------------------------------------------------------------------
    # 2. R_cg contributions
    # ------------------------------------------------------------------

    contributions = load_event_contributions()

    print(
        "\nEvent contribution rows:",
        len(contributions)
    )

    print(
        "Contribution events:",
        contributions["event_id"].nunique()
    )

    # Event alignment
    q_events = set(q.event_id)
    r_events = set(contributions.event_id)

    if q_events != r_events:
        raise RuntimeError(
            "22.31 q events and 22.32 contribution events do not align."
        )

    print("q / R event alignment: PASS")

    # ------------------------------------------------------------------
    # 3. R_cg
    # ------------------------------------------------------------------

    R_cg = reconstruct_R_cg(
        contributions
    )

    print(
        "\nPositive R_cg cells:",
        int(
            (R_cg.R_cg > 0).sum()
        )
    )

    # ------------------------------------------------------------------
    # 4. Support
    # ------------------------------------------------------------------

    support = calculate_support(
        contributions
    )

    R_cg = R_cg.merge(
        support,
        on=[
            "category",
            "popularity_group",
        ],
        how="left",
    )

    # ------------------------------------------------------------------
    # 5. rho
    # ------------------------------------------------------------------

    rho = load_rho()

    print(
        "\nReconstructed rho cells:",
        int(
            np.isfinite(
                rho.rho_cg
            ).sum()
        )
    )

    # ------------------------------------------------------------------
    # 6. Compare R_cg with 22.32 if available
    # ------------------------------------------------------------------

    comparison = R_cg.merge(
        rho,
        on=[
            "category",
            "popularity_group",
        ],
        how="left",
    )

    comparison[
        "R_cg_difference_vs_2232"
    ] = (
        comparison["R_cg"]
        - comparison["R_cg_2232"]
    )

    comparison[
        "R_cg_matches_2232"
    ] = np.isclose(
        comparison["R_cg"],
        comparison["R_cg_2232"],
        equal_nan=True,
        atol=1e-10,
        rtol=1e-8,
    )

    # ------------------------------------------------------------------
    # 7. omega
    # ------------------------------------------------------------------

    omega = determine_omega(
        R_cg
    )

    # ------------------------------------------------------------------
    # 8. Diagnostic variance
    # ------------------------------------------------------------------

    diagnostic_variance, diagnostic_status = (
        calculate_equal_weight_diagnostic(
            rho
        )
    )

    # ------------------------------------------------------------------
    # 9. Final scientific decision
    # ------------------------------------------------------------------

    rho_available = (
        np.isfinite(
            rho["rho_cg"]
        ).sum()
        > 0
    )

    R_available = (
        len(R_cg) > 0
        and (
            R_cg["R_cg"] > 0
        ).any()
    )

    omega_available = (
        omega["omega_c"]
        .notna()
        .all()
    )

    if R_available and rho_available:
        R_status = "RECONSTRUCTED"
        rho_status = "RECONSTRUCTED"
    else:
        R_status = "NOT_RECONSTRUCTABLE"
        rho_status = "NOT_RECONSTRUCTABLE"

    if omega_available:
        omega_status = "SPECIFIED"
    else:
        omega_status = "NOT_OPERATIONALLY_SPECIFIED"

    if (
        R_available
        and rho_available
        and omega_available
    ):
        L_status = "CALCULABLE"
        decision = "PROCEED_TO_LOSS_IMPLEMENTATION"
    else:
        L_status = "NOT_FINAL_IDENTIFIABLE"

        if (
            R_available
            and rho_available
            and not omega_available
        ):
            decision = (
                "ONLY_MISSING_PROPOSAL_INPUT_IS_OMEGA_C"
            )
        else:
            decision = (
                "RELEVANCE_OR_RHO_INPUT_STILL_MISSING"
            )

    # ------------------------------------------------------------------
    # 10. Save R_cg
    # ------------------------------------------------------------------

    R_cg.to_csv(
        OUT / "R_cg_reconstructed.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 11. Save rho / comparison
    # ------------------------------------------------------------------

    comparison.to_csv(
        OUT / "R_cg_rho_alignment.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 12. Save omega
    # ------------------------------------------------------------------

    omega.to_csv(
        OUT / "omega_c_status.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 13. Save support
    # ------------------------------------------------------------------

    support.to_csv(
        OUT / "R_cg_support.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 14. Save diagnostic variance
    # ------------------------------------------------------------------

    if diagnostic_variance is not None:
        with open(
            OUT / "diagnostic_equal_weight_variance.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                diagnostic_variance,
                f,
                indent=2,
            )

    # ------------------------------------------------------------------
    # 15. Final decision
    # ------------------------------------------------------------------

    decision_payload = {
        "experiment": "22.33",
        "read_only": True,
        "training": False,
        "retrieval": False,
        "ranking_changes": False,
        "target_used": False,

        "q_status":
            "OPERATIONAL_LEAKAGE_FREE_SIGNAL",

        "R_cg_status": R_status,
        "rho_cg_status": rho_status,
        "omega_c_status": omega_status,
        "L_fair_status": L_status,

        "positive_R_cg_cells": int(
            (R_cg["R_cg"] > 0).sum()
        ),

        "rho_cells": int(
            np.isfinite(
                rho["rho_cg"]
            ).sum()
        ),

        "omega_category_count": int(
            len(omega)
        ),

        "decision": decision,

        "diagnostic_equal_weight_variance":
            diagnostic_variance,
    }

    with open(
        OUT / "decision.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            decision_payload,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 16. Manifest
    # ------------------------------------------------------------------

    manifest = {
        "experiment":
            "22.33_proposal_fairness_loss_completion",

        "inputs": {
            "q":
                str(Q_PATH),
            "R_event_contributions":
                str(R_EVENT_PATH),
            "R_rho":
                str(RHO_PATH),
            "category_profiles":
                str(CATEGORY_PATH),
        },

        "method": {
            "R_cg":
                "sum of 22.32 R_cg_contribution within category/group",
            "rho_cg":
                "reused from exact 22.32 reconstruction",
            "omega_c":
                "not invented; explicitly unresolved",
        },

        "scientific_status": decision_payload,
    }

    with open(
        OUT / "experiment_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            manifest,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # FINAL PRINT
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("22.33 COMPLETE")
    print("=" * 72)

    print(
        "R_cg:",
        R_status
    )

    print(
        "rho_cg:",
        rho_status
    )

    print(
        "omega_c:",
        omega_status
    )

    print(
        "L_fair:",
        L_status
    )

    print(
        "Positive R_cg cells:",
        int(
            (R_cg["R_cg"] > 0).sum()
        )
    )

    if diagnostic_variance is not None:
        print(
            "Equal-weight diagnostic variance:",
            f"{diagnostic_variance['equal_category_weight_variance']:.12g}",
        )

    print(
        "Decision:",
        decision
    )

    print(
        "\nRead-only: YES"
        " | Training: NO"
        " | Retrieval: NO"
        " | Ranking changes: NO"
    )

    print(
        "\nOutput:",
        OUT.resolve()
    )


if __name__ == "__main__":
    main()