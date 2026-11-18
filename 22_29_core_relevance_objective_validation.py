#!/usr/bin/env python
"""
Experiment 22.29
Core Relevance / Fairness-Objective Decision Gate

Purpose
-------
Determine whether the frozen experimental evidence contains a defensible,
leakage-free contextual relevance q[e,i] sufficient to construct:

    q[e,i]
        ->
    R[c,g]
        ->
    rho[c,g]
        ->
    L_fair

This is a READ-ONLY decision gate.

No:
    - training
    - retrieval
    - ranking changes
    - parameter tuning
    - target-assisted relevance
    - independent-test rerun

Proposal basis:
    Eq. (7):  q[e,i] = Rel(H_e, i, K)
    Eq. (21): R[c,g]
    Eq. (23): rho[c,g]
    Eq. (24)-(25): L_fair

Important:
-----------
Existing 22.18 / 22.28 signals such as:

    - candidate-pool category prevalence
    - KG rank
    - exp(-KG_rank / tau)

are NOT automatically accepted as q[e,i].
They are only evidence/proxies unless a validated contextual relevance
definition is available.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Frozen project inputs
# ---------------------------------------------------------------------

ROOT = Path(".")

KGSF_TOP500 = (
    ROOT
    / "results_final_test_m2"
    / "experiment_22_27A"
    / "kgsf_independent_test_top500.csv"
)

KG_RANKS = (
    ROOT
    / "results_final_test_m2"
    / "experiment_22_27A2"
    / "kgsf_independent_test_kg_ranks.csv"
)

EVENT_MAPPING = (
    ROOT
    / "results_final_test_m2"
    / "m2_test_event_conversation_mapping.csv"
)

RESULT_22_28 = (
    ROOT
    / "results_final_test_m2"
    / "experiment_22_28"
)

OUT = (
    ROOT
    / "results_final_test_m2"
    / "experiment_22_29"
)


EXPECTED_EVENTS = 4003
EXPECTED_CONVERSATIONS = 1283
EXPECTED_CANDIDATES = 500


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def parse_list(value):
    """Parse JSON/Python-list representation."""
    if isinstance(value, list):
        return value

    if isinstance(value, (tuple, np.ndarray)):
        return list(value)

    text = str(value).strip()

    try:
        obj = json.loads(text)
    except Exception:
        obj = ast.literal_eval(text)

    if not isinstance(obj, (list, tuple)):
        raise ValueError(f"Expected list, got: {type(obj).__name__}")

    return list(obj)


def normalize_columns(df):
    df = df.copy()
    df.columns = [
        str(c).strip().lower().replace(" ", "_")
        for c in df.columns
    ]
    return df


def find_relevance_columns(columns):
    """
    Search for an actual contextual relevance/q field.

    Deliberately conservative:
    category prevalence and KG confidence are NOT accepted as q.
    """

    cols = {str(c).lower(): c for c in columns}

    accepted = [
        "q",
        "q_e_i",
        "q_ei",
        "contextual_relevance",
        "context_relevance",
        "relevance",
        "relevance_score",
        "contextual_relevance_score",
    ]

    found = []

    for name in accepted:
        if name in cols:
            found.append(cols[name])

    return found


def forbidden_relevance_columns(columns):
    """
    Signals that must not be silently promoted to q[e,i].
    """

    names = {
        "kg_rank",
        "kg_confidence",
        "kg_confidence_proxy",
        "context_relevance_proxy",
        "legacy_context_kg_relevance_proxy",
        "category_prevalence",
        "candidate_category_prevalence",
        "target_movie_id",
        "target",
        "label",
    }

    return [
        c for c in columns
        if str(c).lower() in names
    ]


def target_like_columns(columns):
    names = {
        "target_movie_id",
        "target",
        "label",
        "ground_truth",
        "ground_truth_movie",
        "target_item",
    }

    return [
        c for c in columns
        if str(c).lower() in names
    ]


# ---------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------

def validate_frozen_inputs():

    required = [
        KGSF_TOP500,
        KG_RANKS,
        EVENT_MAPPING,
    ]

    missing = [str(p) for p in required if not p.is_file()]

    if missing:
        raise FileNotFoundError(
            "Missing required frozen inputs:\n"
            + "\n".join(missing)
        )


def load_frozen_inputs():

    mapping = normalize_columns(
        pd.read_csv(EVENT_MAPPING)
    )

    kgsf = normalize_columns(
        pd.read_csv(KGSF_TOP500)
    )

    kg = normalize_columns(
        pd.read_csv(KG_RANKS)
    )

    required_mapping = {
        "event_id",
        "conv_id",
        "target_movie_id",
    }

    required_kgsf = {
        "event_id",
    }

    required_kg = {
        "event_id",
    }

    if not required_mapping.issubset(mapping.columns):
        raise RuntimeError(
            "Event mapping missing columns: "
            + str(sorted(
                required_mapping - set(mapping.columns)
            ))
        )

    if not required_kgsf.issubset(kgsf.columns):
        raise RuntimeError(
            "KGSF export missing event_id."
        )

    if not required_kg.issubset(kg.columns):
        raise RuntimeError(
            "KG export missing event_id."
        )

    return mapping, kgsf, kg


# ---------------------------------------------------------------------
# Search existing relevance evidence
# ---------------------------------------------------------------------

def inspect_22_28():

    result = {
        "directory_exists": RESULT_22_28.is_dir(),
        "candidate_relevance_file": None,
        "proposal_q_available": False,
        "q_columns": [],
        "forbidden_proxy_columns": [],
    }

    if not RESULT_22_28.is_dir():
        return result

    for path in sorted(
        RESULT_22_28.glob("*.csv")
    ):

        try:
            header = pd.read_csv(
                path,
                nrows=0
            )

            header = normalize_columns(header)

        except Exception:
            continue

        q_cols = find_relevance_columns(
            header.columns
        )

        forbidden = forbidden_relevance_columns(
            header.columns
        )

        if q_cols:

            result["candidate_relevance_file"] = str(
                path
            )

            result["proposal_q_available"] = True

            result["q_columns"] = [
                str(c) for c in q_cols
            ]

            result["forbidden_proxy_columns"] = [
                str(c) for c in forbidden
            ]

            break

    return result


# ---------------------------------------------------------------------
# Search previous relevance experiments
# ---------------------------------------------------------------------

def search_previous_relevance_outputs():

    roots = [
        ROOT / "results_final_test_m2" / "experiment_22_18",
        ROOT / "results_final_test_m2" / "experiment_22_21",
        ROOT / "results_corrected",
    ]

    discovered = []

    for root in roots:

        if not root.exists():
            continue

        for path in root.rglob("*.csv"):

            try:
                header = pd.read_csv(
                    path,
                    nrows=0
                )

                header = normalize_columns(header)

            except Exception:
                continue

            q_cols = find_relevance_columns(
                header.columns
            )

            forbidden = forbidden_relevance_columns(
                header.columns
            )

            if q_cols or forbidden:

                discovered.append({
                    "file": str(path),
                    "q_columns": [
                        str(x) for x in q_cols
                    ],
                    "proxy_columns": [
                        str(x) for x in forbidden
                    ],
                })

    return discovered


# ---------------------------------------------------------------------
# Leakage assessment
# ---------------------------------------------------------------------

def assess_q_file(path, q_columns):

    df = pd.read_csv(path)

    df = normalize_columns(df)

    target_columns = target_like_columns(
        df.columns
    )

    result = {
        "file": str(path),
        "q_columns": [
            str(x) for x in q_columns
        ],
        "target_like_columns": [
            str(x) for x in target_columns
        ],
        "leakage_status": "PASS",
    }

    # A target column existing in the file does not automatically mean
    # leakage, but a q construction based on it cannot be accepted.
    if target_columns:

        result["leakage_status"] = (
            "REQUIRES_MANUAL_PROVENANCE_CHECK"
        )

    return result


# ---------------------------------------------------------------------
# Proposal R_cg calculation
# ---------------------------------------------------------------------

def calculate_R_cg(q_df):

    """
    Calculate proposal Eq. (21):

        R[c,g] =
            sum_e sum_i
            q[e,i] A[i,c]
            I[g(i)=g]
            I[i in I_e]

    This function is only called when validated q[e,i] exists.

    The actual item->category and item->group mappings must be present
    in q_df.
    """

    required = {
        "event_id",
        "movie_id",
        "q",
        "group",
        "category",
    }

    missing = required - set(q_df.columns)

    if missing:
        return None, {
            "status": "NOT_COMPUTABLE",
            "missing_fields": sorted(missing),
        }

    work = q_df.copy()

    work["q"] = pd.to_numeric(
        work["q"],
        errors="coerce"
    )

    work["group"] = pd.to_numeric(
        work["group"],
        errors="coerce"
    )

    work = work.dropna(
        subset=[
            "event_id",
            "movie_id",
            "q",
            "group",
            "category",
        ]
    )

    if work.empty:

        return None, {
            "status": "NOT_COMPUTABLE",
            "reason": "No valid q[e,i] rows.",
        }

    R = (
        work
        .groupby(
            ["category", "group"],
            as_index=False
        )["q"]
        .sum()
        .rename(
            columns={"q": "R_cg"}
        )
    )

    return R, {
        "status": "COMPUTABLE",
        "positive_cells": int(
            (R["R_cg"] > 0).sum()
        ),
    }


# ---------------------------------------------------------------------
# L_fair calculation
# ---------------------------------------------------------------------

def calculate_L_fair(
    R_df,
    exposure_df,
    omega,
    min_support=0.0,
):

    """
    Proposal Eq. (23)-(25).

    rho[c,g] = E[c,g] / (R[c,g] + eps)

    p[c,g] = R[c,g] / sum_h R[c,h]

    rho_bar[c] = sum_g p[c,g] rho[c,g]

    L_fair =
        sum_c omega[c]
        sum_g p[c,g] (rho[c,g] - rho_bar[c])^2

    omega and min_support are explicitly supplied because the proposal
    does not specify unique numerical values for them.
    """

    if R_df is None or R_df.empty:
        return None

    rows = []

    for category, group_df in R_df.groupby(
        "category"
    ):

        group_df = group_df.copy()

        group_df = group_df[
            group_df["R_cg"] > min_support
        ]

        if len(group_df) < 2:
            continue

        total_R = group_df["R_cg"].sum()

        if total_R <= 0:
            continue

        group_df["p_cg"] = (
            group_df["R_cg"] / total_R
        )

        group_df["E_cg"] = 0.0

        for idx, row in group_df.iterrows():

            mask = (
                (exposure_df["category"] == category)
                & (
                    exposure_df["group"]
                    == row["group"]
                )
            )

            if mask.any():

                group_df.loc[
                    idx,
                    "E_cg"
                ] = float(
                    exposure_df.loc[
                        mask,
                        "E_cg"
                    ].iloc[0]
                )

        group_df["rho_cg"] = (
            group_df["E_cg"]
            / (group_df["R_cg"] + 1e-12)
        )

        rho_bar = float(
            (
                group_df["p_cg"]
                * group_df["rho_cg"]
            ).sum()
        )

        variance = float(
            (
                group_df["p_cg"]
                * (
                    group_df["rho_cg"]
                    - rho_bar
                ) ** 2
            ).sum()
        )

        rows.append({
            "category": category,
            "weighted_variance": variance,
            "active_groups": len(group_df),
            "relevance_mass": float(
                total_R
            ),
        })

    if not rows:
        return None

    result = pd.DataFrame(rows)

    if omega is None:
        return {
            "status": "NOT_FULLY_IDENTIFIABLE",
            "reason": (
                "R_cg exists but omega_c is not "
                "numerically specified."
            ),
            "category_variances": result,
        }

    result["omega"] = result[
        "category"
    ].map(omega)

    valid = result[
        result["omega"].notna()
    ].copy()

    if valid.empty:
        return {
            "status": "NOT_FULLY_IDENTIFIABLE",
            "reason": "No valid omega_c values.",
            "category_variances": result,
        }

    L = float(
        (
            valid["omega"]
            * valid["weighted_variance"]
        ).sum()
        / valid["omega"].sum()
    )

    return {
        "status": "COMPUTABLE_WITH_EXPLICIT_ASSUMPTIONS",
        "L_fair": L,
        "category_variances": valid,
    }


# ---------------------------------------------------------------------
# Main decision gate
# ---------------------------------------------------------------------

def main():

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 72)
    print("EXPERIMENT 22.29")
    print("CORE RELEVANCE / FAIRNESS-OBJECTIVE DECISION GATE")
    print("=" * 72)

    validate_frozen_inputs()

    mapping, kgsf, kg = (
        load_frozen_inputs()
    )

    # -------------------------------------------------------------
    # Frozen-data integrity
    # -------------------------------------------------------------

    event_count = len(mapping)

    conversation_count = int(
        mapping["conv_id"].nunique()
    )

    unique_events = int(
        mapping["event_id"].nunique()
    )

    integrity = {
        "events": event_count,
        "expected_events": EXPECTED_EVENTS,
        "unique_events": unique_events,
        "conversations": conversation_count,
        "expected_conversations":
            EXPECTED_CONVERSATIONS,
        "event_count_pass":
            event_count == EXPECTED_EVENTS,
        "unique_event_pass":
            unique_events == EXPECTED_EVENTS,
        "conversation_count_pass":
            conversation_count
            == EXPECTED_CONVERSATIONS,
        "candidate_pool_expected":
            EXPECTED_CANDIDATES,
    }

    # -------------------------------------------------------------
    # Existing 22.28 evidence
    # -------------------------------------------------------------

    evidence_22_28 = inspect_22_28()

    previous_outputs = (
        search_previous_relevance_outputs()
    )

    # -------------------------------------------------------------
    # Decision on q[e,i]
    # -------------------------------------------------------------

    q_available = bool(
        evidence_22_28[
            "proposal_q_available"
        ]
    )

    q_source = (
        evidence_22_28[
            "candidate_relevance_file"
        ]
        if q_available
        else None
    )

    leakage = {
        "status": "NOT_ASSESSED",
    }

    if q_available:

        leakage = assess_q_file(
            q_source,
            evidence_22_28["q_columns"],
        )

    # -------------------------------------------------------------
    # Critical scientific decision
    # -------------------------------------------------------------

    if not q_available:

        q_status = (
            "UNAVAILABLE_AS_VALIDATED_CONTEXTUAL_RELEVANCE"
        )

        R_status = "NOT_IDENTIFIABLE"

        L_status = "NOT_IDENTIFIABLE"

        minimum_missing = [
            "A validated contextual relevance q[e,i].",
            (
                "Its provenance must use only information "
                "available at recommendation event e."
            ),
            (
                "A validated relevance eligibility rule "
                "for constructing I_e."
            ),
        ]

    else:

        if leakage["status"] == "PASS":

            q_status = (
                "AVAILABLE_BUT_REQUIRES_SCHEMA_MAPPING"
            )

        else:

            q_status = (
                "AVAILABLE_BUT_REQUIRES_LEAKAGE_REVIEW"
            )

        R_status = (
            "REQUIRES_EXPLICIT_ITEM_CATEGORY_GROUP_MAPPING"
        )

        L_status = (
            "REQUIRES_R_cg_PLUS_omega_c_PLUS_SUPPORT_POLICY"
        )

        minimum_missing = [
            "Validated q[e,i] schema mapping.",
            "Item-to-category membership A[i,c].",
            "Training-only popularity group g(i).",
            "Predeclared relevance-support rule.",
            "Numerical category weighting omega_c.",
        ]

    # -------------------------------------------------------------
    # Save outputs
    # -------------------------------------------------------------

    relevance_inventory = pd.DataFrame([
        {
            "source":
                "22.28",
            "proposal_q_available":
                q_available,
            "q_source":
                q_source,
            "q_columns":
                json.dumps(
                    evidence_22_28[
                        "q_columns"
                    ]
                ),
            "interpretation":
                (
                    "Validated q[e,i]"
                    if q_available
                    else
                    "No validated contextual q[e,i]"
                ),
        }
    ])

    relevance_inventory.to_csv(
        OUT / "relevance_objective_validation.csv",
        index=False
    )

    leakage_validation = {
        "q_status": q_status,
        "leakage": leakage,
        "previous_relevance_outputs":
            previous_outputs,
    }

    (OUT / "leakage_validation.json").write_text(
        json.dumps(
            leakage_validation,
            indent=2,
        ),
        encoding="utf-8",
    )

    missing_inputs = {
        "q_status": q_status,
        "R_cg_status": R_status,
        "L_fair_status": L_status,
        "minimum_additional_inputs":
            minimum_missing,
        "existing_proxy_signals_rejected_as_q": [
            "candidate-pool category prevalence",
            "KG rank",
            "KG confidence",
            "exp(-KG_rank/tau)",
            "legacy context × KG relevance proxy",
        ],
    }

    (OUT / "missing_inputs.json").write_text(
        json.dumps(
            missing_inputs,
            indent=2,
        ),
        encoding="utf-8",
    )

    objective_feasibility = {
        "proposal_q": q_status,
        "R_cg": R_status,
        "L_fair": L_status,
        "leakage": leakage["status"],
        "read_only": True,
        "training": False,
        "retrieval": False,
        "ranking_changes": False,
        "target_assisted_relevance": False,
        "decision": (
            "PROCEED_TO_22_30"
            if (
                q_available
                and leakage["status"] == "PASS"
            )
            else
            "IMPLEMENT_MINIMUM_MISSING_RELEVANCE_COMPONENT"
        ),
    }

    (OUT / "objective_feasibility.json").write_text(
        json.dumps(
            objective_feasibility,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "experiment":
            "22.29_core_relevance_objective_validation",
        "read_only": True,
        "training": False,
        "retrieval": False,
        "ranking_changes": False,
        "target_assisted_relevance": False,
        "frozen_inputs": [
            str(KGSF_TOP500),
            str(KG_RANKS),
            str(EVENT_MAPPING),
            str(RESULT_22_28),
        ],
        "integrity": integrity,
        "decision":
            objective_feasibility["decision"],
    }

    (OUT / "experiment_manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    # -------------------------------------------------------------
    # Terminal decision
    # -------------------------------------------------------------

    print()
    print("FROZEN DATA")
    print(
        f"Events: {event_count}/{EXPECTED_EVENTS}"
    )
    print(
        f"Conversations: "
        f"{conversation_count}/{EXPECTED_CONVERSATIONS}"
    )

    print()
    print("CORE DECISION")
    print(
        f"q[e,i]: {q_status}"
    )
    print(
        f"R[c,g]: {R_status}"
    )
    print(
        f"L_fair: {L_status}"
    )
    print(
        f"Leakage: "
        f"{leakage['status']}"
    )

    print()
    print("MINIMUM MISSING INPUT")
    for item in minimum_missing:
        print(f"  - {item}")

    print()
    print(
        "DECISION: "
        + objective_feasibility["decision"]
    )

    print()
    print(
        "Output:",
        OUT.resolve()
    )


if __name__ == "__main__":
    main()