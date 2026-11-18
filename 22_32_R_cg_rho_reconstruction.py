#!/usr/bin/env python
"""
Experiment 22.32
Exact Proposal R_{c,g} and rho_{c,g} Reconstruction

Purpose
-------
Reconstruct the proposal-defined:

    R_{c,g}
    rho_{c,g}

from the frozen Experiment 22.31 q[e,i] signal and the frozen
baseline Top-10 rankings.

The mathematical definitions are taken directly from Appendix B /
Section 4.4 of the November 17 KG-FairCRS proposal.

Proposal definitions
--------------------

R_{c,g}
    = sum_e sum_i q_{e,i} A_{i,c}
      I(g(i)=g) I(i in I_e)

E_{c,g}
    = sum_e sum_r w(r) A_{pi_e[r],c}
      I(g(pi_e[r])=g)

where

    w(r) = 1 / log2(r + 1)

rho_{c,g}
    = E_{c,g} / (R_{c,g} + epsilon_0)

This experiment does NOT calculate L_fair because omega_c is not
numerically specified in the proposal.

Scientific constraints
----------------------
- Read-only
- No training
- No retrieval
- No ranking changes
- No target use in q construction
- No target use in R_{c,g}
- No target use in rho_{c,g}
- No parameter optimization
- Uses frozen 22.31 q[e,i]
- Uses frozen baseline Top-10 rankings
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ================================================================
# Constants
# ================================================================

EXPECTED_EVENTS = 4003
EXPECTED_CONVERSATIONS = 1283
CATALOG_SIZE = 6924
TOP_K = 10

EPSILON_0 = 1e-12


# ================================================================
# Frozen inputs
# ================================================================

DEFAULT_Q = Path(
    "results_final_test_m2"
    "/experiment_22_31"
    "/event_candidate_q_ei.csv"
)

DEFAULT_BASELINE = Path(
    "results_final_test_m2"
    "/experiment_22_24B"
    "/baseline_event_results.csv"
)

DEFAULT_CATALOG = Path(
    "catalog_audit_corrected.pkl"
)

DEFAULT_PROFILES = Path(
    "results_category_audit"
    "/experiment_21_22A"
    "/catalog_item_category_profiles.csv"
)

DEFAULT_OUT = Path(
    "results_final_test_m2"
    "/experiment_22_32"
)


# ================================================================
# Utility
# ================================================================

def parse_bool(value):
    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def parse_list(value):
    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if pd.isna(value):
        return []

    text = str(value).strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    try:
        return ast.literal_eval(text)
    except Exception as exc:
        raise ValueError(
            f"Unable to parse list: {text[:200]}"
        ) from exc


def parse_categories(value, delimiter=None):
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    if delimiter is None:
        delimiter = "|" if "|" in text else ";"

    return sorted({
        " ".join(
            token.strip().casefold().split()
        )
        for token in text.split(delimiter)
        if token.strip()
    })


# ================================================================
# Load frozen catalog
# ================================================================

def load_catalog(path):

    with path.open("rb") as f:
        metadata = pickle.load(f)

    movie_ids = [
        int(x)
        for x in metadata["movie_ids"]
    ]

    if len(movie_ids) != CATALOG_SIZE:
        raise RuntimeError(
            f"Expected {CATALOG_SIZE} catalog items; "
            f"got {len(movie_ids)}"
        )

    if len(set(movie_ids)) != CATALOG_SIZE:
        raise RuntimeError(
            "Catalog movie IDs are not unique."
        )

    item2group = {
        int(k): int(v)
        for k, v in metadata["item2group"].items()
    }

    if set(item2group) != set(movie_ids):
        raise RuntimeError(
            "item2group does not exactly match frozen catalog."
        )

    if any(
        group not in range(5)
        for group in item2group.values()
    ):
        raise RuntimeError(
            "Popularity groups must be 0..4."
        )

    return movie_ids, item2group


# ================================================================
# Load validated categories
# ================================================================

def load_categories(path, catalog):

    profiles = pd.read_csv(path)

    required = {
        "movie_entity_id",
        "canonical_genres",
        "imdb_genres",
        "imdb_genre_evidence",
        "imdb_conflicting_records",
        "accepted_genre_evidence",
        "identity_resolution_status",
    }

    missing = required - set(profiles.columns)

    if missing:
        raise RuntimeError(
            "Category profile missing columns: "
            f"{sorted(missing)}"
        )

    profiles["movie_entity_id"] = (
        profiles["movie_entity_id"].astype(int)
    )

    if profiles["movie_entity_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate movie IDs in category profiles."
        )

    catalog_set = set(catalog)

    categories = {
        movie_id: []
        for movie_id in catalog
    }

    for row in profiles.itertuples(index=False):

        movie_id = int(row.movie_entity_id)

        if movie_id not in catalog_set:
            continue

        identity_status = (
            str(row.identity_resolution_status)
            .strip()
            .upper()
        )

        if identity_status in {
            "IDENTITY_REVIEW_REQUIRED",
            "AMBIGUOUS_REQUIRES_REVIEW",
        }:
            continue

        accepted = []

        # --------------------------------------------------------
        # IMDb accepted evidence
        # --------------------------------------------------------

        if (
            parse_bool(row.imdb_genre_evidence)
            and not parse_bool(
                row.imdb_conflicting_records
            )
        ):

            raw = row.imdb_genres

            if not pd.isna(raw):

                accepted.extend(
                    parse_categories(raw)
                )

        # --------------------------------------------------------
        # Accepted DBpedia / canonical evidence
        # --------------------------------------------------------

        if parse_bool(
            row.accepted_genre_evidence
        ):

            accepted.extend(
                parse_categories(
                    row.canonical_genres,
                    ";",
                )
            )

        categories[movie_id] = sorted(
            set(accepted)
        )

    return categories, profiles


# ================================================================
# Load q[e,i]
# ================================================================

def load_q(path):

    q = pd.read_csv(path)

    required = {
        "event_id",
        "candidate_rank",
        "movie_id",
        "q_ei",
    }

    missing = required - set(q.columns)

    if missing:
        raise RuntimeError(
            "22.31 q export missing columns: "
            f"{sorted(missing)}"
        )

    q["event_id"] = q["event_id"].astype(int)
    q["movie_id"] = q["movie_id"].astype(int)
    q["candidate_rank"] = (
        q["candidate_rank"].astype(int)
    )
    q["q_ei"] = q["q_ei"].astype(float)

    if len(q) != EXPECTED_EVENTS * 500:
        raise RuntimeError(
            "Unexpected q row count: "
            f"{len(q)}; expected "
            f"{EXPECTED_EVENTS * 500}"
        )

    if q.event_id.nunique() != EXPECTED_EVENTS:
        raise RuntimeError(
            "Unexpected number of q events."
        )

    counts = (
        q.groupby("event_id")
        .size()
    )

    if not counts.eq(500).all():
        raise RuntimeError(
            "Every event must contain exactly "
            "500 q candidates."
        )

    if q.q_ei.isna().any():
        raise RuntimeError(
            "q[e,i] contains NaN."
        )

    if (q.q_ei < 0).any():
        raise RuntimeError(
            "q[e,i] contains negative values."
        )

    # Proposal R uses event-time candidate set I_e.
    # 22.31 q is normalized across the 500 frozen candidates.
    event_mass = (
        q.groupby("event_id")["q_ei"]
        .sum()
    )

    if not np.allclose(
        event_mass.to_numpy(),
        1.0,
        atol=1e-8,
    ):
        raise RuntimeError(
            "q[e,i] is not normalized to one per event."
        )

    return q


# ================================================================
# Load frozen baseline Top-10
# ================================================================

def load_baseline(path):

    df = pd.read_csv(path)

    required = {
        "event_id",
        "conv_id",
        "target_movie_id",
        "top10_movie_ids",
        "top10_group_ids",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "Baseline export missing columns: "
            f"{sorted(missing)}"
        )

    df["event_id"] = df["event_id"].astype(int)
    df["conv_id"] = df["conv_id"].astype(int)

    if len(df) != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} baseline events; "
            f"got {len(df)}"
        )

    if df.event_id.duplicated().any():
        raise RuntimeError(
            "Duplicate baseline event IDs."
        )

    df = (
        df.sort_values("event_id")
        .reset_index(drop=True)
    )

    return df


# ================================================================
# Validate event alignment
# ================================================================

def validate_alignment(q, baseline):

    q_events = set(
        q.event_id.astype(int)
    )

    baseline_events = set(
        baseline.event_id.astype(int)
    )

    if q_events != baseline_events:
        raise RuntimeError(
            "22.31 q events and baseline events "
            "do not match."
        )

    if len(q_events) != EXPECTED_EVENTS:
        raise RuntimeError(
            "Unexpected event universe."
        )

    if baseline.conv_id.nunique() != (
        EXPECTED_CONVERSATIONS
    ):
        raise RuntimeError(
            "Unexpected number of conversations."
        )


# ================================================================
# R_{c,g}
# ================================================================

def reconstruct_R_cg(
    q,
    categories,
    item2group,
):

    records = []

    R = defaultdict(float)

    # ------------------------------------------------------------
    # Important:
    #
    # A multi-label item contributes to every category according
    # to A_{i,c} in the proposal.
    #
    # We DO NOT divide by category count here because Appendix B
    # defines A_{i,c} as category membership.
    # ------------------------------------------------------------

    for row in q.itertuples(index=False):

        event_id = int(row.event_id)
        movie_id = int(row.movie_id)
        q_value = float(row.q_ei)

        movie_categories = categories.get(
            movie_id,
            [],
        )

        if not movie_categories:
            continue

        group = int(
            item2group[movie_id]
        )

        for category in movie_categories:

            contribution = q_value

            R[
                category,
                group,
            ] += contribution

            records.append(
                {
                    "event_id": event_id,
                    "movie_id": movie_id,
                    "category": category,
                    "popularity_group": group,
                    "q_ei": q_value,
                    "category_membership": 1,
                    "R_cg_contribution": contribution,
                }
            )

    event_level = pd.DataFrame(records)

    return R, event_level


# ================================================================
# E_{c,g}
# ================================================================

def reconstruct_E_cg(
    baseline,
    categories,
    item2group,
):

    E = defaultdict(float)

    records = []

    for row in baseline.itertuples(index=False):

        event_id = int(row.event_id)

        movie_ids = [
            int(x)
            for x in parse_list(
                row.top10_movie_ids
            )
        ]

        if len(movie_ids) != TOP_K:
            raise RuntimeError(
                f"Event {event_id}: "
                "Top-10 movie count is not 10."
            )

        for rank, movie_id in enumerate(
            movie_ids,
            start=1,
        ):

            movie_categories = categories.get(
                movie_id,
                [],
            )

            if not movie_categories:
                continue

            group = int(
                item2group[movie_id]
            )

            # Exact proposal Eq. (20)/(22)
            w = 1.0 / math.log2(
                rank + 1.0
            )

            for category in movie_categories:

                contribution = w

                E[
                    category,
                    group,
                ] += contribution

                records.append(
                    {
                        "event_id": event_id,
                        "rank": rank,
                        "movie_id": movie_id,
                        "category": category,
                        "popularity_group": group,
                        "position_weight": w,
                        "E_cg_contribution": contribution,
                    }
                )

    return E, pd.DataFrame(records)


# ================================================================
# rho_{c,g}
# ================================================================

def calculate_rho(
    R,
    E,
):

    cells = sorted(
        set(R.keys()) |
        set(E.keys())
    )

    rows = []

    by_category = defaultdict(list)

    for category, group in cells:

        r_value = float(
            R.get(
                (category, group),
                0.0,
            )
        )

        e_value = float(
            E.get(
                (category, group),
                0.0,
            )
        )

        rho = (
            e_value
            / (r_value + EPSILON_0)
        )

        by_category[category].append(
            (
                group,
                r_value,
                e_value,
                rho,
            )
        )

    # ------------------------------------------------------------
    # Proposal p_cg and rho_bar_c
    # ------------------------------------------------------------

    category_summary = []

    for category, group_rows in (
        by_category.items()
    ):

        total_R = sum(
            max(r, 0.0)
            for _, r, _, _ in group_rows
        )

        if total_R <= EPSILON_0:
            continue

        rho_bar = 0.0

        for (
            group,
            r_value,
            e_value,
            rho,
        ) in group_rows:

            p = (
                max(r_value, 0.0)
                / total_R
            )

            rho_bar += p * rho

            rows.append(
                {
                    "category": category,
                    "popularity_group": group,
                    "R_cg": r_value,
                    "E_cg": e_value,
                    "rho_cg": rho,
                    "p_cg": p,
                }
            )

        category_summary.append(
            {
                "category": category,
                "total_R_c": total_R,
                "rho_bar_c": rho_bar,
                "positive_groups": sum(
                    r > EPSILON_0
                    for _, r, _, _ in group_rows
                ),
            }
        )

    return (
        pd.DataFrame(rows),
        pd.DataFrame(category_summary),
    )


# ================================================================
# Support summary
# ================================================================

def build_support_summary(
    rho_df,
    event_mass,
):

    if rho_df.empty:

        return {
            "R_positive_cells": 0,
            "categories_with_R": 0,
            "groups_with_R": 0,
            "rho_cells": 0,
            "min_R_cg": None,
            "median_R_cg": None,
            "max_R_cg": None,
        }

    positive = rho_df[
        rho_df["R_cg"] > EPSILON_0
    ]

    return {
        "R_positive_cells": int(
            len(positive)
        ),
        "categories_with_R": int(
            positive["category"].nunique()
        ),
        "groups_with_R": int(
            positive["popularity_group"].nunique()
        ),
        "rho_cells": int(
            len(rho_df)
        ),
        "min_R_cg": float(
            positive["R_cg"].min()
        ) if not positive.empty else None,
        "median_R_cg": float(
            positive["R_cg"].median()
        ) if not positive.empty else None,
        "max_R_cg": float(
            positive["R_cg"].max()
        ) if not positive.empty else None,
    }


# ================================================================
# Main
# ================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--q",
        type=Path,
        default=DEFAULT_Q,
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
    )

    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG,
    )

    parser.add_argument(
        "--category-profiles",
        type=Path,
        default=DEFAULT_PROFILES,
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print(
        "EXPERIMENT 22.32"
    )
    print(
        "EXACT PROPOSAL R_cg / rho_cg RECONSTRUCTION"
    )
    print("=" * 72)

    # ------------------------------------------------------------
    # Load catalog
    # ------------------------------------------------------------

    print("\nLoading frozen catalog...")

    movie_ids, item2group = load_catalog(
        args.catalog
    )

    print(
        f"Catalog: {len(movie_ids)}"
    )

    # ------------------------------------------------------------
    # Load categories
    # ------------------------------------------------------------

    print(
        "\nLoading validated category mappings..."
    )

    categories, profiles = load_categories(
        args.category_profiles,
        movie_ids,
    )

    mapped_items = sum(
        bool(v)
        for v in categories.values()
    )

    print(
        f"Items with accepted categories: "
        f"{mapped_items}/{len(movie_ids)}"
    )

    # ------------------------------------------------------------
    # Load q
    # ------------------------------------------------------------

    print(
        "\nLoading Experiment 22.31 q[e,i]..."
    )

    q = load_q(args.q)

    print(
        f"q events: {q.event_id.nunique()}"
    )

    print(
        f"q candidate instances: {len(q)}"
    )

    # ------------------------------------------------------------
    # Load baseline
    # ------------------------------------------------------------

    print(
        "\nLoading frozen baseline Top-10..."
    )

    baseline = load_baseline(
        args.baseline
    )

    print(
        f"Baseline events: {len(baseline)}"
    )

    # ------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------

    print(
        "\nValidating event alignment..."
    )

    validate_alignment(
        q,
        baseline,
    )

    print(
        "Event alignment: PASS"
    )

    # ------------------------------------------------------------
    # R_cg
    # ------------------------------------------------------------

    print(
        "\nReconstructing R_cg..."
    )

    R, R_event = reconstruct_R_cg(
        q,
        categories,
        item2group,
    )

    print(
        f"Positive R_cg cells: "
        f"{sum(v > EPSILON_0 for v in R.values())}"
    )

    # ------------------------------------------------------------
    # E_cg
    # ------------------------------------------------------------

    print(
        "\nReconstructing E_cg..."
    )

    E, E_event = reconstruct_E_cg(
        baseline,
        categories,
        item2group,
    )

    print(
        f"Positive E_cg cells: "
        f"{sum(v > EPSILON_0 for v in E.values())}"
    )

    # ------------------------------------------------------------
    # rho
    # ------------------------------------------------------------

    print(
        "\nCalculating rho_cg..."
    )

    rho_df, category_summary = calculate_rho(
        R,
        E,
    )

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    if rho_df.empty:
        raise RuntimeError(
            "No rho_cg cells were reconstructed."
        )

    if rho_df["R_cg"].isna().any():
        raise RuntimeError(
            "R_cg contains NaN."
        )

    if rho_df["E_cg"].isna().any():
        raise RuntimeError(
            "E_cg contains NaN."
        )

    if rho_df["rho_cg"].isna().any():
        raise RuntimeError(
            "rho_cg contains NaN."
        )

    if (rho_df["R_cg"] < 0).any():
        raise RuntimeError(
            "R_cg contains negative values."
        )

    # ------------------------------------------------------------
    # Save event-level reconstruction
    # ------------------------------------------------------------

    R_event.to_csv(
        args.out
        / "R_cg_event_contributions.csv",
        index=False,
    )

    E_event.to_csv(
        args.out
        / "E_cg_event_contributions.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Save cell-level reconstruction
    # ------------------------------------------------------------

    rho_df.to_csv(
        args.out
        / "R_cg_rho_cg_reconstruction.csv",
        index=False,
    )

    category_summary.to_csv(
        args.out
        / "category_rho_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Support summary
    # ------------------------------------------------------------

    event_mass = (
        q.groupby("event_id")["q_ei"]
        .sum()
    )

    support = build_support_summary(
        rho_df,
        event_mass,
    )

    with open(
        args.out / "support_summary.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            support,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Mathematical definition manifest
    # ------------------------------------------------------------

    manifest = {

        "experiment":
            "22.32",

        "description":
            "Exact Appendix-B reconstruction of "
            "R_cg and rho_cg.",

        "read_only":
            True,

        "training":
            False,

        "retrieval":
            False,

        "ranking_changes":
            False,

        "parameter_optimization":
            False,

        "events":
            EXPECTED_EVENTS,

        "conversations":
            EXPECTED_CONVERSATIONS,

        "catalog_items":
            CATALOG_SIZE,

        "q_source":
            str(args.q),

        "exposure_source":
            str(args.baseline),

        "R_cg_definition":
            "sum_e sum_i q_ei A_ic I(g(i)=g) I(i in I_e)",

        "E_cg_definition":
            "sum_e sum_r w(r) A_pi[r],c "
            "I(g(pi[r])=g)",

        "position_weight":
            "w(r)=1/log2(r+1)",

        "rho_cg_definition":
            "E_cg/(R_cg+epsilon_0)",

        "epsilon_0":
            EPSILON_0,

        "multi_label_membership":
            "A_ic=1 for every accepted category membership",

        "category_fractional_division":
            False,

        "omega_c":
            "NOT SPECIFIED NUMERICALLY IN PROPOSAL",

        "L_fair":
            "NOT CALCULATED IN 22.32",

        "target_used_for_R":
            False,

        "target_used_for_rho":
            False,

        "future_dialogue_used":
            False,

        "support":
            support,
    }

    with open(
        args.out / "experiment_manifest.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Decision
    # ------------------------------------------------------------

    decision = {

        "R_cg_reconstructable":
            True,

        "rho_cg_reconstructable":
            True,

        "R_cg_is_proxy":
            False,

        "rho_cg_is_proxy":
            False,

        "mathematical_definition":
            "Appendix B exact definition",

        "q_status":
            "OPERATIONAL_LEAKAGE_FREE_SIGNAL",

        "q_calibration_status":
            "NOT_YET_VALIDATED_AS_CONTEXTUAL_RELEVANCE",

        "omega_c_defined":
            False,

        "L_fair_identifiable":
            False,

        "next_step":
            (
                "Validate q_ei as contextual relevance, "
                "then specify omega_c and implement L_fair."
            ),
    }

    with open(
        args.out / "decision.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            decision,
            f,
            indent=2,
        )

    # ------------------------------------------------------------
    # Console
    # ------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("22.32 COMPLETE")
    print("=" * 72)

    print(
        f"Events: {EXPECTED_EVENTS}"
    )

    print(
        f"Conversations: "
        f"{EXPECTED_CONVERSATIONS}"
    )

    print(
        f"Accepted-category items: "
        f"{mapped_items}/{CATALOG_SIZE}"
    )

    print(
        f"Positive R_cg cells: "
        f"{support['R_positive_cells']}"
    )

    print(
        f"Reconstructed rho_cg cells: "
        f"{support['rho_cells']}"
    )

    print(
        "R_cg: RECONSTRUCTED_FROM_EXACT_PROPOSAL_DEFINITION"
    )

    print(
        "rho_cg: RECONSTRUCTED_FROM_EXACT_PROPOSAL_DEFINITION"
    )

    print(
        "Target leakage: NO"
    )

    print(
        "Training: NO"
    )

    print(
        "Retrieval: NO"
    )

    print(
        "Ranking changes: NO"
    )

    print(
        "omega_c: NOT SPECIFIED"
    )

    print(
        "L_fair: NOT CALCULATED"
    )

    print(
        "Output:",
        args.out.resolve(),
    )


if __name__ == "__main__":
    main()