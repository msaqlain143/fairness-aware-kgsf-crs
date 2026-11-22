#!/usr/bin/env python
"""
Experiment 22.28
Fairness Objective Feasibility and Relevance-Mass Reconstruction Audit.

READ-ONLY AUDIT.

Purpose
-------
Determine whether the mathematical fairness objective proposed in Appendix B
can be reconstructed from the existing frozen evidence without modifying
ranking, retrieval, training, or model parameters.

The audit deliberately distinguishes:

    1. Legacy candidate-opportunity-normalized fairness quantities.
    2. Existing contextual/KG relevance proxies.
    3. The proposal's contextual relevance q_{e,i}.
    4. The proposal's relevance mass R_{c,g}.
    5. The proposal's weighted fairness objective L_fair.

Important
---------
The existing Experiment 22.18 "context" is candidate-pool category prevalence,
not semantic dialogue relevance. KG rank confidence is retrieval support, not
semantic relevance. Therefore these signals are reported as proxies and are
NOT silently promoted to the proposal's q_{e,i}.

No:
    - training
    - retrieval
    - ranking changes
    - parameter tuning
    - test-target use for relevance
    - independent-test rerun

is performed.
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


EPS = 1e-12
TOP_K = 10
CATALOG_SIZE = 6924
EXPECTED_EVENTS = 4003
EXPECTED_CONVERSATIONS = 1283


# ---------------------------------------------------------------------
# Frozen input defaults
# ---------------------------------------------------------------------

DEFAULT_CATALOG = Path("catalog_audit_corrected.pkl")

DEFAULT_PROFILES = Path(
    "results_category_audit/experiment_21_22A/"
    "catalog_item_category_profiles.csv"
)

DEFAULT_MAPPING = Path(
    "results_final_test_m2/"
    "m2_test_event_conversation_mapping.csv"
)

DEFAULT_KBRD = Path(
    "results_final_test_m2/experiment_22_27A/"
    "kgsf_independent_test_top500.csv"
)

# Verified Experiment 22.24A wide-format independent-test KG export.
# Its R1_raw_candidate_kg_ranks vector is positionally aligned with the
# frozen KGSF Top-500 candidate order from Experiment 22.27A.
DEFAULT_KG = Path(
    "results_final_test_m2/experiment_22_24A/"
    "independent_test_kg_candidate_ranks.csv"
)

DEFAULT_BASELINE = Path(
    "results_final_test_m2/experiment_22_24B/"
    "baseline_event_results.csv"
)

DEFAULT_PROPOSED = Path(
    "results_final_test_m2/experiment_22_24B/"
    "proposed_event_results.csv"
)

DEFAULT_OUT = Path(
    "results_final_test_m2/experiment_22_28"
)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def parse_list(value, field):
    if isinstance(value, list):
        result = value
    elif isinstance(value, (tuple, np.ndarray)):
        result = list(value)
    else:
        text = str(value).strip()

        try:
            result = json.loads(text)
        except Exception:
            try:
                result = ast.literal_eval(text)
            except Exception as exc:
                raise ValueError(
                    f"Cannot parse {field}: {text[:100]}"
                ) from exc

    if not isinstance(result, (list, tuple)):
        raise ValueError(f"{field} is not a list")

    return list(result)


def as_bool(value):
    if pd.isna(value):
        return False

    if isinstance(value, (bool, np.bool_)):
        return bool(value)

    return str(value).strip().lower() in {
        "true", "1", "yes", "y"
    }


def parse_categories(value, delimiter=None):
    if value is None or pd.isna(value):
        return []

    text = str(value).strip()

    if not text:
        return []

    if delimiter is None:
        delimiter = "|" if "|" in text else ";"

    return sorted({
        " ".join(token.strip().casefold().split())
        for token in text.split(delimiter)
        if token.strip()
    })


def safe_float(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------
# Catalog and category reconstruction
# ---------------------------------------------------------------------

def load_catalog_metadata(path):
    with path.open("rb") as f:
        metadata = pickle.load(f)

    movie_ids = [int(x) for x in metadata["movie_ids"]]

    if len(movie_ids) != CATALOG_SIZE:
        raise RuntimeError(
            f"Expected {CATALOG_SIZE} catalog items; got {len(movie_ids)}"
        )

    if len(set(movie_ids)) != CATALOG_SIZE:
        raise RuntimeError("Catalog movie IDs are not unique")

    item2group = {
        int(k): int(v)
        for k, v in metadata["item2group"].items()
    }

    if set(item2group) != set(movie_ids):
        raise RuntimeError(
            "item2group IDs do not exactly match frozen catalog"
        )

    if any(group not in range(5) for group in item2group.values()):
        raise RuntimeError(
            "Popularity group IDs must be in [0,4]"
        )

    return set(movie_ids), item2group


def load_categories(profile_path, catalog):
    profiles = pd.read_csv(profile_path)

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
            f"Category profile missing columns: {sorted(missing)}"
        )

    profiles["movie_entity_id"] = profiles["movie_entity_id"].astype(int)

    if profiles["movie_entity_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate movie IDs in category profiles"
        )

    categories = {
        movie_id: []
        for movie_id in catalog
    }

    status = {}

    for row in profiles.itertuples(index=False):

        movie_id = int(row.movie_entity_id)

        if movie_id not in catalog:
            continue

        identity_status = str(
            row.identity_resolution_status
        ).upper()

        status[movie_id] = identity_status

        if identity_status == "IDENTITY_REVIEW_REQUIRED":
            continue

        accepted = []

        # Existing 22.16/22.18 evidence rule.
        if (
            as_bool(row.imdb_genre_evidence)
            and not as_bool(row.imdb_conflicting_records)
        ):
            accepted.extend(
                parse_categories(row.imdb_genres)
            )

        if as_bool(row.accepted_genre_evidence):
            accepted.extend(
                parse_categories(
                    row.canonical_genres,
                    ";"
                )
            )

        categories[movie_id] = sorted(set(accepted))

    return categories, status, profiles


# ---------------------------------------------------------------------
# Frozen candidate reconstruction
# ---------------------------------------------------------------------

def load_kbrd_candidates(path):
    df = pd.read_csv(path)

    required = {
        "event_id",
        "conv_id",
        "target_movie_id",
        "top_movie_ids",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"KBRD export missing columns: {sorted(missing)}"
        )

    if len(df) != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} KBRD events; got {len(df)}"
        )

    events = {}

    for row in df.itertuples(index=False):

        event_id = int(row.event_id)

        ids = [
            int(x)
            for x in parse_list(
                row.top_movie_ids,
                "top_movie_ids"
            )
        ]

        if len(ids) != 500:
            raise RuntimeError(
                f"Event {event_id}: expected 500 candidates"
            )

        if len(set(ids)) != 500:
            raise RuntimeError(
                f"Event {event_id}: duplicate candidates"
            )

        events[event_id] = {
            "conv_id": int(row.conv_id),
            "target_movie_id": int(row.target_movie_id),
            "ids": ids,
        }

    if len(events) != EXPECTED_EVENTS:
        raise RuntimeError(
            "Duplicate event IDs in KBRD export"
        )

    return events, df


def load_kg_ranks(path, candidates=None):
    """
    Load KG ranks and align them to the frozen KGSF Top-500 candidate order.

    Preferred input: the verified 22.24A wide-format export, where
    R1_raw_candidate_kg_ranks is a 500-element vector already aligned
    positionally with each event's frozen Top-500 candidates.

    A legacy candidate_kg_ranks schema is also accepted for compatibility.
    When candidate_movie_ids are present in that legacy schema, ranks are
    explicitly realigned by movie ID rather than trusting row order.
    """

    df = pd.read_csv(path)

    if "event_id" not in df.columns:
        raise RuntimeError(
            f"KG rank export {path} is missing required column: event_id"
        )

    rank_map = {}

    # --------------------------------------------------------------
    # Preferred schema: verified 22.24A wide-format export
    # --------------------------------------------------------------
    if "R1_raw_candidate_kg_ranks" in df.columns:
        required = {
            "event_id",
            "conv_id",
            "target_movie_id",
            "candidate_count",
            "R1_raw_candidate_kg_ranks",
        }
        missing = required - set(df.columns)
        if missing:
            raise RuntimeError(
                "22.24A KG rank export missing required columns: "
                f"{sorted(missing)}"
            )

        if df["event_id"].duplicated().any():
            raise RuntimeError(
                "Duplicate event_id values in 22.24A KG rank export."
            )

        if len(df) != EXPECTED_EVENTS:
            raise RuntimeError(
                f"Expected {EXPECTED_EVENTS} 22.24A KG events; "
                f"found {len(df)}."
            )

        for column in ("event_id", "conv_id", "target_movie_id", "candidate_count"):
            df[column] = pd.to_numeric(
                df[column], errors="raise"
            ).astype(np.int64)

        if not df["candidate_count"].eq(500).all():
            bad = df.loc[
                df["candidate_count"] != 500, "event_id"
            ].tolist()[:10]
            raise RuntimeError(
                "22.24A export contains events without exactly 500 candidates. "
                f"Example event IDs: {bad}"
            )

        if candidates is not None:
            kg_events = set(df["event_id"].astype(int))
            candidate_events = set(candidates)
            if kg_events != candidate_events:
                missing_kg = sorted(candidate_events - kg_events)[:10]
                extra_kg = sorted(kg_events - candidate_events)[:10]
                raise RuntimeError(
                    "KBRD/22.24A KG event IDs do not match. "
                    f"Missing KG events (first 10): {missing_kg}; "
                    f"extra KG events (first 10): {extra_kg}"
                )

        for row in df.itertuples(index=False):
            event_id = int(row.event_id)

            if candidates is not None:
                expected = candidates[event_id]
                if int(row.conv_id) != int(expected["conv_id"]):
                    raise RuntimeError(
                        f"Event {event_id}: conv_id mismatch between "
                        "KBRD Top-500 export and 22.24A KG export."
                    )
                if int(row.target_movie_id) != int(expected["target_movie_id"]):
                    raise RuntimeError(
                        f"Event {event_id}: target_movie_id mismatch between "
                        "KBRD Top-500 export and 22.24A KG export."
                    )

            values = parse_list(
                row.R1_raw_candidate_kg_ranks,
                "R1_raw_candidate_kg_ranks",
            )
            raw = np.asarray(values)

            if raw.shape != (500,):
                raise RuntimeError(
                    f"Event {event_id}: expected 500 R1_raw KG ranks; "
                    f"found shape {raw.shape}."
                )

            try:
                numeric = raw.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Event {event_id}: KG ranks contain non-numeric values."
                ) from exc

            if not np.isfinite(numeric).all():
                raise RuntimeError(
                    f"Event {event_id}: KG ranks contain non-finite values."
                )
            if not np.equal(numeric, np.floor(numeric)).all():
                raise RuntimeError(
                    f"Event {event_id}: KG ranks contain non-integer values."
                )
            if np.any(numeric < 0):
                raise RuntimeError(
                    f"Event {event_id}: negative KG ranks detected."
                )

            rank_map[event_id] = numeric.astype(np.int32)

    # --------------------------------------------------------------
    # Compatibility schema: legacy candidate_kg_ranks export
    # --------------------------------------------------------------
    elif "candidate_kg_ranks" in df.columns:
        legacy = df.copy()

        # The historical 22.18 export can contain multiple methods.
        # Use the same frozen A1_frequency_3 method used by that workflow.
        if "method" in legacy.columns:
            if not legacy["method"].astype(str).eq("A1_frequency_3").any():
                raise RuntimeError(
                    "Legacy KG export has a method column but no "
                    "A1_frequency_3 rows."
                )
            legacy = legacy[
                legacy["method"].astype(str) == "A1_frequency_3"
            ].copy()

        if legacy["event_id"].duplicated().any():
            raise RuntimeError(
                "Duplicate event_id values in selected legacy KG export."
            )

        if len(legacy) != EXPECTED_EVENTS:
            raise RuntimeError(
                f"Expected {EXPECTED_EVENTS} legacy KG events; "
                f"found {len(legacy)}."
            )

        if candidates is not None:
            kg_events = set(pd.to_numeric(
                legacy["event_id"], errors="raise"
            ).astype(int))
            candidate_events = set(candidates)
            if kg_events != candidate_events:
                raise RuntimeError(
                    "KBRD/legacy KG event IDs do not match."
                )

        for row in legacy.itertuples(index=False):
            event_id = int(row.event_id)
            ranks = np.asarray(
                parse_list(row.candidate_kg_ranks, "candidate_kg_ranks")
            )

            if hasattr(row, "candidate_movie_ids") and candidates is not None:
                movie_ids = np.asarray(
                    parse_list(row.candidate_movie_ids, "candidate_movie_ids"),
                    dtype=np.int64,
                )
                if len(movie_ids) != len(ranks):
                    raise RuntimeError(
                        f"Event {event_id}: candidate_movie_ids and "
                        "candidate_kg_ranks have different lengths."
                    )
                if len(np.unique(movie_ids)) != len(movie_ids):
                    raise RuntimeError(
                        f"Event {event_id}: duplicate movie IDs in legacy KG export."
                    )

                rank_by_movie = {
                    int(movie_id): int(rank)
                    for movie_id, rank in zip(movie_ids, ranks)
                }
                frozen_ids = candidates[event_id]["ids"]
                missing_ids = [
                    int(movie_id) for movie_id in frozen_ids
                    if int(movie_id) not in rank_by_movie
                ]
                if missing_ids:
                    raise RuntimeError(
                        f"Event {event_id}: legacy KG export does not cover "
                        f"all frozen Top-500 candidates; examples: {missing_ids[:10]}"
                    )
                ranks = np.asarray(
                    [rank_by_movie[int(movie_id)] for movie_id in frozen_ids],
                    dtype=np.int64,
                )
            else:
                if ranks.shape != (500,):
                    raise RuntimeError(
                        f"Event {event_id}: expected 500 KG ranks; "
                        f"found shape {ranks.shape}."
                    )

            try:
                numeric = ranks.astype(np.float64)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Event {event_id}: KG ranks contain non-numeric values."
                ) from exc

            if (
                not np.isfinite(numeric).all()
                or not np.equal(numeric, np.floor(numeric)).all()
                or np.any(numeric < 0)
            ):
                raise RuntimeError(
                    f"Event {event_id}: KG ranks must be finite, "
                    "non-negative integers."
                )

            if numeric.shape != (500,):
                raise RuntimeError(
                    f"Event {event_id}: aligned KG rank vector must "
                    f"contain 500 values; found {numeric.shape}."
                )

            rank_map[event_id] = numeric.astype(np.int32)

    else:
        available = list(df.columns)
        raise RuntimeError(
            "Unrecognized KG rank export schema. Expected either "
            "'R1_raw_candidate_kg_ranks' (verified 22.24A wide format) "
            "or 'candidate_kg_ranks' (legacy format). "
            f"Available columns: {available}"
        )

    if len(rank_map) != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} KG events; got {len(rank_map)}."
        )

    return rank_map


# ---------------------------------------------------------------------
# Category context used by frozen 22.18
# ---------------------------------------------------------------------

def category_context(ids, categories):
    counts = defaultdict(float)

    categorized = 0

    for movie_id in ids:

        movie_categories = categories.get(
            int(movie_id),
            []
        )

        if not movie_categories:
            continue

        categorized += 1

        share = 1.0 / len(movie_categories)

        for category in movie_categories:
            counts[category] += share

    total = float(sum(counts.values()))

    if total <= EPS:
        return {}, categorized

    context = {
        category: value / total
        for category, value in counts.items()
    }

    return context, categorized


# ---------------------------------------------------------------------
# Exposure reconstruction
# ---------------------------------------------------------------------

def exposure_weight(rank):
    return 1.0 / math.log2(
        float(rank) + 1.0
    )


def reconstruct_exposure(
    method_name,
    event_results,
    categories,
    item2group,
):
    """
    Reconstruct proposal-style E_{c,g} from saved Top-10 rankings.

    Exposure is divided equally among an item's verified categories,
    matching the existing multi-label fractional accounting.
    """

    event_rows = []
    conversation_rows = []
    aggregate = defaultdict(float)

    for row in event_results.itertuples(index=False):

        event_id = int(row.event_id)
        conv_id = int(row.conv_id)

        ranked_ids = [
            int(x)
            for x in parse_list(
                row.top10_movie_ids,
                "top10_movie_ids"
            )
        ]

        ranked_ids = ranked_ids[:TOP_K]

        for rank, movie_id in enumerate(
            ranked_ids,
            start=1
        ):

            movie_categories = categories.get(
                movie_id,
                []
            )

            if not movie_categories:
                continue

            group = int(item2group[movie_id])

            contribution = (
                exposure_weight(rank)
                / len(movie_categories)
            )

            for category in movie_categories:

                aggregate[
                    category,
                    group
                ] += contribution

                conversation_rows.append({
                    "method": method_name,
                    "conv_id": conv_id,
                    "category": category,
                    "popularity_group": group,
                    "exposure_contribution": contribution,
                })

    return (
        aggregate,
        pd.DataFrame(conversation_rows)
    )


# ---------------------------------------------------------------------
# Opportunity / candidate mass reconstruction
# ---------------------------------------------------------------------

def opportunity_weighted_candidate_mass(
    candidate_ids,
    categories,
    item2group,
):
    """
    Reconstruct candidate opportunity using the same discounted
    candidate-pool weighting used by the legacy ExposureLedger.

    This is deliberately NOT called proposal relevance mass.
    """

    n = len(candidate_ids)

    weights = np.asarray([
        1.0 / math.log2(
            float(rank) + 1.0
        )
        for rank in range(1, n + 1)
    ])

    weights /= (
        float(weights.sum())
        + EPS
    )

    result = defaultdict(float)

    for index, movie_id in enumerate(candidate_ids):

        movie_categories = categories.get(
            int(movie_id),
            []
        )

        if not movie_categories:
            continue

        group = int(item2group[movie_id])

        contribution = (
            float(weights[index])
            / len(movie_categories)
        )

        for category in movie_categories:
            result[
                category,
                group
            ] += contribution

    return result


# ---------------------------------------------------------------------
# Relevance proxy audit
# ---------------------------------------------------------------------

def reconstruct_relevance_proxies(
    candidates,
    kg_ranks,
    categories,
    tau=250.0,
):
    """
    Reconstruct the signals that actually exist in the frozen
    Experiment 22.18 evidence.

    proxy_context:
        normalized category prevalence in the frozen candidate pool.

    kg_confidence:
        exp(-KG_rank / tau).

    legacy_context_kg_proxy:
        proxy_context * kg_confidence.

    These are NOT declared to be the proposal q_{e,i}.
    """

    rows = []

    for event_id in sorted(candidates):

        candidate_ids = candidates[event_id]["ids"]

        ranks = kg_ranks[event_id]

        context, _ = category_context(
            candidate_ids,
            categories
        )

        for index, movie_id in enumerate(
            candidate_ids
        ):

            movie_categories = categories.get(
                int(movie_id),
                []
            )

            rank = int(ranks[index])

            if movie_categories:

                context_relevance = float(
                    np.mean([
                        context.get(
                            category,
                            0.0
                        )
                        for category in movie_categories
                    ])
                )

            else:
                context_relevance = 0.0

            kg_confidence = (
                math.exp(
                    -float(rank) / tau
                )
                if rank > 0
                else 0.0
            )

            legacy_proxy = (
                context_relevance
                * kg_confidence
            )

            rows.append({
                "event_id": event_id,
                "movie_id": int(movie_id),
                "category_count": len(
                    movie_categories
                ),
                "kg_rank": rank,
                "kg_supported": bool(rank > 0),
                "context_relevance_proxy":
                    context_relevance,
                "kg_confidence_proxy":
                    kg_confidence,
                "legacy_context_kg_relevance_proxy":
                    legacy_proxy,
                "proposal_q_available":
                    False,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Proposal R_{c,g} reconstruction
# ---------------------------------------------------------------------

def reconstruct_R_cg(
    candidates,
    item2group,
    categories,
    relevance_df,
):
    """
    Attempt to reconstruct:

        R_{c,g}
        =
        sum_e sum_i q_{e,i} A_{i,c}
        1[g(i)=g] 1[i in I_e]

    Because the proposal q_{e,i} is not currently available as a
    validated contextual relevance quantity, this function reports
    the legacy proxy reconstruction separately.

    It never labels the proxy as the proposal's validated R_{c,g}.
    """

    proxy_lookup = {
        (
            int(row.event_id),
            int(row.movie_id)
        ):
            float(
                row.legacy_context_kg_relevance_proxy
            )
        for row in relevance_df.itertuples(index=False)
    }

    R_proxy = defaultdict(float)

    event_mass_rows = []

    for event_id in sorted(candidates):

        candidate_ids = candidates[event_id]["ids"]

        for movie_id in candidate_ids:

            movie_categories = categories.get(
                int(movie_id),
                []
            )

            if not movie_categories:
                continue

            proxy = proxy_lookup[
                event_id,
                int(movie_id)
            ]

            group = int(
                item2group[int(movie_id)]
            )

            contribution = (
                proxy
                / len(movie_categories)
            )

            for category in movie_categories:

                R_proxy[
                    category,
                    group
                ] += contribution

                event_mass_rows.append({
                    "event_id": event_id,
                    "category": category,
                    "popularity_group": group,
                    "movie_id": int(movie_id),
                    "relevance_signal":
                        "legacy_context_kg_proxy",
                    "relevance_value":
                        proxy,
                    "category_fraction":
                        1.0 / len(movie_categories),
                    "R_proxy_contribution":
                        contribution,
                })

    return (
        R_proxy,
        pd.DataFrame(event_mass_rows)
    )


# ---------------------------------------------------------------------
# Proposal fairness calculations
# ---------------------------------------------------------------------

def calculate_rho_and_variance(
    exposure,
    R,
    r_min,
):
    """
    Calculate proposal-style rho, p, category weighted variance
    where R is available.

    This calculation is performed on the supplied R quantity.
    The caller is responsible for identifying whether R is validated
    proposal relevance mass or only a proxy.
    """

    rows = []

    by_category = defaultdict(list)

    for (category, group), r_value in R.items():

        e_value = float(
            exposure.get(
                (category, group),
                0.0
            )
        )

        by_category[category].append(
            (
                int(group),
                float(r_value),
                e_value
            )
        )

    category_variances = {}

    for category, cells in by_category.items():

        total_R = sum(
            max(r, 0.0)
            for _, r, _ in cells
        )

        if total_R <= EPS:
            continue

        for group, r_value, e_value in cells:

            p = max(
                r_value,
                0.0
            ) / total_R

            rho = (
                e_value
                / (r_value + EPS)
            )

            rows.append({
                "category": category,
                "popularity_group": group,
                "R_cg": r_value,
                "E_cg": e_value,
                "p_cg": p,
                "rho_cg": rho,
                "support_pass":
                    bool(r_value >= r_min),
            })

        supported = [
            x for x in cells
            if x[1] >= r_min
        ]

        if not supported:
            continue

        support_total = sum(
            max(x[1], 0.0)
            for x in supported
        )

        if support_total <= EPS:
            continue

        rho_bar = sum(
            (
                max(r, 0.0)
                / support_total
            )
            * (
                e
                / (r + EPS)
            )
            for _, r, e in supported
        )

        variance = sum(
            (
                max(r, 0.0)
                / support_total
            )
            * (
                e / (r + EPS)
                - rho_bar
            ) ** 2
            for _, r, e in supported
        )

        category_variances[category] = {
            "rho_bar": rho_bar,
            "variance": variance,
            "supported_groups": len(
                supported
            ),
        }

    return (
        pd.DataFrame(rows),
        category_variances
    )


# ---------------------------------------------------------------------
# Support sensitivity
# ---------------------------------------------------------------------

def support_sensitivity(
    R,
    thresholds,
):
    rows = []

    for threshold in thresholds:

        positive = [
            value
            for value in R.values()
            if value > EPS
        ]

        supported = [
            value
            for value in positive
            if value >= threshold
        ]

        rows.append({
            "R_min": threshold,
            "positive_cells": len(
                positive
            ),
            "supported_cells": len(
                supported
            ),
            "support_fraction": (
                len(supported)
                / len(positive)
                if positive
                else 0.0
            ),
            "minimum_positive_R": (
                min(positive)
                if positive
                else np.nan
            ),
            "median_positive_R": (
                float(np.median(positive))
                if positive
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Category coverage
# ---------------------------------------------------------------------

def category_coverage_audit(
    catalog,
    profiles,
    categories,
    status,
):
    rows = []

    profile_ids = set(
        profiles.movie_entity_id.astype(int)
    )

    for movie_id in sorted(catalog):

        movie_categories = categories.get(
            movie_id,
            []
        )

        rows.append({
            "movie_entity_id": movie_id,
            "in_category_profile":
                movie_id in profile_ids,
            "identity_resolution_status":
                status.get(
                    movie_id,
                    "MISSING_PROFILE"
                ),
            "category_count":
                len(movie_categories),
            "has_category_mapping":
                bool(movie_categories),
            "categories":
                "|".join(movie_categories),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------

def leakage_audit(
    candidates,
    mapping,
    relevance_df,
):
    candidate_events = set(
        candidates
    )

    mapping_events = set(
        mapping.event_id.astype(int)
    )

    relevance_events = set(
        relevance_df.event_id.astype(int)
    )

    return {
        "candidate_events": len(
            candidate_events
        ),
        "mapping_events": len(
            mapping_events
        ),
        "relevance_events": len(
            relevance_events
        ),
        "event_sets_match":
            candidate_events
            == mapping_events
            == relevance_events,
        "target_loaded_for_alignment":
            True,
        "target_used_to_construct_q":
            False,
        "target_used_to_construct_R":
            False,
        "target_used_for_category_mapping":
            False,
        "target_used_for_exposure":
            False,
        "target_used_for_opportunity":
            False,
        "training_performed":
            False,
        "retrieval_performed":
            False,
        "ranking_changed":
            False,
    }


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--catalog",
        type=Path,
        default=DEFAULT_CATALOG
    )

    parser.add_argument(
        "--profiles",
        type=Path,
        default=DEFAULT_PROFILES
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        default=DEFAULT_MAPPING
    )

    parser.add_argument(
        "--kbrd",
        type=Path,
        default=DEFAULT_KBRD
    )

    parser.add_argument(
        "--kg",
        type=Path,
        default=DEFAULT_KG
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE
    )

    parser.add_argument(
        "--proposed",
        type=Path,
        default=DEFAULT_PROPOSED
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT
    )

    parser.add_argument(
        "--tau",
        type=float,
        default=250.0
    )

    parser.add_argument(
        "--r-min-values",
        default="0,0.001,0.005,0.01,0.025,0.05"
    )

    args = parser.parse_args()

    args.out.mkdir(
        parents=True,
        exist_ok=True
    )

    required_paths = [
        args.catalog,
        args.profiles,
        args.mapping,
        args.kbrd,
        args.kg,
        args.baseline,
        args.proposed,
    ]

    for path in required_paths:

        if not path.is_file():
            raise FileNotFoundError(
                f"Required frozen input not found: {path}"
            )

    # --------------------------------------------------------------
    # Load frozen metadata
    # --------------------------------------------------------------

    catalog, item2group = load_catalog_metadata(
        args.catalog
    )

    categories, identity_status, profiles = (
        load_categories(
            args.profiles,
            catalog
        )
    )

    mapping = pd.read_csv(
        args.mapping
    )

    candidates, kbrd_df = (
        load_kbrd_candidates(
            args.kbrd
        )
    )

    kg_ranks = load_kg_ranks(
        args.kg,
        candidates=candidates,
    )

    baseline = pd.read_csv(
        args.baseline
    )

    proposed = pd.read_csv(
        args.proposed
    )

    # --------------------------------------------------------------
    # Validate alignment
    # --------------------------------------------------------------

    if set(candidates) != set(kg_ranks):
        raise RuntimeError(
            "KBRD and KG event IDs do not match"
        )

    if len(mapping) != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} mapping rows"
        )

    if mapping.event_id.duplicated().any():
        raise RuntimeError(
            "Duplicate event IDs in mapping"
        )

    # --------------------------------------------------------------
    # A. Category feasibility
    # --------------------------------------------------------------

    coverage = category_coverage_audit(
        catalog,
        profiles,
        categories,
        identity_status
    )

    coverage.to_csv(
        args.out / "category_coverage_audit.csv",
        index=False
    )

    # --------------------------------------------------------------
    # B. Existing relevance signals
    # --------------------------------------------------------------

    relevance_df = (
        reconstruct_relevance_proxies(
            candidates,
            kg_ranks,
            categories,
            tau=args.tau
        )
    )

    relevance_df.to_csv(
        args.out / "relevance_signal_audit.csv",
        index=False
    )

    # --------------------------------------------------------------
    # C. Existing exposure reconstruction
    # --------------------------------------------------------------

    baseline_exposure, baseline_conv = (
        reconstruct_exposure(
            "baseline",
            baseline,
            categories,
            item2group
        )
    )

    proposed_exposure, proposed_conv = (
        reconstruct_exposure(
            "proposed",
            proposed,
            categories,
            item2group
        )
    )

    conversation_exposure = pd.concat(
        [
            baseline_conv,
            proposed_conv,
        ],
        ignore_index=True
    )

    conversation_exposure.to_csv(
        args.out / "conversation_exposure_reconstruction.csv",
        index=False
    )

    # --------------------------------------------------------------
    # D. Candidate opportunity reconstruction
    # --------------------------------------------------------------

    opportunity_rows = []

    for event_id in sorted(candidates):

        event = candidates[event_id]

        mass = opportunity_weighted_candidate_mass(
            event["ids"],
            categories,
            item2group
        )

        for (category, group), value in mass.items():

            opportunity_rows.append({
                "event_id": event_id,
                "conv_id": event["conv_id"],
                "category": category,
                "popularity_group": group,
                "opportunity_contribution": value,
            })

    opportunity_df = pd.DataFrame(
        opportunity_rows
    )

    opportunity_df.to_csv(
        args.out / "event_opportunity_reconstruction.csv",
        index=False
    )

    # --------------------------------------------------------------
    # E. Proposal R proxy reconstruction
    # --------------------------------------------------------------

    R_proxy, event_R_proxy = (
        reconstruct_R_cg(
            candidates,
            item2group,
            categories,
            relevance_df
        )
    )

    event_R_proxy.to_csv(
        args.out / "relevance_mass_event_level.csv",
        index=False
    )

    R_rows = []

    for (category, group), value in sorted(
        R_proxy.items()
    ):
        R_rows.append({
            "category": category,
            "popularity_group": group,
            "R_cg_signal":
                "legacy_context_kg_proxy",
            "R_cg_value": value,
            "proposal_R_cg_validated":
                False,
        })

    R_df = pd.DataFrame(
        R_rows
    )

    R_df.to_csv(
        args.out / "R_c_g_reconstruction.csv",
        index=False
    )

    # --------------------------------------------------------------
    # F. Support sensitivity
    # --------------------------------------------------------------

    thresholds = sorted(set(
        float(x.strip())
        for x in args.r_min_values.split(",")
        if x.strip()
    ))

    sensitivity = support_sensitivity(
        R_proxy,
        thresholds
    )

    sensitivity.to_csv(
        args.out / "R_min_sensitivity.csv",
        index=False
    )

    # --------------------------------------------------------------
    # G. Proposal-style variance using proxy R
    # --------------------------------------------------------------

    variance_rows = []

    for r_min in thresholds:

        for method, exposure in [
            ("baseline", baseline_exposure),
            ("proposed", proposed_exposure),
        ]:

            cell_df, category_variances = (
                calculate_rho_and_variance(
                    exposure,
                    R_proxy,
                    r_min
                )
            )

            for category, values in (
                category_variances.items()
            ):

                variance_rows.append({
                    "method": method,
                    "R_min": r_min,
                    "category": category,
                    "rho_bar":
                        values["rho_bar"],
                    "category_variance":
                        values["variance"],
                    "supported_groups":
                        values["supported_groups"],
                    "R_source":
                        "legacy_context_kg_proxy",
                    "proposal_q_validated":
                        False,
                    "proposal_L_fair_validated":
                        False,
                })

    variance_df = pd.DataFrame(
        variance_rows
    )

    variance_df.to_csv(
        args.out / "category_variance.csv",
        index=False
    )

    # --------------------------------------------------------------
    # H. Legacy vs proposal estimand comparison
    # --------------------------------------------------------------

    legacy_aggregate = []

    for method, frame in [
        ("baseline", baseline),
        ("proposed", proposed),
    ]:

        if "fairness_MAD" in frame.columns:

            value = safe_float(
                frame["fairness_MAD"].iloc[-1]
            )

        else:
            value = None

        legacy_aggregate.append({
            "method": method,
            "legacy_metric":
                "category_group_log_rate_MAD",
            "legacy_value":
                value,
            "proposal_quantity":
                "R_cg_relevance_normalized_weighted_variance",
            "proposal_q_status":
                "UNAVAILABLE_AS_VALIDATED_CONTEXTUAL_RELEVANCE",
            "proposal_R_status":
                "PROXY_ONLY",
            "omega_c_status":
                "UNSPECIFIED_IN_PROPOSAL",
        })

    comparison = pd.DataFrame(
        legacy_aggregate
    )

    comparison.to_csv(
        args.out / "legacy_vs_proposal_estimand.csv",
        index=False
    )

    # --------------------------------------------------------------
    # I. Leakage / mathematical definition audit
    # --------------------------------------------------------------

    leakage = leakage_audit(
        candidates,
        mapping,
        relevance_df
    )

    positive_R = [
        value
        for value in R_proxy.values()
        if value > EPS
    ]

    mathematical_audit = {

        "experiment":
            "22.28_fairness_objective_feasibility_audit",

        "read_only":
            True,

        "training_performed":
            False,

        "retrieval_performed":
            False,

        "ranking_changed":
            False,

        "catalog_items":
            len(catalog),

        "events":
            len(candidates),

        "conversations":
            int(mapping.conv_id.nunique()),

        "candidate_pool":
            500,

        "category_mappings":
            int(
                coverage.has_category_mapping.sum()
            ),

        "category_mapping_rate":
            float(
                coverage.has_category_mapping.mean()
            ),

        "proposal_q_status":
            "UNAVAILABLE_AS_VALIDATED_CONTEXTUAL_RELEVANCE",

        "existing_context_signal":
            "normalized category prevalence in frozen Top-500 pool",

        "existing_kg_signal":
            "exp(-KG_rank/tau) retrieval-support proxy",

        "legacy_relevance_proxy":
            "context_relevance_proxy * kg_confidence_proxy",

        "legacy_proxy_is_proposal_q":
            False,

        "R_cg_status":
            "PROXY_RECONSTRUCTION_ONLY",

        "positive_R_proxy_cells":
            len(positive_R),

        "R_min_status":
            "NUMERIC_THRESHOLD_NOT_SPECIFIED_BY_PROPOSAL",

        "omega_c_status":
            "WEIGHTING_RULE_NOT_SPECIFIED_BY_PROPOSAL",

        "L_fair_status":
            "NOT_FULLY_IDENTIFIABLE",

        "legacy_MAD_interchangeable_with_L_fair":
            False,

        "leakage_audit":
            leakage,
    }

    with (
        args.out
        / "mathematical_definition_audit.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            mathematical_audit,
            f,
            indent=2
        )

    # --------------------------------------------------------------
    # J. Experiment manifest
    # --------------------------------------------------------------

    manifest = {

        "experiment":
            "22.28_fairness_objective_feasibility_audit",

        "purpose":
            "Read-only feasibility audit of proposal fairness objective",

        "catalog":
            str(args.catalog),

        "category_profiles":
            str(args.profiles),

        "kbrd_candidates":
            str(args.kbrd),

        "kg_ranks":
            str(args.kg),

        "baseline_rankings":
            str(args.baseline),

        "proposed_rankings":
            str(args.proposed),

        "events":
            len(candidates),

        "catalog_items":
            len(catalog),

        "candidate_pool":
            500,

        "tau":
            args.tau,

        "R_min_values":
            thresholds,

        "target_used_for_relevance":
            False,

        "target_used_for_R":
            False,

        "training":
            False,

        "retrieval":
            False,

        "ranking_changes":
            False,

        "proposal_q_validated":
            False,

        "proposal_L_fair_validated":
            False,

        "omega_c_defined":
            False,

        "outputs": [
            "category_coverage_audit.csv",
            "relevance_signal_audit.csv",
            "event_opportunity_reconstruction.csv",
            "conversation_exposure_reconstruction.csv",
            "relevance_mass_event_level.csv",
            "R_c_g_reconstruction.csv",
            "R_min_sensitivity.csv",
            "category_variance.csv",
            "legacy_vs_proposal_estimand.csv",
            "mathematical_definition_audit.json",
            "experiment_manifest.json",
            "estimand_feasibility_report.txt",
        ],
    }

    with (
        args.out
        / "experiment_manifest.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )

    # --------------------------------------------------------------
    # K. Human-readable report
    # --------------------------------------------------------------

    category_count = int(
        coverage.has_category_mapping.sum()
    )

    category_rate = (
        category_count
        / len(coverage)
    )

    kg_supported = int(
        relevance_df.kg_supported.sum()
    )

    proxy_positive = int(
        (
            relevance_df[
                "legacy_context_kg_relevance_proxy"
            ] > EPS
        ).sum()
    )

    report = f"""
Experiment 22.28
Fairness Objective Feasibility and Relevance-Mass Reconstruction Audit

STATUS
------
Read-only: YES
Training: NO
Retrieval: NO
Ranking changes: NO

FROZEN DATA
-----------
Catalog items: {len(catalog)}
Events: {len(candidates)}
Conversations: {mapping.conv_id.nunique()}
Candidate pool: 500

CATEGORY FEASIBILITY
--------------------
Items with accepted category mapping:
    {category_count} / {len(catalog)}

Category coverage:
    {category_rate:.6f}

RELEVANCE SIGNAL AUDIT
----------------------
KG-supported candidate instances:
    {kg_supported}

Positive legacy context x KG relevance-proxy instances:
    {proxy_positive}

Existing contextual signal:
    Candidate-pool category prevalence.

Existing KG signal:
    exp(-KG_rank / tau).

Important:
    These signals are not equivalent to validated contextual dialogue
    relevance q_(e,i) in the proposal.

PROPOSAL q_(e,i)
----------------
Status:
    UNAVAILABLE AS A VALIDATED CONTEXTUAL RELEVANCE QUANTITY.

Reason:
    Experiment 22.18 explicitly defines its context as category prevalence
    in the frozen candidate pool rather than semantic dialogue relevance.
    KG rank is used as retrieval-support confidence.

R_(c,g)
-------
A proxy reconstruction was produced using:

    context_relevance_proxy * KG_confidence_proxy

and fractional multi-label category membership.

This is reported as:
    R_cg proxy

It is NOT reported as:
    validated proposal R_cg.

R_MIN
-----
The proposal requires a support condition R_cg >= R_min,
but does not specify a unique numerical R_min.

Therefore this audit reports sensitivity across:
    {thresholds}

OMEGA_c
--------
The proposal contains omega_c in L_fair but does not provide
a unique numerical construction rule for omega_c.

Therefore no single value is silently imposed.

L_FAIR
------
The full proposal objective cannot yet be claimed as empirically
identified because both of the following remain unresolved:

    1. validated contextual relevance q_(e,i)
    2. numerical category weighting omega_c

LEGACY METRIC
-------------
The existing category-group log-rate MAD remains a separate estimand.

It must not be described as equivalent to L_fair.

RECOMMENDATION
--------------
Do not modify the reranking algorithm yet.

The next research task should establish and validate a leakage-safe
contextual relevance estimator q_(e,i), followed by a principled
definition of omega_c and a predeclared R_min support policy.
"""

    (
        args.out
        / "estimand_feasibility_report.txt"
    ).write_text(
        report.strip() + "\n",
        encoding="utf-8"
    )

    print("=" * 72)
    print("EXPERIMENT 22.28 COMPLETE")
    print("=" * 72)

    print(
        f"Category coverage: "
        f"{category_count}/{len(catalog)} "
        f"({category_rate:.2%})"
    )

    print(
        "Proposal q status: "
        "UNAVAILABLE_AS_VALIDATED_CONTEXTUAL_RELEVANCE"
    )

    print(
        f"Proxy R_cg positive cells: "
        f"{len(positive_R)}"
    )

    print(
        "R_min: sensitivity only"
    )

    print(
        "omega_c: UNSPECIFIED"
    )

    print(
        "L_fair: NOT_FULLY_IDENTIFIABLE"
    )

    print(
        f"Output: {args.out.resolve()}"
    )


if __name__ == "__main__":
    main()
