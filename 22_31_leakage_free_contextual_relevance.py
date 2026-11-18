"""
Experiment 22.31
Leakage-Free Contextual Relevance Construction

Purpose
-------
Construct an operational q[e,i] from the frozen independent-test KGSF
scores and frozen KG evidence.

Important:
- Read-only
- No training
- No retrieval
- No ranking changes
- No target information used to construct q
- No future dialogue used
- q is an operational relevance signal, NOT automatically a validated
  contextual relevance probability.

Critical alignment:
The KGSF full-score NPZ contains:
    scores
    movie_ids
    event_ids

The movie_ids array defines the frozen score-column ordering.
Therefore candidate movie IDs MUST be mapped through movie_ids before
accessing the score matrix.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


# =====================================================================
# PATHS
# =====================================================================

ROOT = Path(".")

TOP500_PATH = (
    ROOT
    / "results_final_test_m2"
    / "experiment_22_27A"
    / "kgsf_independent_test_top500.csv"
)

FULL_SCORE_PATH = (
    ROOT
    / "results_final_test_m2"
    / "experiment_22_27A"
    / "kgsf_independent_test_full_scores.npz"
)

KG_RANK_PATH = (
    ROOT
    / "results_final_test_m2"
    / "experiment_22_27A2"
    / "kgsf_independent_test_kg_ranks.csv"
)

MAPPING_PATH = (
    ROOT
    / "results_final_test_m2"
    / "m2_test_event_conversation_mapping.csv"
)

OUT_DIR = (
    ROOT
    / "results_final_test_m2"
    / "experiment_22_31"
)


# =====================================================================
# EXPECTED FROZEN DATA
# =====================================================================

EXPECTED_EVENTS = 4003
EXPECTED_CATALOG = 6924
TOP_K = 500

# KG confidence:
#
#     KG_confidence = exp(-rank / tau)
#
# Rank 1 therefore receives the highest confidence.
KG_TAU = 50.0

EPS = 1e-12


# =====================================================================
# GENERAL UTILITIES
# =====================================================================

def parse_list(value):
    """Parse JSON/Python-list representation."""

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if pd.isna(value):
        return []

    value = str(value).strip()

    try:
        return json.loads(value)
    except Exception:
        pass

    try:
        return ast.literal_eval(value)
    except Exception:
        pass

    raise ValueError(
        f"Unable to parse list value:\n{value[:300]}"
    )


def find_column(df, candidates, required=True):

    for column in candidates:
        if column in df.columns:
            return column

    if required:
        raise RuntimeError(
            "\nRequired column not found.\n"
            f"Candidates: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def softmax(values):

    x = np.asarray(values, dtype=np.float64)

    if len(x) == 0:
        return x

    x = x - np.max(x)

    exp_x = np.exp(x)

    denominator = np.sum(exp_x)

    if (
        denominator <= 0
        or not np.isfinite(denominator)
    ):
        return np.ones(len(x), dtype=np.float64) / len(x)

    return exp_x / denominator


# =====================================================================
# EVENT MAPPING
# =====================================================================

def load_mapping():

    df = pd.read_csv(MAPPING_PATH)

    print("\n========== EVENT MAPPING ==========")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    required = {
        "event_id",
        "conv_id",
        "target_movie_id",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Event mapping missing columns: {sorted(missing)}"
        )

    if len(df) != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} mapping rows; "
            f"got {len(df)}."
        )

    df = df.sort_values("event_id").reset_index(drop=True)

    if df["event_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate event IDs found in mapping."
        )

    return df


# =====================================================================
# TOP-500
# =====================================================================

def load_top500():

    df = pd.read_csv(TOP500_PATH)

    print("\n========== TOP-500 EXPORT ==========")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    required = {
        "event_id",
        "conv_id",
        "target_movie_id",
        "top_movie_ids",
        "top_scores",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Top-500 export missing columns: {sorted(missing)}"
        )

    df = df.sort_values("event_id").reset_index(drop=True)

    if len(df) != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} top-500 rows; "
            f"got {len(df)}."
        )

    candidate_ids = df["top_movie_ids"].map(parse_list)
    candidate_scores = df["top_scores"].map(parse_list)

    id_lengths = candidate_ids.map(len)
    score_lengths = candidate_scores.map(len)

    if not id_lengths.eq(TOP_K).all():
        raise RuntimeError(
            "Not every event contains exactly 500 candidate IDs."
        )

    if not score_lengths.eq(TOP_K).all():
        raise RuntimeError(
            "Not every event contains exactly 500 candidate scores."
        )

    return df, candidate_ids, candidate_scores


# =====================================================================
# KG RANK EXPORT
# =====================================================================

def load_kg_ranks():

    df = pd.read_csv(KG_RANK_PATH)

    print("\n========== KG RANK EXPORT ==========")
    print("Rows:", len(df))
    print("Columns:", list(df.columns))

    if "event_id" not in df.columns:
        raise RuntimeError(
            "KG export does not contain event_id."
        )

    df = df.sort_values("event_id").reset_index(drop=True)

    if len(df) != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} KG rows; "
            f"got {len(df)}."
        )

    rank_columns = [
        c
        for c in df.columns
        if (
            "candidate_kg_ranks" in c
            or "kg_ranks" in c
        )
    ]

    if not rank_columns:
        raise RuntimeError(
            "No candidate KG-rank column found.\n"
            f"Available columns: {list(df.columns)}"
        )

    print("KG rank columns:", rank_columns)

    # Prefer the raw KG evidence used by 22.27A2.
    preferred = None

    preference_order = [
        "R1_raw_candidate_kg_ranks",
        "R0_filtered_candidate_kg_ranks",
    ]

    for preferred_name in preference_order:
        if preferred_name in df.columns:
            preferred = preferred_name
            break

    if preferred is None:
        preferred = rank_columns[0]

    print("Selected KG evidence:", preferred)

    ranks = df[preferred].map(parse_list)

    lengths = ranks.map(len)

    if not lengths.eq(TOP_K).all():
        raise RuntimeError(
            "KG rank arrays are not all length 500."
        )

    return df, ranks, preferred


# =====================================================================
# FULL SCORE EXPORT
# =====================================================================

def load_full_scores():

    data = np.load(
        FULL_SCORE_PATH,
        allow_pickle=False
    )

    print("\n========== FULL SCORE EXPORT ==========")

    print("Keys:")

    for key in data.files:

        arr = data[key]

        print(
            f"  {key}: "
            f"shape={arr.shape}, "
            f"dtype={arr.dtype}"
        )

    required = {
        "scores",
        "movie_ids",
        "event_ids",
    }

    missing = required - set(data.files)

    if missing:
        raise RuntimeError(
            "Full-score export is missing required arrays: "
            f"{sorted(missing)}"
        )

    scores = np.asarray(
        data["scores"],
        dtype=np.float64
    )

    movie_ids = np.asarray(
        data["movie_ids"],
        dtype=np.int64
    )

    event_ids = np.asarray(
        data["event_ids"],
        dtype=np.int64
    )

    print("\nSelected score matrix:")
    print("Key: scores")
    print("Shape:", scores.shape)

    # ---------------------------------------------------------------
    # Shape validation
    # ---------------------------------------------------------------

    if scores.ndim != 2:
        raise RuntimeError(
            f"Expected 2-D score matrix; "
            f"got {scores.ndim}-D."
        )

    if scores.shape != (
        EXPECTED_EVENTS,
        EXPECTED_CATALOG
    ):
        raise RuntimeError(
            "Unexpected score matrix shape: "
            f"{scores.shape}; expected "
            f"({EXPECTED_EVENTS}, {EXPECTED_CATALOG})."
        )

    if len(movie_ids) != EXPECTED_CATALOG:
        raise RuntimeError(
            f"Expected {EXPECTED_CATALOG} movie IDs; "
            f"got {len(movie_ids)}."
        )

    if len(event_ids) != EXPECTED_EVENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_EVENTS} event IDs; "
            f"got {len(event_ids)}."
        )

    if len(np.unique(movie_ids)) != EXPECTED_CATALOG:
        raise RuntimeError(
            "Frozen score-export movie_ids are not unique."
        )

    if len(np.unique(event_ids)) != EXPECTED_EVENTS:
        raise RuntimeError(
            "Frozen score-export event_ids are not unique."
        )

    return scores, movie_ids, event_ids


# =====================================================================
# CRITICAL FIX:
# MOVIE ID -> SCORE COLUMN
# =====================================================================

def build_movie_to_score_column(movie_ids):

    """
    Build the exact mapping:

        movie/entity ID -> frozen KGSF score column

    This is necessary because movie IDs such as 59913 are NOT
    necessarily valid zero-based score-column indices.

    The NPZ movie_ids array defines the frozen score ordering.
    """

    movie_to_column = {}

    for column_index, movie_id in enumerate(movie_ids):

        movie_id = int(movie_id)

        if movie_id in movie_to_column:
            raise RuntimeError(
                f"Duplicate movie ID in score export: {movie_id}"
            )

        movie_to_column[movie_id] = column_index

    return movie_to_column


def validate_candidate_score_alignment(
    candidate_ids,
    movie_to_column
):

    missing_ids = []

    for event_candidates in candidate_ids:

        for movie_id in event_candidates:

            movie_id = int(movie_id)

            if movie_id not in movie_to_column:
                missing_ids.append(movie_id)

    if missing_ids:

        missing_unique = sorted(
            set(missing_ids)
        )

        raise RuntimeError(
            "\nCandidate movie IDs cannot be mapped to "
            "the frozen KGSF score matrix.\n"
            f"Missing IDs: {missing_unique[:50]}\n"
            f"Number missing: {len(missing_unique)}"
        )

    print(
        "\n========== SCORE ALIGNMENT =========="
    )

    print(
        "Candidate movie IDs successfully mapped "
        "to frozen score columns."
    )

    print(
        "Unique catalog movie IDs:",
        len(movie_to_column)
    )

    print(
        "Candidate alignment: PASS"
    )


# =====================================================================
# EVENT ALIGNMENT
# =====================================================================

def validate_event_alignment(
    mapping,
    top500,
    kg_df,
    score_event_ids
):

    mapping_events = (
        mapping["event_id"]
        .astype(int)
        .to_numpy()
    )

    top_events = (
        top500["event_id"]
        .astype(int)
        .to_numpy()
    )

    kg_events = (
        kg_df["event_id"]
        .astype(int)
        .to_numpy()
    )

    score_events = (
        np.asarray(score_event_ids)
        .astype(int)
    )

    if not np.array_equal(
        mapping_events,
        top_events
    ):
        raise RuntimeError(
            "Mapping and Top-500 event IDs do not align."
        )

    if not np.array_equal(
        mapping_events,
        kg_events
    ):
        raise RuntimeError(
            "Mapping and KG event IDs do not align."
        )

    if not np.array_equal(
        mapping_events,
        score_events
    ):
        raise RuntimeError(
            "Mapping and full-score event IDs do not align."
        )

    print(
        "\n========== EVENT ALIGNMENT =========="
    )

    print(
        "Mapping events:",
        len(mapping_events)
    )

    print(
        "Top-500 events:",
        len(top_events)
    )

    print(
        "KG events:",
        len(kg_events)
    )

    print(
        "KGSF score events:",
        len(score_events)
    )

    print(
        "Event alignment: PASS"
    )


# =====================================================================
# q[e,i] CONSTRUCTION
# =====================================================================

def construct_q(
    candidate_ids,
    candidate_scores,
    kg_ranks,
    score_matrix,
    movie_to_column
):

    """
    Construct operational q[e,i].

    For each event e and candidate i:

        p_KGSF(e,i)
            = softmax(frozen KGSF score)

        KG_conf(e,i)
            = exp(-KG_rank / tau), if KG-supported
              0 otherwise

        q_raw(e,i)
            = p_KGSF(e,i) * (1 + KG_conf(e,i))

        q(e,i)
            = q_raw(e,i) / sum_i q_raw(e,i)

    No target information is used.
    """

    records = []

    for event_index in range(EXPECTED_EVENTS):

        ids = [
            int(x)
            for x in candidate_ids.iloc[event_index]
        ]

        exported_scores = np.asarray(
            [
                float(x)
                for x in candidate_scores.iloc[event_index]
            ],
            dtype=np.float64
        )

        ranks = np.asarray(
            [
                int(x)
                for x in kg_ranks.iloc[event_index]
            ],
            dtype=np.int64
        )

        if len(ids) != TOP_K:
            raise RuntimeError(
                f"Event {event_index + 1}: "
                "invalid candidate count."
            )

        if len(exported_scores) != TOP_K:
            raise RuntimeError(
                f"Event {event_index + 1}: "
                "invalid candidate-score count."
            )

        if len(ranks) != TOP_K:
            raise RuntimeError(
                f"Event {event_index + 1}: "
                "invalid KG-rank count."
            )

        # -----------------------------------------------------------
        # CRITICAL FIX
        #
        # Convert movie IDs into frozen KGSF score-column positions.
        # -----------------------------------------------------------

        score_columns = np.asarray(
            [
                movie_to_column[movie_id]
                for movie_id in ids
            ],
            dtype=np.int64
        )

        frozen_scores = np.asarray(
            score_matrix[event_index, score_columns],
            dtype=np.float64
        )

        # -----------------------------------------------------------
        # Optional consistency check:
        #
        # The exported Top-500 scores should correspond to the same
        # frozen KGSF score matrix.
        #
        # We do NOT fail merely because of tiny numerical differences.
        # -----------------------------------------------------------

        if not np.allclose(
            frozen_scores,
            exported_scores,
            rtol=1e-5,
            atol=1e-5
        ):

            max_difference = float(
                np.max(
                    np.abs(
                        frozen_scores
                        - exported_scores
                    )
                )
            )

            print(
                f"WARNING: event {event_index + 1} "
                f"Top-500 score discrepancy. "
                f"max_abs_diff={max_difference:.8g}"
            )

        # -----------------------------------------------------------
        # KGSF probability
        # -----------------------------------------------------------

        p = softmax(
            frozen_scores
        )

        # -----------------------------------------------------------
        # KG confidence
        # -----------------------------------------------------------

        kg_confidence = np.zeros(
            TOP_K,
            dtype=np.float64
        )

        supported = ranks > 0

        kg_confidence[supported] = np.exp(
            -ranks[supported].astype(np.float64)
            / KG_TAU
        )

        # -----------------------------------------------------------
        # Operational q
        # -----------------------------------------------------------

        q_raw = (
            p
            * (1.0 + kg_confidence)
        )

        q_sum = float(
            np.sum(q_raw)
        )

        if (
            q_sum <= 0
            or not np.isfinite(q_sum)
        ):
            q = (
                np.ones(TOP_K)
                / TOP_K
            )

        else:
            q = q_raw / q_sum

        # -----------------------------------------------------------
        # Store
        # -----------------------------------------------------------

        for j in range(TOP_K):

            records.append(
                {
                    "event_id": event_index + 1,
                    "candidate_rank": j + 1,
                    "movie_id": ids[j],

                    "k_gsf_score": float(
                        frozen_scores[j]
                    ),

                    "k_gsf_candidate_probability": float(
                        p[j]
                    ),

                    "kg_rank": int(
                        ranks[j]
                    ),

                    "kg_confidence": float(
                        kg_confidence[j]
                    ),

                    "q_ei": float(
                        q[j]
                    ),
                }
            )

    return pd.DataFrame(records)


# =====================================================================
# q VALIDATION
# =====================================================================

def validate_q(q_df):

    event_mass = (
        q_df
        .groupby("event_id")["q_ei"]
        .sum()
    )

    if len(event_mass) != EXPECTED_EVENTS:
        raise RuntimeError(
            "q does not contain all 4,003 events."
        )

    if not np.allclose(
        event_mass.to_numpy(),
        1.0,
        atol=1e-8
    ):
        raise RuntimeError(
            "q[e,i] does not sum to 1 within every event."
        )

    if not np.isfinite(
        q_df["q_ei"]
    ).all():
        raise RuntimeError(
            "Non-finite q values detected."
        )

    if (
        q_df["q_ei"] < 0
    ).any():
        raise RuntimeError(
            "Negative q values detected."
        )

    stats = {

        "events":
            int(
                q_df["event_id"].nunique()
            ),

        "candidate_instances":
            int(len(q_df)),

        "q_sum_min":
            float(event_mass.min()),

        "q_sum_max":
            float(event_mass.max()),

        "q_sum_mean":
            float(event_mass.mean()),

        "q_min":
            float(q_df["q_ei"].min()),

        "q_max":
            float(q_df["q_ei"].max()),

        "q_mean":
            float(q_df["q_ei"].mean()),

        "positive_q_instances":
            int(
                (
                    q_df["q_ei"] > 0
                ).sum()
            ),
    }

    return stats


# =====================================================================
# LEAKAGE AUDIT
# =====================================================================

def leakage_audit():

    return {

        "target_movie_id_present_in_input":
            True,

        "target_movie_id_used_to_construct_q":
            False,

        "future_dialogue_used":
            False,

        "training_performed":
            False,

        "retrieval_performed":
            False,

        "ranking_changed":
            False,

        "parameter_optimization_on_independent_test":
            False,

        "q_source":
            [
                "frozen KGSF event-time scores",
                "frozen KG evidence"
            ],

        "leakage_status":
            "NO_TARGET_OR_FUTURE_CONTEXT_USED",
    }


# =====================================================================
# MANIFEST
# =====================================================================

def write_outputs(
    q_df,
    q_stats,
    leakage,
    score_matrix,
    score_movie_ids,
    score_event_ids,
    kg_column
):

    q_path = (
        OUT_DIR
        / "event_candidate_q_ei.csv"
    )

    q_df.to_csv(
        q_path,
        index=False
    )

    with open(
        OUT_DIR / "q_statistics.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            q_stats,
            f,
            indent=2
        )

    with open(
        OUT_DIR / "leakage_audit.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            leakage,
            f,
            indent=2
        )

    manifest = {

        "experiment":
            "22.31",

        "description":
            "Leakage-free operational contextual relevance "
            "construction from frozen KGSF and KG evidence.",

        "read_only":
            True,

        "training":
            False,

        "retrieval":
            False,

        "ranking_changes":
            False,

        "target_used_for_q":
            False,

        "future_dialogue_used":
            False,

        "events":
            EXPECTED_EVENTS,

        "catalog_items":
            EXPECTED_CATALOG,

        "candidate_pool":
            TOP_K,

        "kgsf_score_shape":
            list(score_matrix.shape),

        "kgsf_movie_id_count":
            int(len(score_movie_ids)),

        "kgsf_event_id_count":
            int(len(score_event_ids)),

        "kg_evidence_column":
            kg_column,

        "kg_tau":
            KG_TAU,

        "q_definition":
            (
                "softmax(frozen KGSF candidate scores) "
                "times (1 + exp(-KG_rank/tau)), "
                "followed by event-level normalization."
            ),

        "q_status":
            "OPERATIONAL_LEAKAGE_FREE_SIGNAL",

        "validated_contextual_relevance":
            False,

        "scientific_limitation":
            (
                "q is an operational relevance signal derived "
                "from frozen model and KG evidence. It has not "
                "yet been externally calibrated as a contextual "
                "relevance probability."
            ),

        "q_statistics":
            q_stats,

        "leakage_audit":
            leakage,
    }

    with open(
        OUT_DIR / "experiment_manifest.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            manifest,
            f,
            indent=2
        )


# =====================================================================
# MAIN
# =====================================================================

def main():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 72)
    print("EXPERIMENT 22.31")
    print("LEAKAGE-FREE CONTEXTUAL RELEVANCE CONSTRUCTION")
    print("=" * 72)

    # ---------------------------------------------------------------
    # Input validation
    # ---------------------------------------------------------------

    for path in [
        TOP500_PATH,
        FULL_SCORE_PATH,
        KG_RANK_PATH,
        MAPPING_PATH,
    ]:

        if not path.is_file():

            raise FileNotFoundError(
                f"Required frozen input not found:\n{path}"
            )

    # ---------------------------------------------------------------
    # Load frozen inputs
    # ---------------------------------------------------------------

    mapping = load_mapping()

    top500, candidate_ids, candidate_scores = (
        load_top500()
    )

    kg_df, kg_ranks, kg_column = (
        load_kg_ranks()
    )

    (
        score_matrix,
        score_movie_ids,
        score_event_ids,
    ) = load_full_scores()

    # ---------------------------------------------------------------
    # Event alignment
    # ---------------------------------------------------------------

    validate_event_alignment(
        mapping,
        top500,
        kg_df,
        score_event_ids
    )

    # ---------------------------------------------------------------
    # CRITICAL MOVIE-ID ALIGNMENT
    # ---------------------------------------------------------------

    movie_to_column = (
        build_movie_to_score_column(
            score_movie_ids
        )
    )

    validate_candidate_score_alignment(
        candidate_ids,
        movie_to_column
    )

    # ---------------------------------------------------------------
    # Construct q
    # ---------------------------------------------------------------

    print(
        "\n========== CONSTRUCTING q[e,i] =========="
    )

    print(
        "Target information: NOT USED"
    )

    print(
        "Future dialogue: NOT USED"
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

    q_df = construct_q(
        candidate_ids,
        candidate_scores,
        kg_ranks,
        score_matrix,
        movie_to_column
    )

    # ---------------------------------------------------------------
    # Validate q
    # ---------------------------------------------------------------

    q_stats = validate_q(
        q_df
    )

    leakage = leakage_audit()

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    write_outputs(
        q_df,
        q_stats,
        leakage,
        score_matrix,
        score_movie_ids,
        score_event_ids,
        kg_column
    )

    # ---------------------------------------------------------------
    # Decision
    # ---------------------------------------------------------------

    decision = {

        "operational_q_available":
            True,

        "q_event_count":
            EXPECTED_EVENTS,

        "q_candidate_instances":
            EXPECTED_EVENTS * TOP_K,

        "q_normalization":
            "PASS",

        "q_nonnegative":
            True,

        "q_finite":
            True,

        "q_target_leakage":
            False,

        "q_future_context_leakage":
            False,

        "q_is_calibrated_contextual_relevance":
            False,

        "q_is_operational_relevance_signal":
            True,

        "q_can_support_R_cg_reconstruction":
            True,

        "ranking_modified":
            False,

        "training_performed":
            False,

        "retrieval_performed":
            False,

        "next_experiment":
            (
                "22.32 — reconstruct R_cg and rho_cg "
                "from q_ei without modifying rankings."
            ),
    }

    with open(
        OUT_DIR / "decision.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            decision,
            f,
            indent=2
        )

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------

    print("\n")
    print("=" * 72)
    print("22.31 COMPLETE")
    print("=" * 72)

    print(
        f"Events: "
        f"{q_stats['events']}"
    )

    print(
        f"Candidate instances: "
        f"{q_stats['candidate_instances']}"
    )

    print(
        f"q range: "
        f"{q_stats['q_min']:.10f} "
        f"to "
        f"{q_stats['q_max']:.10f}"
    )

    print(
        f"Mean q: "
        f"{q_stats['q_mean']:.10f}"
    )

    print(
        "q normalization: PASS"
    )

    print(
        "Movie-ID → KGSF score-column alignment: PASS"
    )

    print(
        "Target leakage: NO"
    )

    print(
        "Future-context leakage: NO"
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
        "q status: OPERATIONAL_LEAKAGE_FREE_SIGNAL"
    )

    print(
        "Calibration status: "
        "NOT YET VALIDATED AS CONTEXTUAL RELEVANCE"
    )

    print(
        "Next: "
        "22.32 — reconstruct R_cg and rho_cg"
    )

    print(
        "Output:",
        OUT_DIR.resolve()
    )


if __name__ == "__main__":
    main()