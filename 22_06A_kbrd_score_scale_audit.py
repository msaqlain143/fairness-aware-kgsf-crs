
import ast
import json
import numpy as np
import pandas as pd

from pathlib import Path


# ============================================================
# EXPERIMENT 22.6A
# KBRD SCORE-SCALE AND EVENT-ALIGNMENT AUDIT
# ============================================================

ROOT = Path("results_corrected/experiment_22_04B")

SCORE_FILE = ROOT / "kbrd_validation_full_scores.npz"
TOP_FILE = ROOT / "kbrd_validation_top500.csv"
MAPPING_FILE = Path(
    "results_corrected/m2_event_conversation_mapping.csv"
)

OUTPUT_DIR = Path(
    "results_corrected/experiment_22_06A"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LAMBDA_REFERENCE = 1.2
TOP_K = 500


def parse_array(value, dtype):
    return np.asarray(ast.literal_eval(value), dtype=dtype)


def percentile_summary(values):
    values = np.asarray(values, dtype=np.float64)

    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


print("=" * 70)
print("EXPERIMENT 22.6A — KBRD SCORE-SCALE AUDIT")
print("=" * 70)


# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

z = np.load(SCORE_FILE)

scores = z["scores"]
movie_ids = z["movie_ids"].astype(np.int64)

top = pd.read_csv(TOP_FILE)
mapping = pd.read_csv(MAPPING_FILE)

print("\n========== DATA DIMENSIONS ==========")

print("Score matrix:", scores.shape)
print("Catalog size:", len(movie_ids))
print("Candidate records:", len(top))
print("Mapping records:", len(mapping))


# ------------------------------------------------------------
# 2. BASIC INTEGRITY CHECKS
# ------------------------------------------------------------

assert scores.ndim == 2
assert scores.shape[1] == len(movie_ids)
assert scores.shape[0] == len(top)

assert np.isfinite(scores).all()
assert len(np.unique(movie_ids)) == len(movie_ids)

assert top["event_id"].is_unique
assert mapping["event_id"].is_unique

assert np.array_equal(
    top["event_id"].to_numpy(),
    np.arange(1, len(top) + 1)
)

print("\nBasic matrix integrity: PASS")


# ------------------------------------------------------------
# 3. EVENT AND CONVERSATION ALIGNMENT
# ------------------------------------------------------------

merged = top.merge(
    mapping,
    on="event_id",
    how="left",
    suffixes=("_export", "_mapping"),
    validate="one_to_one"
)

assert len(merged) == len(top)

assert merged["conv_id_mapping"].notna().all()
assert merged["target_movie_id_mapping"].notna().all()

conversation_match = np.array_equal(
    merged["conv_id_export"].to_numpy(),
    merged["conv_id_mapping"].to_numpy()
)

target_match = np.array_equal(
    merged["target_movie_id_export"].to_numpy(),
    merged["target_movie_id_mapping"].to_numpy()
)

print("Conversation alignment:", conversation_match)
print("Target alignment:", target_match)

assert conversation_match
assert target_match


# ------------------------------------------------------------
# 4. VERIFY TOP-500 SCORES AGAINST FULL MATRIX
# ------------------------------------------------------------

movie_id_to_col = {
    int(mid): j
    for j, mid in enumerate(movie_ids)
}

score_match_errors = 0
ordering_errors = 0
membership_errors = 0
target_rank_errors = 0

event_statistics = []

for i, row in top.iterrows():

    exported_ids = parse_array(
        row["top_movie_ids"], np.int64
    )

    exported_scores = parse_array(
        row["top_scores"], np.float64
    )

    assert len(exported_ids) == TOP_K
    assert len(exported_scores) == TOP_K

    if len(np.unique(exported_ids)) != TOP_K:
        membership_errors += 1
        continue

    if not np.all(exported_scores[:-1] >= exported_scores[1:]):
        ordering_errors += 1

    try:
        cols = np.array([
            movie_id_to_col[int(mid)]
            for mid in exported_ids
        ])
    except KeyError:
        membership_errors += 1
        continue

    row_scores = scores[i]

    actual_scores = row_scores[cols]

    if not np.allclose(
        exported_scores,
        actual_scores,
        rtol=1e-6,
        atol=1e-7
    ):
        score_match_errors += 1

    # Strict-greater target rank
    target_id = int(row["target_movie_id"])

    if target_id not in movie_id_to_col:
        target_rank_errors += 1
        continue

    target_col = movie_id_to_col[target_id]
    target_score = row_scores[target_col]

    target_rank = 1 + int(
        np.sum(row_scores > target_score)
    )

    if target_rank != int(row["target_rank_full_catalog"]):
        target_rank_errors += 1

    # --------------------------------------------------------
    # SCORE DISTRIBUTION
    # --------------------------------------------------------

    top500_cutoff = float(exported_scores[-1])

    row_std = float(np.std(row_scores))
    row_range = float(np.max(row_scores) - np.min(row_scores))
    top_gap = float(exported_scores[0] - top500_cutoff)

    event_statistics.append({

        "event_id": int(row["event_id"]),
        "conv_id": int(row["conv_id"]),

        "score_min": float(np.min(row_scores)),
        "score_max": float(np.max(row_scores)),
        "score_mean": float(np.mean(row_scores)),
        "score_std": row_std,
        "score_range": row_range,

        "top1_score": float(exported_scores[0]),
        "top500_cutoff": top500_cutoff,
        "top1_top500_gap": top_gap,

        "lambda_over_std": (
            LAMBDA_REFERENCE / row_std
            if row_std > 0 else np.nan
        ),

        "lambda_over_top_gap": (
            LAMBDA_REFERENCE / top_gap
            if top_gap > 0 else np.nan
        ),

        "target_rank": target_rank
    })


# ------------------------------------------------------------
# 5. SAVE EVENT-LEVEL AUDIT
# ------------------------------------------------------------

event_df = pd.DataFrame(event_statistics)

event_df.to_csv(
    OUTPUT_DIR / "kbrd_event_score_statistics.csv",
    index=False
)


# ------------------------------------------------------------
# 6. SCORE SCALE SUMMARY
# ------------------------------------------------------------

summary = {

    "experiment": "22.6A_KBRD_score_scale_audit",

    "events": int(len(top)),
    "catalog_size": int(len(movie_ids)),

    "matrix_finite": bool(np.isfinite(scores).all()),
    "catalog_ids_unique": bool(
        len(np.unique(movie_ids)) == len(movie_ids)
    ),

    "conversation_alignment": bool(conversation_match),
    "target_alignment": bool(target_match),

    "score_match_errors": int(score_match_errors),
    "ordering_errors": int(ordering_errors),
    "membership_errors": int(membership_errors),
    "target_rank_errors": int(target_rank_errors),

    "lambda_reference": LAMBDA_REFERENCE,

    "score_std_distribution": percentile_summary(
        event_df["score_std"]
    ),

    "score_range_distribution": percentile_summary(
        event_df["score_range"]
    ),

    "top1_top500_gap_distribution": percentile_summary(
        event_df["top1_top500_gap"]
    ),

    "lambda_over_std_distribution": percentile_summary(
        event_df["lambda_over_std"]
    ),

    "lambda_over_top_gap_distribution": percentile_summary(
        event_df["lambda_over_top_gap"]
    ),

    "fraction_lambda_exceeds_event_std": float(
        np.mean(
            event_df["lambda_over_std"] > 1.0
        )
    ),

    "fraction_lambda_exceeds_top500_gap": float(
        np.mean(
            event_df["lambda_over_top_gap"] > 1.0
        )
    )
}


with open(
    OUTPUT_DIR / "kbrd_score_scale_audit_summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(summary, f, indent=4)


# ------------------------------------------------------------
# 7. FINAL REPORT
# ------------------------------------------------------------

print("\n========== SCORE SCALE SUMMARY ==========")

print(
    "Score standard deviation:",
    summary["score_std_distribution"]
)

print(
    "Full score range:",
    summary["score_range_distribution"]
)

print(
    "Top-1 to Top-500 gap:",
    summary["top1_top500_gap_distribution"]
)

print(
    "Lambda / score std:",
    summary["lambda_over_std_distribution"]
)

print(
    "Lambda exceeds event std:",
    summary["fraction_lambda_exceeds_event_std"]
)

print(
    "Lambda exceeds Top-500 gap:",
    summary["fraction_lambda_exceeds_top500_gap"]
)


print("\n========== INTEGRITY RESULTS ==========")

print("Score mismatch errors:", score_match_errors)
print("Ordering errors:", ordering_errors)
print("Membership errors:", membership_errors)
print("Target rank errors:", target_rank_errors)


passed = (
    conversation_match
    and target_match
    and score_match_errors == 0
    and ordering_errors == 0
    and membership_errors == 0
    and target_rank_errors == 0
    and len(event_df) == len(top)
)

print("\n========== FINAL STATUS ==========")

if passed:

    print("PASS: KBRD score and event integrity verified.")

    print(
        "Next: inspect score-scale statistics before "
        "designing the calibration experiment."
    )

else:

    print("FAIL: Investigate integrity discrepancies.")

print("\nOutput directory:", OUTPUT_DIR)
print("=" * 70)
