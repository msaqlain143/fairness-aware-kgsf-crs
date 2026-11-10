
import ast
import numpy as np
import pandas as pd
from pathlib import Path

# ============================================================
# EXPERIMENT 22.5C — KBRD TIE-AWARE TOP-500 DIAGNOSIS
# ============================================================

root = Path("results_corrected/experiment_22_04B")

# Load exported full score matrix
z = np.load(root / "kbrd_validation_full_scores.npz")

scores = z["scores"]
movie_ids = z["movie_ids"]

# Load exported Top-500 recommendations
top = pd.read_csv(root / "kbrd_validation_top500.csv")

# ============================================================
# BASIC INTEGRITY CHECKS
# ============================================================

if scores.shape[0] != len(top):
    raise ValueError("Score matrix rows and Top-500 records do not match.")

if scores.shape[1] != len(movie_ids):
    raise ValueError("Score matrix columns and movie IDs do not match.")

if not np.isfinite(scores).all():
    raise ValueError("Non-finite values found in score matrix.")

if len(np.unique(movie_ids)) != len(movie_ids):
    raise ValueError("Duplicate movie IDs in catalog.")

# IMPORTANT: Build dictionary ONCE, outside the loop.
movie_id_to_col = {
    int(mid): j for j, mid in enumerate(movie_ids)
}

tie_rows = []
membership_errors = []
ordering_differences = 0

# ============================================================
# TOP-500 DIAGNOSIS
# ============================================================

for i, row in top.iterrows():

    exported_ids = np.array(
        ast.literal_eval(row["top_movie_ids"]),
        dtype=np.int64
    )

    exported_scores = np.array(
        ast.literal_eval(row["top_scores"]),
        dtype=np.float64
    )

    if len(exported_ids) != 500 or len(exported_scores) != 500:
        raise ValueError(f"Invalid Top-500 length at row {i}.")

    # Reconstruct score matrix columns
    try:
        exported_cols = np.array(
            [movie_id_to_col[int(mid)] for mid in exported_ids]
        )
    except KeyError as exc:
        raise ValueError(
            f"Exported movie ID absent from catalog at row {i}: {exc}"
        ) from exc

    # Actual score matrix values
    row_scores = scores[i]

    # Verify exported scores against original score matrix
    actual_exported_scores = row_scores[exported_cols]

    if not np.allclose(
        exported_scores,
        actual_exported_scores,
        rtol=1e-6,
        atol=1e-7
    ):
        raise ValueError(f"Exported scores mismatch at row {i}.")

    # Reconstruct expected Top-500 using stable sorting
    expected_indices = np.argsort(
        -row_scores,
        kind="stable"
    )[:500]

    expected_ids = movie_ids[expected_indices]

    if not np.array_equal(exported_ids, expected_ids):
        ordering_differences += 1

    # --------------------------------------------------------
    # TIE-AWARE CUTOFF ANALYSIS
    # --------------------------------------------------------

    cutoff = float(exported_scores[-1])

    strictly_above = int(np.sum(row_scores > cutoff))
    equal_cutoff = int(np.sum(row_scores == cutoff))

    selected_at_cutoff = int(
        np.sum(actual_exported_scores == cutoff)
    )

    exported_set = set(map(int, exported_ids))

    omitted_above = [
        int(movie_ids[j])
        for j in np.where(row_scores > cutoff)[0]
        if int(movie_ids[j]) not in exported_set
    ]

    selected_below = int(
        np.sum(actual_exported_scores < cutoff)
    )

    if omitted_above or selected_below:
        membership_errors.append({
            "row": int(i),
            "event_id": int(row["event_id"]),
            "omitted_above": len(omitted_above),
            "selected_below": selected_below
        })

    if equal_cutoff > 1:
        tie_rows.append({
            "event_id": int(row["event_id"]),
            "cutoff_score": cutoff,
            "strictly_above": strictly_above,
            "equal_cutoff": equal_cutoff,
            "selected_at_cutoff": selected_at_cutoff,
            "omitted_above": len(omitted_above)
        })

# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("EXPERIMENT 22.5C — TIE-AWARE DIAGNOSIS")
print("=" * 70)

print("Total events:", len(top))
print("Catalog size:", len(movie_ids))
print("Exact ordering differences:", ordering_differences)
print("Events with cutoff ties:", len(tie_rows))
print("Actual Top-500 membership errors:", len(membership_errors))

print("\n========== FIRST 10 TIE CASES ==========")

for item in tie_rows[:10]:
    print(item)

print("\n========== MEMBERSHIP ERROR CASES ==========")

for item in membership_errors[:20]:
    print(item)

print("\n========== FINAL STATUS ==========")

if not membership_errors:
    print("PASS: TOP-500 MEMBERSHIP IS VALID.")
    print("Ordering differences, if any, require tie-aware interpretation.")
else:
    print("FAIL: INVESTIGATION REQUIRED.")
    print("Exported candidates violate the Top-500 cutoff.")

print("\n" + "=" * 70)
