
import ast
import numpy as np
import pandas as pd
from pathlib import Path

root = Path("results_corrected/experiment_22_04B")

npz = np.load(
    root / "kbrd_validation_full_scores.npz"
)

df = pd.read_csv(
    root / "kbrd_validation_top500.csv"
)

scores = npz["scores"]
movie_ids = npz["movie_ids"]
event_ids = npz["event_id"]

movie_to_col = {
    int(mid): i
    for i, mid in enumerate(movie_ids)
}

event_to_row = {
    int(eid): i
    for i, eid in enumerate(event_ids)
}

max_diffs = []
correlations = []
top500_matches = []
ordering_matches = []

for _, row in df.iterrows():

    event_id = int(row["event_id"])
    matrix_row = event_to_row[event_id]

    ids = ast.literal_eval(row["top_movie_ids"])
    exported = np.asarray(
        ast.literal_eval(row["top_scores"]),
        dtype=float
    )

    columns = np.asarray([
        movie_to_col[int(mid)]
        for mid in ids
    ])

    reconstructed = scores[matrix_row, columns]

    max_diffs.append(
        np.max(np.abs(reconstructed - exported))
    )

    correlations.append(
        np.corrcoef(reconstructed, exported)[0, 1]
    )

    # Check whether the candidate ordering agrees
    # with the full frozen score matrix.
    full_order = np.argsort(
        scores[matrix_row]
    )[::-1][:500]

    full_top_ids = movie_ids[full_order]

    top500_matches.append(
        set(map(int, ids)) == set(map(int, full_top_ids))
    )

    ordering_matches.append(
        np.array_equal(
            np.asarray(ids),
            full_top_ids
        )
    )

print("=" * 65)
print("22.35 SCORE CONSISTENCY AUDIT")
print("=" * 65)

print("Events:", len(df))
print("Maximum absolute score difference:", max(max_diffs))
print("Mean absolute score difference:", np.mean(max_diffs))
print("Median score correlation:", np.median(correlations))
print("Matching Top-500 candidate sets:", sum(top500_matches))
print("Matching candidate ordering:", sum(ordering_matches))

print("\nScore difference percentiles:")
print(np.percentile(max_diffs, [0, 25, 50, 75, 95, 100]))

print("\nCOMPLETE")
