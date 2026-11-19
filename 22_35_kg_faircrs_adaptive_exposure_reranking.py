
import numpy as np
import pandas as pd
from pathlib import Path

root = Path("results_corrected/experiment_22_04B")

score_path = root / "kbrd_validation_full_scores.npz"
top_path = root / "kbrd_validation_top500.csv"

print("=" * 70)
print("VALIDATION KGSF SCORE ALIGNMENT AUDIT")
print("=" * 70)

data = np.load(score_path, allow_pickle=False)

print("\nNPZ KEYS:")
for key in data.files:
    arr = data[key]
    print(
        key,
        "shape =", arr.shape,
        "dtype =", arr.dtype
    )

print("\nTOP-500 EXPORT:")
df = pd.read_csv(top_path)

print("Rows:", len(df))
print("Columns:", list(df.columns))

print("\nFIRST EVENT:")
for col in [
    "event_id",
    "conv_id",
    "target_movie_id",
    "top_movie_ids",
    "top_scores"
]:
    if col in df.columns:
        print(col, "=", str(df.iloc[0][col])[:300])

print("\nEVENT ID RANGE:")
print(df.event_id.min(), df.event_id.max())
print("Unique events:", df.event_id.nunique())

print("\nSCORE MATRIX ALIGNMENT:")
for key in ["scores", "movie_ids", "event_ids"]:
    if key in data.files:
        print(key, data[key][:5])

print("\nCOMPLETE")



