
import ast
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path("results_final_test_m2")

files = {
    "top500": BASE / "experiment_22_27A/kgsf_independent_test_top500.csv",
    "kg": BASE / "experiment_22_27A2/kgsf_independent_test_kg_ranks.csv",
    "qei": BASE / "experiment_22_31/event_candidate_q_ei.csv",
    "category": BASE / "experiment_22_32/R_cg_rho_cg_reconstruction.csv",
}

def parse_list(x):
    if isinstance(x, list):
        return x
    return ast.literal_eval(x)

print("\n=== 22.35: CROSS-STAGE DATA INTEGRITY AUDIT ===")

# 1. KGSF candidate and score alignment
df = pd.read_csv(files["top500"])

length_errors = 0
duplicate_candidates = 0
target_missing = 0
score_order_errors = 0

for _, row in df.iterrows():
    ids = parse_list(row["top_movie_ids"])
    scores = parse_list(row["top_scores"])

    length_errors += len(ids) != len(scores)
    duplicate_candidates += len(ids) != len(set(ids))
    target_missing += row["target_movie_id"] not in ids
    score_order_errors += not all(
        scores[i] >= scores[i + 1]
        for i in range(len(scores) - 1)
    )

print("\n[1] KGSF")
print("Events:", len(df))
print("Candidate-score length errors:", length_errors)
print("Duplicate candidate rows:", duplicate_candidates)
print("Targets missing from top-500:", target_missing)
print("Score ordering errors:", score_order_errors)

# 2. KG rank-to-movie alignment
kg = pd.read_csv(files["kg"])
rank_errors = 0
length_errors = 0
positive_rank_counts = {}

for _, row in kg.iterrows():
    candidates = parse_list(row["candidate_movie_ids"])

    for rank_col, ids_col in [
        ("R0_filtered_candidate_kg_ranks",
         "R0_filtered_retrieved_movie_ids"),
        ("R1_raw_candidate_kg_ranks",
         "R1_raw_retrieved_movie_ids"),
    ]:
        ranks = parse_list(row[rank_col])
        retrieved = parse_list(row[ids_col])

        length_errors += len(candidates) != len(ranks)

        for movie_id, rank in zip(candidates, ranks):
            if rank > 0:
                if rank > len(retrieved):
                    rank_errors += 1
                elif retrieved[rank - 1] != movie_id:
                    rank_errors += 1

        positive_rank_counts[rank_col] = (
            positive_rank_counts.get(rank_col, 0)
            + sum(r > 0 for r in ranks)
        )

print("\n[2] KG RANKS")
print("Rows:", len(kg))
print("Candidate-rank length errors:", length_errors)
print("Rank-to-movie mismatches:", rank_errors)
print("Positive candidate ranks:", positive_rank_counts)

# 3. Candidate relevance
qei = pd.read_csv(files["qei"])

print("\n[3] Q_EI")
print("Rows:", len(qei))
print("Duplicate event-candidate pairs:",
      qei.duplicated(["event_id", "movie_id"]).sum())
print("Missing values:\n", qei.isna().sum())
print("Q_EI statistics:\n", qei["q_ei"].describe())
print("KG confidence statistics:\n",
      qei["kg_confidence"].describe())

# 4. Category reconstruction
cat = pd.read_csv(files["category"])

print("\n[4] CATEGORY RECONSTRUCTION")
print("Rows:", len(cat))
print("Categories:", cat["category"].nunique())
print("Missing values:\n", cat.isna().sum())
print("Negative relevance values:",
      (cat["R_cg"] < 0).sum())
print("Negative exposure values:",
      (cat["E_cg"] < 0).sum())
print("Zero-exposure rows:",
      (cat["E_cg"] == 0).sum())
print("Positive-relevance, zero-exposure rows:",
      ((cat["R_cg"] > 0) & (cat["E_cg"] == 0)).sum())

print("\nCategory sample:")
print(cat.head(15).to_string(index=False))

print("\n=== AUDIT COMPLETE ===")

