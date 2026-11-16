
import ast
import json
from pathlib import Path

import pandas as pd

INPUT = Path(
    "results_final_test_m2/experiment_22_27A2/"
    "kgsf_independent_test_kg_ranks.csv"
)

df = pd.read_csv(INPUT)


def parse_array(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    return ast.literal_eval(value)


checks = {
    "candidate_length_errors": 0,
    "R0_rank_length_errors": 0,
    "R1_rank_length_errors": 0,
    "R0_retrieval_count_errors": 0,
    "R1_retrieval_count_errors": 0,
    "R0_rank_positive_mismatch": 0,
    "R1_rank_positive_mismatch": 0,
    "R0_R1_rank_differences": 0,
    "R0_R1_retrieval_differences": 0,
}

for _, row in df.iterrows():

    candidates = parse_array(row["candidate_movie_ids"])
    r0_ranks = parse_array(
        row["R0_filtered_candidate_kg_ranks"]
    )
    r1_ranks = parse_array(
        row["R1_raw_candidate_kg_ranks"]
    )

    r0_ids = parse_array(
        row["R0_filtered_retrieved_movie_ids"]
    )
    r1_ids = parse_array(
        row["R1_raw_retrieved_movie_ids"]
    )

    if len(candidates) != row["candidate_count"]:
        checks["candidate_length_errors"] += 1

    if len(r0_ranks) != len(candidates):
        checks["R0_rank_length_errors"] += 1

    if len(r1_ranks) != len(candidates):
        checks["R1_rank_length_errors"] += 1

    if len(r0_ids) != row["R0_filtered_retrieved_count"]:
        checks["R0_retrieval_count_errors"] += 1

    if len(r1_ids) != row["R1_raw_retrieved_count"]:
        checks["R1_retrieval_count_errors"] += 1

    if len(r0_ranks) == len(candidates):
        if sum(x > 0 for x in r0_ranks) != sum(
            x in set(r0_ids) for x in candidates
        ):
            checks["R0_rank_positive_mismatch"] += 1

    if len(r1_ranks) == len(candidates):
        if sum(x > 0 for x in r1_ranks) != sum(
            x in set(r1_ids) for x in candidates
        ):
            checks["R1_rank_positive_mismatch"] += 1

    if r0_ranks != r1_ranks:
        checks["R0_R1_rank_differences"] += 1

    if r0_ids != r1_ids:
        checks["R0_R1_retrieval_differences"] += 1


print("\n========== RETRIEVAL INTEGRITY AUDIT ==========")
print("Total events:", len(df))

for name, value in checks.items():
    print(f"{name}: {value}")

print("\n========== SUMMARY ==========")

summary = {
    "events": len(df),
    "checks": checks,
    "R0_R1_ranks_identical": (
        checks["R0_R1_rank_differences"] == 0
    ),
    "R0_R1_retrievals_identical": (
        checks["R0_R1_retrieval_differences"] == 0
    ),
}

output = Path("results_corrected/experiment_22_08")
output.mkdir(parents=True, exist_ok=True)

with open(
    output / "retrieval_integrity_summary.json",
    "w",
    encoding="utf-8"
) as f:
    json.dump(summary, f, indent=4)

print("\nSaved:", output)
