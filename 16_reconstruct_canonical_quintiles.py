import pickle
import numpy as np

with open("catalog_audit_corrected.pkl", "rb") as f:
    c = pickle.load(f)

movie_ids = list(map(int, c["movie_ids"]))

freq = {
    int(k): int(v)
    for k, v in c["freq_dict"].items()
}

stored = {
    int(k): int(v)
    for k, v in c["item2group"].items()
}

sorted_items = list(map(int, c["sorted_items"]))

N = len(movie_ids)

print("=" * 70)
print("EXPERIMENT 16: INDEPENDENT QUINTILE RECONSTRUCTION")
print("=" * 70)

# 1. Reconstruct frequency/entity-ID ordering.
independent_order = sorted(
    movie_ids,
    key=lambda x: (freq[x], x)
)

print("\n1. SORTING VALIDATION")

print("Catalog size:", N)
print("Stored ordering matches independent ordering:",
      sorted_items == independent_order)

if sorted_items != independent_order:
    mismatches = sum(
        a != b for a, b in zip(sorted_items, independent_order)
    )
    print("Ordering mismatches:", mismatches)

# 2. Reconstruct using equal-sized contiguous partitions.
chunks = np.array_split(
    np.asarray(independent_order, dtype=np.int64),
    5
)

split_groups = {}

for group, chunk in enumerate(chunks):
    for movie in chunk:
        split_groups[int(movie)] = group

print("\n2. ARRAY-SPLIT RECONSTRUCTION")

print("Group sizes:", [len(x) for x in chunks])

split_mismatches = [
    movie for movie in movie_ids
    if stored[movie] != split_groups[movie]
]

print("Mismatched assignments:", len(split_mismatches))

# 3. Alternative rank-based formula.
rank_groups = {
    movie: min((rank * 5) // N, 4)
    for rank, movie in enumerate(independent_order)
}

rank_mismatches = [
    movie for movie in movie_ids
    if stored[movie] != rank_groups[movie]
]

print("\n3. RANK-FORMULA RECONSTRUCTION")

print("Mismatched assignments:", len(rank_mismatches))

# 4. Examine observed group boundaries.
print("\n4. OBSERVED GROUP BOUNDARIES")

for group in range(5):

    members = [
        movie for movie in independent_order
        if stored[movie] == group
    ]

    print(
        f"G{group}: size={len(members)}, "
        f"first={members[0]}, "
        f"last={members[-1]}, "
        f"frequency_range="
        f"{freq[members[0]]}-{freq[members[-1]]}"
    )

# 5. Final diagnostic.
print("\n5. FINAL DIAGNOSTIC")

if sorted_items != independent_order:
    print("FAIL: Sorting procedure is not independently reproduced.")

elif len(split_mismatches) == 0:
    print("PASS: Stored groups match independent array_split.")

elif len(rank_mismatches) == 0:
    print("PASS: Stored groups match rank-based floor partition.")

else:
    print("UNRESOLVED: Neither reconstruction matches.")
    print("Inspect the original catalog construction procedure.")

print("\nEXPERIMENT 16 DIAGNOSTIC COMPLETE.")
