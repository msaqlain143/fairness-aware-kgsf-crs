"""Paired conversation-clustered held-out comparison:
A1_frequency_3 versus original KGSF_only.
No model inference or training.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("results_final_test_m2")
INPUT = ROOT / "adaptive_m2_with_kgsf_baseline_events.csv"
MAPPING = ROOT / "m2_test_event_conversation_mapping.csv"
ORIGINAL = ROOT / "adaptive_m2_final_test_events_clustered.csv"
OUTPUT = ROOT / "m2_test_vs_kgsf_cluster_bootstrap.csv"

BASELINE = "KGSF_only"
PROPOSED = "A1_frequency_3"
GROUPS = ("all", "zero_shot", "one_shot", "other")
METRICS = ("hit10", "hit50", "ndcg10")

REPS = 10000
SEED = 20261028
BATCH_SIZE = 100

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} already exists; refusing to overwrite."
    )

def read_rows(path):
    with path.open(
        newline="", encoding="utf-8-sig"
    ) as f:
        return list(csv.DictReader(f))

rows = read_rows(INPUT)
map_rows = read_rows(MAPPING)
original_rows = read_rows(ORIGINAL)

if len(rows) != 24018:
    raise RuntimeError(
        f"Expected 24018 baseline-evaluation rows; found {len(rows)}."
    )

# Validate the previously verified event-conversation mapping.
mapping = {}

for row in map_rows:
    eid = int(row["event_id"])

    if eid in mapping:
        raise RuntimeError(f"Duplicate mapping: {eid}")

    mapping[eid] = (
        str(row["conv_id"]),
        int(row["target_movie_id"]),
    )

if set(mapping) != set(range(1, 4004)):
    raise RuntimeError("Mapping does not contain exactly 4003 events.")

# Verify the proposed method against the original frozen test CSV.
historical = {}

for row in original_rows:
    if row["method"] != PROPOSED:
        continue

    eid = int(row["event_id"])
    if eid in historical:
        raise RuntimeError(f"Duplicate historical event {eid}")
    historical[eid] = row

if len(historical) != 4003:
    raise RuntimeError("Historical proposed-method rows incomplete.")

events = defaultdict(dict)

for row in rows:
    method = row["method"]

    if method not in (BASELINE, PROPOSED):
        continue

    eid = int(row["event_id"])

    if method in events[eid]:
        raise RuntimeError(
            f"Duplicate {method} for event {eid}"
        )

    events[eid][method] = row

if set(events) != set(mapping):
    raise RuntimeError("Event IDs differ from mapping.")

for eid, pair in events.items():
    if set(pair) != {BASELINE, PROPOSED}:
        raise RuntimeError(f"Incomplete pair for event {eid}")

    a = pair[BASELINE]
    b = pair[PROPOSED]
    conv_id, target = mapping[eid]
    old = historical[eid]

    for row in (a, b):
        if int(row["target_movie_id"]) != target:
            raise RuntimeError(f"Target mismatch: event {eid}")

    if a["frequency_group"] != b["frequency_group"]:
        raise RuntimeError(f"Group mismatch: event {eid}")

    for field in (
        "target_movie_id",
        "target_frequency",
        "frequency_group",
        "rank_top50",
        "hit10",
        "hit50",
        "ndcg10",
        "top10_movie_ids",
        "top10_group_ids",
    ):
        if b[field] != old[field]:
            raise RuntimeError(
                f"Historical prediction changed: "
                f"event {eid}, field {field}"
            )

    if old["conv_id"] != conv_id:
        raise RuntimeError(
            f"Conversation mismatch: event {eid}"
        )

conv_ids = sorted({
    mapping[eid][0]
    for eid in events
})

conv_position = {
    cid: index
    for index, cid in enumerate(conv_ids)
}

C = len(conv_ids)
G = len(GROUPS)
M = len(METRICS)

counts = np.zeros((C, G), dtype=np.int64)
sums_base = np.zeros((C, G, M), dtype=np.float64)
sums_new = np.zeros((C, G, M), dtype=np.float64)

group_positions = {
    name: index
    for index, name in enumerate(GROUPS)
}

for eid, pair in events.items():
    a = pair[BASELINE]
    b = pair[PROPOSED]

    ci = conv_position[mapping[eid][0]]
    group = a["frequency_group"]

    if group not in ("zero_shot", "one_shot", "other"):
        raise RuntimeError(
            f"Unexpected frequency group: {group}"
        )

    applicable = (
        group_positions["all"],
        group_positions[group],
    )

    for gi in applicable:
        counts[ci, gi] += 1

        for mi, metric in enumerate(METRICS):
            av = float(a[metric])
            bv = float(b[metric])

            if not np.isfinite(av) or not np.isfinite(bv):
                raise RuntimeError("Non-finite metric encountered.")

            sums_base[ci, gi, mi] += av
            sums_new[ci, gi, mi] += bv

if C != 1283 or int(counts[:, 0].sum()) != 4003:
    raise RuntimeError("Conversation/event accounting mismatch.")

group_totals = counts.sum(axis=0)
if any(group_totals == 0):
    raise RuntimeError("Empty evaluation group.")

observed_base = sums_base.sum(axis=0) / group_totals[:, None]
observed_new = sums_new.sum(axis=0) / group_totals[:, None]
observed_delta = observed_new - observed_base

rng = np.random.default_rng(SEED)
replicates = np.empty(
    (REPS, G, M), dtype=np.float64
)

for start in range(0, REPS, BATCH_SIZE):
    end = min(start + BATCH_SIZE, REPS)
    n = end - start

    # Resample entire conversations, preserving within-conversation
    # events and pairing the two recommendation methods.
    sampled = rng.integers(
        0, C, size=(n, C)
    )

    denominator = counts[sampled].sum(axis=1)
    new_sum = sums_new[sampled].sum(axis=1)
    base_sum = sums_base[sampled].sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        replicates[start:end] = (
            (new_sum - base_sum)
            / denominator[:, :, None]
        )

if not np.all(np.isfinite(replicates)):
    raise RuntimeError(
        "Invalid bootstrap replicates encountered."
    )

print("\n=== HELD-OUT: KG-FAIRCRS VS ORIGINAL KGSF ===")
print("Baseline:", BASELINE)
print("Proposed:", PROPOSED)
print("Conversations:", C)
print("Events:", len(events))
print("Bootstrap replicates:", REPS)
print("Random seed:", SEED)
print("Historical proposed predictions: VERIFIED")

output_rows = []

for gi, group in enumerate(GROUPS):
    n_events = int(group_totals[gi])
    n_convs = int((counts[:, gi] > 0).sum())

    print(
        f"\n{group}: "
        f"events={n_events}, conversations={n_convs}"
    )

    for mi, metric in enumerate(METRICS):
        low, high = np.percentile(
            replicates[:, gi, mi], [2.5, 97.5]
        )

        av = float(observed_base[gi, mi])
        bv = float(observed_new[gi, mi])
        delta = float(observed_delta[gi, mi])

        unit = 100 if metric != "ndcg10" else 1
        label = "pp" if metric != "ndcg10" else "absolute"

        print(
            f"  {metric:6s}  "
            f"KGSF={av * unit:.5f}  "
            f"Ours={bv * unit:.5f}  "
            f"delta={delta * unit:+.5f} {label}  "
            f"95% CI=[{low * unit:+.5f}, "
            f"{high * unit:+.5f}]"
        )

        output_rows.append({
            "baseline": BASELINE,
            "method": PROPOSED,
            "group": group,
            "metric": metric,
            "n_events": n_events,
            "n_conversations": n_convs,
            "baseline_value": av,
            "proposed_value": bv,
            "delta": delta,
            "ci95_lower": float(low),
            "ci95_upper": float(high),
            "bootstrap_replicates": REPS,
            "seed": SEED,
            "resampling_unit": "conversation",
            "interval_type": "percentile",
            "multiple_comparison_adjusted": False,
        })

with OUTPUT.open(
    "x", newline="", encoding="utf-8"
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(output_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(output_rows)

print("\nPASS: Paired analysis completed.")
print("Saved:", OUTPUT)
print("Frozen test predictions unchanged.")
