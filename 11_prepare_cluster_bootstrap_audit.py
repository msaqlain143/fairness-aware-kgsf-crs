
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("results_final_test_m2")

EVENTS = ROOT / "adaptive_m2_with_kgsf_baseline_events.csv"
MAPPING = ROOT / "m2_test_event_conversation_mapping.csv"
REPORT = ROOT / "adaptive_m2_with_kgsf_baseline.json"

METHODS = [
    "KGSF_only",
    "A0_fixed_025",
    "A1_frequency_1",
    "A2_frequency_context_1",
    "A1_frequency_3",
    "A2_frequency_context_3",
]

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

events = read_csv(EVENTS)
mapping_rows = read_csv(MAPPING)

with REPORT.open(encoding="utf-8") as f:
    report = json.load(f)

event_to_conv = {}
conv_to_events = defaultdict(list)

for row in mapping_rows:
    eid = int(row["event_id"])
    cid = int(row["conv_id"])

    if eid in event_to_conv:
        raise ValueError(f"Duplicate event mapping: {eid}")

    event_to_conv[eid] = cid
    conv_to_events[cid].append(eid)

print("========== 1. MAPPING INTEGRITY ==========")
print("Mapping rows:", len(mapping_rows))
print("Unique events:", len(event_to_conv))
print("Unique conversations:", len(conv_to_events))
print("Events per conversation:", Counter(
    len(x) for x in conv_to_events.values()
))

assert len(event_to_conv) == 4003
assert len(conv_to_events) == 1283

records = {}

for row in events:
    eid = int(row["event_id"])
    method = row["method"]
    key = (eid, method)

    if key in records:
        raise ValueError(f"Duplicate event-method: {key}")

    records[key] = row

expected = {
    (eid, method)
    for eid in event_to_conv
    for method in METHODS
}

assert set(records) == expected

print("\n========== 2. METHOD COMPLETENESS ==========")
print("Actual rows:", len(records))
print("Expected rows:", len(expected))
print("Methods:", METHODS)

print("\n========== 3. PAIRED EVENT CHECK ==========")

for eid in event_to_conv:
    baseline = records[(eid, "KGSF_only")]

    for method in METHODS[1:]:
        current = records[(eid, method)]

        for field in [
            "target_movie_id",
            "frequency_group",
            "target_frequency",
            "target_context_overlap",
        ]:
            if baseline[field] != current[field]:
                raise ValueError(
                    f"Mismatch: event {eid}, method {method}, field {field}"
                )

print("All six methods share identical event targets and metadata.")

print("\n========== 4. CONVERSATION SUBGROUPS ==========")

for cid, eids in conv_to_events.items():
    if not eids:
        raise ValueError(f"Empty conversation: {cid}")

print("Minimum events/conversation:", min(map(len, conv_to_events.values())))
print("Maximum events/conversation:", max(map(len, conv_to_events.values())))

print("\n========== 5. FROZEN TEST METADATA ==========")
print("Reported events:", report["events"])
print("Candidate movies:", report["candidate_movies"])

assert report["events"] == 4003
assert report["candidate_movies"] == 6924

print("\nALL STRUCTURAL CHECKS PASSED.")
