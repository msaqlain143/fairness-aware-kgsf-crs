#!/usr/bin/env python
"""Verify whether raw ReDial validation conversations align with CRSLab's
flattened validation records. Read-only diagnostic; writes no project files.
"""
import json
import pickle
from pathlib import Path
from collections import Counter

BASE = Path("data/dataset/redial/nltk")
RAW_VALID = BASE / "valid_data.json"
PICKLE = BASE / "all_data.pkl"
MAPPING = Path("results_corrected/m2_event_conversation_mapping.csv")

def merged_turn_count(dialog):
    """Mirror CRSLab _merge_conv_data role-merging behavior."""
    if not dialog:
        return 0
    groups = 0
    last_role = None
    for utt in dialog:
        role = utt.get("role")
        if role != last_role:
            groups += 1
        last_role = role
    # _augment_and_add emits one record for every merged turn after the first.
    return max(groups - 1, 0)

def main():
    for p in (RAW_VALID, PICKLE):
        if not p.is_file():
            raise FileNotFoundError(p)

    raw = json.loads(RAW_VALID.read_text(encoding="utf-8"))
    with PICKLE.open("rb") as f:
        data = pickle.load(f)

    processed_valid = data[1]
    print("RAW VALIDATION FILE:", RAW_VALID)
    print("RAW CONVERSATIONS:", len(raw))
    print("PROCESSED VALIDATION RECORDS:", len(processed_valid))

    if raw:
        first = raw[0]
        print("\nFIRST RAW CONVERSATION KEYS:", list(first.keys()))
        for key in ("conversationId", "conversation_id", "id", "dialog"):
            if key in first:
                value = first[key]
                if key == "dialog":
                    print("dialog length:", len(value))
                    if value:
                        print("first dialogue turn keys:", list(value[0].keys()))
                        print("first dialogue turn preview:", str(value[0])[:500])
                else:
                    print(f"{key}:", value)

    id_keys = ("conversationId", "conversation_id", "id")
    found_key = next((k for k in id_keys if raw and k in raw[0]), None)
    print("\nDETECTED CONVERSATION ID KEY:", found_key or "NOT FOUND")

    expected_counts = [merged_turn_count(c.get("dialog", [])) for c in raw]
    expected_total = sum(expected_counts)
    print("\n========== FLATTENING COUNT CHECK ==========")
    print("Expected augmented records from raw validation:", expected_total)
    print("Actual processed validation records:", len(processed_valid))
    print("Count equality:", expected_total == len(processed_valid))
    print("Conversations producing zero records:", sum(n == 0 for n in expected_counts))
    print("Expected records/conversation min/max:",
          (min(expected_counts), max(expected_counts)) if expected_counts else None)

    if expected_total == len(processed_valid):
        print("\n========== SEQUENTIAL GROUP ALIGNMENT ==========")
        offset = 0
        failures = []
        for i, (conv, count) in enumerate(zip(raw, expected_counts)):
            chunk = processed_valid[offset:offset + count]
            if len(chunk) != count:
                failures.append((i, "short chunk", count, len(chunk)))
            offset += count
        print("Final processed-record offset:", offset)
        print("Sequential partition valid:", offset == len(processed_valid))
        print("Partition errors:", failures[:10])

        if found_key:
            ids = [str(c[found_key]) for c in raw]
            print("Unique raw conversation IDs:", len(set(ids)))
            print("Duplicate raw conversation IDs:", len(ids) - len(set(ids)))
            print("First 10 raw conversation IDs:", ids[:10])

            if MAPPING.is_file():
                import pandas as pd
                m = pd.read_csv(MAPPING)
                mapped = set(m["conv_id"].astype(str))
                raw_ids = set(ids)
                print("\n========== EVENT MAPPING COVERAGE ==========")
                print("Mapping rows/events:", len(m))
                print("Mapping unique conversations:", m["conv_id"].nunique())
                print("Mapped conversation IDs found in raw validation:",
                      len(mapped & raw_ids), "/", len(mapped))
                print("Mapped IDs absent from raw validation:",
                      sorted(mapped - raw_ids)[:20])
                print("Raw validation IDs absent from mapping:",
                      len(raw_ids - mapped))
    else:
        print("\nSTOP: Do not infer sequential alignment. Investigate preprocessing,")
        print("split provenance, or a mismatch between the raw and restored dataset.")

if __name__ == "__main__":
    main()
