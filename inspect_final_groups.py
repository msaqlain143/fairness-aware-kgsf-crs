import pickle

with open("catalog_audit_corrected.pkl", "rb") as f:
    meta = pickle.load(f)

print("========== GROUP DEFINITIONS ==========")
print(meta["group_meta"])

print("\n========== ITEM GROUP DISTRIBUTION ==========")
from collections import Counter
print(Counter(meta["item2group"].values()))

print("\n========== ITEM GROUP SAMPLE ==========")
print(list(meta["item2group"].items())[:15])

print("\n========== QUINTILE DEFINITION ==========")
print(meta["quintile_definition"])

print("\n========== FREQUENCY GROUP CROSS-CHECK ==========")
for name in ["zero_shot_ids", "one_shot_ids", "low_interaction_ids"]:
    ids = meta[name]
    print(name, ":", len(ids))
