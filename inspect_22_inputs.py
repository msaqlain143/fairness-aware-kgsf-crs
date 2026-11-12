import pandas as pd

base = "results_corrected/adaptive_m2_validation_export22"

c = pd.read_csv(base + "_candidates.csv")
e = pd.read_csv(base + "_events.csv")

p = pd.read_csv(
    "results_category_audit/experiment_21_22A/"
    "catalog_item_category_profiles.csv"
)

print("========== CANDIDATE SCHEMA ==========")
print(c.columns.tolist())
print(c.iloc[0].to_dict())

print("\n========== EVENT SCHEMA ==========")
print(e.columns.tolist())
print(e.iloc[0].to_dict())

print("\n========== CATEGORY PROFILE SCHEMA ==========")
print(p.columns.tolist())
print(p.head(2).to_string(index=False))

print("\n========== CATEGORY ACCEPTANCE COUNTS ==========")
for col in [
    "accepted_genre_evidence",
    "accepted_hybrid_evidence",
    "identity_resolution_status",
    "category_availability"
]:
    if col in p.columns:
        print("\n", col)
        print(p[col].value_counts(dropna=False).to_dict())
