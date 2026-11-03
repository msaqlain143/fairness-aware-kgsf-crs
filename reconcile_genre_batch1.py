
import pandas as pd
import html
import re
from pathlib import Path

BASE = Path(
    "results_category_audit/dbpedia_pilot/"
    "genre_manual_validation_cleaned.csv"
)

BATCH1 = Path(
    "results_category_audit/dbpedia_pilot/"
    "genre_validation_archive/"
    "genre_manual_validation_batch1_updated.csv"
)

OUTPUT = Path(
    "results_category_audit/dbpedia_pilot/"
    "genre_manual_validation_reconciled_batch1.csv"
)

ANNOTATION_COLUMNS = [
    "validated_type",
    "canonical_genre",
    "parent_genre",
    "temporal_modifier",
    "geographical_modifier",
    "manual_decision",
    "review_notes",
]

EXPECTED_BATCH1 = 40


def normalize_header(value):
    value = html.unescape(str(value))
    value = re.sub(r"^\*+", "", value)
    return value.strip()


def normalize_uri(value):
    value = str(value).strip()
    value = value.replace("\\", "")
    return value


def normalize_category(value):
    return " ".join(str(value).strip().split())


def normalize_decision(value):
    value = str(value).strip()
    value = value.strip("*").strip()
    return value.upper()


def load_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, dtype=str).fillna("")
    df.columns = [normalize_header(c) for c in df.columns]

    if len(set(df.columns)) != len(df.columns):
        raise ValueError(
            f"Duplicate column names after normalization: {path}"
        )

    required = [
        "original_category",
        "value_uri_or_literal",
        "manual_decision",
    ]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )

    df["original_category"] = (
        df["original_category"].map(normalize_category)
    )

    df["value_uri_or_literal"] = (
        df["value_uri_or_literal"].map(normalize_uri)
    )

    df["manual_decision"] = (
        df["manual_decision"].map(normalize_decision)
    )

    return df


print("=== BATCH 1 RECONCILIATION ===")

base = load_csv(BASE)
batch = load_csv(BATCH1)

print("Clean master rows:", len(base))
print("Batch archive rows:", len(batch))

if len(base) != 501:
    raise ValueError(
        "Unexpected master row count. Expected 501."
    )

# Ensure required annotation columns exist.
for col in ANNOTATION_COLUMNS:
    if col not in base.columns:
        base[col] = ""

    if col not in batch.columns:
        batch[col] = ""

# Select genuinely completed annotations.
completed = batch[
    batch["manual_decision"].ne("")
    & batch["manual_decision"].ne("PENDING")
].copy()

print("\n=== BATCH 1 SOURCE STATUS ===")
print("Completed annotations:", len(completed))
print(completed["manual_decision"].value_counts().to_string())

# Strict safety check.
if len(completed) != EXPECTED_BATCH1:
    raise ValueError(
        f"Expected {EXPECTED_BATCH1} completed annotations, "
        f"but found {len(completed)}. "
        "No output file was created."
    )

# Allowed decisions.
allowed = {
    "ACCEPT_GENRE",
    "ACCEPT_HYBRID",
    "ATTRIBUTE_ONLY",
    "EXCLUDE",
    "NEEDS_REVIEW",
}

unknown = set(completed["manual_decision"]) - allowed

if unknown:
    raise ValueError(
        f"Unexpected decision values: {unknown}"
    )

# Build lookup indexes using category + URI.
base_pairs = {}

for idx, row in base.iterrows():
    key = (
        row["original_category"],
        row["value_uri_or_literal"],
    )
    base_pairs.setdefault(key, []).append(idx)

base_categories = {}

for idx, row in base.iterrows():
    key = row["original_category"]
    base_categories.setdefault(key, []).append(idx)

matched_indices = []
unmatched = []
conflicts = []

# Reconcile each completed annotation.
for _, annotation in completed.iterrows():

    category = annotation["original_category"]
    uri = annotation["value_uri_or_literal"]

    key = (category, uri)

    candidates = base_pairs.get(key, [])

    # Fallback to category only if it is unique.
    if len(candidates) == 0:
        candidates = base_categories.get(category, [])

    if len(candidates) != 1:
        unmatched.append({
            "category": category,
            "uri": uri,
            "candidate_count": len(candidates),
        })
        continue

    target_idx = candidates[0]

    if target_idx in matched_indices:
        conflicts.append(
            f"Multiple annotations matched row {target_idx}"
        )
        continue

    # Prevent accidental overwriting of existing annotations.
    existing = base.at[target_idx, "manual_decision"]

    if existing not in ("", "PENDING"):
        conflicts.append(
            f"Target row {target_idx} already has decision "
            f"{existing}"
        )
        continue

    matched_indices.append(target_idx)

    # Copy the complete annotation.
    for col in ANNOTATION_COLUMNS:
        value = annotation[col]

        if col == "manual_decision":
            value = normalize_decision(value)

        base.at[target_idx, col] = value

# Stop if any reconciliation issue occurred.
if unmatched or conflicts:
    print("\n=== RECONCILIATION PROBLEMS ===")

    print("Unmatched records:", len(unmatched))
    print("Conflicts:", len(conflicts))

    for item in unmatched:
        print(item)

    for item in conflicts:
        print(item)

    raise RuntimeError(
        "Reconciliation stopped. No output file was created."
    )

# Final integrity checks.
if len(matched_indices) != EXPECTED_BATCH1:
    raise RuntimeError("Not all 40 annotations were matched.")

if len(base) != 501:
    raise RuntimeError("Master row count changed unexpectedly.")

if base["manual_decision"].eq("").any():
    raise RuntimeError("Some decision fields are empty.")

# Save a new master.
base.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n=== RECONCILIATION SUCCESSFUL ===")
print("Total master records:", len(base))
print("Batch 1 annotations restored:", len(matched_indices))

print("\n=== FINAL DECISION COUNTS ===")
print(base["manual_decision"].value_counts().to_string())

pending = base["manual_decision"].eq("PENDING").sum()

print("\nPending records:", pending)
print("Completed records:", len(base) - pending)

print("\nOutput:", OUTPUT)
print("Original files were NOT modified.")
