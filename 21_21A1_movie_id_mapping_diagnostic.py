
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent

FILES = [
    ROOT / "data/dataset/redial/nltk/movie_ids.json",
    ROOT / "data/dataset/redial/nltk/entity2id.json",
    ROOT / "results_category_audit/movie_entity_mapping.csv",
    ROOT / "results_category_audit/experiment_21_19C/catalog_item_category_profiles.csv",
]

print("=" * 70)
print("M2.1A.1: MOVIE ID MAPPING DIAGNOSTIC")
print("=" * 70)

for path in FILES:

    print("\nFILE:", path)
    print("EXISTS:", path.exists())

    if not path.exists():
        continue

    if path.suffix == ".json":

        data = json.loads(path.read_text(encoding="utf-8"))

        print("TYPE:", type(data).__name__)
        print("RECORDS:", len(data))
        print("SAMPLE:", list(data.items())[:5]
              if isinstance(data, dict) else data[:5])

    elif path.suffix == ".csv":

        df = pd.read_csv(path)

        print("ROWS:", len(df))
        print("COLUMNS:", df.columns.tolist())
        print(df.head().to_string(index=False))

print("\nMAPPING DIAGNOSTIC COMPLETED")
