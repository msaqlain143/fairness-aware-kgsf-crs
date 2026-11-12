
import pickle
from pathlib import Path

# Find processed dataset files
for p in Path(".").rglob("all_data.pkl"):
    if ".venv" in str(p):
        continue
    print("DATASET:", p)

    try:
        with open(p, "rb") as f:
            data = pickle.load(f)

        print("TYPE:", type(data))

        if isinstance(data, dict):
            print("KEYS:", data.keys())

        break
    except Exception as e:
        print("ERROR:", e)

# Inspect available validation dataset structure
