
from calibrate_and_evaluate import initialize
import torch

# ============================================================
# PHASE 5.2 — KGSF RECOMMENDATION BATCH INSPECTION
# ============================================================

CONFIG = "config/crs/kgsf/redial.yaml"
GPU = "0"  # Change to "-1" if running on CPU

print("=" * 70)
print("PHASE 5.2 — KGSF RECOMMENDATION BATCH INSPECTION")
print("=" * 70)

# Initialize the existing trained KGSF model and dataloaders
system, valid, test, side_data = initialize(CONFIG, GPU)

# ------------------------------------------------------------
# 1. DATASET INFORMATION
# ------------------------------------------------------------

print("\n===== 1. DATASET INFORMATION =====")

for name, loader in [("VALIDATION", valid), ("TEST", test)]:
    print(f"\n{name}")

    for attr in ["data", "dataset"]:
        if hasattr(loader, attr):
            obj = getattr(loader, attr)
            try:
                print(f"{attr} length:", len(obj))
            except TypeError:
                print(f"{attr} type:", type(obj))

print("\nDevice:", system.device)

# ------------------------------------------------------------
# 2. KNOWLEDGE GRAPH INFORMATION
# ------------------------------------------------------------

print("\n===== 2. KNOWLEDGE GRAPH INFORMATION =====")

print("Side data keys:", list(side_data.keys()))

if "entity_kg" in side_data:
    print("Entity KG keys:", list(side_data["entity_kg"].keys()))

if "item_entity_ids" in side_data:
    print("Official movie count:", len(side_data["item_entity_ids"]))

print("Padding entity ID:", valid.pad_entity_idx)

# ------------------------------------------------------------
# 3. INSPECT ACTUAL RECOMMENDATION BATCH
# ------------------------------------------------------------

print("\n===== 3. RECOMMENDATION BATCH STRUCTURE =====")

batch_iterator = iter(
    valid.get_rec_data(batch_size=2, shuffle=False)
)

batch = next(batch_iterator)

print("Batch type:", type(batch))
print("Number of batch fields:", len(batch))

# Inspect every field without assuming its name or position.
for i, value in enumerate(batch):

    print(f"\nField {i}")
    print("Type:", type(value))

    if torch.is_tensor(value):
        print("Shape:", tuple(value.shape))
        print("Dtype:", value.dtype)
        print("Device:", value.device)

        if value.ndim > 0 and value.shape[0] > 0:
            print("First event:", value[0].detach().cpu().tolist())

    else:
        print("Value:", repr(value)[:500])

# ------------------------------------------------------------
# 4. IDENTIFY CONTEXT AND TARGET FIELDS
# ------------------------------------------------------------

print("\n===== 4. FIRST RECOMMENDATION EVENT =====")

if len(batch) >= 2:

    print("Context field (batch[0]):")
    print(batch[0][0].detach().cpu().tolist()
          if torch.is_tensor(batch[0]) else repr(batch[0]))

    print("\nGround-truth target (batch[-1]):")
    print(batch[-1][0].detach().cpu().tolist()
          if torch.is_tensor(batch[-1]) else repr(batch[-1]))

# ------------------------------------------------------------
# 5. FINAL VALIDATION
# ------------------------------------------------------------

print("\n===== 5. INSPECTION SUMMARY =====")

print("Model:", type(system.model).__name__)
print("Validation dataloader:", type(valid).__name__)
print("Test dataloader:", type(test).__name__)
print("Batch fields:", len(batch))

print("\nInspection completed successfully.")
