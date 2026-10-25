
from calibrate_and_evaluate import initialize

CONFIG = "config/crs/kgsf/redial.yaml"

system, valid, test, side_data = initialize(CONFIG, "0")

print("\nPHASE 5.2 — KGSF BATCH INSPECTION")

print("Validation records:", len(valid.dataset))
print("Test records:", len(test.dataset))

print("Entity KG keys:", list(side_data["entity_kg"].keys()))
print("Official movie count:", len(side_data["item_entity_ids"]))

batch = next(
    iter(valid.get_rec_data(batch_size=2, shuffle=False))
)

print("\nRecommendation batch structure:")
print("Number of tensors:", len(batch))

names = [
    "context_entities",
    "context_words",
    "context_entity_onehot",
    "target_movie",
]

for i, tensor in enumerate(batch):
    print(
        f"{i}: {names[i]} | "
        f"shape={tuple(tensor.shape)} | "
        f"dtype={tensor.dtype}"
    )

print("\nFirst recommendation event:")
print("Context entity tensor:", batch[0][0].tolist())
print("Context word tensor:", batch[1][0].tolist())
print("Ground-truth movie:", int(batch[3][0]))

print("\nPadding entity ID:")
print(valid.pad_entity_idx)

print("\nInspection completed.")
