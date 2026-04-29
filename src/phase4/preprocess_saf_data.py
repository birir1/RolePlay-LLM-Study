import os
import json
from datasets import Dataset, DatasetDict

INPUT_PATH = "data/processed/saf_training/saf_train.json"
OUTPUT_DIR = "data/processed/saf_training_clean"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# FILTERING LOGIC
# =========================
def is_valid(x):
    inp = x.get("input", "").strip()
    tgt = x.get("target", "").strip()
    grounding = x.get("grounding", 0)

    return (
        len(inp) > 20 and
        len(tgt) > 20 and
        inp != tgt and
        grounding > 0.6
    )


# =========================
# MAIN
# =========================
def main():
    with open(INPUT_PATH, "r") as f:
        data = json.load(f)

    print(f"[INFO] Loaded raw samples: {len(data)}")

    # Filter
    filtered = [x for x in data if is_valid(x)]
    print(f"[INFO] After filtering: {len(filtered)}")

    if len(filtered) < 1000:
        print("[WARN] Dataset too small after filtering")

    # Convert to HF dataset
    dataset = Dataset.from_list(filtered)

    # Train / Validation split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    print(f"[INFO] Train size: {len(dataset['train'])}")
    print(f"[INFO] Val size: {len(dataset['test'])}")

    # Save
    dataset.save_to_disk(OUTPUT_DIR)

    # Save preview
    preview_path = os.path.join(OUTPUT_DIR, "preview.json")
    with open(preview_path, "w") as f:
        json.dump(filtered[:5], f, indent=2)

    print(f"[SUCCESS] Clean dataset saved → {OUTPUT_DIR}")


if __name__ == "__main__":
    main()