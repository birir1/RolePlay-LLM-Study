import os
import json
import random
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = Path("data")
RAW_DIR = BASE_DIR / "raw"
PROCESSED_DIR = BASE_DIR / "processed" / "phase1"

TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "val"
TEST_DIR = PROCESSED_DIR / "test"
PLOT_DIR = PROCESSED_DIR / "plots"
META_DIR = BASE_DIR / "processed" / "metadata"

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, PLOT_DIR, META_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------------------
# CONFIG
# -------------------------------
MIN_INPUT_LEN = 3
MAX_INPUT_LEN = 512
MAX_TARGET_LEN = 256

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15

# -------------------------------
# LOAD CSV
# -------------------------------
def load_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        print(f"[ERROR] Loading {file_path}: {e}")
        return None


# -------------------------------
# STANDARDIZE COLUMNS
# -------------------------------
def standardize_columns(df):
    columns = df.columns.tolist()

    # common mappings
    mapping_candidates = [
        ("question", "answer"),
        ("prompt", "response"),
        ("input", "target"),
        ("instruction", "output"),
    ]

    for inp, tgt in mapping_candidates:
        if inp in columns and tgt in columns:
            df = df.rename(columns={inp: "input", tgt: "target"})
            break
    else:
        # fallback: first two columns
        if len(columns) < 2:
            raise ValueError("Dataset has less than 2 columns")
        df = df.rename(columns={columns[0]: "input", columns[1]: "target"})

    df = df[["input", "target"]]

    df["input"] = df["input"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()

    return df


# -------------------------------
# CLEAN DATA
# -------------------------------
def clean_dataset(df):
    # remove empty
    df = df[(df["input"] != "") & (df["target"] != "")]

    # remove duplicates
    df = df.drop_duplicates(subset=["input", "target"])

    # length filtering
    df["input_len"] = df["input"].apply(lambda x: len(x.split()))
    df["target_len"] = df["target"].apply(lambda x: len(x.split()))

    df = df[
        (df["input_len"] >= MIN_INPUT_LEN) &
        (df["input_len"] <= MAX_INPUT_LEN) &
        (df["target_len"] <= MAX_TARGET_LEN)
    ]

    df = df.drop(columns=["input_len", "target_len"])

    return df.reset_index(drop=True)


# -------------------------------
# SPLIT
# -------------------------------
def split_dataset(df):
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    return df[:train_end], df[train_end:val_end], df[val_end:]


# -------------------------------
# SAVE
# -------------------------------
def save_splits(name, train_df, val_df, test_df):
    # CSV
    train_df.to_csv(TRAIN_DIR / f"{name}_train.csv", index=False)
    val_df.to_csv(VAL_DIR / f"{name}_val.csv", index=False)
    test_df.to_csv(TEST_DIR / f"{name}_test.csv", index=False)

    # JSON (for training)
    train_df.to_json(TRAIN_DIR / f"{name}_train.json", orient="records", lines=True)
    val_df.to_json(VAL_DIR / f"{name}_val.json", orient="records", lines=True)
    test_df.to_json(TEST_DIR / f"{name}_test.json", orient="records", lines=True)


# -------------------------------
# PLOTS
# -------------------------------
def plot_distribution(name, df):
    lengths = df["input"].apply(lambda x: len(x.split()))

    plt.figure()
    plt.hist(lengths, bins=50)
    plt.title(f"{name} Input Length Distribution")
    plt.xlabel("Token Count")
    plt.ylabel("Frequency")

    plt.savefig(PLOT_DIR / f"{name}_length_distribution.png")
    plt.close()


# -------------------------------
# MAIN PROCESS
# -------------------------------
def process_all_datasets():
    stats = {}
    combined_train = []
    combined_val = []
    combined_test = []

    for file in RAW_DIR.glob("*.csv"):
        print(f"[INFO] Processing {file.name}")

        df = load_csv(file)
        if df is None or len(df) < 20:
            print(f"[WARNING] Skipping {file.name} (too small or invalid)")
            continue

        try:
            df = standardize_columns(df)
            df = clean_dataset(df)
        except Exception as e:
            print(f"[ERROR] {file.name}: {e}")
            continue

        if len(df) < 20:
            print(f"[WARNING] Skipping {file.name} after cleaning")
            continue

        train_df, val_df, test_df = split_dataset(df)

        dataset_name = file.stem

        save_splits(dataset_name, train_df, val_df, test_df)
        plot_distribution(dataset_name, df)

        # collect for combined dataset
        combined_train.append(train_df)
        combined_val.append(val_df)
        combined_test.append(test_df)

        stats[dataset_name] = {
            "total": len(df),
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
            "avg_input_len": float(df["input"].apply(lambda x: len(x.split())).mean())
        }

    # -------------------------------
    # SAVE COMBINED DATASET (IMPORTANT)
    # -------------------------------
    print("[INFO] Creating combined datasets...")

    combined_train_df = pd.concat(combined_train).reset_index(drop=True)
    combined_val_df = pd.concat(combined_val).reset_index(drop=True)
    combined_test_df = pd.concat(combined_test).reset_index(drop=True)

    combined_train_df.to_json(TRAIN_DIR / "train_combined.json", orient="records", lines=True)
    combined_val_df.to_json(VAL_DIR / "val_combined.json", orient="records", lines=True)
    combined_test_df.to_json(TEST_DIR / "test_combined.json", orient="records", lines=True)

    # -------------------------------
    # METADATA
    # -------------------------------
    with open(META_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=4)

    with open(META_DIR / "split_config.json", "w") as f:
        json.dump({
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "test_ratio": 1 - (TRAIN_RATIO + VAL_RATIO),
            "seed": SEED,
            "filters": {
                "min_input_len": MIN_INPUT_LEN,
                "max_input_len": MAX_INPUT_LEN,
                "max_target_len": MAX_TARGET_LEN
            }
        }, f, indent=4)

    print("[SUCCESS] Phase 1 dataset preparation complete.")


# -------------------------------
# ENTRY
# -------------------------------
if __name__ == "__main__":
    process_all_datasets()