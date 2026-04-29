import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Paths
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

datasets = [
    "Friends.csv",
    "Natural-Questions-Base.csv",
    "Natural-Questions-Filtered.csv",
    "personality.csv",
    "OpenSubtitles.eu-ko.ko",
    "articles.csv",
    "editors.csv",
    "language_coverage.csv"
]

for dataset in datasets:
    raw_path = os.path.join(RAW_DIR, dataset)
    
    # 1. Handle the "Directory pretending to be a file" issue
    if os.path.isdir(raw_path):
        nested_file = os.path.join(raw_path, dataset)
        if os.path.isfile(nested_file):
            raw_path = nested_file
        else:
            print(f" Skipping {dataset}: It's a directory and doesn't contain the expected file.")
            continue

    if not os.path.exists(raw_path):
        print(f" Skipping {dataset}: file not found.")
        continue
    
    try:
        # 2. Handle non-CSV files (like .ko)
        if not dataset.endswith('.csv'):
            # If it's a text file, you might need to read it differently
            # For now, we skip to prevent pd.read_csv from crashing
            print(f"ℹ Skipping {dataset}: Not a CSV file.")
            continue

        # Load CSV
        df = pd.read_csv(raw_path)
        
        # Split into 80% train, 10% val, 10% test
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Save
        name = os.path.splitext(dataset)[0]
        train_df.to_csv(os.path.join(PROCESSED_DIR, f"{name}_train.csv"), index=False)
        val_df.to_csv(os.path.join(PROCESSED_DIR, f"{name}_val.csv"), index=False)
        test_df.to_csv(os.path.join(PROCESSED_DIR, f"{name}_test.csv"), index=False)
        
        print(f" Split {dataset} -> train/val/test saved in {PROCESSED_DIR}/")
        
    except Exception as e:
        print(f" Error processing {dataset}: {e}")