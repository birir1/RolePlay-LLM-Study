# -------------------------------
# 🔥 Multiprocessing Fix
# -------------------------------
import multiprocessing as mp
mp.set_start_method("fork", force=True)

# -------------------------------
# IMPORTS
# -------------------------------
import os
from pathlib import Path
import pandas as pd
import torch
import transformers
import wandb

from datasets import Dataset, Features, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

# -------------------------------
# CONFIG
# -------------------------------
MODEL_NAME = "google/mt5-small"

BASE_DIR = Path("data/processed/phase1")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

OUTPUT_DIR = Path("models/mt5_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRAIN_SAMPLES = 100_000
MAX_VAL_SAMPLES = 10_000

SEED = 42

print(f"[INFO] Transformers version: {transformers.__version__}")

# -------------------------------
# W&B INIT
# -------------------------------
if wandb.run is None:
    wandb.init(
        project="safrag-roleplay-study",
        name="mt5-baseline-stable",
        config={
            "model": MODEL_NAME,
            "epochs": 2,
            "batch_size": 4,
            "lr": 1e-5,
        }
    )

# -------------------------------
# DATA FORMAT
# -------------------------------
FEATURES = Features({
    "input": Value("string"),
    "target": Value("string")
})

# -------------------------------
# LOAD DATA
# -------------------------------
def load_split(split_dir, split_type="train"):
    file = split_dir / f"{split_type}_combined.json"
    print(f"[INFO] Loading {file}")

    df = pd.read_json(file, lines=True)

    # Clean data
    df = df.dropna()
    df["input"] = df["input"].astype(str)
    df["target"] = df["target"].astype(str)

    df = df[
        (df["input"].str.strip() != "") &
        (df["target"].str.strip() != "")
    ]

    dataset = Dataset.from_pandas(df, features=FEATURES)

    # Subsample
    if split_type == "train" and len(dataset) > MAX_TRAIN_SAMPLES:
        dataset = dataset.shuffle(seed=SEED).select(range(MAX_TRAIN_SAMPLES))

    if split_type == "val" and len(dataset) > MAX_VAL_SAMPLES:
        dataset = dataset.shuffle(seed=SEED).select(range(MAX_VAL_SAMPLES))

    print(f"[INFO] Cleaned {split_type} size: {len(dataset)}")
    return dataset


# -------------------------------
# TOKENIZATION (FIXED)
# -------------------------------
def tokenize_function(examples, tokenizer):
    inputs = tokenizer(
        examples["input"],
        max_length=256,
        truncation=True,
        padding=False
    )

    targets = tokenizer(
        examples["target"],
        max_length=128,
        truncation=True,
        padding=False
    )

    new_labels = []

    for seq in targets["input_ids"]:

        # 🔥 FIX 1: empty sequence
        if len(seq) == 0:
            seq = [tokenizer.eos_token_id]

        # 🔥 FIX 2: replace pad with -100
        seq = [
            t if t != tokenizer.pad_token_id else -100
            for t in seq
        ]

        # 🔥 FIX 3: avoid all -100 → causes zero loss
        if all(t == -100 for t in seq):
            seq[0] = tokenizer.eos_token_id

        new_labels.append(seq)

    inputs["labels"] = new_labels
    return inputs


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("[INFO] Loading datasets...")

    train_dataset = load_split(TRAIN_DIR, "train")
    val_dataset = load_split(VAL_DIR, "val")

    print("[INFO] Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=False  # 🔥 critical for mT5
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 🔥 Stability fixes
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    model.config.tie_word_embeddings = False

    print("[INFO] Tokenizing datasets...")

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["input", "target"],
        num_proc=1
    )

    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["input", "target"],
        num_proc=1
    )

    # -------------------------------
    # DATA COLLATOR
    # -------------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        pad_to_multiple_of=8
    )

    # -------------------------------
    # TRAINING ARGS (VERSION SAFE)
    # -------------------------------
    try:
        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,

            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,

            num_train_epochs=2,
            learning_rate=1e-5,
            weight_decay=0.01,
            max_grad_norm=1.0,

            save_total_limit=2,
            load_best_model_at_end=True,

            fp16=False,  # 🔥 disable for stability

            dataloader_num_workers=0,
            dataloader_pin_memory=False,

            report_to="wandb",
            run_name="mt5-baseline-stable",
            seed=SEED
        )
    except TypeError:
        print("[WARN] Using OLD transformers API")

        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,

            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,

            num_train_epochs=2,
            learning_rate=1e-5,
            weight_decay=0.01,
            max_grad_norm=1.0,

            save_total_limit=2,
            load_best_model_at_end=True,

            fp16=False,

            dataloader_num_workers=0,
            dataloader_pin_memory=False,

            report_to="wandb",
            run_name="mt5-baseline-stable",
            seed=SEED
        )

    # -------------------------------
    # TRAINER
    # -------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator
    )

    print("[INFO] Training...")
    trainer.train()

    print("[INFO] Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    wandb.finish()
    print("[SUCCESS] Training complete.")


# -------------------------------
if __name__ == "__main__":
    main()