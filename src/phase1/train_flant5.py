# -------------------------------
# 🔥 CRITICAL FIX (Python 3.14 multiprocessing issue)
# -------------------------------
import multiprocessing as mp
mp.set_start_method("fork", force=True)

# -------------------------------
# IMPORTS
# -------------------------------
import os
from pathlib import Path
import pandas as pd
from datasets import Dataset, concatenate_datasets, Features, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
import torch
import wandb
import transformers

# -------------------------------
# CONFIG
# -------------------------------
MODEL_NAME = "google/flan-t5-base"

BASE_DIR = Path("data/processed/phase1")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

OUTPUT_DIR = Path("models/transformer_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRAIN_SAMPLES = 100_000
MAX_VAL_SAMPLES = 10_000

SEED = 42
torch.manual_seed(SEED)

print(f"[INFO] Transformers version: {transformers.__version__}")

# -------------------------------
# W&B INIT
# -------------------------------
if wandb.run is None:
    wandb.init(
        project="safrag-roleplay-study",
        name="baseline-flan-t5-stable",
        config={
            "model": MODEL_NAME,
            "epochs": 3,
            "batch_size": 4,   # 🔥 safer
            "lr": 3e-5,
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
    combined_file = split_dir / f"{split_type}_combined.json"

    if combined_file.exists():
        print(f"[INFO] Loading {combined_file}")
        df = pd.read_json(combined_file, lines=True)
    else:
        datasets_list = []
        pattern = "*_train.json" if split_type == "train" else "*_val.json"

        for file in split_dir.glob(pattern):
            print(f"[INFO] Loading {file.name}")
            df = pd.read_json(file, lines=True)
            datasets_list.append(df)

        if len(datasets_list) == 0:
            raise ValueError(f"No datasets found in {split_dir}")

        df = pd.concat(datasets_list, ignore_index=True)

    # clean
    df = df.loc[:, ~df.columns.str.contains("^__index_level_0__")]
    df["input"] = df["input"].astype(str)
    df["target"] = df["target"].astype(str)

    dataset = Dataset.from_pandas(df, features=FEATURES)

    # subsample
    if split_type == "train" and len(dataset) > MAX_TRAIN_SAMPLES:
        dataset = dataset.shuffle(seed=SEED).select(range(MAX_TRAIN_SAMPLES))

    if split_type == "val" and len(dataset) > MAX_VAL_SAMPLES:
        dataset = dataset.shuffle(seed=SEED).select(range(MAX_VAL_SAMPLES))

    return dataset


# -------------------------------
# TOKENIZATION (FIXED)
# -------------------------------
def tokenize_function(examples, tokenizer):
    model_inputs = tokenizer(
        examples["input"],
        max_length=256,
        truncation=True,
    )

    # 🔥 CRITICAL FIX: use text_target (T5 correct way)
    labels = tokenizer(
        text_target=examples["target"],
        max_length=128,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# -------------------------------
# MAIN
# -------------------------------
def main():
    print("[INFO] Loading datasets...")

    train_dataset = load_split(TRAIN_DIR, "train")
    val_dataset = load_split(VAL_DIR, "val")

    print(f"[INFO] Train size: {len(train_dataset)}")
    print(f"[INFO] Val size: {len(val_dataset)}")

    print("[INFO] Loading model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 🔥 Disable problematic gradient checkpointing (causing instability)
    # model.gradient_checkpointing_enable()

    print("[INFO] Tokenizing datasets...")

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["input", "target"],
        num_proc=2
    )

    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["input", "target"],
        num_proc=2
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    # -------------------------------
    # TRAINING ARGS (STABLE)
    # -------------------------------
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        # compatibility
        eval_strategy="epoch",
        save_strategy="epoch",

        logging_steps=50,

        # 🔥 safer batch
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,

        gradient_accumulation_steps=4,

        num_train_epochs=3,
        learning_rate=3e-5,
        weight_decay=0.01,

        # 🔥 stability
        max_grad_norm=1.0,
        warmup_ratio=0.05,

        save_total_limit=2,
        load_best_model_at_end=True,

        # 🔥 TURN OFF FP16 (fix NaN loss)
        fp16=False,

        # 🔥 multiprocessing fix
        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        report_to="wandb",
        run_name="baseline-flan-t5-stable",

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

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    wandb.finish()

    print("[SUCCESS] Training complete.")


# -------------------------------
if __name__ == "__main__":
    main()