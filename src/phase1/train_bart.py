# -------------------------------
# 🔥 CRITICAL FIX (Python 3.14 multiprocessing)
# -------------------------------
import multiprocessing as mp
mp.set_start_method("fork", force=True)

# -------------------------------
# 🔥 ENV FIXES (stability)
# -------------------------------
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

# -------------------------------
# IMPORTS
# -------------------------------
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

# -------------------------------
# CONFIG
# -------------------------------
MODEL_NAME = "facebook/bart-base"

BASE_DIR = Path("data/processed/phase1")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

OUTPUT_DIR = Path("models/bart_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_TRAIN_SAMPLES = 50_000   # 🔥 reduced for speed
MAX_VAL_SAMPLES = 10_000

# -------------------------------
# W&B INIT
# -------------------------------
if wandb.run is None:
    wandb.init(
        project="safrag-roleplay-study",
        name="bart-baseline-stable",
        config={
            "model": MODEL_NAME,
            "epochs": 2,
            "batch_size": 4,
            "lr": 3e-5
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

    if not combined_file.exists():
        raise FileNotFoundError(f"{combined_file} not found")

    print(f"[INFO] Loading {combined_file}")
    df = pd.read_json(combined_file, lines=True)

    # cleanup
    df = df.loc[:, ~df.columns.str.contains("^__index_level_0__")]
    df["input"] = df["input"].astype(str)
    df["target"] = df["target"].astype(str)

    dataset = Dataset.from_pandas(df, features=FEATURES)

    # 🔥 subsample
    if split_type == "train":
        dataset = dataset.shuffle(seed=SEED).select(range(min(len(dataset), MAX_TRAIN_SAMPLES)))
    else:
        dataset = dataset.shuffle(seed=SEED).select(range(min(len(dataset), MAX_VAL_SAMPLES)))

    return dataset

# -------------------------------
# TOKENIZATION
# -------------------------------
def tokenize_function(examples, tokenizer):
    model_inputs = tokenizer(
        examples["input"],
        max_length=256,
        truncation=True
    )

    labels = tokenizer(
        examples["target"],
        max_length=128,
        truncation=True
    )

    labels_ids = labels["input_ids"]

    # mask padding
    labels_ids = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in labels_ids
    ]

    model_inputs["labels"] = labels_ids
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

    # 🔥 speed + stability
    model.config.use_cache = False

    print("[INFO] Tokenizing datasets...")

    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["input", "target"]
    )

    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=["input", "target"]
    )

    # -------------------------------
    # DATA COLLATOR
    # -------------------------------
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    # -------------------------------
    # TRAINING ARGS (STABLE)
    # -------------------------------
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        # 🔥 checkpointing
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,

        logging_steps=100,

        # 🔥 memory-safe batch
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,

        # 🔥 training
        num_train_epochs=2,
        learning_rate=3e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,

        # 🔥 optimizer fix
        optim="adamw_torch",

        # 🔥 stability fixes
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        # 🔥 checkpoint control
        save_total_limit=2,
        load_best_model_at_end=True,

        report_to="wandb",
        run_name="bart-baseline-stable",
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