# -------------------------------
# 🔥 Multiprocessing Fix
# -------------------------------
import multiprocessing as mp
mp.set_start_method("fork", force=True)

# -------------------------------
# ENV FIXES
# -------------------------------
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["ACCELERATE_DISABLE_RICH"] = "1"

# -------------------------------
# IMPORTS
# -------------------------------
from pathlib import Path
import pandas as pd
from datasets import Dataset, Features, Value
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
MODEL_NAME = "t5-small"

BASE_DIR = Path("data/processed/phase1")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"

OUTPUT_DIR = Path("models/t5_small_baseline")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_TRAIN_SAMPLES = 50_000
MAX_VAL_SAMPLES = 10_000

# -------------------------------
# W&B
# -------------------------------
if wandb.run is None:
    wandb.init(
        project="safrag-roleplay-study",
        name="t5-small-baseline",
        config={"model": MODEL_NAME}
    )

# -------------------------------
# DATA FORMAT
# -------------------------------
FEATURES = Features({
    "input": Value("string"),
    "target": Value("string")
})

def load_split(split_dir, split_type):
    file = split_dir / f"{split_type}_combined.json"

    print(f"[INFO] Loading {file}")
    df = pd.read_json(file, lines=True)

    df = df.loc[:, ~df.columns.str.contains("^__index_level_0__")]
    df["input"] = df["input"].astype(str)
    df["target"] = df["target"].astype(str)

    dataset = Dataset.from_pandas(df, features=FEATURES)

    if split_type == "train":
        dataset = dataset.shuffle(seed=SEED).select(range(min(len(dataset), MAX_TRAIN_SAMPLES)))
    else:
        dataset = dataset.shuffle(seed=SEED).select(range(min(len(dataset), MAX_VAL_SAMPLES)))

    return dataset

def tokenize(examples, tokenizer):
    inputs = tokenizer(examples["input"], max_length=256, truncation=True)
    targets = tokenizer(examples["target"], max_length=128, truncation=True)

    inputs["labels"] = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in targets["input_ids"]
    ]
    return inputs

# -------------------------------
# MAIN
# -------------------------------
def main():
    train = load_split(TRAIN_DIR, "train")
    val = load_split(VAL_DIR, "val")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    model.config.use_cache = False

    train = train.map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["input", "target"])
    val = val.map(lambda x: tokenize(x, tokenizer), batched=True, remove_columns=["input", "target"])

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,

        logging_steps=100,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,

        num_train_epochs=2,
        learning_rate=3e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,

        optim="adamw_torch",

        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,

        save_total_limit=2,
        load_best_model_at_end=True,

        report_to="wandb",
        run_name="t5-small-baseline",
        seed=SEED
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model)
    )

    print("[INFO] Training T5-small...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    wandb.finish()

if __name__ == "__main__":
    main()