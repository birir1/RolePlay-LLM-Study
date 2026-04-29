import os
import json
import torch
import wandb

from datasets import Dataset, load_from_disk
from transformers import (
    TrainingArguments,
    DataCollatorForSeq2Seq
)

from src.training.saf_trainer import SAFTrainer
from src.models.saf_rag.saf_rag import SAFRAG  # ✅ FIXED IMPORT


# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

USE_PREPROCESSED = True

RAW_DATA_PATH = "data/processed/saf_training/saf_train.json"
PROCESSED_DATA_PATH = "data/processed/saf_training_clean"

BASE_MODEL_PATH = "google/flan-t5-base"
OUTPUT_DIR = "models/saf_rag_model"

MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# DATA LOADING
# =========================
def load_dataset():

    if USE_PREPROCESSED and os.path.exists(PROCESSED_DATA_PATH):
        print("[INFO] Loading preprocessed dataset...")
        dataset = load_from_disk(PROCESSED_DATA_PATH)
        return dataset["train"], dataset["test"]

    print("[INFO] Loading raw dataset...")

    with open(RAW_DATA_PATH, "r") as f:
        data = json.load(f)

    filtered = []
    for x in data:
        inp = str(x.get("input", "")).strip()
        tgt = str(x.get("target", "")).strip()

        if len(inp.split()) < 6:
            continue
        if len(tgt.split()) < 5:
            continue
        if inp == tgt:
            continue
        if x.get("grounding", 0) < 0.7:
            continue

        filtered.append({"input": inp, "target": tgt})

    print(f"[INFO] Filtered dataset size: {len(filtered)}")

    dataset = Dataset.from_list(filtered)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)

    return dataset["train"], dataset["test"]


# =========================
# MODEL LOADING
# =========================
def load_model():

    print("[INFO] Loading SAF-RAG model...")

    model = SAFRAG(
        model_name=BASE_MODEL_PATH,
        device=DEVICE,
        use_fusion_gate=True
    )

    # 🔥 Fix T5 tied weights warning
    if hasattr(model, "model") and hasattr(model.model, "config"):
        model.model.config.tie_word_embeddings = False

    tokenizer = model.tokenizer

    return tokenizer, model


# =========================
# TOKENIZATION
# =========================
def tokenize_function(example, tokenizer):

    model_inputs = tokenizer(
        example["input"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding=False
    )

    target = example["target"].strip()

    if len(target) < 3:
        return None

    labels = tokenizer(
        target,
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding=False
    )["input_ids"]

    labels = [
        (l if l != tokenizer.pad_token_id else -100)
        for l in labels
    ]

    if all(l == -100 for l in labels):
        return None

    model_inputs["labels"] = labels

    return model_inputs


# =========================
# SAFE MAP
# =========================
def safe_map(dataset, tokenizer):

    mapped = dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        remove_columns=dataset.column_names
    )

    mapped = mapped.filter(
        lambda x: (
            x["labels"] is not None and
            any(l != -100 for l in x["labels"])
        )
    )

    return mapped


# =========================
# MAIN
# =========================
def main():

    train_dataset, val_dataset = load_dataset()
    tokenizer, model = load_model()

    print("[INFO] Tokenizing dataset...")

    train_dataset = safe_map(train_dataset, tokenizer)
    val_dataset = safe_map(val_dataset, tokenizer)

    print(f"[INFO] Train size: {len(train_dataset)}")
    print(f"[INFO] Val size: {len(val_dataset)}")

    # =========================
    # WANDB
    # =========================
    wandb.init(
        project="saf-rag",
        name="saf_rag_final_loss_v2",  # 🔥 updated name for new loss
        mode="online"
    )

    # =========================
    # TRAINING ARGS
    # =========================
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,

        learning_rate=1e-5,
        num_train_epochs=3,

        warmup_steps=200,

        logging_steps=25,

        do_eval=True,

        save_steps=500,
        save_total_limit=2,

        max_grad_norm=1.0,

        fp16=False,

        report_to="wandb"
    )

    # =========================
    # DATA COLLATOR
    # =========================
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model.model,
        label_pad_token_id=-100,
        padding=True
    )

    # =========================
    # TRAINER
    # =========================
    trainer = SAFTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    print("[INFO] Starting SAF-RAG training...")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"[SUCCESS] Saved → {OUTPUT_DIR}")


# =========================
# ENTRY
# =========================
if __name__ == "__main__":
    main()