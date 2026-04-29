# -------------------------------
# 🔥 FIX multiprocessing (Python 3.14)
# -------------------------------
import multiprocessing as mp
mp.set_start_method("fork", force=True)

# -------------------------------
# IMPORTS
# -------------------------------
import torch
from pathlib import Path
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model
import wandb

# -------------------------------
# CONFIG
# -------------------------------
MODEL_NAME = "mistralai/Mistral-7B-v0.1"

BASE_DIR = Path("data/processed/phase1")
TRAIN_FILE = BASE_DIR / "train" / "train_combined.json"
VAL_FILE = BASE_DIR / "val" / "val_combined.json"

OUTPUT_DIR = Path("models/mistral_lora")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MAX_TRAIN = 50000
MAX_VAL = 5000

# -------------------------------
# LOAD DATA
# -------------------------------
def load_data(path, max_samples):
    df = pd.read_json(path, lines=True)

    df["text"] = df["input"] + "\n" + df["target"]
    df = df[["text"]]

    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)

    return Dataset.from_pandas(df)

# -------------------------------
# TOKENIZATION
# -------------------------------
def tokenize(example, tokenizer):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512
    )

# -------------------------------
# MAIN
# -------------------------------
def main():
    wandb.init(project="safrag-roleplay-study", name="mistral-qlora")

    print("[INFO] Loading dataset...")
    train_ds = load_data(TRAIN_FILE, MAX_TRAIN)
    val_ds = load_data(VAL_FILE, MAX_VAL)

    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("[INFO] Loading model (4-bit)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )

    print("[INFO] Applying LoRA...")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    print("[INFO] Tokenizing...")
    train_ds = train_ds.map(lambda x: tokenize(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize(x, tokenizer), batched=True)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print("[INFO] Training...")

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,

        num_train_epochs=2,
        learning_rate=2e-4,

        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",

        fp16=True,

        dataloader_num_workers=0,

        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator
    )

    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    wandb.finish()

    print("[SUCCESS] Mistral training complete")


if __name__ == "__main__":
    main()