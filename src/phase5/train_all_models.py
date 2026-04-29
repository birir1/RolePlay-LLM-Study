import os
import torch
import wandb
from datasets import load_from_disk

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)

from peft import LoraConfig, get_peft_model, TaskType

# -------------------------------
# CONFIG
# -------------------------------
DATA_PATH = "data/processed/phase1/train_combined"
OUTPUT_ROOT = "models/comparison"

os.makedirs(OUTPUT_ROOT, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# LOAD DATA
# -------------------------------
dataset = load_from_disk(DATA_PATH)

# small subset for fair comparison
dataset = dataset.shuffle(seed=42).select(range(50000))

# -------------------------------
# TOKENIZATION
# -------------------------------
def tokenize_seq2seq(examples, tokenizer):
    inputs = tokenizer(examples["input"], truncation=True, max_length=256)
    labels = tokenizer(examples["target"], truncation=True, max_length=128)

    labels_ids = labels["input_ids"]
    labels_ids = [
        [(t if t != tokenizer.pad_token_id else -100) for t in seq]
        for seq in labels_ids
    ]

    inputs["labels"] = labels_ids
    return inputs


def tokenize_causal(examples, tokenizer):
    texts = [f"{i}\n{t}" for i, t in zip(examples["input"], examples["target"])]

    outputs = tokenizer(
        texts,
        truncation=True,
        max_length=256,
        padding="max_length"
    )

    outputs["labels"] = outputs["input_ids"].copy()
    return outputs


# -------------------------------
# TRAIN FUNCTION
# -------------------------------
def train_model(model_name, model_type, use_lora=False):
    print(f"\n🚀 Training {model_name} | LoRA={use_lora}")

    wandb.init(
        project="safrag-roleplay-study",
        name=f"{model_name.replace('/', '_')}_{'lora' if use_lora else 'full'}",
        reinit=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenized = dataset.map(
            lambda x: tokenize_seq2seq(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenized = dataset.map(
            lambda x: tokenize_causal(x, tokenizer),
            batched=True,
            remove_columns=dataset.column_names
        )
        collator = None

    # -------------------------------
    # LoRA
    # -------------------------------
    if use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM if model_type == "causal" else TaskType.SEQ_2_SEQ_LM,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1
        )
        model = get_peft_model(model, peft_config)

    model.to(device)

    # -------------------------------
    # TRAINING ARGS (compatible)
    # -------------------------------
    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_ROOT, model_name.replace("/", "_")),
        eval_strategy="no",
        save_strategy="epoch",

        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,

        num_train_epochs=1,   # keep fast for comparison
        learning_rate=2e-5,

        logging_steps=50,

        fp16=torch.cuda.is_available(),

        report_to="wandb"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=collator
    )

    trainer.train()
    trainer.save_model()

    wandb.finish()


# -------------------------------
# RUN ALL MODELS
# -------------------------------
def main():

    # 1. FLAN-T5 (FULL)
    train_model(
        model_name="google/flan-t5-base",
        model_type="seq2seq",
        use_lora=False
    )

    # 2. Phi-3 (FULL + LoRA)
    train_model(
        model_name="microsoft/phi-3-mini-4k-instruct",
        model_type="causal",
        use_lora=False
    )

    train_model(
        model_name="microsoft/phi-3-mini-4k-instruct",
        model_type="causal",
        use_lora=True
    )

    # 3. Mistral (LoRA ONLY)
    train_model(
        model_name="mistralai/Mistral-7B-v0.1",
        model_type="causal",
        use_lora=True
    )

    # 4. LLaMA 3 (LoRA ONLY)
    train_model(
        model_name="meta-llama/Meta-Llama-3-8B",
        model_type="causal",
        use_lora=True
    )


if __name__ == "__main__":
    main()