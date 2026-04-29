import os
import torch
import wandb
from datasets import load_from_disk
from transformers import (
    RagTokenizer,
    RagRetriever,
    RagTokenForGeneration,
    Trainer,
    TrainingArguments
)

# -------------------------------
# Paths
# -------------------------------
DATASET_PATH = "data/rag_index/my_dataset"
INDEX_PATH = "data/rag_index/hf_dataset_index.faiss"
OUTPUT_DIR = "outputs/rag_finetuned"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------
# 1. Load dataset + FAISS index
# -------------------------------
dataset = load_from_disk(DATASET_PATH)
dataset.load_faiss_index("embeddings", INDEX_PATH)

print("✅ Dataset + FAISS index loaded.")

# -------------------------------
# 2. Load tokenizer + model
# -------------------------------
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-base")

retriever = RagRetriever.from_pretrained(
    "facebook/rag-token-base",
    index_name="custom",
    passages_path=DATASET_PATH,
    index_path=INDEX_PATH
)

model = RagTokenForGeneration.from_pretrained(
    "facebook/rag-token-base",
    retriever=retriever
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# -------------------------------
# 3. FIXED tokenization (CRITICAL)
# -------------------------------
def tokenize_fn(batch):
    # 👇 You MUST define a query vs target
    # For now: simulate QA (you should replace later with real supervision)

    questions = ["What is this about?"] * len(batch["text"])
    answers = batch["text"]

    inputs = tokenizer.question_encoder(
        questions,
        truncation=True,
        padding="max_length",
        max_length=64
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer.generator(
            answers,
            truncation=True,
            padding="max_length",
            max_length=128
        )

    inputs["labels"] = labels["input_ids"]
    return inputs

dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=dataset.column_names
)

# -------------------------------
# 4. WandB
# -------------------------------
wandb.init(project="roleplay-rag", name="rag_finetune_v2")

# -------------------------------
# 5. Training arguments
# -------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,
    num_train_epochs=2,
    logging_steps=50,
    save_steps=500,
    fp16=torch.cuda.is_available(),
    report_to="wandb",
    remove_unused_columns=False  # 🔥 IMPORTANT for RAG
)

# -------------------------------
# 6. Trainer
# -------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# -------------------------------
# 7. Train
# -------------------------------
trainer.train()

# -------------------------------
# 8. Save
# -------------------------------
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"✅ Model saved to {OUTPUT_DIR}")