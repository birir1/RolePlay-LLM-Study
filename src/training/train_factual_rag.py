import os
import torch
import faiss
import numpy as np
from tqdm import tqdm
import wandb

from datasets import load_from_disk
from transformers import (
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

# -------------------------------
# Paths
# -------------------------------
DATASET_PATH = "data/rag_index/my_dataset"
INDEX_PATH = "data/rag_index/hf_dataset_index.faiss"
OUTPUT_DIR = "outputs/rag_factual"

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------
# 1. Load dataset
# -------------------------------
dataset = load_from_disk(DATASET_PATH)
print(f"✅ Loaded dataset: {len(dataset)} passages")

index = faiss.read_index(INDEX_PATH)
print("✅ FAISS index loaded")

# -------------------------------
# 2. Load models
# -------------------------------
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
)

question_encoder = DPRQuestionEncoder.from_pretrained(
    "facebook/dpr-question_encoder-single-nq-base"
).to(device)

generator_tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

generator = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-base"
).to(device)

generator.gradient_checkpointing_enable()

# -------------------------------
# 3. Retrieval
# -------------------------------
def retrieve(query, top_k=3):
    inputs = question_tokenizer(
        query,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        query_vec = question_encoder(**inputs).pooler_output.cpu().numpy()

    scores, indices = index.search(query_vec, top_k)

    docs = [dataset[int(i)]["text"] for i in indices[0]]
    return docs


# -------------------------------
# 4. Build training examples
# -------------------------------
def build_examples(batch):
    inputs = []
    targets = []

    for text in batch["text"]:
        docs = retrieve(text)

        context = " ".join(docs)

        # 🔥 INPUT
        input_text = f"question: {text} context: {context}"

        # 🔥 TARGET (IMPORTANT FIX)
        # Instead of copying text, we make it summarization-style
        target_text = context[:256]  # crude factual grounding proxy

        inputs.append(input_text)
        targets.append(target_text)

    model_inputs = generator_tokenizer(
        inputs,
        truncation=True,
        max_length=256
    )

    labels = generator_tokenizer(
        targets,
        truncation=True,
        max_length=256
    )

    # mask padding
    labels_ids = labels["input_ids"]
    labels_ids = [
        [(tok if tok != generator_tokenizer.pad_token_id else -100) for tok in seq]
        for seq in labels_ids
    ]

    model_inputs["labels"] = labels_ids
    return model_inputs


print("⚙️ Building training dataset...")

# 🔥 LIMIT DATASET (VERY IMPORTANT)
train_dataset = dataset.shuffle(seed=42).select(range(50000))

train_dataset = train_dataset.map(
    build_examples,
    batched=True,
    batch_size=4,   # keep small (retrieval inside)
    remove_columns=train_dataset.column_names,
    desc="Building RAG samples"
)

# -------------------------------
# 5. Training
# -------------------------------
wandb.init(project="factual-rag", name="rag_factual_v2")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=generator_tokenizer,
    model=generator
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,

    learning_rate=3e-5,
    num_train_epochs=2,

    logging_steps=50,
    save_steps=500,

    fp16=torch.cuda.is_available(),

    dataloader_num_workers=2,
    dataloader_pin_memory=True,

    report_to="wandb"
)

trainer = Trainer(
    model=generator,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=generator_tokenizer,
    data_collator=data_collator
)

# -------------------------------
# 6. Train
# -------------------------------
trainer.train()

# -------------------------------
# 7. Save
# -------------------------------
trainer.save_model(OUTPUT_DIR)
generator_tokenizer.save_pretrained(OUTPUT_DIR)

print(f" Model saved at {OUTPUT_DIR}")