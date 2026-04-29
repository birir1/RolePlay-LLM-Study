import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch

MODEL_NAME = "microsoft/phi-3-mini-4k-instruct"


def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)

    texts = []
    for d in data:
        prompt = f"""<|user|>
{d['query']}

Context:
{d['context']}

<|assistant|>
{d['response']}"""
        texts.append({"text": prompt})

    return Dataset.from_list(texts)


def tokenize(dataset, tokenizer):
    return dataset.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            padding="max_length",
            max_length=512
        ),
        batched=True
    )


def main():
    train = load_data("data/processed/train.json")
    val = load_data("data/processed/val.json")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    train = tokenize(train, tokenizer)
    val = tokenize(val, tokenizer)

    training_args = TrainingArguments(
        output_dir="outputs/phi3",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="steps",
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=val,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    trainer.train()
    trainer.save_model("outputs/phi3/final")


if __name__ == "__main__":
    main()