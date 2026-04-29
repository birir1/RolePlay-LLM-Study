import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
import torch

MODEL_NAME = "meta-llama/Llama-2-7b-hf"


def load_data(path):
    with open(path, "r") as f:
        data = json.load(f)

    formatted = []
    for d in data:
        text = f"""### Instruction:
{d['query']}

### Context:
{d['context']}

### Response:
{d['response']}"""
        formatted.append({"text": text})

    return Dataset.from_list(formatted)


def tokenize(ds, tokenizer):
    return ds.map(
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        load_in_8bit=True,
        device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    train = tokenize(train, tokenizer)

    args = TrainingArguments(
        output_dir="outputs/llama_lora",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train
    )

    trainer.train()
    model.save_pretrained("outputs/llama_lora/final")


if __name__ == "__main__":
    main()