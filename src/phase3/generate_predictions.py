import os
import json
from tqdm import tqdm
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoConfig
)

# =========================
# CONFIG
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_INPUT_LENGTH = 512
MAX_OUTPUT_LENGTH = 128
BATCH_SIZE = 4

DATA_PATH = "data/processed/roleplay_benchmark/roleplay_test.json"

MODEL_PATHS = {
    "mt5_baseline": "models/mt5_baseline",
    "transformer_baseline": "models/transformer_baseline",
    "transformer_prompted": "models/transformer_prompted"
}

OUTPUT_DIR = "outputs/predictions"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD DATASET
# =========================
def load_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)

    print(f"[INFO] Loaded dataset: {len(data)} samples")
    return data


# =========================
# LOAD MODEL
# =========================
def load_model(model_path):
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if config.is_encoder_decoder:
        print("[INFO] Detected Seq2Seq model")

        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model_type = "seq2seq"

        tokenizer.padding_side = "right"

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    else:
        print("[INFO] Detected Causal LM model")

        model = AutoModelForCausalLM.from_pretrained(model_path)
        model_type = "causal"

        tokenizer.padding_side = "left"

        if tokenizer.pad_token is None:
            print("[WARN] No pad_token found. Using eos_token as pad_token.")
            tokenizer.pad_token = tokenizer.eos_token

        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id

    model.to(DEVICE)
    model.eval()

    return tokenizer, model, model_type


# =========================
# GENERATION (IMPROVED)
# =========================
def generate_batch(tokenizer, model, inputs, model_type):
    encodings = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        if model_type == "seq2seq":
            outputs = model.generate(
                **encodings,
                max_length=MAX_OUTPUT_LENGTH,
                num_beams=4,
                early_stopping=True
            )
        else:
            outputs = model.generate(
                **encodings,
                max_length=MAX_OUTPUT_LENGTH,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )

    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded


# =========================
# RUN INFERENCE
# =========================
def run_inference(model_name, model_path, dataset):
    print(f"\n[INFO] Loading model: {model_name}")
    tokenizer, model, model_type = load_model(model_path)

    results = []

    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):
        batch = dataset[i:i + BATCH_SIZE]

        inputs = [item["input"] for item in batch]
        outputs = generate_batch(tokenizer, model, inputs, model_type)

        for j, output in enumerate(outputs):
            sample = batch[j]

            results.append({
                # 🔥 CORE STRUCTURE FOR EVALUATION
                "query": sample["input"],        # misleading prompt
                "context": sample["target"],     # ground truth
                "prediction": output,

                # metadata (for analysis later)
                "source": sample.get("source", "unknown"),
                "type": sample.get("type", "misleading_roleplay")
            })

        # Limit for fast experiments
        if len(results) >= 1500:
            break

    save_path = os.path.join(OUTPUT_DIR, f"{model_name}.json")

    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[SUCCESS] Saved predictions → {save_path}")
    print(f"[INFO] Total predictions: {len(results)}")

    return results


# =========================
# MAIN
# =========================
def main():
    dataset = load_dataset(DATA_PATH)

    for model_name, model_path in MODEL_PATHS.items():
        try:
            run_inference(model_name, model_path, dataset)
        except Exception as e:
            print(f"[ERROR] Failed on {model_name}: {str(e)}")
            continue


if __name__ == "__main__":
    main()