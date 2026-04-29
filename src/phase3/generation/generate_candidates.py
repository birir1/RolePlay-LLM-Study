from typing import List, Dict, Any, Optional, Tuple
import os
import random
import torch
from functools import lru_cache

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM
)

# =========================================================
# MODEL REGISTRY (CAN BE OVERRIDDEN VIA ENV)
# =========================================================
DEFAULT_MODEL_PATHS = {
    "mt5": "google/mt5-small",
    "transformer": "gpt2"
}


def get_model_registry() -> Dict[str, str]:
    """
    Allows safe override without code changes.
    """
    return {
        "mt5": os.getenv("MT5_MODEL", DEFAULT_MODEL_PATHS["mt5"]),
        "transformer": os.getenv("TRANSFORMER_MODEL", DEFAULT_MODEL_PATHS["transformer"]),
    }


# =========================================================
# DEVICE
# =========================================================
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# MODEL LOADER (CACHED → IMPORTANT FOR PERFORMANCE)
# =========================================================
@lru_cache(maxsize=4)
def load_model(model_name: str) -> Tuple[Any, Any, torch.device]:
    """
    Loads and caches model + tokenizer.
    Avoids reloading during multi-candidate generation.
    """

    registry = get_model_registry()

    if model_name not in registry:
        raise ValueError(f"[ERROR] Unknown model: {model_name}")

    model_id = registry[model_name]
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # GPT2 fix: no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_name == "mt5":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)

    model.to(device)
    model.eval()

    return tokenizer, model, device


# =========================================================
# PROMPT BUILDER
# =========================================================
def build_prompt(model_name: str, query: str, context: str = "") -> str:
    context = context.strip() if context else ""

    if model_name == "mt5":
        return f"question: {query} context: {context}".strip()

    return (
        "You are a factual assistant.\n\n"
        f"Question: {query}\n"
        f"Context: {context}\n\n"
        "Answer:"
    )


# =========================================================
# SAFE GENERATION CORE
# =========================================================
def generate_single(
    tokenizer,
    model,
    device,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int
) -> str:

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text.strip()


# =========================================================
# MAIN GENERATION PIPELINE
# =========================================================
def generate_candidates(
    query: str,
    context: str = "",
    models: Optional[List[str]] = None,
    n_per_model: int = 2,
    generation_config: Optional[Dict[str, float]] = None
) -> List[Dict[str, Any]]:
    """
    SAF-RAG Candidate Generator

    Returns:
        List of:
        {
            "text": str,
            "model": str,
            "query": str,
            "context": str,
            "temperature": float,
            "top_p": float
        }
    """

    if models is None:
        models = list(get_model_registry().keys())

    # default safe decoding config
    cfg = generation_config or {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 120
    }

    candidates: List[Dict[str, Any]] = []

    for model_name in models:

        try:
            tokenizer, model, device = load_model(model_name)
        except Exception as e:
            print(f"[WARN] Failed to load {model_name}: {e}")
            continue

        prompt = build_prompt(model_name, query, context)

        for _ in range(n_per_model):

            temp = round(random.uniform(0.6, 0.9), 2)
            top_p = cfg.get("top_p", 0.9)

            try:
                output = generate_single(
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    prompt=prompt,
                    temperature=temp,
                    top_p=top_p,
                    max_new_tokens=cfg.get("max_new_tokens", 120)
                )

                candidates.append({
                    "text": output,
                    "model": model_name,
                    "query": query,
                    "context": context,
                    "temperature": temp,
                    "top_p": top_p
                })

            except Exception as e:
                print(f"[ERROR] generation failed ({model_name}): {e}")

    return candidates


# =========================================================
# CLI TEST (OPTIONAL)
# =========================================================
if __name__ == "__main__":

    q = "Where is the Eiffel Tower located?"
    ctx = ""

    results = generate_candidates(
        query=q,
        context=ctx,
        models=["mt5", "transformer"],
        n_per_model=2
    )

    for i, r in enumerate(results):
        print(f"\n--- CANDIDATE {i} ---")
        print("MODEL:", r["model"])
        print(r["text"])