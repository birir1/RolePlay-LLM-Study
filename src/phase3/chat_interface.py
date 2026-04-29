import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# SAF PIPELINE IMPORTS
from src.retrieval.generate_candidates import generate_candidates
from src.retrieval.saf_reranker import SAFReranker


# =========================================================
# MODEL CONFIG
# =========================================================
MODEL_PATHS = {
    "mt5": "google/mt5-small",
    "transformer": "gpt2"
}


# =========================================================
# LOAD MODEL
# =========================================================
def load_model(name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------
    # SAF is NOT a model
    # -------------------------
    if name == "saf":
        print("[INFO] SAF selected → using transformer generator + reranker pipeline")
        name = "transformer"

    model_name = MODEL_PATHS[name]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # GPT2 fix: missing pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if name == "mt5":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)

    model.to(device)
    model.eval()

    return tokenizer, model, device


# =========================================================
# PROMPT FORMATTING
# =========================================================
def format_prompt(model_name, query, context):
    context = context.strip() if context else ""

    if model_name == "mt5":
        return f"question: {query} context: {context}"

    return f"""You are a factual assistant.

Question: {query}
Context: {context}

Answer:"""


# =========================================================
# GENERATION
# =========================================================
def generate_answer(model_name, tokenizer, model, device, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# =========================================================
# SAF PIPELINE
# =========================================================
def saf_generate(query, tokenizer, model, device):
    """
    SAF = Retrieve → Rerank → Generate
    """

    # 1. retrieve candidates
    candidates = generate_candidates(query)

    # 2. rerank
    reranker = SAFReranker()
    top_docs = reranker.rerank(query, candidates, top_k=3)

    # 3. build context
    context = "\n".join([d.get("text", "") for d in top_docs])

    # 4. prompt
    prompt = format_prompt("transformer", query, context)

    # 5. generate
    return generate_answer("transformer", tokenizer, model, device, prompt)


# =========================================================
# MAIN INTERFACE
# =========================================================
def main():
    print("\n==============================")
    print("  ROLEPLAY MODEL INTERFACE")
    print("==============================\n")

    print("Available models:", list(MODEL_PATHS.keys()) + ["saf"])

    model_name = input("Select model: ").strip().lower()

    tokenizer, model, device = load_model(model_name)

    print("\nType 'exit' to quit\n")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() in ["exit", "quit"]:
            break

        context = input("Context (optional): ").strip()

        print("\n[GENERATING...]\n")

        # -------------------------
        # SAF PIPELINE
        # -------------------------
        if model_name == "saf":
            answer = saf_generate(query, tokenizer, model, device)
            print(f"SAF:\n{answer}")
        else:
            prompt = format_prompt(model_name, query, context)
            answer = generate_answer(model_name, tokenizer, model, device, prompt)
            print(f"{model_name.upper()}:\n{answer}")


if __name__ == "__main__":
    main()