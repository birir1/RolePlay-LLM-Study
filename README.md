# RolePlay-LLM-Study

AI research framework for analyzing and improving reliability in role-playing Large Language Models (LLMs), focusing on hallucination, sycophancy, and persona drift.

---

## Overview

This project investigates key failure modes in role-playing LLMs:

- **Hallucination** – generating false or unsupported information  
- **Sycophancy** – agreeing with user even when incorrect  
- **Persona Drift** – losing consistency in role-play  

We introduce a **SAF-RAG (Safety-Aware Retrieval-Augmented Generation)** pipeline to mitigate these issues using:
- Retrieval grounding
- Safety filtering
- Behavioral evaluation

---

## System Architecture

> (Replace with your diagram)

![Architecture](docs/images/architecture.png)

---

## Pipeline Flow

> (Replace with your flow diagram)

![Pipeline](docs/images/pipeline.png)

---

## Project Structure

---

## Installation

### 1. Clone repo
```bash
git clone https://github.com/birir1/RolePlay-LLM-Study.git
cd RolePlay-LLM-Study

```

## Create environment
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# or
.venv\Scripts\activate      # Windows


## Install Dependencies
> pip install -r requirements.txt

## Running the Full Pipeline
---
Step 1: Prepare Data
 > python src/phase1/prepare_dataset.py

(Optional: build roleplay dataset)

python src/phase1/build_roleplay_dataset.py

---
Step 2: Train Models
 > python src/phase1/train_mt5.py
 > python src/phase1/train_bart.py
 > python src/phase1/train_t5_small.py

---
Step 3: Retrieval + RAG Pipeline
 > python src/phase2/run_rag_pipeline.py

---
Step 4: Generate Predictions
 > python src/phase3/generate_predictions.py

---
Step 5: Evaluate Models
 > python src/phase3/run_evaluation.py

Sycophancy-specific:

python src/phase3/run_sycophancy_evaluation.py
🔹 Step 6: SAF-RAG Training
python src/phase4/train_saf_rag.py
🔹 Step 7: SAF-RAG Evaluation
python src/phase4/evaluate_saf_rag.py
🔹 Step 8: Final Analysis
python src/phase5/analyze_results.py
