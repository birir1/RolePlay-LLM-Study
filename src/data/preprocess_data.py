import os
import json
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

class RolePlayDataProcessor:
    def __init__(self, model_name="gpt2", max_length=1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Ensure gpt2 has a pad token to avoid errors during batching
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.max_length = max_length
        self.raw_dir = "data/raw"
        self.processed_dir = "data/processed"
        self.tokenized_dir = "data/tokenized"
        self.rag_dir = "data/rag_corpus"
        
        for d in [self.tokenized_dir, self.rag_dir]:
            os.makedirs(d, exist_ok=True)

    def clean_text_columns(self, df):
        """Silences Pandas4Warning and handles column merging"""
        # Explicitly include 'string' and 'object' per the new Pandas migration guide
        cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
        if not cols:
            return None
        # Efficiently join text columns
        return df[cols].apply(lambda x: " ".join(x.dropna().astype(str)), axis=1).tolist()

    def batch_tokenize(self, texts):
        """Uses the fast tokenizer's batch abilities (much faster than a loop)"""
        return self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=False,  # Save space in JSON by not padding here
            add_special_tokens=True
        )["input_ids"]

    def process_csvs(self):
        csv_files = [f for f in os.listdir(self.processed_dir) if f.endswith(".csv")]
        rag_passages = []

        for file in tqdm(csv_files, desc="Processing CSVs"):
            path = os.path.join(self.processed_dir, file)
            df = pd.read_csv(path)
            
            texts = self.clean_text_columns(df)
            if texts:
                # Tokenize and save
                tokenized = self.batch_tokenize(texts)
                save_name = f"{os.path.splitext(file)[0]}_tokenized.json"
                
                with open(os.path.join(self.tokenized_dir, save_name), "w") as f:
                    json.dump(tokenized, f)
                
                # Prepare RAG passages
                for t in texts:
                    rag_passages.append({"source": file, "text": t})
        
        return rag_passages

    def process_text_files(self, filenames, rag_passages):
        for fname in tqdm(filenames, desc="Processing Text Files"):
            path = os.path.join(self.raw_dir, fname)
            if not os.path.exists(path):
                continue
            
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                lines = [line.strip() for line in f if line.strip()]
            
            if lines:
                tokenized = self.batch_tokenize(lines)
                with open(os.path.join(self.tokenized_dir, f"{fname}_tokenized.json"), "w") as f:
                    json.dump(tokenized, f)
                
                for i, line in enumerate(lines):
                    rag_passages.append({"source": f"{fname}_{i}", "text": line})
        
        return rag_passages

    def run(self):
        # 1. Process CSVs
        passages = self.process_csvs()
        
        # 2. Process Raw Text (OpenSubtitles)
        text_files = ["OpenSubtitles.eu-ko.ko"]
        passages = self.process_text_files(text_files, passages)
        
        # 3. Save RAG Corpus
        with open(os.path.join(self.rag_dir, "rag_passages.json"), "w", encoding="utf-8") as f:
            json.dump(passages, f, ensure_ascii=False, indent=2)
            
        print(f"\n🚀 Preprocessing Complete.")
        print(f"Total RAG Passages: {len(passages)}")

if __name__ == "__main__":
    processor = RolePlayDataProcessor(model_name="gpt2")
    processor.run()