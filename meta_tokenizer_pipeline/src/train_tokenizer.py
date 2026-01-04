import argparse
import os
import pandas as pd
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

def train_tokenizer(input_file, output_path, vocab_size=32000, columns=["song.title", "artist.name"]):
    """
    Trains a BPE tokenizer on multiple CSV file columns (Unified Vocabulary).
    """
    print(f"Loading data from {input_file}...")
    
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")

    all_texts = []

    # 1. Extract Text
    if input_file.endswith(".csv"):
        df = pd.read_csv(input_file)
        
        for col in columns:
            if col not in df.columns:
                print(f"Warning: Requested column '{col}' not found. Skipping.")
                continue
            
            print(f"  - Extracting column: {col}")
            col_texts = df[col].dropna().astype(str).tolist()
            all_texts.extend(col_texts)
            
        if not all_texts:
             raise ValueError("No valid text found in any of the specified columns.")
             
    else:
        # Assume raw text file
        with open(input_file, "r", encoding="utf-8") as f:
            all_texts = [line.strip() for line in f if line.strip()]

    print(f"Extracted {len(all_texts)} total items for training.")
    
    # 2. Prepare Temporary Training File
    # The Rust tokenizer trainer expects a file on disk
    temp_train_file = "temp_train_corpus.txt"
    with open(temp_train_file, "w", encoding="utf-8") as f:
        for t in all_texts:
            f.write(t + "\n")

    # 3. Configure Tokenizer
    print(f"Training BPE Tokenizer (Vocab Size: {vocab_size})...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        min_frequency=2, 
        special_tokens=["<UNK>", "<PAD>"]
    )

    # 4. Train
    tokenizer.train([temp_train_file], trainer)
    
    # 5. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    
    # Cleanup
    if os.path.exists(temp_train_file):
        os.remove(temp_train_file)
        
    print(f"Success! Tokenizer saved to: {output_path}")
    print(f"Vocab Size: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE Tokenizer for Music Metadata")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV or TXT file")
    parser.add_argument("--output", type=str, required=True, help="Path to save tokenizer.json")
    parser.add_argument("--columns", nargs='+', default=["song.title", "artist.name"], help="List of CSV columns to train on")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size (max 65535)")
    
    args = parser.parse_args()
    
    train_tokenizer(
        input_file=args.input,
        output_path=args.output,
        vocab_size=args.vocab_size,
        columns=args.columns
    )
