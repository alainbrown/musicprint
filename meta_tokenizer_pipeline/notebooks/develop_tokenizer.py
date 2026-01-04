# %% [markdown]
# # Music Metadata Tokenizer Development
# 
# This notebook develops the **BPE (Byte Pair Encoding)** tokenizer for the MusicPrint project.
# 
# **Goal:** Compress song titles and artist names by ~60-80% to fit 100M songs on an iPhone.
# **Method:** 
# 1. Download a real-world dataset (Million Song Dataset subset).
# 2. Train a specialized BPE tokenizer on the text.
# 3. Measure the compression ratio (Raw Bytes vs. Token IDs).

# %% [markdown]
# ## 1. Setup & Data Ingestion
# We use the CORGIS 'Music' dataset (derived from the Million Song Dataset) as a proxy for our 100M track catalog.

# %%
import os
import requests
import pandas as pd
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

# Configuration
DATA_URL = "https://corgis-edu.github.io/corgis/datasets/csv/music/music.csv"
DATA_DIR = "../data"
RAW_FILE = os.path.join(DATA_DIR, "music.csv")
TITLES_FILE = os.path.join(DATA_DIR, "titles_for_training.txt")

os.makedirs(DATA_DIR, exist_ok=True)

# %% 
# Download the dataset if not present
if not os.path.exists(RAW_FILE):
    print(f"Downloading {DATA_URL}...")
    try:
        response = requests.get(DATA_URL)
        response.raise_for_status()
        with open(RAW_FILE, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download: {e}")
else:
    print("Dataset already exists.")

# %% 
# Load and Inspect
if os.path.exists(RAW_FILE):
    df = pd.read_csv(RAW_FILE)
    print(f"Loaded {len(df)} songs.")
    print("\nColumns:", df.columns.tolist())
    
    # DEBUG: Inspect the actual content
    print("\n--- DATA INSPECTION ---")
    print(df[['song.title', 'release.name', 'artist.name']].head(10))
    print("-----------------------")
else:
    print("Error: Dataset file not found.")
    df = pd.DataFrame()
    
    
    
    # %% [markdown]
# ## 2. Data Preparation
# We need a plain text file for the Tokenizer trainer. We will verify the column names and export the titles.

# %%
if not df.empty:
    # 'song.title' and 'release.name' are corrupted (zeros) in this specific dataset subset.
    # We use 'artist.name' as a proxy for text compression analysis.
    target_col = 'artist.name'
    
    print(f"Using '{target_col}' for tokenizer training (proxy for song titles).")
            
    if target_col and target_col in df.columns:
        titles = df[target_col].dropna().astype(str).tolist()
        
        # Save to disk for the Rust-based trainer
        with open(TITLES_FILE, "w", encoding="utf-8") as f:
            for t in titles:
                f.write(t + "\n")
                
        print(f"Exported {len(titles)} text items to {TITLES_FILE}")
        print(f"Sample items: {titles[:5]}")
    else:
        print(f"Error: Target column '{target_col}' not found.")
        titles = []
else:
    titles = []

# %% [markdown]
# ## 3. Train BPE Tokenizer
# We utilize the `tokenizers` library (HuggingFace) to learn the subword vocabulary.

# %% 
if os.path.exists(TITLES_FILE):
    # Initialize a BPE Tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Pre-tokenizer: Split by whitespace first (standard for English-like text)
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Trainer
    # vocab_size: 30,000 is the sweet spot. 
    # It fits in a 2-byte integer (max 65535) and covers 99% of words.
    trainer = trainers.BpeTrainer(
        vocab_size=32000, 
        min_frequency=2, 
        special_tokens=["<UNK>", "<PAD>"]
    )
    
    # Train!
    print("Training tokenizer...")
    tokenizer.train([TITLES_FILE], trainer)
    print("Training complete.")
    
    # Save the model
    tokenizer.save(os.path.join(DATA_DIR, "music_title_tokenizer.json"))
    print("Tokenizer saved.")
else:
    print("Skipping training: No titles file.")

# %% [markdown]
# ## 4. Analyze Compression
# Let's calculate the compression ratio.

# %% 
def get_varint_size(token_id):
    """Estimate size if using Variable Length Integers"""
    if token_id < 128:
        return 1
    return 2 # Simple VarInt (for < 32k vocab)

def analyze_compression(text_list, tokenizer):
    raw_bytes = 0
    token_bytes = 0
    token_count = 0
    
    for text in text_list:
        encoded = tokenizer.encode(text)
        ids = encoded.ids
        
        # Raw UTF-8 length
        raw_bytes += len(text.encode('utf-8'))
        
        # Compressed length
        for tid in ids:
            token_bytes += get_varint_size(tid)
            
        token_count += len(ids)
        
    return raw_bytes, token_bytes, token_count

if 'tokenizer' in locals() and 'titles' in locals():
    # Run analysis on a subset
    sample_subset = titles[:1000]
    raw_size, compressed_size, num_tokens = analyze_compression(sample_subset, tokenizer)
    
    print(f"--- Compression Results (Sample of {len(sample_subset)} titles) ---")
    print(f"Raw UTF-8 Size:       {raw_size / 1024:.2f} KB")
    print(f"Tokenized Size:       {compressed_size / 1024:.2f} KB")
    print(f"Compression Ratio:    {raw_size / compressed_size:.2f}x")
    print(f"Space Saving:         {100 * (1 - compressed_size/raw_size):.2f}%")
    print(f"Avg Tokens per Title: {num_tokens / len(sample_subset):.2f}")

# %% [markdown]
# ## 5. Visual Inspection
# Let's look at what tokens are actually being learned.

# %% 
def show_tokens(text):
    if 'tokenizer' in locals():
        encoded = tokenizer.encode(text)
        print(f"Original: '{text}'")
        print(f"Tokens:   {encoded.tokens}")
        print(f"IDs:      {encoded.ids}")
        print("-" * 30)

if 'tokenizer' in locals():
    print("--- Tokenization Examples ---")
    show_tokens("Symphony No. 5")
    show_tokens("Love Remix")
    show_tokens("Concerto in C Minor")
    show_tokens("Taylor Swift") 
    
    # Inspect the vocabulary
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    print(f"\nTop 10 Tokens: {sorted_vocab[:10]}")