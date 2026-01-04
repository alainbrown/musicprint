# %% [markdown]
# # Music Metadata Tokenizer Development (Spotify Dataset)
# 
# This notebook develops the **BPE (Byte Pair Encoding)** tokenizer for the MusicPrint project.
# 
# **Goal:** Compress song titles, artists, and album names by ~60-80% to fit 100M songs on an iPhone.
# **Dataset:** Spotify Tracks Dataset (114k entries) from Hugging Face.

# %% [markdown]
# ## 1. Setup & Data Ingestion
# We use a real-world Spotify dataset containing track names, artists, and album names.

# %%
import os
import requests
import pandas as pd
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

# Configuration
DATA_URL = "https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset/resolve/main/dataset.csv"
DATA_DIR = "../data"
RAW_FILE = os.path.join(DATA_DIR, "spotify_dataset.csv")
CORPUS_FILE = os.path.join(DATA_DIR, "unified_corpus.txt")

os.makedirs(DATA_DIR, exist_ok=True)

# %% 
# Download the dataset if not present
if not os.path.exists(RAW_FILE):
    print(f"Downloading {DATA_URL}...")
    try:
        # Use a user-agent to avoid potential blocks
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(DATA_URL, headers=headers, stream=True)
        response.raise_for_status()
        with open(RAW_FILE, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download: {e}")
else:
    print("Dataset already exists.")

# %% 
# Load and Inspect
if os.path.exists(RAW_FILE):
    # This dataset is ~20MB, easy for pandas
    df = pd.read_csv(RAW_FILE)
    print(f"Loaded {len(df)} tracks.")
    
    # Identify relevant columns
    cols = ['track_name', 'artists', 'album_name']
    print("\n--- DATA PREVIEW ---")
    print(df[cols].head())
else:
    print("Error: Dataset file not found.")
    df = pd.DataFrame()

# %% [markdown]
# ## 2. Data Preparation
# We combine all text fields (Track, Artist, Album) into a single corpus for "Unified Tokenization".

# %% 
if not df.empty:
    all_texts = []
    for col in ['track_name', 'artists', 'album_name']:
        print(f"Extracting {col}...")
        all_texts.extend(df[col].dropna().astype(str).tolist())
    
    # Save to disk for the Rust-based trainer
    with open(CORPUS_FILE, "w", encoding="utf-8") as f:
        for line in all_texts:
            f.write(line + "\n")
            
    print(f"\nExported {len(all_texts)} strings to {CORPUS_FILE}")
    print(f"Sample: {all_texts[:5]}")

# %% [markdown]
# ## 3. Train Unified BPE Tokenizer
# We learn a vocabulary of 32,000 "Music Tokens".

# %% 
if os.path.exists(CORPUS_FILE):
    # Initialize a BPE Tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Pre-tokenizer: ByteLevel handles whitespaces and casing well
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=32000, 
        min_frequency=2, 
        special_tokens=["<UNK>", "<PAD>"]
    )
    
    # Train!
    print("Training tokenizer (Rust backend)...")
    tokenizer.train([CORPUS_FILE], trainer)
    print("Training complete.")
    
    # Save the model
    MODEL_PATH = os.path.join(DATA_DIR, "spotify_unified_tokenizer.json")
    tokenizer.save(MODEL_PATH)
    print(f"Tokenizer saved to {MODEL_PATH}")
else:
    print("Skipping training: No corpus file.")

# %% [markdown]
# ## 4. Analyze Compression
# Measuring the efficiency gain of our unified vocabulary.

# %% 
def get_varint_size(token_id):
    """Estimate size if using Variable Length Integers (0-127 = 1b, rest = 2b)"""
    return 1 if token_id < 128 else 2

def analyze_compression(text_list, tokenizer):
    raw_bytes = 0
    token_bytes = 0
    token_count = 0
    
    for text in text_list:
        encoded = tokenizer.encode(text)
        ids = encoded.ids
        raw_bytes += len(text.encode('utf-8'))
        for tid in ids:
            token_bytes += get_varint_size(tid)
        token_count += len(ids)
        
    return raw_bytes, token_bytes, token_count

if 'tokenizer' in locals() and not df.empty:
    # Test on 1000 random songs from the dataset
    test_samples = df.sample(1000)
    test_texts = []
    for _, row in test_samples.iterrows():
        test_texts.extend([str(row['track_name']), str(row['artists']), str(row['album_name'])])
        
    raw_size, compressed_size, num_tokens = analyze_compression(test_texts, tokenizer)
    
    print(f"--- Compression Results (3000 field samples) ---")
    print(f"Raw UTF-8 Size:       {raw_size / 1024:.2f} KB")
    print(f"Tokenized Size:       {compressed_size / 1024:.2f} KB")
    print(f"Compression Ratio:    {raw_size / compressed_size:.2f}x")
    print(f"Space Saving:         {100 * (1 - compressed_size/raw_size):.2f}%")
    print(f"Avg Tokens per Field: {num_tokens / len(test_texts):.2f}")

# %% [markdown]
# ## 5. Performance Stress Test
# Inspecting how the tokenizer handles complex, commercial song titles.

# %% 
def show_tokens(text):
    if 'tokenizer' in locals():
        encoded = tokenizer.encode(text)
        print(f"'{text}'")
        print(f"  Tokens: {encoded.tokens}")
        print(f"  IDs:    {encoded.ids} (Bytes: {sum(get_varint_size(i) for i in encoded.ids)})")
        print("-" * 30)

if 'tokenizer' in locals():
    print("--- Edge Case Examples ---")
    show_tokens("Can't Help Falling In Love")
    show_tokens("Ghost - Acoustic")
    show_tokens("Symphony No. 5 in C Minor, Op. 67: I. Allegro con brio")
    show_tokens("Taylor Swift feat. Bon Iver")
    show_tokens("Remastered 2022 (Live at Wembley)")
    
    # Inspect the vocabulary
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    print(f"\nTop 10 Tokens (learned music vocabulary): {sorted_vocab[:10]}")
