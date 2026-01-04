# %% [markdown]
# # Music Metadata Tokenizer Development (DB-Backed)
# 
# This notebook connects to the local MusicBrainz Postgres database to perform:
# 1.  **Vocabulary Analysis:** Determine the optimal vocab size.
# 2.  **Compression Tournament:** Compare Fixed-Width (uint16) vs VarInt strategies.
# 3.  **Tokenizer Training:** Train the winner.

# %%
import os
import psycopg2
import pandas as pd
from collections import Counter
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

# Config
DB_HOST = os.environ.get("DB_HOST", "mb_db")
DB_NAME = os.environ.get("DB_NAME", "musicbrainz_db")
DB_USER = os.environ.get("DB_USER", "musicbrainz")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "musicbrainz")

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

# %% [markdown]
# ## 1. Frequency Analysis
# We stream text from the DB to understand the "Long Tail" of music vocabulary.

# %% 
def stream_word_counts(limit=None):
    conn = get_db_connection()
    cur = conn.cursor(name='eda_cursor') # Server-side cursor
    
    query = """
        SELECT name FROM musicbrainz.artist WHERE name IS NOT NULL
        UNION ALL
        SELECT name FROM musicbrainz.recording WHERE name IS NOT NULL
    """
    
    print("Executing query...")
    cur.execute(query)
    
    word_counts = Counter()
    processed = 0
    
    while True:
        rows = cur.fetchmany(50000)
        if not rows:
            break
            
        for row in rows:
            text = row[0]
            # Simple pre-tokenization (whitespace)
            words = text.split() 
            word_counts.update(words)
            
        processed += len(rows)
        if processed % 1000000 == 0:
            print(f"Processed {processed/1000000:.1f}M rows...")
            
        if limit and processed >= limit:
            break
            
    cur.close()
    conn.close()
    return word_counts

# Run analysis on a subset (e.g., 1M rows) for interactivity.
print("Starting Frequency Analysis (Sample 1M)...")
counts = stream_word_counts(limit=1000000)
print(f"Total Unique Words: {len(counts)}")
print(f"Top 10 Words: {counts.most_common(10)}")

# %% [markdown]
# ## 2. Coverage Analysis
# How many words do we need to cover X% of the corpus?

# %% 
total_occurrences = sum(counts.values())
sorted_counts = counts.most_common()

cumulative = 0
vocab_needed_90 = 0
vocab_needed_95 = 0
vocab_needed_99 = 0

for rank, (word, count) in enumerate(sorted_counts):
    cumulative += count
    percent = cumulative / total_occurrences
    
    if percent >= 0.90 and vocab_needed_90 == 0:
        vocab_needed_90 = rank + 1
    if percent >= 0.95 and vocab_needed_95 == 0:
        vocab_needed_95 = rank + 1
    if percent >= 0.99 and vocab_needed_99 == 0:
        vocab_needed_99 = rank + 1
        break

print(f"Vocab size for 90% coverage: {vocab_needed_90}")
print(f"Vocab size for 95% coverage: {vocab_needed_95}")
print(f"Vocab size for 99% coverage: {vocab_needed_99}")

# %% [markdown]
# ## 3. Compression Tournament
# Comparing **Fixed uint16 (65k)** vs **VarInt (16k)**.
# 
# *   **Fixed:** Every token = 2 bytes. Vocab = 65,535.
# *   **VarInt:** IDs < 128 = 1 byte. IDs > 127 = 2 bytes. Vocab = 16,383 (Limit for 2 bytes).

# %% 
def simulate_size(vocab_size, use_varint, text_sample):
    # Train a temp tokenizer with this config
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<UNK>"])
    
    # We need a file for training
    with open("temp_sample.txt", "w") as f:
        f.write("\n".join(text_sample))
        
    tokenizer.train(["temp_sample.txt"], trainer)
    
    total_bytes = 0
    token_count = 0
    
    for text in text_sample:
        ids = tokenizer.encode(text).ids
        token_count += len(ids)
        
        if use_varint:
            for i in ids:
                if i < 128: total_bytes += 1
                elif i < 16384: total_bytes += 2
                else: total_bytes += 3 # Should not happen if vocab < 16k
        else:
            # Fixed uint16
            total_bytes += len(ids) * 2
            
    return total_bytes, token_count

# Generate sample for simulation from the elements of our counter
sample_texts = [x for x in list(counts.elements())[:50000]] 

print("\n--- Running Tournament ---")

# Scenario A: VarInt (Max 16k)
size_a, count_a = simulate_size(16000, True, sample_texts)
print(f"Scenario A (VarInt 16k): {size_a / 1024:.2f} KB")

# Scenario B: Fixed (Max 65k)
size_b, count_b = simulate_size(60000, False, sample_texts)
print(f"Scenario B (Fixed 60k):  {size_b / 1024:.2f} KB")

winner = "VarInt" if size_a < size_b else "Fixed"
print(f"\nWINNER: {winner}")