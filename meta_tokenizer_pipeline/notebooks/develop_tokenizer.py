# %% [markdown]
# # System-Wide Tokenizer Optimizer
# 
# This notebook determines the mathematically optimal **Vocabulary Size** for the 100M track database.
# It minimizes the total storage footprint:
# 
# **Total = Model Weights + Search Index + Metadata Database**

# %%
import os
import psycopg2
import pandas as pd
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

# Database Configuration
DB_HOST = os.environ.get("DB_HOST", "mb_db")
DB_NAME = os.environ.get("DB_NAME", "musicbrainz_db")
DB_USER = os.environ.get("DB_USER", "musicbrainz")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "musicbrainz")

def get_db_connection():
    return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)

# %% [markdown]
# ## 1. Load Real-World Data Sample
# We fetch (Artist, Album, Title) triplets to simulate the clustered storage architecture.

# %% 
def get_clustered_sample(sample_size=100000):
    conn = get_db_connection()
    cur = conn.cursor()
    # Join Recording -> Track -> Medium -> Release -> ArtistCredit
    query = f"""
        SELECT 
            ac.name as artist_name,
            r.name as album_name,
            rec.name as song_title
        FROM musicbrainz.recording rec
        JOIN musicbrainz.track t ON t.recording = rec.id
        JOIN musicbrainz.medium m ON t.medium = m.id
        JOIN musicbrainz.release r ON m.release = r.id
        JOIN musicbrainz.artist_credit ac ON rec.artist_credit = ac.id
        LIMIT {sample_size}
    """
    print(f"Fetching {sample_size} triplets from MusicBrainz...")
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return pd.DataFrame(rows, columns=['artist', 'album', 'title'])

df_sample = get_clustered_sample(100000)
print(f"Sample loaded. Unique Artists: {df_sample['artist'].nunique()}, Unique Albums: {df_sample['album'].nunique()}")

# %% [markdown]
# ## 2. The Cost Model
# 
# ### A. Model Weight Penalty
# The final layer of the MERT model is a linear projection: `768 -> VocabSize`.
# *   Size = `VocabSize * 768 * 2 bytes (Float16)`
# 
# ### B. Database Storage Strategy
# We determine the byte-width based on the vocabulary size:
# *   **V < 128:** 1 byte (7-bit)
# *   **V < 16,384:** 2 bytes (14-bit VarInt)
# *   **V < 65,536:** 2 bytes (16-bit Fixed)
# *   **V > 65,536:** 3 bytes (21-bit VarInt)

# %% 
def get_token_byte_width(vocab_size):
    if vocab_size < 128: return 1
    if vocab_size < 16384: return 2
    if vocab_size < 65536: return 2 # Fixed width 16-bit is more efficient than 3-byte VarInt
    return 3

def estimate_total_footprint(vocab_size, df):
    # 1. Train Tokenizer
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<UNK>"])
    
    # Combined corpus for training
    all_text = pd.concat([df['artist'], df['album'], df['title']]).astype(str).unique()
    temp_corpus = "/tmp/temp_corpus.txt"
    with open(temp_corpus, "w") as f:
        f.write("\n".join(all_text))
    tokenizer.train([temp_corpus], trainer)
    
    # 2. Measure Average Bytes (Clustered Model)
    byte_width = get_token_byte_width(vocab_size)
    
    def get_set_size(series):
        unique_vals = series.unique()
        total_tokens = sum(len(tokenizer.encode(str(x)).ids) for x in unique_vals)
        return total_tokens * byte_width, len(unique_vals)

    artist_payload, num_unique_artists = get_set_size(df['artist'])
    album_payload, num_unique_albums = get_set_size(df['album'])
    
    # Titles are stored for EVERY song (not deduplicated)
    title_tokens = sum(len(tokenizer.encode(str(x)).ids) for x in df['title'])
    title_payload = title_tokens * byte_width
    
    # 3. Project to 100 Million tracks
    # We maintain the ratios found in the sample
    artist_ratio = num_unique_artists / len(df)
    album_ratio = num_unique_albums / len(df)
    
    proj_artist = (artist_payload / num_unique_artists) * (100_000_000 * artist_ratio)
    proj_album = (album_payload / num_unique_albums) * (100_000_000 * album_ratio)
    proj_title = (title_payload / len(df)) * 100_000_000
    
    db_size_mb = (proj_artist + proj_album + proj_title) / (1024 * 1024)
    
    # 4. Model Weight Size
    # Base MERT (60MB) + Linear Head
    model_size_mb = 60 + (vocab_size * 768 * 2) / (1024 * 1024)
    
    # 5. Index Size (Fixed PQ)
    index_size_mb = 400 
    
    total_size_mb = db_size_mb + model_size_mb + index_size_mb
    
    return {
        "vocab_size": vocab_size,
        "db_mb": db_size_mb,
        "model_mb": model_size_mb,
        "total_mb": total_size_mb
    }

# %% [markdown]
# ## 3. Optimization Sweep
# We run the simulation across the major bit-width breakpoints.

# %% 
# Breakpoints and intermediate steps
sweep_values = [2000, 8000, 16383, 32000, 65535, 100000, 150000]
results = []

print("Starting Optimization Sweep...")
for v in sweep_values:
    print(f"Testing Vocab Size: {v}...")
    res = estimate_total_footprint(v, df_sample)
    results.append(res)

df_results = pd.DataFrame(results)

# %% [markdown]
# ## 4. The Recommendation
# Finding the global minimum.

# %% 
winner = df_results.loc[df_results['total_mb'].idxmin()]

print("\n" + "="*40)
print("FINAL OPTIMIZATION REPORT (100M Tracks)")
print("="*40)
print(df_results.to_string(index=False))
print("-" * 40)
print(f"OPTIMAL VOCAB SIZE: {winner['vocab_size']:.0f}")
print(f"TOTAL APP FOOTPRINT: {winner['total_mb']:.2f} MB")
print("="*40)

# Cleanup

if os.path.exists("/tmp/temp_corpus.txt"):

    os.remove("/tmp/temp_corpus.txt")
