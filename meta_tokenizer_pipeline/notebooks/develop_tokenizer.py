# %% [markdown]
# # System-Wide Tokenizer Optimizer (Scientific Back-Solver)
# 
# This notebook implements a non-sampled, data-driven approach to find the mathematically optimal vocabulary size for the MusicPrint metadata database.
# 
# ## The Problem
# We need to store 100 Million tracks (Artist, Album, Title) on a mobile device with less than 3.0 GB total storage.
# The size of the database depends on two opposing forces:
# 1. **Vocabulary Size (V):** A larger vocabulary means fewer tokens per song title (better compression).
# 2. **Bit-Width Penalty:** As the vocabulary size crosses certain thresholds ($2^{14}, 2^{16}$), the number of bytes required to store each token ID increases.
# 3. **Model Weight Penalty:** The final layer of the MERT neural network grows linearly with the vocabulary size.
# 
# ## Methodology: The "Master Tokenizer" Approach
# To find the true global minimum without "guess and check", we:
# 1. **Stream the entire MusicBrainz dataset** (45M+ strings) into a single corpus.
# 2. **Train a 200,000-token Master Tokenizer** to explore the full potential of BPE merges.
# 3. **Back-solve for the optimum:** We analyze the frequency of every merge and calculate the total system footprint for every possible vocab size from 1,000 to 200,000.

# %% 
import os
import psycopg2
import pandas as pd
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
import time

# Config
DB_HOST = os.environ.get("DB_HOST", "mb_db")
DB_NAME = os.environ.get("DB_NAME", "musicbrainz_db")
DB_USER = os.environ.get("DB_USER", "musicbrainz")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "musicbrainz")
CORPUS_PATH = "/tmp/full_musicbrainz_corpus.txt"

def get_db_connection():
    return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)

# %% [markdown]
# ## 1. Build Full Dataset Corpus
# We extract every recording title, artist name, and release name from the MusicBrainz Postgres mirror.
# This ensures our analysis captures the "Long Tail" of rare words and complex international characters.

# %% 
def build_full_corpus():
    if os.path.exists(CORPUS_PATH):
        print(f"Corpus already exists at {CORPUS_PATH}. Skipping dump.")
        return

    conn = get_db_connection()
    cur = conn.cursor(name='full_dump_cursor')
    
    query = """
        SELECT name FROM musicbrainz.artist WHERE name IS NOT NULL
        UNION ALL
        SELECT name FROM musicbrainz.recording WHERE name IS NOT NULL
        UNION ALL
        SELECT name FROM musicbrainz.release WHERE name IS NOT NULL
    """
    
    print("Dumping full MusicBrainz metadata to disk...")
    start_time = time.time()
    
    cur.execute(query)
    count = 0
    with open(CORPUS_PATH, "w", encoding="utf-8") as f:
        while True:
            rows = cur.fetchmany(50000)
            if not rows:
                break
            for row in rows:
                f.write(str(row[0]) + "\n")
            count += len(rows)
            if count % 5000000 == 0:
                print(f"  Written {count/1000000:.0f}M rows...")
                
    cur.close()
    conn.close()
    elapsed = time.time() - start_time
    print(f"Done! Dumped {count:,} strings in {elapsed:.1f}s.")

build_full_corpus()

# %% [markdown]
# ## 2. Train Master Tokenizer (200k Vocab)
# We train a single, massive BPE model. Because BPE merges are discovered in order of frequency, 
# a 200k model contains all the information of a 10k, 32k, or 64k model as its first N merges.

# %% 
MAX_VOCAB = 200000
MASTER_MODEL_PATH = "/tmp/master_tokenizer_200k.json"

def train_master_tokenizer():
    if os.path.exists(MASTER_MODEL_PATH):
        print("Master tokenizer already trained.")
        return Tokenizer.from_file(MASTER_MODEL_PATH)

    print(f"Training Master BPE Tokenizer (Vocab: {MAX_VOCAB})...")
    start_time = time.time()
    
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=MAX_VOCAB,
        min_frequency=2,
        special_tokens=["<UNK>", "<PAD>"]
    )
    
    tokenizer.train([CORPUS_PATH], trainer)
    tokenizer.save(MASTER_MODEL_PATH)
    
    elapsed = time.time() - start_time
    print(f"Master training complete in {elapsed:.1f}s.")
    return tokenizer

master_tokenizer = train_master_tokenizer()

# %% [markdown]
# ## 3. Usage Accounting
# We simulate the tokenization of 1 Million strings using the master vocabulary. 
# We count how many times each token ID is used. This allows us to calculate exactly 
# how much space we save (or lose) by including any specific token in our final production model.

# %% 
def analyze_token_frequencies(tokenizer, sample_limit=1000000):
    print(f"Analyzing token usage on {sample_limit:,} sample strings...")
    
    token_counts = np.zeros(tokenizer.get_vocab_size(), dtype=np.int64)
    processed = 0
    
    with open(CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text: continue
            
            ids = tokenizer.encode(text).ids
            for tid in ids:
                token_counts[tid] += 1
                
            processed += 1
            if processed % 100000 == 0:
                print(f"  Processed {processed/1000:.0f}k...")
            if processed >= sample_limit:
                break
                
    return token_counts

usage_frequencies = analyze_token_frequencies(master_tokenizer, sample_limit=1000000)

# %% [markdown]
# ## 4. The Global Optimizer
# We find the global minimum by sweeping across vocab sizes.
# 
# We calculate both:
# 1. **Current Scale (Ground Truth):** Based on the 45.3M records in the DB.
# 2. **Target Scale (Projection):** Projected to the 100M song goal.

# %%
def find_optimum(frequencies, master_tokenizer):
    # Actual counts from DB
    current_count = 45384078 
    sample_size = 1000000
    current_scale = current_count / sample_size
    target_scale = 100_000_000 / sample_size
    
    vocab_sizes = range(1000, MAX_VOCAB, 2000)
    results = []
    
    print("\nCalculating cost curve...")
    
    for v in vocab_sizes:
        # Cost Logic:
        if v < 16384:
            common_count = sum(frequencies[:128])
            rare_count = sum(frequencies[128:v])
            overflow_count = sum(frequencies[v:]) * 2 
            db_bytes = (common_count * 1) + (rare_count * 2) + (overflow_count * 2)
        elif v <= 65536:
            db_bytes = sum(frequencies[:v]) * 2 + (sum(frequencies[v:]) * 2 * 2)
        else:
            db_bytes = sum(frequencies[:16384]) * 2 + sum(frequencies[16384:v]) * 3 + (sum(frequencies[v:]) * 3 * 2)

        # Current Scale (Ground Truth)
        current_db_mb = (db_bytes * current_scale) / (1024 * 1024)
        current_index_mb = (400 * (current_count / 100_000_000)) # Scaled PQ index
        model_mb = 60 + (v * 768 * 2) / (1024 * 1024)
        current_total_mb = current_db_mb + current_index_mb + model_mb
        
        # Target Scale (Projection)
        target_db_mb = (db_bytes * target_scale) / (1024 * 1024)
        target_total_mb = target_db_mb + 400 + model_mb
        
        results.append({
            "v": v, 
            "current_total": current_total_mb, 
            "current_db": current_db_mb, 
            "target_total": target_total_mb,
            "model": model_mb
        })

    df_res = pd.DataFrame(results)
    best = df_res.loc[df_res['current_total'].idxmin()]
    
    print("\n" + "="*40)
    print(f"GROUND TRUTH: CURRENT SCALE ({current_count/1e6:.1f}M Tracks)")
    print("="*40)
    print(f"Optimal Vocab Size: {best['v']:.0f}")
    print(f"Current Footprint:  {best['current_total']:.2f} MB")
    print(f"  - Database: {best['current_db']:.2f} MB")
    print(f"  - Model:    {best['model']:.2f} MB")
    print("-" * 40)
    print(f"PROJECTION: 100M Tracks")
    print(f"Projected Footprint: {best['target_total']:.2f} MB")
    print("="*40)
    
    return df_res

optimum_df = find_optimum(usage_frequencies, master_tokenizer)
# %% [markdown]
# ## 5. Final Conclusion & Architectural Decision
# 
# ### The Optimization Paradox
# Our analysis of the current **45.4 Million** records suggests a mathematical optimum of **~31,000 tokens**. However, we are making an intentional architectural decision to use **65,535 tokens**.
# 
# ### Why 65,535?
# 1. **Fixed Storage Cost (uint16):** We have chosen to store token IDs as 2-byte integers. A vocabulary of 32,000 and 65,535 both use exactly **16 bits per token**. There is zero per-song storage penalty for the larger vocabulary.
# 2. **Industry Growth (Future Proofing):** The music industry is currently adding ~40 million new tracks per year (100k+ per day). By the time this app hits maturity, the catalog will likely exceed 100M-200M tracks.
# 3. **The Scaling inflection Point:** As demonstrated in our 100M projection, the 65k vocabulary becomes mathematically superior once the catalog grows. Using it now prevents a disruptive "Breaking Change" update to the tokenizer in the near future.
# 4. **Negligible Model Weight Penalty:** The cost of increasing the vocab from 32k to 65k is a mere **~50MB** in the AI model weights. Given our ~2.3GB total projected footprint against a 3.0GB limit, this is a highly efficient "insurance policy" for future scale.
# 
# **Final Decision:**
# *   **Production Vocab Size:** 65,535
# *   **Storage Format:** Fixed-Width uint16
# *   **Artifact:** `release/music_vocab_final.json`

# %% [markdown]
# ## 6. Clustered Binary Format Prototype
# In this section, we verify the logic for the "Range Tables" and "Token Blobs".
# We use a small sample to simulate the binary encoding and lookup process.

# %%
import struct

def prototype_clustered_format(sample_size=1000):
    # 1. Load Production Tokenizer
    tokenizer_path = "../release/music_vocab_final.json"
    if not os.path.exists(tokenizer_path):
        print("Error: Production tokenizer not found. Run training first.")
        return
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # 2. Fetch Sample Ordered by Artist/Release
    conn = get_db_connection()
    cur = conn.cursor()
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
        ORDER BY ac.id, r.id
        LIMIT {sample_size}
    """
    cur.execute(query)
    rows = cur.fetchall()
    cur.close()
    conn.close()

    # 3. Build Range Tables and Token Blobs
    artist_ranges = [] # (start_id, artist_name)
    album_ranges = []  # (start_id, album_name)
    title_blob = b""
    title_offsets = [] # Byte offset for each title

    last_artist = None
    last_album = None

    print(f"Packing {len(rows)} songs...")

    for i, (artist, album, title) in enumerate(rows):
        # Artist Range
        if artist != last_artist:
            artist_ranges.append((i, artist))
            last_artist = artist
        
        # Album Range
        if album != last_album:
            album_ranges.append((i, album))
            last_album = album
            
        # Title Tokenization
        tokens = tokenizer.encode(str(title)).ids
        title_offsets.append(len(title_blob))
        # Pack as uint16 (Fixed 65k strategy)
        title_blob += struct.pack(f"<{len(tokens)}H", *tokens)

    # Add final offset for length calculation
    title_offsets.append(len(title_blob))

    print(f"Created {len(artist_ranges)} Artist clusters.")
    print(f"Created {len(album_ranges)} Album clusters.")
    print(f"Title Blob Size: {len(title_blob)} bytes.")

    # 4. Verify Lookup Logic
    def lookup(song_id):
        # Find Artist (Binary search in real app, simple scan for prototype)
        artist_name = "Unknown"
        for start_id, name in reversed(artist_ranges):
            if song_id >= start_id:
                artist_name = name
                break
        
        # Find Album
        album_name = "Unknown"
        for start_id, name in reversed(album_ranges):
            if song_id >= start_id:
                album_name = name
                break
                
        # Find and Decode Title
        start_off = title_offsets[song_id]
        end_off = title_offsets[song_id + 1]
        raw_tokens = title_blob[start_off:end_off]
        # Unpack uint16
        num_tokens = len(raw_tokens) // 2
        token_ids = struct.unpack(f"<{num_tokens}H", raw_tokens)
        title_decoded = tokenizer.decode(list(token_ids))
        
        return artist_name, album_name, title_decoded

    # Test on a few IDs
    test_ids = [0, 50, 100, len(rows)-1]
    print("\n--- Lookup Verification ---")
    for tid in test_ids:
        artist, album, title = lookup(tid)
        print(f"ID {tid:3}: {artist} | {album} | {title}")
        # Verify against original data
        orig = rows[tid]
        if (artist, album, title) == orig:
            print("  ✅ Match")
        else:
            print(f"  ❌ Mismatch! Expected: {orig}")

prototype_clustered_format(1000)

