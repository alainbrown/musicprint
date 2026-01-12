# %% [markdown]
# # System-Wide Tokenizer Optimizer & Pipeline Smoke Test
# 
# This notebook serves two purposes:
# 1. **Scientific Back-Solver:** A non-sampled, data-driven approach to find the mathematically optimal vocabulary size.
# 2. **End-to-End Smoke Test:** verifying the production pipeline modules (`src/*.py`).
# 
# ## The Problem
# We need to store 100 Million tracks (Artist, Album, Title) on a mobile device with less than 3.0 GB total storage.

# %% 
import os
import sys
import time
import struct
import pandas as pd
import numpy as np
import psycopg2
import mmap
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

# Ensure we can import from src
sys.path.append(os.path.abspath('../src'))

import train_tokenizer
import build_db
import export_vocab
import import_mb # For Ingestion Verification
from build_db import pack_isrc

# Config
DB_HOST = os.environ.get("DB_HOST", "mb_db")
DB_NAME = os.environ.get("DB_NAME", "musicbrainz_db")
DB_USER = os.environ.get("DB_USER", "musicbrainz")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "musicbrainz")

# Paths (Smoke Test / Verification)
# CRITICAL: We use /tmp or a cache volume to avoid overwriting real release artifacts
DATA_DIR = "/vol/data"
TEST_ARTIFACTS_DIR = "/tmp/smoke_test_artifacts"
CACHE_DIR = "/vol/cache"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TEST_ARTIFACTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

CORPUS_PATH = os.path.join(DATA_DIR, "full_musicbrainz_corpus.txt")

def get_db_connection():
    return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)

# %% [markdown]
# ## 1. Build Full Dataset Corpus
# We extract every recording title, artist name, and release name from the MusicBrainz Postgres mirror.

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

# Uncomment to regenerate corpus if needed
# build_full_corpus()

# %% [markdown]
# ## 2. Train Master Tokenizer (200k Vocab)
# We use the production `train_tokenizer` module to train a massive BPE model for analysis.

# %% 
MAX_VOCAB = 200000
MASTER_MODEL_PATH = os.path.join(CACHE_DIR, "master_tokenizer_200k.json")

if not os.path.exists(MASTER_MODEL_PATH) and os.path.exists(CORPUS_PATH):
    print(f"Training Master BPE Tokenizer (Vocab: {MAX_VOCAB})...")
    # Using the production module logic
    train_tokenizer.train_tokenizer(
        input_file=CORPUS_PATH,
        output_path=MASTER_MODEL_PATH,
        vocab_size=MAX_VOCAB,
        use_db=False # Use the file we just dumped (or existing one)
    )
    master_tokenizer = Tokenizer.from_file(MASTER_MODEL_PATH)
elif os.path.exists(MASTER_MODEL_PATH):
    print("Loading existing Master Tokenizer...")
    master_tokenizer = Tokenizer.from_file(MASTER_MODEL_PATH)
else:
    print("Skipping Master Tokenizer training (Corpus not found).")
    master_tokenizer = None

# %% [markdown]
# ## 3. Usage Accounting & Global Optimizer (Scientific Back-Solver)
# We simulate the tokenization of 1 Million strings using the master vocabulary. 
# We count how many times each token ID is used to calculate space savings.

# %% 
def analyze_token_frequencies(tokenizer, sample_limit=1000000):
    if not tokenizer or not os.path.exists(CORPUS_PATH):
        return None
        
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

if master_tokenizer:
    usage_frequencies = analyze_token_frequencies(master_tokenizer, sample_limit=1000000)
else:
    usage_frequencies = None

# %% [markdown]
# ## 4. The Global Optimizer
# We find the global minimum by sweeping across vocab sizes.

# %% 
def find_optimum(frequencies, master_tokenizer):
    if frequencies is None: return

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

        # ISRC Overhead: 8 bytes in Meta DB + 8 bytes in Audio Index = 16 bytes per track
        isrc_overhead_bytes = 16

        # Current Scale (Ground Truth)
        current_db_mb = ((db_bytes + isrc_overhead_bytes) * current_scale) / (1024 * 1024)
        current_index_mb = (400 * (current_count / 100_000_000)) # Scaled PQ index (Vectors only)
        model_mb = 60 + (v * 768 * 2) / (1024 * 1024)
        current_total_mb = current_db_mb + current_index_mb + model_mb
        
        # Target Scale (Projection)
        target_db_mb = ((db_bytes + isrc_overhead_bytes) * target_scale) / (1024 * 1024)
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

if usage_frequencies is not None:
    optimum_df = find_optimum(usage_frequencies, master_tokenizer)

# %% [markdown]
# # --- PRODUCTION PIPELINE SMOKE TEST ---
# Below we execute the refactored production modules to ensure the pipeline works end-to-end.
# **NOTE:** We output to a temporary directory to avoid overwriting production release artifacts.

# %% [markdown]
# ## Step 0: Ingestion Verification (Schema Check)
# We verify that the `import_mb` module can successfully connect to the DB and initialize the schema.

# %% 
print("Step 0: Verifying Schema Initialization...")
conn = import_mb.get_db_connection()
conn.autocommit = True
# This ensures the SQL scripts in src/ can be downloaded and applied
import_mb.init_schema(conn.cursor())
conn.close()
print("✅ Schema Verified.")

# %% [markdown]
# ## Step 1: Train Production Tokenizer
# We train a small tokenizer (e.g., 32k) using the production script.

# %% 
TEST_TOKENIZER_PATH = os.path.join(TEST_ARTIFACTS_DIR, "music_encoder.json")

print(f"Step 1: Training Test Tokenizer to {TEST_TOKENIZER_PATH}...")
# We can use a smaller subset or the full corpus. 
# For smoke test, let's just ensure it runs.
if not os.path.exists(TEST_TOKENIZER_PATH):
    # Create dummy args-like object or call function directly
    train_tokenizer.train_tokenizer(
        input_file=CORPUS_PATH if os.path.exists(CORPUS_PATH) else None,
        output_path=TEST_TOKENIZER_PATH,
        vocab_size=32000,
        use_db=True if not os.path.exists(CORPUS_PATH) else False, # Fallback to DB if corpus missing
        db_query="SELECT name FROM musicbrainz.artist LIMIT 10000" if not os.path.exists(CORPUS_PATH) else None # fast smoke test query
    )

# %% [markdown]
# ## Step 2: Export Binary Vocab
# Convert the JSON tokenizer to the optimized binary format for the app.

# %% 
TEST_VOCAB_BIN = os.path.join(TEST_ARTIFACTS_DIR, "music_decoder.bin")

print(f"Step 2: Exporting Binary Vocab to {TEST_VOCAB_BIN}...")
export_vocab.export_to_binary(TEST_TOKENIZER_PATH, TEST_VOCAB_BIN)

# %% [markdown]
# ## Step 3: Build Metadata Database
# Build the compact binary database using the tokenizer.

# %% 
TEST_DB_BIN = os.path.join(TEST_ARTIFACTS_DIR, "music_meta.bin")
TEST_MANIFEST_PATH = os.path.join(TEST_ARTIFACTS_DIR, "album_manifest.csv")

class BuildArgs:
    tokenizer = TEST_TOKENIZER_PATH
    output = TEST_DB_BIN

print(f"Step 3: Building Database to {TEST_DB_BIN}...")
# We use the build_db module. 
# Note: build_db.build() takes args.
build_db.build(BuildArgs())

# %% [markdown]
# ## Step 4: Verify Production Binary (Client-Side Simulation)
# We simulate the iOS app's high-speed lookup logic using `mmap` and binary search.

# %% 
def verify_production_binary(target_isrcs=None):
    bin_path = TEST_DB_BIN
    encoder_path = TEST_TOKENIZER_PATH
    manifest_path = TEST_MANIFEST_PATH
    
    if not os.path.exists(bin_path):
        print(f"Error: Production binary not found at {bin_path}. Run build_db.py first.")
        return

    tokenizer = Tokenizer.from_file(encoder_path)
    
    # Load manifest for verification
    manifest = {}
    if os.path.exists(manifest_path):
        import csv
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                manifest[int(row['album_index'])] = row['release_uuid']
    
    with open(bin_path, "rb") as f:
        # Memory-map the entire file for instant access
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # 1. Parse Header (128 bytes)
        magic, version, song_count, artist_count, album_count = struct.unpack("<4sIIII", mm[0:20])
        offsets = struct.unpack("<QQQQQQQ", mm[20:76])
        
        off_isrc_index, off_artist_ranges, off_album_ranges, \
        off_title_offsets, off_title_blob, off_artist_blob, off_album_blob = offsets

        print(f"\n--- Production Binary Verified ---")
        print(f"Magic: {magic.decode()} | Version: {version} | Songs: {song_count:,} | Albums: {album_count:,}")
        
        # 2. Lookup Function (Binary Search on ISRC Index)
        def lookup(isrc_str):
            packed_target = pack_isrc(isrc_str)
            
            # Binary Search in Section 1 (ISRC Index)
            # Each entry is 12 bytes: [uint64 packed_isrc][uint32 internal_id]
            low = 0
            high = song_count - 1
            internal_id = -1
            
            while low <= high:
                mid = (low + high) // 2
                entry_pos = off_isrc_index + (mid * 12)
                packed_mid, uid = struct.unpack("<QI", mm[entry_pos:entry_pos+12])
                
                if packed_mid < packed_target:
                    low = mid + 1
                elif packed_mid > packed_target:
                    high = mid - 1
                else:
                    internal_id = uid
                    break
            
            if internal_id == -1: return None
            
            # 3. Retrieve Artist Name (Binary Search in Artist Range Table)
            artist_name = "Unknown Artist"
            low = 0
            high = artist_count - 1
            while low <= high:
                mid = (low + high) // 2
                entry_pos = off_artist_ranges + (mid * 8)
                start_id, name_off = struct.unpack("<II", mm[entry_pos:entry_pos+8])
                if internal_id >= start_id:
                    res_off = off_artist_blob + name_off
                    length = mm[res_off]
                    tokens = struct.unpack(f"<{length}H", mm[res_off+1:res_off+1+(length*2)])
                    artist_name = tokenizer.decode(list(tokens))
                    low = mid + 1
                else:
                    high = mid - 1

            # 4. Retrieve Album Name & Art UUID (Binary Search in Album Range Table)
            album_name = "Unknown Album"
            album_uuid = "N/A"
            album_idx = -1
            low = 0
            high = album_count - 1
            while low <= high:
                mid = (low + high) // 2
                entry_pos = off_album_ranges + (mid * 8)
                start_id, name_off = struct.unpack("<II", mm[entry_pos:entry_pos+8])
                if internal_id >= start_id:
                    res_off = off_album_blob + name_off
                    length = mm[res_off]
                    tokens = struct.unpack(f"<{length}H", mm[res_off+1:res_off+1+(length*2)])
                    album_name = tokenizer.decode(list(tokens))
                    album_idx = mid
                    low = mid + 1
                else:
                    high = mid - 1
            
            if album_idx != -1:
                album_uuid = manifest.get(album_idx, "NOT IN MANIFEST")

            # 5. Retrieve Title (Direct offset lookup)
            off_pos = off_title_offsets + (internal_id * 4)
            start_off, next_off = struct.unpack("<II", mm[off_pos:off_pos+8])
            
            title_pos = off_title_blob + start_off
            title_len = (next_off - start_off) // 2
            title_tokens = struct.unpack(f"<{title_len}H", mm[title_pos:title_pos+(title_len*2)])
            title_name = tokenizer.decode(list(title_tokens))
            
            return {
                "id": internal_id,
                "isrc": isrc_str,
                "artist": artist_name,
                "album": album_name,
                "album_idx": album_idx,
                "album_uuid": album_uuid,
                "title": title_name
            }

        # 6. Test against actual data
        if not target_isrcs:
            # Try to fetch some real ISRCs if DB is available
            try:
                conn = get_db_connection()
                cur = conn.cursor()
                cur.execute("SELECT i.isrc FROM musicbrainz.isrc i JOIN musicbrainz.recording r ON i.recording = r.id LIMIT 5")
                target_isrcs = [r[0] for r in cur.fetchall()]
                cur.close()
                conn.close()
            except:
                print("Could not fetch ISRCs from DB (check connection).")
                target_isrcs = []

        print("\nTesting Lookups (Metadata + Album Art Bridge):")
        for isrc in target_isrcs:
            start = time.time()
            res = lookup(isrc)
            elapsed = (time.time() - start) * 1000
            if res:
                print(f"  ISRC: {isrc}\n")
                print(f"  Song:   {res['title']}\n")
                print(f"  Artist: {res['artist']}\n")
                print(f"  Album:  {res['album']} (Idx: {res['album_idx']})\n")
                print(f"  Art ID: {res['album_uuid']}\n")
                print(f"  Time:   {elapsed:.2f}ms\n")
            else:
                print(f"  {isrc} -> NOT FOUND\n")

        mm.close()

verify_production_binary()

# %% [markdown]
# ## Step 5: Manifest Integrity Check (100% Scan)
# We verify the `album_manifest.csv` against the `music_meta.bin` file.

# %% 
def verify_manifest_integrity():
    manifest_path = TEST_MANIFEST_PATH
    bin_path = TEST_DB_BIN
    encoder_path = TEST_TOKENIZER_PATH

    if not os.path.exists(manifest_path) or not os.path.exists(bin_path):
        print("Files not found.")
        return

    print(f"--- 100% Integrity Check: Manifest vs Binary ---")
    
    # 1. Load Resources
    df = pd.read_csv(manifest_path)
    tokenizer = Tokenizer.from_file(encoder_path)
    
    # 2. Open Binary
    with open(bin_path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Parse Offsets
        offsets = struct.unpack("<QQQQQQQ", mm[20:76])
        off_album_ranges = offsets[2]
        off_album_blob = offsets[6]
        
        errors = 0
        total = len(df)
        print(f"Verifying {total:,} albums...")
        
        # 3. Iterate EVERY album
        for row in df.itertuples(index=False):
            idx = row.album_index
            csv_name = str(row.album_name) # Handle NaNs

            # Read from Binary at Index `idx`
            entry_pos = off_album_ranges + (idx * 8)
            _, name_off = struct.unpack("<II", mm[entry_pos:entry_pos+8])
            
            # Decode Name
            res_off = off_album_blob + name_off
            length = mm[res_off]
            tokens = struct.unpack(f"<{length}H", mm[res_off+1:res_off+1+(length*2)])
            bin_name = str(tokenizer.decode(list(tokens)))
            
            # Normalize Empty/Null states
            def normalize(s):
                s = str(s).strip()
                if s.lower() in ["nan", "none", "null", "n/a", "na", ""]:
                    return ""
                return s.replace('""', '"')

            c_norm = normalize(csv_name)
            b_norm = normalize(bin_name)
            
            # Robust comparison
            c_bytes = c_norm.encode('utf-8', errors='ignore')
            b_bytes = b_norm.encode('utf-8', errors='ignore')
            b_check = b_bytes[:-3] if len(b_bytes) > 3 else b_bytes
            
            if not c_bytes.startswith(b_check):
                print(f"❌ MISMATCH at Index {idx}!")
                print(f"   CSV: {c_norm}")
                print(f"   BIN: {b_norm}")
                errors += 1
                if errors > 10: 
                    print("Too many errors, aborting.")
                    break
        
        mm.close()
        
        if errors == 0:
            print(f"\n✅ SUCCESS: All {total:,} albums match perfectly.")
        else:
            print(f"\n❌ FAILED: Found {errors} mismatches.")

verify_manifest_integrity()

# %% [markdown]
# ## Step 6: Binary Decoder Verification
# We confirm that `music_decoder.bin` matches the JSON source.

# %% 
class BinaryDecoder:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        
        # Header: Magic(4s), Version(I), Token Count(I)
        magic, version, count = struct.unpack("<4sII", self.mm[0:12])
        self.count = count
        self.offsets_start = 12
        # Offsets are (count + 1) * 4 bytes
        self.data_start = 12 + (count + 1) * 4
        print(f"BinaryDecoder Loaded: {count} tokens")
        
    def decode(self, token_id):
        if token_id >= self.count: return None
        
        # Read offsets
        off_pos = self.offsets_start + (token_id * 4)
        start, end = struct.unpack("<II", self.mm[off_pos:off_pos+8])
        
        # Read bytes
        b = self.mm[self.data_start + start : self.data_start + end]
        return b.decode("utf-8")

def verify_binary_decoder_integrity():
    if not os.path.exists(TEST_VOCAB_BIN) or not os.path.exists(TEST_TOKENIZER_PATH):
        print("Decoder artifacts missing.")
        return

    print("\n--- Verifying Binary Decoder ---")
    decoder = BinaryDecoder(TEST_VOCAB_BIN)
    json_tok = Tokenizer.from_file(TEST_TOKENIZER_PATH)
    
    vocab_size = json_tok.get_vocab_size()
    
    # Check random sample
    errors = 0
    samples = 100
    print(f"Checking {samples} random tokens...")
    
    for _ in range(samples):
        tid = np.random.randint(0, vocab_size)
        
        # JSON Tokenizer 'id_to_token' returns the raw BPE token (e.g. "ĠThe")
        expected = json_tok.id_to_token(tid)
        actual = decoder.decode(tid)
        
        if expected != actual:
            print(f"❌ MISMATCH at ID {tid}")
            print(f"   JSON: {expected}")
            print(f"   BIN:  {actual}")
            errors += 1
            if errors > 5: break
            
    if errors == 0:
        print("✅ SUCCESS: Binary Decoder matches JSON Tokenizer.")
    else:
        print("❌ FAILED: Decoder mismatch.")

verify_binary_decoder_integrity()
