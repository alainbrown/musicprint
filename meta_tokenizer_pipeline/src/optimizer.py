import os
import numpy as np
import pandas as pd

def analyze_token_frequencies(tokenizer, corpus_path, sample_limit=1000000):
    """
    Simulates tokenization to count total tokens and frequency distribution.
    Returns:
        token_counts (np.array): Histogram of token usage.
        total_tokens (int): Total number of tokens generated for the sample.
    """
    if not tokenizer or not os.path.exists(corpus_path):
        return None, 0
        
    print(f"Analyzing token usage on {sample_limit:,} sample strings...")
    
    token_counts = np.zeros(tokenizer.get_vocab_size(), dtype=np.int64)
    processed = 0
    total_tokens = 0
    
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text: continue
            
            ids = tokenizer.encode(text).ids
            count = len(ids)
            total_tokens += count
            
            for tid in ids:
                token_counts[tid] += 1
                
            processed += 1
            if processed % 100000 == 0:
                print(f"  Processed {processed/1000:.0f}k...")
            if processed >= sample_limit:
                break
                
    return token_counts, total_tokens

def find_optimum(frequencies, total_tokens_in_sample, sample_size, current_db_count=45384078, target_db_count=100000000):
    """
    Calculates the system footprint for various vocabulary sizes based on
    a FIXED-WIDTH (uint16) storage architecture.
    """
    if frequencies is None: return None

    # Scales
    current_scale = current_db_count / sample_size
    target_scale = target_db_count / sample_size
    
    # We simulate different vocab sizes by truncating the frequency list
    # (assuming the BPE merges are ordered by frequency, which is generally true for the 'Master Tokenizer' approach)
    # NOTE: This is an approximation. A true 32k tokenizer might merge differently than the first 32k merges of a 200k tokenizer.
    # But for back-solving, this is the standard heuristic.
    
    # However, strictly speaking, just taking the top N frequencies doesn't give you the token count of a smaller vocab.
    # The token count INCREASES as vocab DECREASES.
    # The 'frequencies' array from the Master Tokenizer (200k) is fixed.
    # We can't easily simulate the *token count* of a 32k vocab from the 200k stats without re-tokenizing.
    
    # RE-EVALUATION: 
    # The previous notebook logic tried to simulate "Rare" vs "Common" tokens to calculate variable byte size.
    # But for Fixed Width, we only care about Total Token Count.
    # The Master Tokenizer approach (training one huge model) allows us to see how many tokens are used.
    # BUT, to know the token count for a *smaller* vocab, we actually have to re-tokenize or have a way to 'un-merge'.
    
    # Given the complexity of re-tokenizing for every data point, and the fact that we confirmed 
    # "Fixed Width" means "Bigger Vocab is Always Better for Storage",
    # the optimization is actually just balancing Model Size vs Token Count.
    
    # Since we can't trivially calculate "Token Count @ 32k" from "Token Count @ 200k" stats without re-running,
    # we will keep the logic simple:
    # We assume the user creates *multiple* tokenizers (e.g. 16k, 32k, 64k) and runs this tool on each,
    # OR we accept that this tool analyzes the *current* tokenizer's efficiency.
    
    # For the purpose of this Refactor, I will implement the Cost Function for the *provided* tokenizer.
    # Use cases:
    # 1. Train 32k tokenizer -> Run Optimizer -> Get Cost.
    # 2. Train 64k tokenizer -> Run Optimizer -> Get Cost.
    # 3. Compare.
    
    vocab_size = len(frequencies)
    
    # 1. Database Storage Cost (Fixed uint16 = 2 bytes)
    # We assume the 'total_tokens_in_sample' is representative.
    # Cost = (Total Tokens * 2 bytes)
    db_bytes_sample = total_tokens_in_sample * 2
    
    # ISRC Overhead: 16 bytes per track (8 Meta + 8 Audio)
    isrc_overhead_sample = sample_size * 16
    
    # Scale to Target
    target_db_mb = ((db_bytes_sample + isrc_overhead_sample) * target_scale) / (1024 * 1024)
    
    # 2. Model Weight Cost
    # MERT Adapter: Fixed 60MB + (Vocab * Embedding(768) * 2 bytes(float16))
    # Note: Embedding might be projected. System design says "Projects 768... to 64".
    # But the *Input Embedding* of the Text Decoder (BPE) is usually Vocab * Hidden.
    # If the system uses a standard Transformer Decoder (like GPT-2 small):
    # Embedding Matrix = Vocab * 768 * 2 bytes.
    model_mb = 60 + (vocab_size * 768 * 2) / (1024 * 1024)
    
    # 3. Audio Index Cost (Fixed ~400MB for 100M tracks)
    index_mb = 400
    
    total_mb = target_db_mb + model_mb + index_mb
    
    return {
        "vocab_size": vocab_size,
        "total_mb": total_mb,
        "db_mb": target_db_mb,
        "model_mb": model_mb,
        "avg_tokens_per_track": total_tokens_in_sample / sample_size
    }
