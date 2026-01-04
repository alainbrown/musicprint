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

# %% [markdown]
# ## 2. Data Preparation
# We need a plain text file for the Tokenizer trainer. We will verify the column names and export the titles.

# %% [markdown]
# ## 3. Train BPE Tokenizer
# We utilize the `tokenizers` library (HuggingFace) to learn the subword vocabulary.

# %% [markdown]
# ## 4. Analyze Compression
# Let's calculate the compression ratio.
# 
# **Metric:**
# *   **Raw Size:** Length of UTF-8 string in bytes.
# *   **Tokenized Size:** Number of Tokens * Variable Integer Size (avg 1.5 bytes?).
#     *   Common tokens (0-127) = 1 byte (if we implement VarInt).
#     *   Other tokens = 2 bytes.

# %% [markdown]
# ## 5. Visual Inspection
# Let's look at what tokens are actually being learned.

