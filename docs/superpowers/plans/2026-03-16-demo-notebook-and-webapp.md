# Demo Notebook & Web App Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a top-level demo notebook that trains, indexes, and verifies the MusicPrint system end-to-end, plus a web app for interactive song identification.

**Architecture:** Two independent deliverables sharing a common Python search module. The notebook runs pipelines as subprocesses and verifies recall. The web app is a Flask server that accepts audio, encodes it, searches the pre-built index, and returns metadata.

**Tech Stack:** Python, PyTorch, Flask, pydub/ffmpeg, NumPy, Jupyter

**Spec:** `docs/superpowers/specs/2026-03-16-demo-notebook-and-webapp-design.md`

---

## File Structure

```
musicprint/
├── demo.ipynb                          # Top-level verification notebook
├── demo_app/
│   ├── app.py                          # Flask backend (audio → identify)
│   ├── search.py                       # Hamming search engine (load index, query)
│   ├── metadata.py                     # Binary metadata reader (music_meta.bin + music_decoder.bin)
│   ├── audio.py                        # Audio decode, resample, window, binarize
│   ├── static/
│   │   └── index.html                  # Frontend (mic + file upload → result card)
│   ├── requirements.txt                # Flask, torch, pydub, numpy
│   ├── Dockerfile                      # Python slim + ffmpeg + CPU PyTorch
│   └── docker-compose.yml              # Mounts release/ read-only, exposes :5000
├── tests/
│   ├── conftest.py                     # Adds demo_app/ to sys.path for imports
│   ├── test_search.py                  # Unit tests for search.py
│   ├── test_metadata.py                # Unit tests for metadata.py
│   └── test_audio.py                   # Unit tests for audio.py
└── release/                            # Artifacts (gitignored, produced by notebook)
    ├── encoder.pt
    ├── audio_index.bin
    ├── music_meta.bin
    └── music_decoder.bin
```

**Shared modules:** `demo_app/search.py`, `demo_app/metadata.py`, and `demo_app/audio.py` are used by both the web app and the notebook (the notebook adds `demo_app/` to `sys.path`).

---

## Chunk 1: Core Modules (search, audio, metadata)

### Task 1: Hamming Search Module

**Files:**
- Create: `demo_app/search.py`
- Create: `tests/test_search.py`

The search module loads `audio_index.bin` (64-byte header + 16-byte entries: 8B hash + 8B packed ISRC) and finds the closest match by Hamming distance.

- [ ] **Step 0: Create tests/conftest.py for import resolution**

All test files import from `demo_app/` using bare imports (e.g., `from search import SearchEngine`). The `app.py` also uses bare imports since it runs from within `demo_app/`. This `conftest.py` adds `demo_app/` to `sys.path` so both contexts work without needing `__init__.py`.

```python
# tests/conftest.py
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "demo_app"))
```

- [ ] **Step 1: Write failing tests**

```python
# tests/test_search.py
import struct
import tempfile
import os
import numpy as np
from search import SearchEngine

def make_test_index(entries, path):
    """Create a minimal audio_index.bin with given (hash, isrc) entries."""
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"MPAF", 1, len(entries)))
        f.write(b"\x00" * 52)  # pad header to 64 bytes
        for h, isrc in entries:
            f.write(struct.pack("<QQ", h, isrc))

def test_search_exact_match():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "index.bin")
        make_test_index([(0xDEADBEEF, 42), (0xCAFEBABE, 99)], path)
        engine = SearchEngine(path)
        result = engine.search(0xDEADBEEF)
        assert result["song_id"] == 42
        assert result["distance"] == 0

def test_search_nearest_match():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "index.bin")
        # 0xFF differs from 0xFE by 1 bit
        make_test_index([(0xFE, 1), (0x00, 2)], path)
        engine = SearchEngine(path)
        result = engine.search(0xFF)
        assert result["song_id"] == 1
        assert result["distance"] == 1

def test_search_empty_index():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "index.bin")
        make_test_index([], path)
        engine = SearchEngine(path)
        result = engine.search(0xFF)
        assert result is None

def test_entry_count():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "index.bin")
        make_test_index([(i, i) for i in range(100)], path)
        engine = SearchEngine(path)
        assert engine.num_entries == 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/bigstore/media/apps/code-server/workspace/musicprint && python -m pytest tests/test_search.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'search')

- [ ] **Step 3: Implement search module**

```python
# demo_app/search.py
import struct
import mmap

HEADER_SIZE = 64
ENTRY_SIZE = 16  # 8 bytes hash + 8 bytes packed ISRC


class SearchEngine:
    def __init__(self, index_path):
        self._file = open(index_path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        file_size = len(self._mm)
        if file_size < HEADER_SIZE:
            self.num_entries = 0
            return

        magic, version, count = struct.unpack_from("<4sII", self._mm, 0)
        self.num_entries = count

    def search(self, query_hash, k=1):
        """Find the closest entry by Hamming distance.

        Args:
            query_hash: uint64 binary hash of the query
            k: number of results (only k=1 supported)

        Returns:
            dict with song_id (packed ISRC) and distance, or None if empty.
        """
        if self.num_entries == 0:
            return None

        best_dist = 65  # max possible is 64
        best_isrc = 0

        for i in range(self.num_entries):
            offset = HEADER_SIZE + i * ENTRY_SIZE
            stored_hash, packed_isrc = struct.unpack_from("<QQ", self._mm, offset)
            dist = bin(stored_hash ^ query_hash).count("1")
            if dist < best_dist:
                best_dist = dist
                best_isrc = packed_isrc

        return {"song_id": best_isrc, "distance": best_dist}

    def close(self):
        self._mm.close()
        self._file.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/bigstore/media/apps/code-server/workspace/musicprint && python -m pytest tests/test_search.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py demo_app/search.py tests/test_search.py
git commit -m "feat(demo): add Hamming search module with tests"
```

---

### Task 2: Audio Processing Module

**Files:**
- Create: `demo_app/audio.py`
- Create: `tests/test_audio.py`

Handles: decode any format → resample to 24kHz mono → window into 5-second chunks → encode → binarize → return list of uint64 hashes.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_audio.py
import numpy as np
from audio import binarize, window_audio

def test_binarize_all_positive():
    # 64 positive floats → all bits set
    vec = np.ones(64, dtype=np.float32)
    h = binarize(vec)
    assert h == (1 << 64) - 1  # 0xFFFFFFFFFFFFFFFF

def test_binarize_all_negative():
    vec = -np.ones(64, dtype=np.float32)
    h = binarize(vec)
    assert h == 0

def test_binarize_mixed():
    vec = np.zeros(64, dtype=np.float32)
    vec[0] = 1.0   # bit 0 set
    vec[63] = 1.0   # bit 63 set
    h = binarize(vec)
    assert h == (1 << 0) | (1 << 63)

def test_window_audio_short():
    # 3 seconds of audio at 24kHz — too short for a single 5s window
    audio = np.zeros(72000, dtype=np.float32)
    windows = window_audio(audio, sr=24000)
    assert len(windows) == 0

def test_window_audio_exact():
    # Exactly 5 seconds → one window
    audio = np.zeros(120000, dtype=np.float32)
    windows = window_audio(audio, sr=24000)
    assert len(windows) == 1
    assert windows[0].shape == (120000,)

def test_window_audio_ten_seconds():
    # 10 seconds → multiple windows with 1s stride
    audio = np.zeros(240000, dtype=np.float32)
    windows = window_audio(audio, sr=24000)
    # 5s window, 1s stride: (240000 - 120000) / 24000 + 1 = 6 windows
    assert len(windows) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/bigstore/media/apps/code-server/workspace/musicprint && python -m pytest tests/test_audio.py -v`
Expected: FAIL

- [ ] **Step 3: Implement audio module**

```python
# demo_app/audio.py
import struct
import numpy as np

SAMPLE_RATE = 24000
WINDOW_SAMPLES = 120000   # 5 seconds at 24kHz
STRIDE_SAMPLES = 24000    # 1 second stride


def binarize(embedding):
    """Convert a 64-dim float embedding to a uint64 hash via sign bits.

    Matches the C++ and index pipeline convention:
    bit[i] = 1 if embedding[i] > 0, packed little-endian.
    """
    bits = (embedding > 0).astype(np.uint8)
    packed = np.packbits(bits, bitorder="little")
    return struct.unpack("<Q", packed.tobytes())[0]


def window_audio(audio, sr=SAMPLE_RATE):
    """Split audio into 5-second windows with 1-second stride.

    Args:
        audio: 1D numpy array of float32 samples
        sr: sample rate (must be 24000)

    Returns:
        List of 1D numpy arrays, each 120000 samples.
    """
    if len(audio) < WINDOW_SAMPLES:
        return []
    windows = []
    start = 0
    while start + WINDOW_SAMPLES <= len(audio):
        windows.append(audio[start : start + WINDOW_SAMPLES])
        start += STRIDE_SAMPLES
    return windows


def load_and_resample(audio_path):
    """Load audio from any format, resample to 24kHz mono.

    Uses pydub+ffmpeg for format support.
    Returns a 1D numpy float32 array normalized to [-1, 1].
    """
    from pydub import AudioSegment

    seg = AudioSegment.from_file(audio_path)
    seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE).set_sample_width(2)
    samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
    samples /= 32768.0
    return samples
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/bigstore/media/apps/code-server/workspace/musicprint && python -m pytest tests/test_audio.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add demo_app/audio.py tests/test_audio.py
git commit -m "feat(demo): add audio processing module with tests"
```

---

### Task 3: Metadata Reader Module

**Files:**
- Create: `demo_app/metadata.py`
- Create: `tests/test_metadata.py`

Reads `music_meta.bin` (128-byte header, ISRC binary search, clustered range tables) and `music_decoder.bin` (BPE token → string). No dependency on the `tokenizers` library.

Reference: `3_meta_tokenizer/src/build_db.py:146-188` for binary layout, `3_meta_tokenizer/src/export_vocab.py:39-51` for decoder layout.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metadata.py
import struct
import tempfile
import os
from metadata import BPEDecoder, MetadataDB

def make_test_decoder(tokens, path):
    """Create a minimal music_decoder.bin.

    Format: Header(magic 4s, version I, count I) + offsets((count+1) * I) + data blob
    """
    encoded = [t.encode("utf-8") for t in tokens]
    offsets = []
    pos = 0
    for e in encoded:
        offsets.append(pos)
        pos += len(e)
    offsets.append(pos)

    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"MPVC", 1, len(tokens)))
        for o in offsets:
            f.write(struct.pack("<I", o))
        for e in encoded:
            f.write(e)

def test_decoder_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "decoder.bin")
        make_test_decoder(["hello", " ", "world"], path)
        dec = BPEDecoder(path)
        assert dec.decode_token(0) == "hello"
        assert dec.decode_token(1) == " "
        assert dec.decode_token(2) == "world"

def test_decoder_sequence():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "decoder.bin")
        make_test_decoder(["Hel", "lo", " World"], path)
        dec = BPEDecoder(path)
        assert dec.decode_tokens([0, 1, 2]) == "Hello World"

def test_decoder_out_of_range():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "decoder.bin")
        make_test_decoder(["a", "b"], path)
        dec = BPEDecoder(path)
        assert dec.decode_token(999) is None


def make_test_meta_db(songs, decoder_path, meta_path):
    """Create a minimal music_meta.bin with given songs.

    songs: list of (packed_isrc, title_tokens, artist_tokens) tuples.
    Songs must be pre-sorted by packed_isrc.
    """
    # Build sections
    song_count = len(songs)
    artist_count = song_count  # one artist range per song for simplicity

    isrc_pairs = []
    artist_ranges = []
    title_blob = bytearray()
    title_offsets = []
    artist_blob = bytearray()

    for i, (packed_isrc, title_toks, artist_toks) in enumerate(songs):
        isrc_pairs.append((packed_isrc, i))

        # Title
        title_offsets.append(len(title_blob))
        title_blob.extend(struct.pack(f"<{len(title_toks)}H", *title_toks))

        # Artist (length-prefixed)
        artist_ranges.append((i, len(artist_blob)))
        artist_blob.extend(struct.pack("<B", len(artist_toks)))
        artist_blob.extend(struct.pack(f"<{len(artist_toks)}H", *artist_toks))

    title_offsets.append(len(title_blob))

    # Calculate offsets
    header_size = 128
    isrc_index_size = song_count * 12
    artist_range_size = artist_count * 8
    album_range_size = 0  # no albums in test
    title_offset_size = len(title_offsets) * 4

    off_isrc = header_size
    off_artist_ranges = off_isrc + isrc_index_size
    off_album_ranges = off_artist_ranges + artist_range_size
    off_title_offsets = off_album_ranges + album_range_size
    off_title_blob = off_title_offsets + title_offset_size
    off_artist_blob = off_title_blob + len(title_blob)
    off_album_blob = off_artist_blob + len(artist_blob)

    with open(meta_path, "wb") as f:
        f.write(struct.pack("<4sIIII", b"MPDB", 3, song_count, artist_count, 0))
        f.write(struct.pack("<QQQQQQQ",
            off_isrc, off_artist_ranges, off_album_ranges,
            off_title_offsets, off_title_blob, off_artist_blob, off_album_blob))
        f.write(b"\x00" * (header_size - f.tell()))

        for packed_isrc, internal_id in isrc_pairs:
            f.write(struct.pack("<QI", packed_isrc, internal_id))
        for start_id, name_off in artist_ranges:
            f.write(struct.pack("<II", start_id, name_off))
        for off in title_offsets:
            f.write(struct.pack("<I", off))
        f.write(title_blob)
        f.write(artist_blob)


def test_metadata_lookup():
    with tempfile.TemporaryDirectory() as td:
        dec_path = os.path.join(td, "decoder.bin")
        meta_path = os.path.join(td, "meta.bin")

        # Vocab: 0="Hel" 1="lo" 2=" " 3="World" 4="Art" 5="ist"
        make_test_decoder(["Hel", "lo", " ", "World", "Art", "ist"], dec_path)
        dec = BPEDecoder(dec_path)

        # Two songs, sorted by packed_isrc
        make_test_meta_db([
            (100, [0, 1, 2, 3], [4, 5]),  # "Hello World" by "Artist"
            (200, [3], [0, 1]),             # "World" by "Hello"
        ], dec_path, meta_path)

        db = MetadataDB(meta_path, dec)

        result = db.lookup(100)
        assert result is not None
        assert result["title"] == "Hello World"
        assert result["artist"] == "Artist"

        result = db.lookup(200)
        assert result is not None
        assert result["title"] == "World"
        assert result["artist"] == "Hello"

        assert db.lookup(999) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /mnt/bigstore/media/apps/code-server/workspace/musicprint && python -m pytest tests/test_metadata.py -v`
Expected: FAIL

- [ ] **Step 3: Implement metadata module**

```python
# demo_app/metadata.py
import struct
import mmap


class BPEDecoder:
    """Reads music_decoder.bin to map 16-bit BPE token IDs to strings.

    Binary format (from export_vocab.py):
      Header: magic(4s) version(I) count(I) = 12 bytes
      Offsets: (count+1) x uint32 = byte offsets into data blob
      Data: concatenated UTF-8 strings
    """

    def __init__(self, path):
        self._file = open(path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        _, _, self.count = struct.unpack_from("<4sII", self._mm, 0)
        self._offsets_start = 12
        self._data_start = 12 + (self.count + 1) * 4

    def decode_token(self, token_id):
        if token_id >= self.count:
            return None
        off_pos = self._offsets_start + token_id * 4
        start, end = struct.unpack_from("<II", self._mm, off_pos)
        return self._mm[self._data_start + start : self._data_start + end].decode("utf-8")

    def decode_tokens(self, token_ids):
        parts = []
        for tid in token_ids:
            s = self.decode_token(tid)
            if s is not None:
                parts.append(s)
        return "".join(parts)

    def close(self):
        self._mm.close()
        self._file.close()


class MetadataDB:
    """Reads music_meta.bin for ISRC → title/artist lookup.

    Binary format (from build_db.py):
      Header (128 bytes):
        magic(4s) version(I) song_count(I) artist_count(I) album_count(I) = 20 bytes
        7 x uint64 section offsets = 56 bytes
        padding to 128 bytes
      Sections:
        1. ISRC Index: song_count x (uint64 packed_isrc + uint32 internal_id) = 12 bytes each
        2. Artist Ranges: artist_count x (uint32 start_id + uint32 name_offset) = 8 bytes each
        3. Album Ranges: album_count x (uint32 start_id + uint32 name_offset) = 8 bytes each
        4. Title Offsets: (song_count+1) x uint32
        5. Title Blob: concatenated 16-bit token sequences
        6. Artist Blob: length-prefixed 16-bit token sequences
        7. Album Blob: length-prefixed 16-bit token sequences
    """

    def __init__(self, meta_path, decoder):
        self._file = open(meta_path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        self.decoder = decoder

        # Parse header
        magic, version, self.song_count, self.artist_count, self.album_count = struct.unpack_from(
            "<4sIIII", self._mm, 0
        )
        offsets = struct.unpack_from("<QQQQQQQ", self._mm, 20)
        (
            self.off_isrc_index,
            self.off_artist_ranges,
            self.off_album_ranges,
            self.off_title_offsets,
            self.off_title_blob,
            self.off_artist_blob,
            self.off_album_blob,
        ) = offsets

    def _binary_search_isrc(self, packed_isrc):
        """Binary search the ISRC index. Returns internal_id or -1."""
        low, high = 0, self.song_count - 1
        while low <= high:
            mid = (low + high) // 2
            pos = self.off_isrc_index + mid * 12
            mid_isrc, mid_id = struct.unpack_from("<QI", self._mm, pos)
            if mid_isrc < packed_isrc:
                low = mid + 1
            elif mid_isrc > packed_isrc:
                high = mid - 1
            else:
                return mid_id
        return -1

    def _read_title(self, internal_id):
        off_pos = self.off_title_offsets + internal_id * 4
        start, end = struct.unpack_from("<II", self._mm, off_pos)
        num_tokens = (end - start) // 2
        blob_pos = self.off_title_blob + start
        tokens = struct.unpack_from(f"<{num_tokens}H", self._mm, blob_pos)
        return self.decoder.decode_tokens(tokens)

    def _read_artist(self, internal_id):
        """Binary search artist ranges for the given internal_id."""
        low, high = 0, self.artist_count - 1
        result_offset = -1
        while low <= high:
            mid = (low + high) // 2
            pos = self.off_artist_ranges + mid * 8
            start_id, name_off = struct.unpack_from("<II", self._mm, pos)
            if internal_id >= start_id:
                result_offset = name_off
                low = mid + 1
            else:
                high = mid - 1
        if result_offset == -1:
            return "Unknown Artist"
        pos = self.off_artist_blob + result_offset
        length = self._mm[pos]
        tokens = struct.unpack_from(f"<{length}H", self._mm, pos + 1)
        return self.decoder.decode_tokens(tokens)

    def lookup(self, packed_isrc):
        """Look up metadata by packed ISRC.

        Returns dict with title and artist, or None if not found.
        """
        internal_id = self._binary_search_isrc(packed_isrc)
        if internal_id == -1:
            return None
        return {
            "title": self._read_title(internal_id),
            "artist": self._read_artist(internal_id),
        }

    def close(self):
        self._mm.close()
        self._file.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /mnt/bigstore/media/apps/code-server/workspace/musicprint && python -m pytest tests/test_metadata.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add demo_app/metadata.py tests/test_metadata.py
git commit -m "feat(demo): add binary metadata reader with tests"
```

---

## Chunk 2: Web App

### Task 4: Flask Backend

**Files:**
- Create: `demo_app/app.py`
- Create: `demo_app/requirements.txt`

The backend loads artifacts on startup, accepts audio via `POST /identify`, and returns JSON.

- [ ] **Step 1: Create requirements.txt**

```
# demo_app/requirements.txt
# Note: torch is installed separately in the Dockerfile from the CPU index
flask>=3.0
numpy
pydub
```

- [ ] **Step 2: Write app.py**

```python
# demo_app/app.py
import os
import sys
import tempfile

import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory

from search import SearchEngine
from metadata import MetadataDB, BPEDecoder
from audio import load_and_resample, window_audio, binarize

RELEASE_DIR = os.environ.get("RELEASE_DIR", os.path.join(os.path.dirname(__file__), "..", "release"))

ENCODER_PATH = os.path.join(RELEASE_DIR, "encoder.pt")
INDEX_PATH = os.path.join(RELEASE_DIR, "audio_index.bin")
META_PATH = os.path.join(RELEASE_DIR, "music_meta.bin")
DECODER_PATH = os.path.join(RELEASE_DIR, "music_decoder.bin")

app = Flask(__name__, static_folder="static")

# Globals loaded at startup
encoder = None
search_engine = None
meta_db = None


def load_artifacts():
    global encoder, search_engine, meta_db

    for path, name in [
        (ENCODER_PATH, "encoder.pt"),
        (INDEX_PATH, "audio_index.bin"),
        (META_PATH, "music_meta.bin"),
        (DECODER_PATH, "music_decoder.bin"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: Missing artifact: {path}", file=sys.stderr)
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading encoder on {device}...")
    encoder = torch.jit.load(ENCODER_PATH, map_location=device)
    encoder.eval()

    search_engine = SearchEngine(INDEX_PATH)
    print(f"Index loaded: {search_engine.num_entries} entries")

    decoder = BPEDecoder(DECODER_PATH)
    meta_db = MetadataDB(META_PATH, decoder)
    print(f"Metadata loaded: {meta_db.song_count} songs")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/identify", methods=["POST"])
def identify():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    suffix = os.path.splitext(audio_file.filename or "audio.webm")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        samples = load_and_resample(tmp_path)
        windows = window_audio(samples)

        if not windows:
            return jsonify({"error": "Audio too short (need at least 5 seconds)"}), 400

        device = next(encoder.parameters()).device
        best_result = None

        for win in windows:
            tensor = torch.from_numpy(win).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = encoder(tensor)
            query_hash = binarize(embedding[0].cpu().numpy())
            result = search_engine.search(query_hash)
            if result and (best_result is None or result["distance"] < best_result["distance"]):
                best_result = result

        if best_result is None:
            return jsonify({"match": False})

        meta = meta_db.lookup(best_result["song_id"])
        return jsonify({
            "match": True,
            "title": meta["title"] if meta else "Unknown",
            "artist": meta["artist"] if meta else "Unknown",
            "isrc": best_result["song_id"],
            "distance": best_result["distance"],
        })
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=False)
```

- [ ] **Step 3: Commit**

```bash
git add demo_app/app.py demo_app/requirements.txt
git commit -m "feat(demo): add Flask backend for song identification"
```

---

### Task 5: Frontend

**Files:**
- Create: `demo_app/static/index.html`

Single page with mic recording (MediaRecorder API) and file upload. Sends audio to `POST /identify`, displays result card.

- [ ] **Step 1: Write the frontend**

```html
<!-- demo_app/static/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MusicPrint Demo</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; background: #111; color: #eee; display: flex; justify-content: center; padding: 40px 20px; }
        .container { max-width: 480px; width: 100%; }
        h1 { font-size: 1.4em; margin-bottom: 24px; text-align: center; }
        .controls { display: flex; gap: 12px; margin-bottom: 24px; }
        button, label.upload { flex: 1; padding: 14px; border: 1px solid #333; border-radius: 8px; background: #1a1a1a; color: #eee; font-size: 1em; cursor: pointer; text-align: center; }
        button:hover, label.upload:hover { background: #252525; }
        button.recording { background: #7f1d1d; border-color: #b91c1c; }
        input[type="file"] { display: none; }
        .status { text-align: center; color: #888; margin-bottom: 16px; min-height: 20px; }
        .result { border: 1px solid #333; border-radius: 8px; padding: 20px; background: #1a1a1a; display: none; }
        .result.visible { display: block; }
        .result .title { font-size: 1.2em; font-weight: 600; margin-bottom: 4px; }
        .result .artist { color: #aaa; margin-bottom: 12px; }
        .result .meta { font-size: 0.85em; color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h1>MusicPrint</h1>
        <div class="controls">
            <button id="mic-btn">Record</button>
            <label class="upload">
                Upload
                <input type="file" id="file-input" accept="audio/*">
            </label>
        </div>
        <div class="status" id="status"></div>
        <div class="result" id="result">
            <div class="title" id="r-title"></div>
            <div class="artist" id="r-artist"></div>
            <div class="meta" id="r-meta"></div>
        </div>
    </div>
    <script>
        const micBtn = document.getElementById('mic-btn');
        const fileInput = document.getElementById('file-input');
        const status = document.getElementById('status');
        const resultDiv = document.getElementById('result');
        let mediaRecorder = null;
        let chunks = [];

        micBtn.addEventListener('click', async () => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
                mediaRecorder.stop();
                return;
            }
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                chunks = [];
                mediaRecorder.ondataavailable = e => chunks.push(e.data);
                mediaRecorder.onstop = () => {
                    stream.getTracks().forEach(t => t.stop());
                    micBtn.textContent = 'Record';
                    micBtn.classList.remove('recording');
                    const blob = new Blob(chunks);
                    sendAudio(blob, 'recording.webm');
                };
                mediaRecorder.start();
                micBtn.textContent = 'Stop';
                micBtn.classList.add('recording');
                status.textContent = 'Listening...';
                resultDiv.classList.remove('visible');
            } catch (e) {
                status.textContent = 'Microphone access denied';
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                sendAudio(fileInput.files[0], fileInput.files[0].name);
                fileInput.value = '';
            }
        });

        async function sendAudio(blob, filename) {
            status.textContent = 'Identifying...';
            resultDiv.classList.remove('visible');
            const form = new FormData();
            form.append('audio', blob, filename);
            try {
                const res = await fetch('/identify', { method: 'POST', body: form });
                const data = await res.json();
                if (data.error) {
                    status.textContent = data.error;
                } else if (!data.match) {
                    status.textContent = 'No match found';
                } else {
                    status.textContent = '';
                    document.getElementById('r-title').textContent = data.title;
                    document.getElementById('r-artist').textContent = data.artist;
                    document.getElementById('r-meta').textContent = `Hamming distance: ${data.distance}`;
                    resultDiv.classList.add('visible');
                }
            } catch (e) {
                status.textContent = 'Request failed';
            }
        }
    </script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add demo_app/static/index.html
git commit -m "feat(demo): add frontend for mic recording and file upload"
```

---

### Task 6: Docker Setup

**Files:**
- Create: `demo_app/Dockerfile`
- Create: `demo_app/docker-compose.yml`

- [ ] **Step 1: Write Dockerfile**

```dockerfile
# demo_app/Dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

- [ ] **Step 2: Write docker-compose.yml**

```yaml
# demo_app/docker-compose.yml
services:
  demo:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ../release:/app/release:ro
    environment:
      - RELEASE_DIR=/app/release
```

- [ ] **Step 3: Commit**

```bash
git add demo_app/Dockerfile demo_app/docker-compose.yml
git commit -m "feat(demo): add Docker setup for web app"
```

---

## Chunk 3: Demo Notebook

### Task 7: Demo Notebook

**Files:**
- Create: `demo.ipynb`

A Jupyter notebook (percent-format `.py` converted to `.ipynb`, or native `.ipynb`) that runs the full pipeline and verification. Runs inside the existing training pipeline container.

- [ ] **Step 1: Write the notebook**

The notebook has 6 cells matching the spec steps. It uses subprocess for pipelines and imports from `demo_app/` for search/audio/metadata.

```python
# Cell 1: Setup & Configuration
# %% [markdown]
# # MusicPrint End-to-End Demo
#
# This notebook trains the encoder, builds the search index, and verifies
# recall on a subset of the catalog.
#
# **Prerequisites:**
# - MP3 files in `music/` named by 12-character ISRC (e.g., `GBAYE0601498.mp3`)
# - Running inside the training pipeline container (`docker compose exec training-pipeline bash`)
#
# **Outputs:** All artifacts written to `release/`.

# %%
import os
import sys
import subprocess
import random
import time
import shutil
import struct

import numpy as np
import torch

# Add demo_app to path for search/audio modules
# In Jupyter, __file__ is not defined, so we use cwd (must be repo root)
ROOT = os.getcwd()
sys.path.insert(0, os.path.join(ROOT, "demo_app"))

from audio import load_and_resample, window_audio, binarize
from search import SearchEngine

MUSIC_DIR = os.path.join(ROOT, "music")
RELEASE_DIR = os.path.join(ROOT, "release")
DATA_DIR = "/tmp/musicprint_data"
CHECKPOINT_DIR = "/tmp/musicprint_checkpoints"

os.makedirs(RELEASE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# List all tracks
tracks = [f for f in os.listdir(MUSIC_DIR) if f.endswith((".mp3", ".flac", ".wav"))]
isrcs = [os.path.splitext(f)[0] for f in tracks]
print(f"Found {len(tracks)} tracks in music/")
```

```python
# Cell 2: Train Encoder
# %% [markdown]
# ## Step 1: Train Encoder

# %%
print("Training encoder (this may take a while)...")
env = os.environ.copy()
env["WANDB_MODE"] = "disabled"

result = subprocess.run(
    [
        sys.executable,
        "1_adapter_training/src/pipeline.py",
        "--source_dir", MUSIC_DIR,
        "--data_dir", DATA_DIR,
        "--checkpoint_dir", CHECKPOINT_DIR,
        "--release_dir", RELEASE_DIR,
    ],
    env=env,
    cwd=ROOT,
)
assert result.returncode == 0, f"Training failed with code {result.returncode}"
assert os.path.exists(os.path.join(RELEASE_DIR, "encoder.pt")), "encoder.pt not found"
print("Encoder trained successfully.")
```

```python
# Cell 3: Build Index
# %% [markdown]
# ## Step 2: Build Index

# %%
print("Building index...")
INDEX_DIR = os.path.join(RELEASE_DIR, "index")

result = subprocess.run(
    [
        sys.executable,
        "2_vector_index/src/pipeline.py",
        "--model_path", os.path.join(RELEASE_DIR, "encoder.pt"),
        "--source_dir", MUSIC_DIR,
        "--data_dir", DATA_DIR,
        "--index_dir", INDEX_DIR,
    ],
    cwd=ROOT,
)
assert result.returncode == 0, f"Indexing failed with code {result.returncode}"

INDEX_PATH = os.path.join(RELEASE_DIR, "audio_index.bin")
assert os.path.exists(INDEX_PATH), "audio_index.bin not found"

# Report index stats
with open(INDEX_PATH, "rb") as f:
    _, _, count = struct.unpack("<4sII", f.read(12))
index_size_mb = os.path.getsize(INDEX_PATH) / 1e6
print(f"Index built: {count} entries, {index_size_mb:.1f} MB")
```

```python
# Cell 4: Copy Metadata
# %% [markdown]
# ## Step 3: Copy Metadata Artifacts

# %%
META_SRC = os.path.join(ROOT, "3_meta_tokenizer", "release")
for fname in ["music_meta.bin", "music_decoder.bin"]:
    src = os.path.join(META_SRC, fname)
    dst = os.path.join(RELEASE_DIR, fname)
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {fname} ({os.path.getsize(dst) / 1024:.0f} KB)")
    else:
        print(f"WARNING: {fname} not found at {src}")
```

```python
# Cell 5: Verification
# %% [markdown]
# ## Step 4 & 5: Clean and Degraded Recall Verification

# %%
# Helper: pack ISRC to uint64 (matches build_db.py and smoke_test_generator.py)
def pack_isrc(isrc_str):
    if not isrc_str or len(isrc_str) != 12:
        return 0
    isrc_str = isrc_str.upper()
    c1, c2 = ord(isrc_str[0]) - ord('A'), ord(isrc_str[1]) - ord('A')
    country = (c1 * 26) + c2
    def c2i(c):
        if 'A' <= c <= 'Z': return ord(c) - ord('A')
        if '0' <= c <= '9': return ord(c) - ord('0') + 26
        return 0
    reg = (c2i(isrc_str[2]) * 36 * 36) + (c2i(isrc_str[3]) * 36) + c2i(isrc_str[4])
    year, desig = int(isrc_str[5:7]), int(isrc_str[7:12])
    return (country << 40) | (reg << 24) | (year << 17) | desig

def degrade_audio(audio, sr=24000):
    """Apply combined degradation: noise + volume + low-pass."""
    # Additive white noise (SNR 5-15 dB)
    snr_db = random.uniform(5, 15)
    signal_power = np.mean(audio ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    audio = audio + np.random.normal(0, np.sqrt(noise_power), len(audio)).astype(np.float32)

    # Volume scaling (-12 to +6 dB)
    gain_db = random.uniform(-12, 6)
    audio = audio * (10 ** (gain_db / 20))

    # Low-pass filter at 4kHz (simple FIR via FFT)
    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    fft[freqs > 4000] = 0
    audio = np.fft.irfft(fft, n=len(audio)).astype(np.float32)

    return audio

def run_recall_test(tracks, isrcs, engine, encoder, device, degrade=False, indices=None):
    """Test recall on a subset of tracks. Returns (correct, total, avg_time)."""
    if indices is None:
        sample_size = max(1, len(tracks) // 20)  # 5%
        indices = random.sample(range(len(tracks)), sample_size)
    correct = 0
    total_time = 0

    for idx in indices:
        track_path = os.path.join(MUSIC_DIR, tracks[idx])
        expected_isrc = pack_isrc(isrcs[idx])

        audio = load_and_resample(track_path)

        # Random 10-second segment
        ten_sec = 24000 * 10
        if len(audio) > ten_sec:
            start = random.randint(0, len(audio) - ten_sec)
            audio = audio[start : start + ten_sec]

        if degrade:
            audio = degrade_audio(audio)

        windows = window_audio(audio)
        if not windows:
            continue

        t0 = time.time()
        best_result = None
        for win in windows:
            tensor = torch.from_numpy(win).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = encoder(tensor)
            query_hash = binarize(embedding[0].cpu().numpy())
            result = engine.search(query_hash)
            if result and (best_result is None or result["distance"] < best_result["distance"]):
                best_result = result
        total_time += time.time() - t0

        if best_result and best_result["song_id"] == expected_isrc:
            correct += 1

    avg_time = total_time / max(len(indices), 1)
    return correct, len(indices), avg_time

# Load encoder and index
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder = torch.jit.load(os.path.join(RELEASE_DIR, "encoder.pt"), map_location=device)
encoder.eval()
engine = SearchEngine(os.path.join(RELEASE_DIR, "audio_index.bin"))

# Select the same 5% sample for both tests (spec: "same 5% of tracks")
sample_size = max(1, len(tracks) // 20)
test_indices = random.sample(range(len(tracks)), sample_size)

# Clean recall
print("Running clean recall test (5% of catalog, 10s clips)...")
clean_correct, clean_total, clean_time = run_recall_test(tracks, isrcs, engine, encoder, device, degrade=False, indices=test_indices)
clean_recall = clean_correct / max(clean_total, 1) * 100
print(f"Clean Recall: {clean_correct}/{clean_total} ({clean_recall:.1f}%) | Avg time: {clean_time:.3f}s")

# Degraded recall (same tracks)
print("\nRunning degraded recall test (noise + volume + low-pass)...")
deg_correct, deg_total, deg_time = run_recall_test(tracks, isrcs, engine, encoder, device, degrade=True, indices=test_indices)
deg_recall = deg_correct / max(deg_total, 1) * 100
print(f"Degraded Recall: {deg_correct}/{deg_total} ({deg_recall:.1f}%) | Avg time: {deg_time:.3f}s")
```

```python
# Cell 6: Summary
# %% [markdown]
# ## Summary

# %%
print("=" * 50)
print("MUSICPRINT VERIFICATION REPORT")
print("=" * 50)
print(f"Catalog:          {len(tracks)} songs")
print(f"Index entries:    {engine.num_entries}")
print(f"Index size:       {os.path.getsize(os.path.join(RELEASE_DIR, 'audio_index.bin')) / 1e6:.1f} MB")
print(f"Clean recall:     {clean_correct}/{clean_total} ({clean_recall:.1f}%)")
print(f"Degraded recall:  {deg_correct}/{deg_total} ({deg_recall:.1f}%)")
print(f"Avg query time:   {clean_time:.3f}s (clean), {deg_time:.3f}s (degraded)")
print("=" * 50)
```

- [ ] **Step 2: Create the notebook as .ipynb**

Convert the percent-format cells above into a proper `.ipynb` file. Use the `nbformat` library or write the JSON directly.

Run: `cd /mnt/bigstore/media/apps/code-server/workspace/musicprint && python -c "import nbformat; print('nbformat available')"` to check if nbformat is available. If not, write the notebook as a `.py` percent-format file (`demo.py`) which JupyterLab can open natively.

- [ ] **Step 3: Commit**

```bash
git add demo.ipynb  # or demo.py
git commit -m "feat(demo): add end-to-end verification notebook"
```

---

### Task 8: Update README and gitignore

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Add release/ to .gitignore if not already present**

Check if `release/` at the repo root is already ignored. The per-pipeline `release/` dirs have their own rules. Add `release/` to the root `.gitignore` if needed (the notebook-generated artifacts should not be tracked).

- [ ] **Step 2: Add demo sections to README**

Add two sections to the end of the README (before License):

```markdown
## Demo Notebook

The `demo.ipynb` notebook trains, indexes, and verifies the full system from a `music/` folder. Run it inside the training pipeline container:

```bash
cd 1_adapter_training
docker compose up --build -d
docker compose exec training-pipeline bash
# Inside container:
cd /workspace
jupyter lab  # or run cells directly
```

It reports clean and degraded recall numbers for a 5% sample of your catalog.

## Web App Demo

A browser-based demo that identifies songs from mic input or file upload.

```bash
cd demo_app
docker compose up --build
# Open http://localhost:5000
```

Requires pre-built artifacts in `release/` (produced by the notebook).
```

- [ ] **Step 3: Commit**

```bash
git add README.md .gitignore
git commit -m "docs: add demo notebook and web app sections to README"
```
