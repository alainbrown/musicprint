import struct
import tempfile
import os
import numpy as np
from search import SearchEngine

def make_test_index(entries, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"MPAF", 1, len(entries)))
        f.write(b"\x00" * 52)
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
