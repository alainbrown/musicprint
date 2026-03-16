import struct
import tempfile
import os
from metadata import BPEDecoder, MetadataDB

def make_test_decoder(tokens, path):
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
    song_count = len(songs)
    artist_count = song_count

    isrc_pairs = []
    artist_ranges = []
    title_blob = bytearray()
    title_offsets = []
    artist_blob = bytearray()

    for i, (packed_isrc, title_toks, artist_toks) in enumerate(songs):
        isrc_pairs.append((packed_isrc, i))
        title_offsets.append(len(title_blob))
        title_blob.extend(struct.pack(f"<{len(title_toks)}H", *title_toks))
        artist_ranges.append((i, len(artist_blob)))
        artist_blob.extend(struct.pack("<B", len(artist_toks)))
        artist_blob.extend(struct.pack(f"<{len(artist_toks)}H", *artist_toks))

    title_offsets.append(len(title_blob))

    header_size = 128
    isrc_index_size = song_count * 12
    artist_range_size = artist_count * 8
    album_range_size = 0
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

        make_test_decoder(["Hel", "lo", " ", "World", "Art", "ist"], dec_path)
        dec = BPEDecoder(dec_path)

        make_test_meta_db([
            (100, [0, 1, 2, 3], [4, 5]),
            (200, [3], [0, 1]),
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
