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
    """Reads music_meta.bin for ISRC -> title/artist lookup.

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
