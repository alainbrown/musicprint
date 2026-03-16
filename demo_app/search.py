import struct
import mmap

HEADER_SIZE = 64
ENTRY_SIZE = 16


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
        if self.num_entries == 0:
            return None

        best_dist = 65
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
