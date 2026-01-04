import os
import psycopg2
import struct
import time
from tokenizers import Tokenizer

# Configuration
DB_HOST = os.environ.get("DB_HOST", "mb_db")
DB_NAME = os.environ.get("DB_NAME", "musicbrainz_db")
DB_USER = os.environ.get("DB_USER", "musicbrainz")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "musicbrainz")

TOKENIZER_PATH = "release/music_encoder.json"
OUTPUT_PATH = "release/music_meta.bin"

# UI-Driven Truncation Limits
LIMIT_ARTIST = 15
LIMIT_ALBUM = 20
LIMIT_TITLE = 25

def pack_isrc(isrc_str):
    if not isrc_str or len(isrc_str) != 12: return 0
    isrc_str = isrc_str.upper()
    try:
        c1, c2 = ord(isrc_str[0]) - ord('A'), ord(isrc_str[1]) - ord('A')
        country = (c1 * 26) + c2
        def c2i(c):
            if 'A' <= c <= 'Z': return ord(c) - ord('A')
            if '0' <= c <= '9': return ord(c) - ord('0') + 26
            return 0
        reg = (c2i(isrc_str[2])*36*36) + (c2i(isrc_str[3])*36) + c2i(isrc_str[4])
        year, desig = int(isrc_str[5:7]), int(isrc_str[7:12])
        return (country << 40) | (reg << 24) | (year << 17) | desig
    except: return 0

def get_db_connection():
    return psycopg2.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASSWORD)

def build_db():
    print(">>> Building Unified Clustered Metadata + ISRC Index...")
    
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Error: Tokenizer not found at {TOKENIZER_PATH}")
        return
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    conn = get_db_connection()
    cur = conn.cursor(name='build_db_cursor')
    
    # IMPORTANT: We sort by Artist/Album to ensure CLUSTERING works.
    query = """
        SELECT 
            i.isrc,
            ac.id as artist_id,
            ac.name as artist_name,
            r.id as release_id,
            r.name as album_name,
            rec.name as song_title
        FROM musicbrainz.recording rec
        JOIN musicbrainz.isrc i ON i.recording = rec.id
        JOIN musicbrainz.track t ON t.recording = rec.id
        JOIN musicbrainz.medium m ON t.medium = m.id
        JOIN musicbrainz.release r ON m.release = r.id
        JOIN musicbrainz.artist_credit ac ON rec.artist_credit = ac.id
        WHERE i.isrc IS NOT NULL
        ORDER BY ac.id, r.id, rec.id
    """
    
    print("Streaming records from Postgres...")
    cur.execute(query)
    
    isrc_id_pairs = [] # List of (packed_isrc, internal_id) 
    
    artist_ranges = []
    album_ranges = []
    title_blob = bytearray()
    title_offsets = [] 
    
    artist_name_blob = bytearray()
    artist_name_map = {}
    album_name_blob = bytearray()
    album_name_map = {}

    last_artist_id = -1
    last_release_id = -1
    song_count = 0
    start_time = time.time()

    while True:
        rows = cur.fetchmany(50000)
        if not rows: break
            
        for isrc, artist_id, artist_name, release_id, album_name, title in rows:
            # 1. Map random ISRC to current sequential Internal ID
            isrc_id_pairs.append((pack_isrc(isrc), song_count))

            # 2. Artist Clustering
            if artist_id != last_artist_id:
                if artist_id not in artist_name_map:
                    tokens = tokenizer.encode(str(artist_name)).ids[:LIMIT_ARTIST]
                    offset = len(artist_name_blob)
                    artist_name_blob.extend(struct.pack("<B", len(tokens)))
                    artist_name_blob.extend(struct.pack(f"<{len(tokens)}H", *tokens))
                    artist_name_map[artist_id] = offset
                artist_ranges.append((song_count, artist_name_map[artist_id]))
                last_artist_id = artist_id
            
            # 3. Album Clustering
            if release_id != last_release_id:
                if release_id not in album_name_map:
                    tokens = tokenizer.encode(str(album_name)).ids[:LIMIT_ALBUM]
                    offset = len(album_name_blob)
                    album_name_blob.extend(struct.pack("<B", len(tokens)))
                    album_name_blob.extend(struct.pack(f"<{len(tokens)}H", *tokens))
                    album_name_map[release_id] = offset
                album_ranges.append((song_count, album_name_map[release_id]))
                last_release_id = release_id
            
            # 4. Title Storage
            tokens = tokenizer.encode(str(title)).ids[:LIMIT_TITLE]
            title_offsets.append(len(title_blob))
            title_blob.extend(struct.pack(f"<{len(tokens)}H", *tokens))
            
            song_count += 1
            if song_count % 1000000 == 0:
                print(f"  Processed {song_count/1000000:.1f}M songs...")

    title_offsets.append(len(title_blob))
    
    # 5. SORT ISRC INDEX
    # This is critical: The iOS app uses binary search on this section.
    print("Sorting ISRC Index...")
    isrc_id_pairs.sort()
    
    # 6. Finalize Binary Layout
    header_size = 64
    isrc_index_size = len(isrc_id_pairs) * 12 # uint64 ISRC + uint32 ID
    artist_range_size = len(artist_ranges) * 8
    album_range_size = len(album_ranges) * 8
    title_offset_size = len(title_offsets) * 4
    
    off_isrc_index = header_size
    off_artist_ranges = off_isrc_index + isrc_index_size
    off_album_ranges = off_artist_ranges + artist_range_size
    off_title_offsets = off_album_ranges + album_range_size
    off_title_blob = off_title_offsets + title_offset_size
    off_artist_blob = off_title_blob + len(title_blob)
    off_album_blob = off_artist_blob + len(artist_name_blob)
    
    with open(OUTPUT_PATH, "wb") as f:
        # Header (Version 3: Clustered + Sorted Index)
        f.write(struct.pack("<4sIIII", b"MPDB", 3, song_count, len(artist_ranges), len(album_ranges)))
        f.write(struct.pack("<QQQQQQ", off_isrc_index, off_artist_ranges, off_album_ranges, off_title_offsets, off_title_blob, off_artist_blob))
        f.write(struct.pack("<Q", off_album_blob))
        f.write(b"\x00" * (header_size - f.tell()))
        
        # Section 1: Sorted ISRC Index (12 bytes per track)
        for packed_isrc, internal_id in isrc_id_pairs:
            f.write(struct.pack("<QI", packed_isrc, internal_id))
            
        # Section 2+: Clustered Metadata
        for start_id, name_off in artist_ranges: f.write(struct.pack("<II", start_id, name_off))
        for start_id, name_off in album_ranges: f.write(struct.pack("<II", start_id, name_off))
        for off in title_offsets: f.write(struct.pack("<I", off))
        f.write(title_blob)
        f.write(artist_name_blob)
        f.write(album_name_blob)

    print(f"\nSUCCESS! Created {OUTPUT_PATH} with {song_count:,} tracks in {time.time()-start_time:.1f}s.")

if __name__ == "__main__":
    build_db()
