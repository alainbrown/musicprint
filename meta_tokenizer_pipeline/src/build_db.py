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

TOKENIZER_PATH = "models/music_vocab.json"
OUTPUT_PATH = "release/music_meta.bin"

# UI-Driven Truncation Limits
LIMIT_ARTIST = 15
LIMIT_ALBUM = 20
LIMIT_TITLE = 25

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

def build_db():
    print(f">>> Building Clustered Binary Database (With UX Truncation)..." )
    
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Error: Tokenizer not found at {TOKENIZER_PATH}")
        return
    
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    conn = get_db_connection()
    cur = conn.cursor(name='build_db_cursor')
    
    query = """
        SELECT 
            ac.id as artist_id,
            ac.name as artist_name,
            r.id as release_id,
            r.name as album_name,
            rec.name as song_title
        FROM musicbrainz.recording rec
        JOIN musicbrainz.track t ON t.recording = rec.id
        JOIN musicbrainz.medium m ON t.medium = m.id
        JOIN musicbrainz.release r ON m.release = r.id
        JOIN musicbrainz.artist_credit ac ON rec.artist_credit = ac.id
        ORDER BY ac.id, r.id, rec.id
    """
    
    print("Executing clustered query (this may take a minute)...")
    cur.execute(query)
    
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

    print(f"Streaming and Packing records (Artist max: {LIMIT_ARTIST}, Album max: {LIMIT_ALBUM}, Title max: {LIMIT_TITLE})...")
    
    while True:
        rows = cur.fetchmany(50000)
        if not rows:
            break
            
        for artist_id, artist_name, release_id, album_name, title in rows:
            # 1. Handle Artist Clustering
            if artist_id != last_artist_id:
                if artist_id not in artist_name_map:
                    # Encode and truncate
                    tokens = tokenizer.encode(str(artist_name)).ids[:LIMIT_ARTIST]
                    offset = len(artist_name_blob)
                    # Store as: [uint8 length, uint16 tokens...]
                    artist_name_blob.extend(struct.pack("<B", len(tokens)))
                    artist_name_blob.extend(struct.pack(f"<{len(tokens)}H", *tokens))
                    artist_name_map[artist_id] = offset
                
                artist_ranges.append((song_count, artist_name_map[artist_id]))
                last_artist_id = artist_id
            
            # 2. Handle Album Clustering
            if release_id != last_release_id:
                if release_id not in album_name_map:
                    # Encode and truncate
                    tokens = tokenizer.encode(str(album_name)).ids[:LIMIT_ALBUM]
                    offset = len(album_name_blob)
                    album_name_blob.extend(struct.pack("<B", len(tokens)))
                    album_name_blob.extend(struct.pack(f"<{len(tokens)}H", *tokens))
                    album_name_map[release_id] = offset
                
                album_ranges.append((song_count, album_name_map[release_id]))
                last_release_id = release_id
            
            # 3. Handle Title
            # Encode and truncate
            tokens = tokenizer.encode(str(title)).ids[:LIMIT_TITLE]
            title_offsets.append(len(title_blob))
            title_blob.extend(struct.pack(f"<{len(tokens)}H", *tokens))
            
            song_count += 1
            if song_count % 1000000 == 0:
                print(f"  Packed {song_count/1000000:.1f}M songs... (Elapsed: {time.time()-start_time:.1f}s)")

    title_offsets.append(len(title_blob))
    
    print(f"Finalizing file: {OUTPUT_PATH}")
    
    # CALCULATE SECTION OFFSETS
    header_size = 64
    artist_range_size = len(artist_ranges) * 8
    album_range_size = len(album_ranges) * 8
    title_offset_size = len(title_offsets) * 4
    
    off_artist_ranges = header_size
    off_album_ranges = off_artist_ranges + artist_range_size
    off_title_offsets = off_album_ranges + album_range_size
    off_title_blob = off_title_offsets + title_offset_size
    off_artist_blob = off_title_blob + len(title_blob)
    off_album_blob = off_artist_blob + len(artist_name_blob)
    
    with open(OUTPUT_PATH, "wb") as f:
        # 1. HEADER (64 bytes)
        f.write(struct.pack("<4s", b"MPDB"))
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<I", song_count))
        f.write(struct.pack("<I", len(artist_ranges)))
        f.write(struct.pack("<I", len(album_ranges)))
        
        f.write(struct.pack("<Q", off_artist_ranges))
        f.write(struct.pack("<Q", off_album_ranges))
        f.write(struct.pack("<Q", off_title_offsets))
        f.write(struct.pack("<Q", off_title_blob))
        f.write(struct.pack("<Q", off_artist_blob))
        f.write(struct.pack("<Q", off_album_blob))
        
        f.write(b"\x00" * (64 - f.tell()))
        
        # 2. SECTIONS
        for start_id, name_off in artist_ranges:
            f.write(struct.pack("<II", start_id, name_off))
            
        for start_id, name_off in album_ranges:
            f.write(struct.pack("<II", start_id, name_off))
            
        for off in title_offsets:
            f.write(struct.pack("<I", off))
            
        f.write(title_blob)
        f.write(artist_name_blob)
        f.write(album_name_blob)

    cur.close()
    conn.close()
    
    elapsed = time.time() - start_time
    final_size = os.path.getsize(OUTPUT_PATH)
    print(f"\nSUCCESS!")
    print(f"Total Songs: {song_count:,}")
    print(f"Final File Size: {final_size / 1024 / 1024:.2f} MB")
    print(f"Average per song: {final_size / song_count:.2f} bytes")
    print(f"Time taken: {elapsed:.1f}s")

if __name__ == "__main__":
    build_db()