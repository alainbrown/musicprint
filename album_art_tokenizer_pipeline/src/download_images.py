
import os
import time
import requests
import pandas as pd
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from io import BytesIO
from PIL import Image

# Configuration
MANIFEST_PATH = "data/album_manifest.csv"
OUTPUT_DIR = "data/covers"
MAX_WORKERS = 24 # Conservative limit for CAA
USER_AGENT = "MusicPrint-Indexer/0.1 ( alain.brown@gmail.com )"
TIMEOUT = 10

def setup_directories():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def get_shard_path(uuid):
    """
    Shards 850k files into 256*256 folders based on hash.
    Actually, using the first 4 chars of the UUID is perfectly uniform for version 4 UUIDs.
    """
    if not uuid or len(uuid) < 4: return "misc"
    folder = os.path.join(OUTPUT_DIR, uuid[:2], uuid[2:4])
    return folder

def download_single(row):
    uuid = row.release_uuid
    idx = row.album_index
    
    # 1. Check if exists
    folder = get_shard_path(uuid)
    filename = os.path.join(folder, f"{uuid}.jpg")
    
    if os.path.exists(filename):
        return "SKIPPED"

    # 2. Create folder if needed (race condition safe)
    os.makedirs(folder, exist_ok=True)
    
    # 3. Download
    url = f"https://coverartarchive.org/release/{uuid}/front-250"
    headers = {"User-Agent": USER_AGENT}
    
    try:
        r = requests.get(url, headers=headers, timeout=TIMEOUT)
        
        if r.status_code == 200:
            # Verify it's a valid image before saving
            try:
                img = Image.open(BytesIO(r.content))
                img.verify() # Fast check
                
                with open(filename, "wb") as f:
                    f.write(r.content)
                return "DOWNLOADED"
            except:
                return "INVALID"
                
        elif r.status_code == 404:
            return "MISSING"
        elif r.status_code in [429, 503]:
            time.sleep(2) # Backoff
            return "RATELIMIT"
        else:
            return f"ERROR_{r.status_code}"
            
    except Exception as e:
        return f"EXCEPTION"

def main():
    print(">>> Album Art Downloader Starting...")
    setup_directories()
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"Manifest not found at {MANIFEST_PATH}")
        return

    df = pd.read_csv(MANIFEST_PATH)
    total = len(df)
    print(f"Loaded {total:,} albums to process.")
    
    stats = {
        "DOWNLOADED": 0,
        "SKIPPED": 0,
        "MISSING": 0,
        "INVALID": 0,
        "RATELIMIT": 0,
        "ERROR": 0
    }
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        futures = {executor.submit(download_single, row): row for row in df.itertuples()}
        
        # Process as they complete
        with tqdm(total=total, unit="img") as pbar:
            for future in as_completed(futures):
                res = future.result()
                
                # Update Stats
                key = res if "ERROR" not in res and "EXCEPTION" not in res else "ERROR"
                if key not in stats: stats[key] = 0
                stats[key] += 1
                
                pbar.update(1)
                pbar.set_postfix(dl=stats["DOWNLOADED"], miss=stats["MISSING"])

    print("\n>>> Download Complete.")
    print(stats)

if __name__ == "__main__":
    main()
