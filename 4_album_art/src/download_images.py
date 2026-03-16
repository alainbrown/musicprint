import os
import time
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from io import BytesIO
from PIL import Image
import random

# Configuration
# Mount path from docker-compose
MANIFEST_PATH = "/vol/meta_release/album_manifest.csv"
OUTPUT_DIR = "/vol/data"

# POLITE SETTINGS
MAX_WORKERS = 4      # Reduced from 24 to stay under IA rate limits
DELAY_BETWEEN = 0.2  # Mandatory small sleep to prevent bursts
USER_AGENT = "MusicPrint-Indexer/1.0 ( alain.brown@gmail.com )"
TIMEOUT = 15

def setup_directories(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

def get_shard_path(uuid, output_dir):
    if not uuid or len(uuid) < 4: return os.path.join(output_dir, "misc")
    folder = os.path.join(output_dir, uuid[:2], uuid[2:4])
    return folder

def download_single(row, output_dir):
    uuid = row.release_uuid
    
    # 1. Check if exists
    folder = get_shard_path(uuid, output_dir)
    filename = os.path.join(folder, f"{uuid}.jpg")
    
    if os.path.exists(filename):
        return "SKIPPED"

    # 2. Create folder
    os.makedirs(folder, exist_ok=True)
    
    # 3. Download with Retries & Exponential Backoff
    url = f"https://coverartarchive.org/release/{uuid}/front-250"
    headers = {"User-Agent": USER_AGENT}
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Polite Delay
            time.sleep(DELAY_BETWEEN + random.uniform(0, 0.1))
            
            r = requests.get(url, headers=headers, timeout=TIMEOUT)
            
            if r.status_code == 200:
                try:
                    img = Image.open(BytesIO(r.content))
                    img.verify()
                    with open(filename, "wb") as f:
                        f.write(r.content)
                    return "DOWNLOADED"
                except:
                    return "INVALID"
                    
            elif r.status_code == 404:
                return "MISSING"
            
            elif r.status_code in [429, 503]:
                # Exponential backoff
                wait = (2 ** attempt) * 5 + random.uniform(0, 2)
                time.sleep(wait)
                continue # Retry
                
            else:
                return f"ERROR_{r.status_code}"
                
        except Exception as e:
            time.sleep(2)
            continue
            
    return "FAILED"

def main():
    print(">>> Polite Album Art Downloader Starting...")
    setup_directories(OUTPUT_DIR)
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"Manifest not found at {MANIFEST_PATH}")
        return

    df = pd.read_csv(MANIFEST_PATH)
    total = len(df)
    print(f"Loaded {total:,} albums to process.")
    
    stats = {
        "DOWNLOADED": 0, "SKIPPED": 0, "MISSING": 0,
        "INVALID": 0, "RATELIMIT": 0, "ERROR": 0, "FAILED": 0
    }
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(download_single, row, OUTPUT_DIR): row for row in df.itertuples()}
        
        with tqdm(total=total, unit="img") as pbar:
            for future in as_completed(futures):
                res = future.result()
                
                # Update Stats
                key = res if "ERROR" not in res else "ERROR"
                if key not in stats: stats[key] = 0
                stats[key] += 1
                
                pbar.update(1)
                pbar.set_postfix(dl=stats["DOWNLOADED"], miss=stats["MISSING"])

    print("\n>>> Download Complete.")
    print(stats)

if __name__ == "__main__":
    main()