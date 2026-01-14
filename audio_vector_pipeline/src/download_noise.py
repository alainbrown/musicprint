
import os
import requests
import zipfile
import io

NOISE_DIR = "/vol/data/noise"

def download_noise():
    """
    Downloads and extracts the DEMAND noise dataset for audio augmentation.
    """
    os.makedirs(NOISE_DIR, exist_ok=True)
    
    # Direct links to specific environmental noise files from DEMAND (Zenodo)
    noise_urls = [
        ("DKITCHEN", "https://zenodo.org/record/1227121/files/DKITCHEN_16k.zip?download=1"), # Domestic Kitchen
        ("PCAFETER", "https://zenodo.org/record/1227121/files/PCAFETER_16k.zip?download=1"), # Busy Cafeteria
        ("NFIELD", "https://zenodo.org/record/1227121/files/NFIELD_16k.zip?download=1"),    # Nature Field
        ("TCAR", "https://zenodo.org/record/1227121/files/TCAR_16k.zip?download=1"),       # Car Interior
    ]

    print(f">>> Checking for Noise Dataset in {NOISE_DIR}...")
    
    downloaded_count = 0
    for name, url in noise_urls:
        target_dir = os.path.join(NOISE_DIR, name)
        
        # Check if dir exists and has files
        if os.path.exists(target_dir) and os.listdir(target_dir):
            print(f"  - {name}: Exists (Skipping)")
            continue
            
        print(f"  - {name}: Downloading...")
        try:
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code != 200:
                print(f"    ❌ Failed to fetch {url} (Status: {r.status_code})")
                continue
                
            os.makedirs(target_dir, exist_ok=True)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            
            # Robust Extraction: Strip internal folders and put everything in target_dir
            count = 0
            for file_info in z.infolist():
                if file_info.filename.endswith('.wav'):
                    # Get just the filename, ignoring internal folders
                    base_name = os.path.basename(file_info.filename)
                    if not base_name: continue
                    
                    target_path = os.path.join(target_dir, base_name)
                    with open(target_path, "wb") as f_out:
                        f_out.write(z.read(file_info))
                    count += 1
            
            print(f"    ✅ Extracted {count} files to {target_dir}")
            downloaded_count += 1
        except Exception as e:
            print(f"    ❌ Exception: {e}")

    if downloaded_count > 0:
        print(">>> Noise Download Complete.")
    else:
        print(">>> Noise Dataset Ready.")

if __name__ == "__main__":
    download_noise()
