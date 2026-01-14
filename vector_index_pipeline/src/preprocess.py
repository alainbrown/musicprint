import os
import glob
import subprocess
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from pathlib import Path

def convert_to_flac(args):
    """
    Worker function to convert a single file to FLAC.
    """
    input_path, output_path, ffmpeg_verbosity = args
    
    # Skip if output exists and is not empty
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return "skipped"

    try:
        # ffmpeg command:
        # -y: overwrite output
        # -i: input
        # -c:a flac: use FLAC codec
        # -ar 24000: resample to 24kHz (Pipeline Standard)
        # -ac 1: mix to mono (Pipeline Standard)
        # -map_metadata 0: copy metadata
        
        cmd = [
            "ffmpeg", "-y", 
            "-i", input_path,
            "-c:a", "flac",
            "-ar", "24000",
            "-ac", "1",
            "-map_metadata", "0",
            output_path
        ]
        
        # Run silent unless error
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode != 0:
            return f"error: {result.stderr}"
            
        return "success"
    except Exception as e:
        return f"exception: {str(e)}"

def preprocess_dataset(data_dir, output_dir, workers=None):
    """
    Main driver to convert entire folder tree to FLAC in a separate directory.
    """
    if workers is None:
        workers = max(1, cpu_count() - 2)
        
    print(f"Scanning {data_dir} for audio files...")
    extensions = ['*.mp3', '*.wav', '*.m4a', '*.aac']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))
        
    tasks = []
    for f in files:
        # Construct mirrored output path
        rel_path = os.path.relpath(f, data_dir)
        output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.flac')
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        tasks.append((f, output_path, "error"))
        
    print(f"Found {len(tasks)} candidates. Starting conversion to {output_dir} with {workers} workers...")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    with Pool(workers) as pool:
        for result in tqdm(pool.imap_unordered(convert_to_flac, tasks), total=len(tasks)):
            if result == "success":
                success_count += 1
            elif result == "skipped":
                skip_count += 1
            else:
                error_count += 1
                
    print(f"\nProcessing Complete.")
    print(f"✅ Converted: {success_count}")
    print(f"⏭️ Skipped:   {skip_count}")
    print(f"❌ Errors:    {error_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/vol/src_music")
    parser.add_argument("--output_dir", type=str, default="/vol/data")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    
    preprocess_dataset(args.data_dir, args.output_dir, args.workers)
