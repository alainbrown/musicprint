import os
import tarfile
import psycopg2
import time
import requests
import re

# Configuration
MB_FTP_BASE = "http://ftp.musicbrainz.org/pub/musicbrainz/data/fullexport"
TYPES_SQL_URL = "https://raw.githubusercontent.com/metabrainz/musicbrainz-server/master/admin/sql/CreateTypes.sql"
TABLES_SQL_URL = "https://raw.githubusercontent.com/metabrainz/musicbrainz-server/master/admin/sql/CreateTables.sql"

INPUT_FILE = "/app/data/mbdump.tar.bz2"
DB_HOST = os.environ.get("DB_HOST", "mb_db")
DB_NAME = os.environ.get("DB_NAME", "musicbrainz_db")
DB_USER = os.environ.get("DB_USER", "musicbrainz")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "musicbrainz")

# Target Tables
TARGET_TABLES = {
    "mbdump/artist": "artist",
    "mbdump/artist_credit": "artist_credit",
    "mbdump/artist_credit_name": "artist_credit_name",
    "mbdump/recording": "recording",
    "mbdump/release": "release",
    "mbdump/track": "track"
}

def get_latest_url():
    print(f"Checking {MB_FTP_BASE}/LATEST ...")
    r = requests.get(f"{MB_FTP_BASE}/LATEST")
    r.raise_for_status()
    version = r.text.strip()
    return f"{MB_FTP_BASE}/{version}/mbdump.tar.bz2"

def download_file(url, dest):
    print(f"Downloading {url} to {dest}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                downloaded += len(chunk)
                if downloaded % (1024 * 1024 * 50) == 0: # Print every 50MB
                    print(f"  Downloaded {downloaded / 1024 / 1024:.0f}MB / {total_size / 1024 / 1024:.0f}MB")
    print("Download complete.")

def get_db_connection():
    max_retries = 30
    for i in range(max_retries):
        try:
            conn = psycopg2.connect(
                host=DB_HOST,
                database=DB_NAME,
                user=DB_USER,
                password=DB_PASSWORD
            )
            return conn
        except psycopg2.OperationalError:
            print(f"Waiting for DB... ({i+1}/{max_retries})")
            time.sleep(2)
    raise Exception("Could not connect to Database")

def init_schema(cur):
    print("Checking Schema...")
    # check if table exists
    cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'musicbrainz' AND table_name = 'recording');")
    if cur.fetchone()[0]:
        print("Schema seems to exist. Skipping initialization.")
        return

    print("Initializing Schema...")
    
    # 1. Setup Namespace & Extensions
    cur.execute('CREATE SCHEMA IF NOT EXISTS musicbrainz')
    cur.execute('SET search_path TO musicbrainz, public')
    cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    cur.execute('CREATE EXTENSION IF NOT EXISTS cube')

    # 2. Types
    print("Downloading and applying CreateTypes.sql...")
    r = requests.get(TYPES_SQL_URL)
    r.raise_for_status()
    sql = r.text
    # Remove psql \set commands
    sql = re.sub(r'^\\.*$', '', sql, flags=re.MULTILINE)
    cur.execute(sql)

    # 3. Tables
    print("Downloading and applying CreateTables.sql...")
    r = requests.get(TABLES_SQL_URL)
    r.raise_for_status()
    sql = r.text
    # Remove psql \set commands and custom collation
    sql = re.sub(r'^\\.*$', '', sql, flags=re.MULTILINE)
    sql = sql.replace('COLLATE musicbrainz', '')
    cur.execute(sql)
    
    print("Schema initialized successfully.")

def import_mb():
    # 1. Ensure Data is present
    if not os.path.exists(INPUT_FILE):
        url = get_latest_url()
        download_file(url, INPUT_FILE)
    else:
        print(f"File {INPUT_FILE} already exists. Skipping download.")

    # 2. Connect & Init Schema
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    
    init_schema(cur)
    
    # 3. Import Data
    cur.execute("SET search_path TO musicbrainz, public;")

    print(f"Opening {INPUT_FILE} for extraction...")
    with tarfile.open(INPUT_FILE, mode="r:bz2") as tar:
        for member in tar:
            if member.name in TARGET_TABLES:
                table_name = TARGET_TABLES[member.name]
                print(f"Importing {member.name} ({member.size / 1024 / 1024:.2f} MB)...")
                
                f = tar.extractfile(member)
                if f:
                    # Clear table first (Idempotency)
                    print(f"  - Truncating {table_name}...")
                    cur.execute(f"TRUNCATE musicbrainz.{table_name} CASCADE;")
                    
                    # MusicBrainz format: Text, Tab-separated, \N for null
                    # We use raw COPY for maximum speed. No QUOTE/ESCAPE in 'text' format.
                    sql = f"COPY musicbrainz.{table_name} FROM STDIN WITH (FORMAT 'text', DELIMITER '\t', NULL '\\N');"
                    
                    try:
                        start_time = time.time()
                        cur.copy_expert(sql, f)
                        print(f"✅ Imported {table_name} in {time.time() - start_time:.2f}s")
                    except Exception as e:
                        print(f"❌ Failed to import {table_name}: {e}")
                        
    conn.close()
    print("Import process finished.")

if __name__ == "__main__":
    print(">>> MusicBrainz Importer Starting...")
    import_mb()