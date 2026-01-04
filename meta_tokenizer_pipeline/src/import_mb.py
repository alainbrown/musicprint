import os
import requests
import tarfile
import psycopg2
import time

# Configuration
MB_FTP_BASE = "http://ftp.musicbrainz.org/pub/musicbrainz/data/fullexport"
SCHEMA_URL = "https://raw.githubusercontent.com/metabrainz/musicbrainz-server/master/admin/sql/CreateTables.sql"
DB_HOST = os.environ.get("DB_HOST", "mb_db")
DB_NAME = os.environ.get("DB_NAME", "musicbrainz_db")
DB_USER = os.environ.get("DB_USER", "musicbrainz")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "musicbrainz")

# We only target the core tables needed for the 100M app
# But we must create ALL tables because CreateTables.sql is monolithic.
# The import loop will only populate these specific ones.
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
    try:
        r = requests.get(f"{MB_FTP_BASE}/LATEST")
        r.raise_for_status()
        version = r.text.strip()
        url = f"{MB_FTP_BASE}/{version}/mbdump.tar.bz2"
        print(f"Latest dump: {url}")
        return url
    except Exception as e:
        print(f"Error fetching LATEST: {e}")
        return None

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
    print("Fetching official schema...")
    r = requests.get(SCHEMA_URL)
    r.raise_for_status()
    sql = r.text
    
    # MusicBrainz schema uses a specific structure.
    # We need to ensure we are in the 'musicbrainz' schema or public.
    # The official SQL often assumes a 'musicbrainz' schema exists.
    cur.execute("CREATE SCHEMA IF NOT EXISTS musicbrainz;")
    cur.execute("SET search_path TO musicbrainz, public;")
    
    print("Executing CreateTables.sql...")
    cur.execute(sql)
    print("Schema initialized.")

def import_dump():
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    
    # 1. Init Schema
    try:
        init_schema(cur)
    except Exception as e:
        print(f"Schema init warning (might already exist): {e}")
    
    # 2. Download and Stream
    url = get_latest_url()
    if not url:
        return

    print("Starting Stream Download & Import...")
    
    # Request stream
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        
        # Open tar stream
        with tarfile.open(fileobj=r.raw, mode="r|bz2") as tar:
            for member in tar:
                if member.name in TARGET_TABLES:
                    table_name = TARGET_TABLES[member.name]
                    print(f"Found {member.name} -> Importing to table 'musicbrainz.{table_name}'...")
                    
                    f = tar.extractfile(member)
                    if f:
                        # Postgres COPY
                        # MusicBrainz format: Text, Tab-separated, \N for null
                        sql = f"COPY musicbrainz.{table_name} FROM STDIN WITH (FORMAT 'text', DELIMITER '\t', NULL '\\N', ESCAPE '\\', QUOTE '\\b');" # \b is a dummy quote char since MB doesn't use standard quoting
                        
                        try:
                            cur.copy_expert(sql, f)
                            print(f"Successfully imported {table_name}")
                        except Exception as e:
                            print(f"Failed to import {table_name}: {e}")
                        
    conn.close()
    print("Import process finished.")

if __name__ == "__main__":
    import_dump()
