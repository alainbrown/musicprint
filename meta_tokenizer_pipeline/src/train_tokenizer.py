import argparse
import os
import pandas as pd
import psycopg2
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

def get_db_connection():
    return psycopg2.connect(
        host=os.environ.get("DB_HOST", "mb_db"),
        database=os.environ.get("DB_NAME", "musicbrainz_db"),
        user=os.environ.get("DB_USER", "musicbrainz"),
        password=os.environ.get("DB_PASSWORD", "musicbrainz")
    )

def train_tokenizer(input_file, output_path, vocab_size=32000, columns=["song.title", "artist.name"], use_db=False, db_query=None):
    """
    Trains a BPE tokenizer on CSV files OR Database Query.
    """
    
    # 1. Prepare Data Source
    temp_train_file = "/tmp/tokenizer_train_corpus.txt"
    
    if use_db:
        print("Loading data from Database...")
        if not db_query:
            # Default Unified Query
            db_query = """
                SELECT name FROM musicbrainz.artist WHERE name IS NOT NULL
                UNION ALL
                SELECT name FROM musicbrainz.recording WHERE name IS NOT NULL
                UNION ALL
                SELECT name FROM musicbrainz.release WHERE name IS NOT NULL
            """
        
        print(f"Executing Query: {db_query[:50]}...")
        
        try:
            conn = get_db_connection()
            # Server-side cursor to handle 40M rows without RAM explosion
            cur = conn.cursor(name='tokenizer_cursor') 
            cur.execute(db_query)
            
            count = 0
            with open(temp_train_file, "w", encoding="utf-8") as f:
                while True:
                    rows = cur.fetchmany(10000)
                    if not rows:
                        break
                    for row in rows:
                        # Assuming 1st column is text
                        if row[0]:
                            f.write(str(row[0]) + "\n")
                    count += len(rows)
                    if count % 1000000 == 0:
                        print(f"  Streamed {count/1000000:.1f}M rows...")
            
            cur.close()
            conn.close()
            print(f"Extracted {count} items from DB.")
            
        except Exception as e:
            print(f"Database Error: {e}")
            return

    else:
        # File-based logic
        print(f"Loading data from {input_file}...")
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found.")

        all_texts = []
        if input_file.endswith(".csv"):
            df = pd.read_csv(input_file)
            for col in columns:
                if col in df.columns:
                    print(f"  - Extracting column: {col}")
                    all_texts.extend(df[col].dropna().astype(str).tolist())
        else:
            with open(input_file, "r", encoding="utf-8") as f:
                all_texts = [line.strip() for line in f if line.strip()]

        print(f"Extracted {len(all_texts)} total items.")
        with open(temp_train_file, "w", encoding="utf-8") as f:
            for t in all_texts:
                f.write(t + "\n")

    # 2. Configure Tokenizer
    print(f"Training BPE Tokenizer (Vocab Size: {vocab_size})...")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size, 
        min_frequency=2, 
        special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
    )

    # 4. Train
    tokenizer.train([temp_train_file], trainer)
    
    # 5. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tokenizer.save(output_path)
    
    # Cleanup
    if os.path.exists(temp_train_file):
        os.remove(temp_train_file)
        
    print(f"Success! Tokenizer saved to: {output_path}")
    print(f"Vocab Size: {tokenizer.get_vocab_size()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BPE Tokenizer for Music Metadata")
    parser.add_argument("--input", type=str, required=False, help="Path to input CSV or TXT file")
    parser.add_argument("--output", type=str, required=True, help="Path to save tokenizer.json")
    parser.add_argument("--columns", nargs='+', default=["song.title", "artist.name"], help="List of CSV columns to train on")
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size (max 65535)")
    parser.add_argument("--use-db", action="store_true", help="Fetch data from Postgres DB")
    parser.add_argument("--db-query", type=str, help="Custom SQL Query")
    
    args = parser.parse_args()
    
    if not args.use_db and not args.input:
        parser.error("Must specify either --input or --use-db")
    
    train_tokenizer(
        input_file=args.input,
        output_path=args.output,
        vocab_size=args.vocab_size,
        columns=args.columns,
        use_db=args.use_db,
        db_query=args.db_query
    )
