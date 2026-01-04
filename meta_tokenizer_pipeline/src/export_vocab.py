import json
import struct
import argparse
import os

def export_to_binary(json_path, bin_path):
    print(f"Loading tokenizer from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Hugging Face BPE JSON structure: model -> vocab
    # vocab is a dict of { "string": id }
    vocab = data['model']['vocab']
    
    # Sort by ID to ensure the offset table is indexable
    # We need to handle the case where some tokens might be added_tokens
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    token_count = len(sorted_vocab)
    print(f"Exporting {token_count} tokens...")
    
    strings = [item[0] for item in sorted_vocab]
    
    # Encode all strings to UTF-8
    encoded_strings = [s.encode('utf-8') for s in strings]
    
    # Calculate offsets
    offsets = []
    current_offset = 0
    for s in encoded_strings:
        offsets.append(current_offset)
        current_offset += len(s)
    
    # Add one final offset representing the end of the last string
    # This allows calculating length of token i as offsets[i+1] - offsets[i]
    offsets.append(current_offset)
    
    print(f"Writing to {bin_path}...")
    with open(bin_path, 'wb') as f:
        # 1. Header: Token Count (4 bytes)
        f.write(struct.pack('<I', token_count))
        
        # 2. Offset Table: (token_count + 1) * 4 bytes
        # Using uint32 for offsets (supports up to 4GB data blob)
        for offset in offsets:
            f.write(struct.pack('<I', offset))
            
        # 3. Data Blob: All strings concatenated
        for s in encoded_strings:
            f.write(s)
            
    # Verification
    file_size = os.path.getsize(bin_path)
    print(f"Success! Binary size: {file_size / 1024:.2f} KB")
    print(f"Estimated savings vs JSON: {(1 - file_size/os.path.getsize(json_path))*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Tokenizer JSON to Production Binary")
    parser.add_argument("--input", type=str, default="release/music_vocab_final.json", help="Input JSON path")
    parser.add_argument("--output", type=str, default="release/music_vocab.bin", help="Output binary path")
    
    args = parser.parse_args()
    
    export_to_binary(args.input, args.output)
