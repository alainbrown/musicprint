import os
import sys
import tempfile

import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory

from search import SearchEngine
from metadata import MetadataDB, BPEDecoder
from audio import load_and_resample, window_audio, binarize

RELEASE_DIR = os.environ.get("RELEASE_DIR", os.path.join(os.path.dirname(__file__), "..", "release"))

ENCODER_PATH = os.path.join(RELEASE_DIR, "encoder.pt")
INDEX_PATH = os.path.join(RELEASE_DIR, "audio_index.bin")
META_PATH = os.path.join(RELEASE_DIR, "music_meta.bin")
DECODER_PATH = os.path.join(RELEASE_DIR, "music_decoder.bin")

app = Flask(__name__, static_folder="static")

encoder = None
search_engine = None
meta_db = None


def load_artifacts():
    global encoder, search_engine, meta_db

    for path, name in [
        (ENCODER_PATH, "encoder.pt"),
        (INDEX_PATH, "audio_index.bin"),
        (META_PATH, "music_meta.bin"),
        (DECODER_PATH, "music_decoder.bin"),
    ]:
        if not os.path.exists(path):
            print(f"ERROR: Missing artifact: {path}", file=sys.stderr)
            sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading encoder on {device}...")
    encoder = torch.jit.load(ENCODER_PATH, map_location=device)
    encoder.eval()

    search_engine = SearchEngine(INDEX_PATH)
    print(f"Index loaded: {search_engine.num_entries} entries")

    decoder = BPEDecoder(DECODER_PATH)
    meta_db = MetadataDB(META_PATH, decoder)
    print(f"Metadata loaded: {meta_db.song_count} songs")


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/identify", methods=["POST"])
def identify():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    suffix = os.path.splitext(audio_file.filename or "audio.webm")[1] or ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        samples = load_and_resample(tmp_path)
        windows = window_audio(samples)

        if not windows:
            return jsonify({"error": "Audio too short (need at least 5 seconds)"}), 400

        device = next(encoder.parameters()).device
        best_result = None

        for win in windows:
            tensor = torch.from_numpy(win).unsqueeze(0).to(device)
            with torch.no_grad():
                embedding = encoder(tensor)
            query_hash = binarize(embedding[0].cpu().numpy())
            result = search_engine.search(query_hash)
            if result and (best_result is None or result["distance"] < best_result["distance"]):
                best_result = result

        if best_result is None:
            return jsonify({"match": False})

        meta = meta_db.lookup(best_result["song_id"])
        return jsonify({
            "match": True,
            "title": meta["title"] if meta else "Unknown",
            "artist": meta["artist"] if meta else "Unknown",
            "isrc": best_result["song_id"],
            "distance": best_result["distance"],
        })
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    load_artifacts()
    app.run(host="0.0.0.0", port=5000, debug=False)
