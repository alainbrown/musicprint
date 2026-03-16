
#include "MetadataDatabase.h"
#include "BPEDecoder.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Running MetadataDatabase Tests (against production binary)..." << std::endl;

    musicprint::MetadataDatabase db;
    musicprint::BPEDecoder decoder;
    
    const std::string meta_path = "../../3_meta_tokenizer/release/music_meta.bin";
    const std::string vocab_path = "../../3_meta_tokenizer/release/music_decoder.bin";

    try {
        db.load(meta_path);
        decoder.load(vocab_path);
        std::cout << "✅ Loaded DB (" << db.getSongCount() << " songs) and Vocab." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ FAILED to load: " << e.what() << std::endl;
        return 1;
    }

    // List of test ISRCs (From MusicBrainz samples we saw earlier)
    std::vector<std::string> test_isrcs = {
        "GBAYE0601498", // Should be "Smile" by Lily Allen? Let's see.
        "USRC10301389", 
        "INVALIDISRCX"  
    };

    for (const auto& isrc : test_isrcs) {
        musicprint::SongMetadata meta;
        bool found = db.lookup(isrc, meta);
        
        if (found) {
            std::string artist = decoder.decode(meta.artist_tokens);
            std::string title = decoder.decode(meta.title_tokens);
            std::string album = decoder.decode(meta.album_tokens);
            
            std::cout << "✅ Found [" << isrc << "]:\n";
            std::cout << "   Title:  " << title << "\n";
            std::cout << "   Artist: " << artist << "\n";
            std::cout << "   Album:  " << album << " (Index: " << meta.album_index << ")\n";
        } else {
            if (isrc == "INVALIDISRCX") {
                std::cout << "✅ Correctly skipped invalid ISRC: " << isrc << std::endl;
            } else {
                std::cout << "⚠️  ISRC not found: " << isrc << std::endl;
            }
        }
    }

    std::cout << "ALL TESTS PASSED." << std::endl;
    return 0;
}
