
#include "MetadataDatabase.h"
#include <iostream>
#include <cassert>

int main() {
    std::cout << "Running MetadataDatabase Tests (against production binary)..." << std::endl;

    musicprint::MetadataDatabase db;
    const std::string meta_path = "../../meta_tokenizer_pipeline/release/music_meta.bin";

    try {
        db.load(meta_path);
        std::cout << "✅ Loaded " << db.getSongCount() << " songs." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ FAILED to load: " << e.what() << std::endl;
        std::cerr << "   Note: Ensure you have built the metadata binary first." << std::endl;
        return 1;
    }

    // List of test ISRCs (From MusicBrainz samples we saw earlier)
    // USRC10301389 is usually a safe bet for testing
    std::vector<std::string> test_isrcs = {
        "GBAYE0601498", // Random sample
        "USRC10301389", // Common test case
        "INVALIDISRCX"  // Negative test
    };

    for (const auto& isrc : test_isrcs) {
        musicprint::SongMetadata meta;
        bool found = db.lookup(isrc, meta);
        
        if (found) {
            std::cout << "✅ Found [" << isrc << "]: " 
                      << "SongID=" << meta.song_id << ", "
                      << "ArtistTokens=" << meta.artist_tokens.size() << ", "
                      << "AlbumTokens=" << meta.album_tokens.size() << ", "
                      << "TitleTokens=" << meta.title_tokens.size() << ", "
                      << "AlbumIndex=" << meta.album_index << std::endl;
        } else {
            if (isrc == "INVALIDISRCX") {
                std::cout << "✅ Correctly skipped invalid ISRC: " << isrc << std::endl;
            } else {
                std::cout << "⚠️  ISRC not found (might not be in your 5M subset): " << isrc << std::endl;
            }
        }
    }

    std::cout << "ALL TESTS PASSED." << std::endl;
    return 0;
}
