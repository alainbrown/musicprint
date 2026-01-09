
#include "Searcher.h"
#include "MetadataDatabase.h"
#include "BPEDecoder.h"
#include <iostream>
#include <vector>
#include <fstream>

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <query.bin> <index.bin> <centroids.bin> <meta.bin> <vocab.bin>" << std::endl;
        return 1;
    }

    const char* query_path = argv[1];
    const char* index_path = argv[2];
    const char* centroids_path = argv[3];
    const char* meta_path = argv[4];
    const char* vocab_path = argv[5];

    try {
        // 1. Load Everything
        musicprint::Searcher searcher;
        musicprint::MetadataDatabase metaDB;
        musicprint::BPEDecoder decoder;

        searcher.load(index_path, centroids_path);
        metaDB.load(meta_path);
        decoder.load(vocab_path);

        // 2. Load Query Vector (64 floats)
        // Note: D=64 matches MERT-v1-95M-Adapter output
        size_t D = 64; 
        std::ifstream fq(query_path, std::ios::binary);
        if (!fq) throw std::runtime_error("Could not open query file");
        
        std::vector<float> query(D);
        fq.read(reinterpret_cast<char*>(query.data()), D * sizeof(float));

        // 3. Search
        auto results = searcher.search(query, 1);
        if (results.empty()) {
            std::cout << "NO_MATCH" << std::endl;
            return 0;
        }

        // 4. Resolve Metadata
        std::string isrc = musicprint::MetadataDatabase::unpackISRC(results[0].song_id);
        musicprint::SongMetadata meta;
        if (!metaDB.lookup(isrc, meta)) {
            std::cout << "MATCH_FOUND_BUT_METADATA_MISSING: " << isrc << std::endl;
            return 0;
        }

        // 5. Decode and Print
        std::string title = decoder.decode(meta.title_tokens);
        std::string artist = decoder.decode(meta.artist_tokens);
        
        std::cout << "MATCH: " << title << " by " << artist << " [" << isrc << "]" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
