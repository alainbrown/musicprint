
#pragma once

#include "BinaryReader.h"
#include <vector>
#include <cstdint>
#include <string>

namespace musicprint {

struct SearchResult {
    uint64_t song_id; // Packed ISRC
    uint32_t distance; // Hamming Distance
};

class Searcher {
public:
    Searcher();
    ~Searcher();

    void load(const std::string& index_path);

    // Search for the top-k nearest neighbors using Hamming Distance
    std::vector<SearchResult> search(uint64_t query_hash, int k = 1) const;

    bool isLoaded() const { return indexReader_.getSize() > 0; }

private:
    BinaryReader indexReader_;

    // Pointers
    const uint8_t* index_data_ = nullptr; // Pointer to start of entries
    uint32_t num_vectors_ = 0;
};

} // namespace musicprint
