
#include "Searcher.h"
#include <algorithm>
#include <queue>
#include <cstring>
#include <bit> // For std::popcount in C++20, or use builtin

namespace musicprint {

Searcher::Searcher() {}
Searcher::~Searcher() {}

void Searcher::load(const std::string& index_path) {
    indexReader_.open(index_path);

    // 1. Setup Index Pointers
    // Format: [Header (64 bytes)][Entries]
    // Entry: [8 bytes Hash][8 bytes Packed ISRC] = 16 bytes
    size_t header_size = 64;
    size_t entry_size = 16;
    
    if (indexReader_.getSize() < header_size) {
        num_vectors_ = 0;
        index_data_ = nullptr;
    } else {
        index_data_ = static_cast<const uint8_t*>(indexReader_.getPointer(header_size));
        num_vectors_ = (indexReader_.getSize() - header_size) / entry_size;
    }
}

std::vector<SearchResult> Searcher::search(uint64_t query_hash, int k) const {
    // Priority Queue to keep Top-K (Max-Heap)
    // We store <Distance, IndexInFile>
    std::priority_queue<std::pair<uint32_t, uint32_t>> pq;
    size_t entry_size = 16;

    for (uint32_t i = 0; i < num_vectors_; ++i) {
        // Read stored hash (first 8 bytes of entry)
        const uint8_t* entry = index_data_ + (i * entry_size);
        uint64_t stored_hash = *reinterpret_cast<const uint64_t*>(entry);

        // Hamming Distance
        uint64_t xor_val = stored_hash ^ query_hash;
        
        // Use builtin for speed (GCC/Clang)
        uint32_t dist = __builtin_popcountll(xor_val);

        if (pq.size() < (size_t)k) {
            pq.push({dist, i});
        } else if (dist < pq.top().first) {
            pq.pop();
            pq.push({dist, i});
        }
    }

    // Extract Results
    std::vector<SearchResult> results;
    results.reserve(pq.size());
    while (!pq.empty()) {
        uint32_t idx = pq.top().second;
        uint32_t dist = pq.top().first;
        
        // Read ISRC from the entry (Offset 8)
        const uint8_t* entry = index_data_ + (idx * entry_size);
        uint64_t packed_isrc = *reinterpret_cast<const uint64_t*>(entry + 8);
        
        results.push_back({packed_isrc, dist});
        pq.pop();
    }
    std::reverse(results.begin(), results.end());

    return results;
}

} // namespace musicprint
