
#include "Searcher.h"
#include <cmath>
#include <algorithm>
#include <queue>
#include <cstring>

namespace musicprint {

Searcher::Searcher() {}
Searcher::~Searcher() {}

void Searcher::load(const std::string& index_path, const std::string& pq_codebook_path) {
    indexReader_.open(index_path);
    codebookReader_.open(pq_codebook_path);

    // 1. Setup Index Pointers
    // Format: [Header (64 bytes)][Entries]
    // Entry: [8 bytes PQ Code][8 bytes Packed ISRC] = 16 bytes
    // We skip the header (64 bytes)
    size_t header_size = 64;
    size_t entry_size = 16;
    
    if (indexReader_.getSize() < header_size) {
        num_vectors_ = 0;
        codes_ = nullptr;
    } else {
        codes_ = static_cast<const uint8_t*>(indexReader_.getPointer(header_size));
        num_vectors_ = (indexReader_.getSize() - header_size) / entry_size;
    }

    // 2. Setup Codebook Pointers
    // Codebook format: [M][K][d_sub] floats
    d_sub_ = D_ / M_;
    centroids_ = static_cast<const float*>(codebookReader_.getPointer(0));
}

std::vector<SearchResult> Searcher::search(const std::vector<float>& query, int k) const {
    if (query.size() != D_) return {};

    // 1. Precompute Distance Table (M x K)
    std::vector<float> dist_table(M_ * K_);

    for (size_t m = 0; m < M_; ++m) {
        size_t query_offset = m * d_sub_;
        size_t table_offset = m * K_;
        size_t centroid_base = m * K_ * d_sub_;

        for (size_t k_idx = 0; k_idx < K_; ++k_idx) {
            float dist = 0.0f;
            size_t centroid_offset = centroid_base + (k_idx * d_sub_);
            for (size_t d = 0; d < d_sub_; ++d) {
                float diff = query[query_offset + d] - centroids_[centroid_offset + d];
                dist += diff * diff;
            }
            dist_table[table_offset + k_idx] = dist;
        }
    }

    // 2. Scan Index (ADC)
    // Priority Queue to keep Top-K (Max-Heap)
    // We store <Distance, IndexInFile> temporarily, then resolve ISRC later
    std::priority_queue<std::pair<float, uint32_t>> pq;
    size_t entry_size = 16;

    for (uint32_t i = 0; i < num_vectors_; ++i) {
        float dist = 0.0f;
        const uint8_t* entry = codes_ + (i * entry_size);
        
        // Unrolled loop for M=8
        dist += dist_table[0 * K_ + entry[0]];
        dist += dist_table[1 * K_ + entry[1]];
        dist += dist_table[2 * K_ + entry[2]];
        dist += dist_table[3 * K_ + entry[3]];
        dist += dist_table[4 * K_ + entry[4]];
        dist += dist_table[5 * K_ + entry[5]];
        dist += dist_table[6 * K_ + entry[6]];
        dist += dist_table[7 * K_ + entry[7]];

        if (pq.size() < k) {
            pq.push({dist, i});
        } else if (dist < pq.top().first) {
            pq.pop();
            pq.push({dist, i});
        }
    }

    // 3. Extract Results
    std::vector<SearchResult> results;
    results.reserve(pq.size());
    while (!pq.empty()) {
        uint32_t idx = pq.top().second;
        float dist = pq.top().first;
        
        // Read ISRC from the entry (Offset 8)
        const uint8_t* entry = codes_ + (idx * entry_size);
        uint64_t packed_isrc = *reinterpret_cast<const uint64_t*>(entry + 8);
        
        results.push_back({packed_isrc, dist});
        pq.pop();
    }
    std::reverse(results.begin(), results.end());

    return results;
}

} // namespace musicprint
