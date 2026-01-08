
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
    // Assuming audio_index.bin is just flat codes (no header for now, or handled externally)
    // Actually, let's assume it's flat 8-byte codes for maximum simplicity.
    codes_ = static_cast<const uint8_t*>(indexReader_.getPointer(0));
    num_vectors_ = indexReader_.getSize() / M_; // 8 bytes per vector

    // 2. Setup Codebook Pointers
    // Codebook format: [M][K][d_sub] floats
    // d_sub = D / M = 768 / 8 = 96
    d_sub_ = D_ / M_;
    centroids_ = static_cast<const float*>(codebookReader_.getPointer(0));
}

std::vector<SearchResult> Searcher::search(const std::vector<float>& query, int k) const {
    if (query.size() != D_) return {};

    // 1. Precompute Distance Table (M x K)
    // Table[m][k] = L2_sq(query_sub[m], centroid[m][k])
    // Size: 8 * 256 = 2048 floats (Small enough for L1 Cache)
    std::vector<float> dist_table(M_ * K_);

    for (size_t m = 0; m < M_; ++m) {
        size_t query_offset = m * d_sub_;
        size_t table_offset = m * K_;
        size_t centroid_base = m * K_ * d_sub_;

        for (size_t k_idx = 0; k_idx < K_; ++k_idx) {
            float dist = 0.0f;
            size_t centroid_offset = centroid_base + (k_idx * d_sub_);
            
            // Compute Euclidean Distance Squared
            for (size_t d = 0; d < d_sub_; ++d) {
                float diff = query[query_offset + d] - centroids_[centroid_offset + d];
                dist += diff * diff;
            }
            dist_table[table_offset + k_idx] = dist;
        }
    }

    // 2. Scan Index (ADC)
    // Priority Queue to keep Top-K (Max-Heap)
    std::priority_queue<std::pair<float, uint32_t>> pq;

    for (uint32_t i = 0; i < num_vectors_; ++i) {
        float dist = 0.0f;
        const uint8_t* code = codes_ + (i * M_);

        // Unrolled loop for M=8
        dist += dist_table[0 * K_ + code[0]];
        dist += dist_table[1 * K_ + code[1]];
        dist += dist_table[2 * K_ + code[2]];
        dist += dist_table[3 * K_ + code[3]];
        dist += dist_table[4 * K_ + code[4]];
        dist += dist_table[5 * K_ + code[5]];
        dist += dist_table[6 * K_ + code[6]];
        dist += dist_table[7 * K_ + code[7]];

        if (pq.size() < k) {
            pq.push({dist, i});
        } else if (dist < pq.top().first) {
            pq.pop();
            pq.push({dist, i});
        }
    }

    // 3. Extract Results (Reverse order because PQ is Max-Heap)
    std::vector<SearchResult> results;
    results.reserve(pq.size());
    while (!pq.empty()) {
        results.push_back({pq.top().second, pq.top().first});
        pq.pop();
    }
    std::reverse(results.begin(), results.end());

    return results;
}

} // namespace musicprint
