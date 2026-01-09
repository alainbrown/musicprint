
#pragma once

#include "BinaryReader.h"
#include <vector>
#include <cstdint>

namespace musicprint {

struct SearchResult {
    uint64_t song_id; // Packed ISRC
    float distance;
};

class Searcher {
public:
    Searcher();
    ~Searcher();

    void load(const std::string& index_path, const std::string& pq_codebook_path);

    // Search for the top-k nearest neighbors
    // query_vector: Must match the dimension (e.g., 768)
    std::vector<SearchResult> search(const std::vector<float>& query_vector, int k = 1) const;

    bool isLoaded() const { return indexReader_.getSize() > 0; }

private:
    BinaryReader indexReader_;
    BinaryReader codebookReader_;

    // PQ Parameters
    size_t M_ = 8;       // Number of sub-quantizers
    size_t K_ = 256;     // Number of centroids per sub-quantizer (8-bit)
    size_t D_ = 64;      // Input dimension (Matches MERTAdapter output)
    size_t d_sub_ = 0;   // Dimension per sub-quantizer (D / M)

    // Pointers
    const uint8_t* codes_ = nullptr; // The massive list of 8-byte codes
    const float* centroids_ = nullptr; // The PQ Codebook (M x K x d_sub)
    uint32_t num_vectors_ = 0;
};

} // namespace musicprint
