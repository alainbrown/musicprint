
#include "ArtDatabase.h"
#include <cstring>
#include <stdexcept>

namespace musicprint {

// Constants from VQ-VAE Architecture
static const size_t TOKENS_PER_ALBUM = 256; // 16x16
static const size_t BYTES_PER_ALBUM = TOKENS_PER_ALBUM * sizeof(uint16_t); // 512 bytes
static const size_t VECTOR_DIM = 64;

ArtDatabase::ArtDatabase() {}
ArtDatabase::~ArtDatabase() {}

void ArtDatabase::load(const std::string& artBinPath, const std::string& codebookPath) {
    artReader_.open(artBinPath);
    codebookReader_.open(codebookPath);
    
    codebookData_ = static_cast<const float*>(codebookReader_.getPointer(0));
    codebookSize_ = codebookReader_.getSize() / (VECTOR_DIM * sizeof(float));
}

std::vector<uint16_t> ArtDatabase::getTokens(uint32_t albumIndex) const {
    size_t offset = albumIndex * BYTES_PER_ALBUM;
    
    // Bounds check
    if (offset + BYTES_PER_ALBUM > artReader_.getSize()) {
        return {}; // Return empty if out of bounds
    }
    
    std::vector<uint16_t> tokens(TOKENS_PER_ALBUM);
    const uint16_t* ptr = static_cast<const uint16_t*>(artReader_.getPointer(offset));
    
    // Check for "Missing Art" placeholder (0xFFFF)
    if (ptr[0] == 0xFFFF) {
        return {}; // Return empty to signal missing
    }
    
    std::memcpy(tokens.data(), ptr, BYTES_PER_ALBUM);
    return tokens;
}

std::vector<float> ArtDatabase::getVectors(uint32_t albumIndex) const {
    auto tokens = getTokens(albumIndex);
    if (tokens.empty()) return {}; // Missing or OOB
    
    // Result: 256 vectors * 64 floats
    std::vector<float> result(TOKENS_PER_ALBUM * VECTOR_DIM);
    
    // Gather Loop: O(256)
    // For each token, copy 64 floats from codebook to result
    for (size_t i = 0; i < TOKENS_PER_ALBUM; ++i) {
        uint16_t tokenID = tokens[i];
        
        if (tokenID >= codebookSize_) {
            // Safety: If token ID is invalid, fill with zeros
            std::memset(&result[i * VECTOR_DIM], 0, VECTOR_DIM * sizeof(float));
            continue;
        }
        
        const float* src = codebookData_ + (tokenID * VECTOR_DIM);
        float* dst = &result[i * VECTOR_DIM];
        std::memcpy(dst, src, VECTOR_DIM * sizeof(float));
    }
    
    return result;
}

} // namespace musicprint
