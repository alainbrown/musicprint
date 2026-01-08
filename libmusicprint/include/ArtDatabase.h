
#pragma once

#include "BinaryReader.h"
#include <vector>
#include <memory>

namespace musicprint {

class ArtDatabase {
public:
    ArtDatabase();
    ~ArtDatabase();

    void load(const std::string& artBinPath, const std::string& codebookPath);

    // Get raw tokens (for debugging or manual lookup)
    // Returns 256 uint16_t values
    std::vector<uint16_t> getTokens(uint32_t albumIndex) const;

    // Get fully resolved vectors ready for CoreML
    // Returns 256 * 64 floats (16x16x64)
    std::vector<float> getVectors(uint32_t albumIndex) const;

    bool isLoaded() const { return artReader_.getSize() > 0; }

private:
    BinaryReader artReader_;
    BinaryReader codebookReader_;
    
    // Cache codebook dimensions
    size_t codebookSize_ = 0; // e.g. 1024
    size_t vectorDim_ = 64;   // Fixed for this model
    const float* codebookData_ = nullptr;
};

} // namespace musicprint
