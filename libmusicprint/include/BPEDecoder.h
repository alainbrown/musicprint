
#pragma once

#include "BinaryReader.h"
#include <string>
#include <vector>
#include <cstdint>

namespace musicprint {

class BPEDecoder {
public:
    BPEDecoder();
    ~BPEDecoder();

    void load(const std::string& path);

    // Decode a single token ID to its string representation
    std::string decodeToken(uint16_t tokenID) const;

    // Decode a sequence of tokens into a full string
    std::string decode(const std::vector<uint16_t>& tokens) const;

    bool isLoaded() const { return reader_.getSize() > 0; }

private:
    BinaryReader reader_;
    uint32_t vocabSize_ = 0;
    
    // Pointers into the mmap
    const uint32_t* offsets_ = nullptr;
    const char* blob_ = nullptr;
};

} // namespace musicprint
