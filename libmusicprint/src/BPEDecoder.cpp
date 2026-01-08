
#include "BPEDecoder.h"
#include <cstring>
#include <stdexcept>
#include <sstream>

namespace musicprint {

BPEDecoder::BPEDecoder() {}
BPEDecoder::~BPEDecoder() {}

void BPEDecoder::load(const std::string& path) {
    reader_.open(path);

    // 1. Parse Header
#pragma pack(push, 1)
    struct Header {
        char magic[4];
        uint32_t version;
        uint32_t count;
    };
#pragma pack(pop)

    const Header& h = reader_.read<Header>(0);
    
    if (std::strncmp(h.magic, "MPVC", 4) != 0) {
        throw std::runtime_error("BPEDecoder: Invalid magic bytes");
    }
    
    vocabSize_ = h.count;
    
    // 2. Setup Pointers
    // Offsets start immediately after header (12 bytes)
    offsets_ = static_cast<const uint32_t*>(reader_.getPointer(sizeof(Header)));
    
    // Blob starts after the offset table
    // Table size: (count + 1) * 4 bytes (The +1 is for the total length)
    size_t tableSize = (vocabSize_ + 1) * sizeof(uint32_t);
    blob_ = static_cast<const char*>(reader_.getPointer(sizeof(Header) + tableSize));
}

std::string BPEDecoder::decodeToken(uint16_t tokenID) const {
    if (tokenID >= vocabSize_) {
        return ""; // Unknown token
    }

    uint32_t start = offsets_[tokenID];
    uint32_t end = offsets_[tokenID + 1];
    size_t len = end - start;

    return std::string(blob_ + start, len);
}

std::string BPEDecoder::decode(const std::vector<uint16_t>& tokens) const {
    std::stringstream ss;
    for (uint16_t t : tokens) {
        ss << decodeToken(t);
    }
    
    // Simple post-processing: BPE often uses 'Ġ' (U+0120) for spaces
    // In a real app, we might replace specific BPE artifacts, but for now
    // we return the raw reconstruction.
    return ss.str();
}

} // namespace musicprint
