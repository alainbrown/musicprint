
#include "MetadataDatabase.h"
#include <cstring>
#include <algorithm>

namespace musicprint {

MetadataDatabase::MetadataDatabase() {}
MetadataDatabase::~MetadataDatabase() {}

void MetadataDatabase::load(const std::string& path) {
    metadataReader_.open(path);
    
    // 1. Parse Header (Ensuring no padding)
#pragma pack(push, 1)
    struct Header {
        char magic[4];
        uint32_t version;
        uint32_t song_count;
        uint32_t artist_count;
        uint32_t album_count;
        uint64_t offsets[7];
    };
#pragma pack(pop)

    const Header& h = metadataReader_.read<Header>(0);
    
    if (std::strncmp(h.magic, "MPDB", 4) != 0) {
        throw std::runtime_error("MetadataDatabase: Invalid magic bytes");
    }
    if (h.version != 3) {
        throw std::runtime_error("MetadataDatabase: Unsupported version (expected 3)");
    }

    songCount_ = h.song_count;
    artistCount_ = h.artist_count;
    albumCount_ = h.album_count;

    off_isrc_index_ = h.offsets[0];
    off_artist_ranges_ = h.offsets[1];
    off_album_ranges_ = h.offsets[2];
    off_title_offsets_ = h.offsets[3];
    off_title_blob_ = h.offsets[4];
    off_artist_blob_ = h.offsets[5];
    off_album_blob_ = h.offsets[6];
}

std::string MetadataDatabase::unpackISRC(uint64_t packed) {
    uint64_t desig = packed & 0x1FFFF;
    uint64_t year = (packed >> 17) & 0x7F;
    uint64_t reg = (packed >> 24) & 0xFFFF;
    uint64_t country = (packed >> 40) & 0x3FF;

    char c1 = static_cast<char>((country / 26) + 'A');
    char c2 = static_cast<char>((country % 26) + 'A');

    auto i2c = [](uint64_t i) -> char {
        return (i < 26) ? static_cast<char>(i + 'A') : static_cast<char>(i - 26 + '0');
    };

    char r1 = i2c(reg / 1296);
    char r2 = i2c((reg / 36) % 36);
    char r3 = i2c(reg % 36);

    char buf[13];
    snprintf(buf, 13, "%c%c%c%c%c%02llu%05llu", c1, c2, r1, r2, r3, year, desig);
    return std::string(buf);
}

uint64_t MetadataDatabase::packISRC(const std::string& isrc_str) const {
    if (isrc_str.length() != 12) return 0;
    
    try {
        uint64_t c1 = std::toupper(isrc_str[0]) - 'A';
        uint64_t c2 = std::toupper(isrc_str[1]) - 'A';
        uint64_t country = (c1 * 26) + c2;

        auto c2i = [](char c) -> uint64_t {
            c = std::toupper(c);
            if (c >= 'A' && c <= 'Z') return c - 'A';
            if (c >= '0' && c <= '9') return c - '0' + 26;
            return 0;
        };

        uint64_t reg = (c2i(isrc_str[2]) * 36 * 36) + (c2i(isrc_str[3]) * 36) + c2i(isrc_str[4]);
        uint64_t year = std::stoull(isrc_str.substr(5, 2));
        uint64_t desig = std::stoull(isrc_str.substr(7, 5));

        return (country << 40) | (reg << 24) | (year << 17) | desig;
    } catch (...) {
        return 0;
    }
}

bool MetadataDatabase::lookup(const std::string& isrc, SongMetadata& out) const {
    uint64_t target = packISRC(isrc);
    if (target == 0) return false;

    // 1. Binary Search on ISRC Index
    // Entry: uint64_t packed_isrc, uint32_t internal_id (12 bytes)
    const uint8_t* indexPtr = static_cast<const uint8_t*>(metadataReader_.getPointer(off_isrc_index_));
    
    int32_t low = 0;
    int32_t high = songCount_ - 1;
    uint32_t internal_id = 0xFFFFFFFF;

    while (low <= high) {
        int32_t mid = low + (high - low) / 2;
        const uint8_t* entry = indexPtr + (mid * 12);
        uint64_t mid_isrc = *reinterpret_cast<const uint64_t*>(entry);
        
        if (mid_isrc < target) {
            low = mid + 1;
        } else if (mid_isrc > target) {
            high = mid - 1;
        } else {
            internal_id = *reinterpret_cast<const uint32_t*>(entry + 8);
            break;
        }
    }

    if (internal_id == 0xFFFFFFFF) return false;
    out.song_id = internal_id;

    // 2. Resolve Artist (Binary Search in Ranges)
    const uint8_t* artistRangePtr = static_cast<const uint8_t*>(metadataReader_.getPointer(off_artist_ranges_));
    low = 0; high = artistCount_ - 1;
    while (low <= high) {
        int32_t mid = low + (high - low) / 2;
        uint32_t start_id = *reinterpret_cast<const uint32_t*>(artistRangePtr + (mid * 8));
        uint32_t name_off = *reinterpret_cast<const uint32_t*>(artistRangePtr + (mid * 8) + 4);
        
        if (internal_id >= start_id) {
            out.artist_tokens = getTokensFromBlob(off_artist_blob_, name_off);
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    // 3. Resolve Album (Binary Search in Ranges)
    const uint8_t* albumRangePtr = static_cast<const uint8_t*>(metadataReader_.getPointer(off_album_ranges_));
    low = 0; high = albumCount_ - 1;
    while (low <= high) {
        int32_t mid = low + (high - low) / 2;
        uint32_t start_id = *reinterpret_cast<const uint32_t*>(albumRangePtr + (mid * 8));
        uint32_t name_off = *reinterpret_cast<const uint32_t*>(albumRangePtr + (mid * 8) + 4);
        
        if (internal_id >= start_id) {
            out.album_tokens = getTokensFromBlob(off_album_blob_, name_off);
            out.album_index = mid; // This index is used for art.bin lookup!
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }

    // 4. Resolve Title (Direct Offset Lookup)
    const uint32_t* titleOffPtr = static_cast<const uint32_t*>(metadataReader_.getPointer(off_title_offsets_));
    uint32_t start_off = titleOffPtr[internal_id];
    uint32_t next_off = titleOffPtr[internal_id + 1];
    
    size_t title_len_bytes = next_off - start_off;
    size_t token_count = title_len_bytes / sizeof(uint16_t);
    
    out.title_tokens.resize(token_count);
    std::memcpy(out.title_tokens.data(), 
                static_cast<const uint8_t*>(metadataReader_.getPointer(off_title_blob_)) + start_off, 
                title_len_bytes);

    return true;
}

std::vector<uint16_t> MetadataDatabase::getTokensFromBlob(uint64_t blobBase, uint32_t nameOffset) const {
    const uint8_t* ptr = static_cast<const uint8_t*>(metadataReader_.getPointer(blobBase + nameOffset));
    uint8_t len = *ptr; // First byte is length
    
    std::vector<uint16_t> tokens(len);
    std::memcpy(tokens.data(), ptr + 1, len * sizeof(uint16_t));
    return tokens;
}

} // namespace musicprint
