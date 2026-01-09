
#pragma once

#include "BinaryReader.h"
#include <string>
#include <vector>
#include <cstdint>

namespace musicprint {

struct SongMetadata {
    uint32_t song_id;
    std::vector<uint16_t> artist_tokens;
    std::vector<uint16_t> album_tokens;
    std::vector<uint16_t> title_tokens;
    uint32_t album_index; // Critical for art lookup
};

class MetadataDatabase {
public:
    MetadataDatabase();
    ~MetadataDatabase();

    void load(const std::string& path);

    // Main lookup: ISRC string -> Metadata tokens
    // Returns true if found, false otherwise
    bool lookup(const std::string& isrc, SongMetadata& out) const;

    uint32_t getSongCount() const { return songCount_; }
    bool isLoaded() const { return metadataReader_.getSize() > 0; }

    // Helper: Unpack uint64 to ISRC string (Static, useful for Searcher results)
    static std::string unpackISRC(uint64_t packed);

private:
    BinaryReader metadataReader_;

    // Header Info
    uint32_t songCount_ = 0;
    uint32_t artistCount_ = 0;
    uint32_t albumCount_ = 0;

    // Section Offsets (pointers)
    uint64_t off_isrc_index_ = 0;
    uint64_t off_artist_ranges_ = 0;
    uint64_t off_album_ranges_ = 0;
    uint64_t off_title_offsets_ = 0;
    uint64_t off_title_blob_ = 0;
    uint64_t off_artist_blob_ = 0;
    uint64_t off_album_blob_ = 0;

    // Helper: Bit-pack ISRC same as Python
    uint64_t packISRC(const std::string& isrc) const;
    
    // Helper: Decode a token list from a specific blob/offset
    std::vector<uint16_t> getTokensFromBlob(uint64_t blobBase, uint32_t nameOffset) const;
};

} // namespace musicprint
