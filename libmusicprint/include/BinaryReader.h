
#pragma once

#include <string>
#include <vector>
#include <stdexcept>

namespace musicprint {

class BinaryReader {
public:
    BinaryReader();
    ~BinaryReader();

    // Prevent copying to avoid double-unmap
    BinaryReader(const BinaryReader&) = delete;
    BinaryReader& operator=(const BinaryReader&) = delete;

    void open(const std::string& path);
    void close();

    // Read a specific type at offset
    template <typename T>
    const T& read(size_t offset) const {
        if (!data_ || offset + sizeof(T) > size_) {
            throw std::out_of_range("BinaryReader: Read out of bounds");
        }
        return *reinterpret_cast<const T*>(static_cast<const uint8_t*>(data_) + offset);
    }

    // Get raw pointer (for bulk reading)
    const void* getPointer(size_t offset) const;
    size_t getSize() const { return size_; }

private:
    void* data_ = nullptr;
    size_t size_ = 0;
    int fd_ = -1;
};

} // namespace musicprint
