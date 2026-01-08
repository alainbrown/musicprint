
#include "BinaryReader.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdexcept>

namespace musicprint {

BinaryReader::BinaryReader() {}

BinaryReader::~BinaryReader() {
    close();
}

void BinaryReader::open(const std::string& path) {
    // 1. Open File
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ == -1) {
        throw std::runtime_error("BinaryReader: Could not open file: " + path);
    }

    // 2. Get Size
    struct stat sb;
    if (fstat(fd_, &sb) == -1) {
        ::close(fd_);
        throw std::runtime_error("BinaryReader: Could not stat file");
    }
    size_ = sb.st_size;

    // 3. Memory Map (Read-Only, Shared)
    data_ = mmap(NULL, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (data_ == MAP_FAILED) {
        ::close(fd_);
        throw std::runtime_error("BinaryReader: mmap failed");
    }
}

void BinaryReader::close() {
    if (data_) {
        munmap(data_, size_);
        data_ = nullptr;
    }
    if (fd_ != -1) {
        ::close(fd_);
        fd_ = -1;
    }
    size_ = 0;
}

const void* BinaryReader::getPointer(size_t offset) const {
    if (!data_ || offset >= size_) {
        return nullptr;
    }
    return static_cast<const uint8_t*>(data_) + offset;
}

} // namespace musicprint
