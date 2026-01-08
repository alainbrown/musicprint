
#include "ArtDatabase.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <vector>

void create_dummy_files() {
    // 1. Create Codebook (2 vectors, dim 64)
    // Vector 0: All 1.0s
    // Vector 1: All 2.0s
    std::vector<float> codebook(2 * 64);
    for(int i=0; i<64; ++i) codebook[i] = 1.0f;
    for(int i=64; i<128; ++i) codebook[i] = 2.0f;
    
    std::ofstream cb("test_codebook.bin", std::ios::binary);
    cb.write(reinterpret_cast<const char*>(codebook.data()), codebook.size() * sizeof(float));
    cb.close();

    // 2. Create Art Index (2 Albums)
    // Album 0: All Token 0
    // Album 1: All Token 1
    std::vector<uint16_t> art(2 * 256);
    for(int i=0; i<256; ++i) art[i] = 0;
    for(int i=256; i<512; ++i) art[i] = 1;
    
    std::ofstream ab("test_art.bin", std::ios::binary);
    ab.write(reinterpret_cast<const char*>(art.data()), art.size() * sizeof(uint16_t));
    ab.close();
}

int main() {
    std::cout << "Running ArtDatabase Tests..." << std::endl;
    create_dummy_files();

    musicprint::ArtDatabase db;
    db.load("test_art.bin", "test_codebook.bin");

    // Test 1: Album 0 (Token 0 -> 1.0f)
    auto vec0 = db.getVectors(0);
    assert(vec0.size() == 256 * 64);
    assert(vec0[0] == 1.0f);
    assert(vec0[63] == 1.0f);
    assert(vec0[64] == 1.0f); // Next token is also 0
    std::cout << "✅ Test 1 Passed (Album 0 lookup)" << std::endl;

    // Test 2: Album 1 (Token 1 -> 2.0f)
    auto vec1 = db.getVectors(1);
    assert(vec1[0] == 2.0f);
    std::cout << "✅ Test 2 Passed (Album 1 lookup)" << std::endl;

    // Test 3: OOB
    auto vecOOB = db.getVectors(99);
    assert(vecOOB.empty());
    std::cout << "✅ Test 3 Passed (Out of bounds)" << std::endl;

    std::cout << "ALL TESTS PASSED." << std::endl;
    return 0;
}
