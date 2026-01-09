#include "Searcher.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstdlib>

// Constants matching production
const int M = 8;
const int K = 256;
const int D = 64; 
const int d_sub = D / M; // 8
const int N = 1000; // 1000 vectors in database

void create_synthetic_search_data() {
    // 1. Generate Random Centroids (M x K x d_sub)
    std::vector<float> centroids(M * K * d_sub);
    for (size_t i = 0; i < centroids.size(); ++i) {
        centroids[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    std::ofstream cb("test_centroids.bin", std::ios::binary);
    cb.write(reinterpret_cast<const char*>(centroids.data()), centroids.size() * sizeof(float));
    cb.close();

    // 2. Generate Index (N vectors)
    // Structure: [Header 64 bytes] [Entry 0] [Entry 1] ...
    // Entry: [8 bytes Code] [8 bytes ISRC]
    
    std::ofstream idx("test_index.bin", std::ios::binary);
    
    // Header
    char magic[4] = {'M', 'P', 'A', 'F'};
    uint32_t version = 1;
    uint32_t count = N;
    idx.write(magic, 4);
    idx.write(reinterpret_cast<const char*>(&version), 4);
    idx.write(reinterpret_cast<const char*>(&count), 4);
    
    // Padding (52 bytes)
    char padding[52] = {0};
    idx.write(padding, 52);

    for (uint32_t i = 0; i < N; ++i) {
        // Generate Random Code
        uint8_t code[8];
        for (int m = 0; m < 8; ++m) code[m] = rand() % K;
        
        // Inject Target
        if (i == 42) {
            for (int m = 0; m < 8; ++m) code[m] = 0; // Matches Centroid 0
        }
        
        // Write Code (8 bytes)
        idx.write(reinterpret_cast<const char*>(code), 8);
        
        // Write ISRC (8 bytes) - Using index as dummy ISRC
        uint64_t dummy_isrc = i;
        idx.write(reinterpret_cast<const char*>(&dummy_isrc), 8);
    }
    
    idx.close();
}

int main() {
    std::cout << "Running Searcher Tests (Synthetic)..." << std::endl;
    create_synthetic_search_data();

    musicprint::Searcher searcher;
    searcher.load("test_index.bin", "test_centroids.bin");

    // 4. Create Query Vector
    // We construct a query that matches Centroid 0 exactly.
    // The Searcher pre-computes distances. 
    // Distance(Query, Centroid0) should be 0.0
    // Distance(Query, Others) should be > 0.0
    // So ID 42 (which uses Centroid 0) should have total distance 0.0.
    
    // Load centroids back to construct query
    std::ifstream cb("test_centroids.bin", std::ios::binary);
    std::vector<float> centroids(M * K * d_sub);
    cb.read(reinterpret_cast<char*>(centroids.data()), centroids.size() * sizeof(float));
    
    std::vector<float> query(D);
    for (int m = 0; m < M; ++m) {
        // Copy Centroid[m][0] into query
        size_t centroid_base = m * K * d_sub; // index 0 is at start
        for (int d = 0; d < d_sub; ++d) {
            query[m * d_sub + d] = centroids[centroid_base + d];
        }
    }

    // 5. Search
    auto results = searcher.search(query, 5); // Top 5

    // 6. Verify
    std::cout << "Top Match ID: " << results[0].song_id << " (Dist: " << results[0].distance << ")" << std::endl;
    
    assert(results[0].song_id == 42);
    assert(std::abs(results[0].distance) < 0.001f); // Should be effectively 0

    std::cout << "✅ Searcher correctly identified the injected target." << std::endl;
    std::cout << "ALL TESTS PASSED." << std::endl;
    return 0;
}