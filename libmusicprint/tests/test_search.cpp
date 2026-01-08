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
const int D = 768; 
const int d_sub = D / M; // 96
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
    std::vector<uint8_t> index(N * M);
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = rand() % K;
    }
    
    // 3. Inject Target at ID 42
    // We set its code to [0, 0, 0, 0, 0, 0, 0, 0]
    for (int m = 0; m < M; ++m) {
        index[42 * M + m] = 0;
    }

    std::ofstream idx("test_index.bin", std::ios::binary);
    idx.write(reinterpret_cast<const char*>(index.data()), index.size() * sizeof(uint8_t));
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