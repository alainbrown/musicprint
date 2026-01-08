cmake_minimum_required(VERSION 3.10)
project(libmusicprint)

set(CMAKE_CXX_STANDARD 17)

# Include Directories
include_directories(include)

# Source Files
file(GLOB SOURCES "src/*.cpp")

# Create Library (Static for iOS)
add_library(musicprint_core STATIC ${SOURCES})

# Tests (Optional, enabled if not on iOS)
if(NOT IOS)
    enable_testing()
    
    add_executable(test_art tests/test_art.cpp)
    target_link_libraries(test_art musicprint_core)

    add_executable(test_meta tests/test_meta.cpp)
    target_link_libraries(test_meta musicprint_core)

    add_executable(test_search tests/test_search.cpp)
    target_link_libraries(test_search musicprint_core)
endif()
