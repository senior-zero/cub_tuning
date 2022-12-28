# 3.18.3 fixed bugs in MSVC + CUDA + C++17 support
cmake_minimum_required(VERSION 3.18.3)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CXX needed for fmt
project(TBM CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/")

include(CPM)

CPMAddPackage(
  NAME fmt
  GITHUB_REPOSITORY fmtlib/fmt
  GIT_TAG 7.1.3
  GIT_SHALLOW TRUE
  OPTIONS
    # Force static to keep fmt internal.
    "BUILD_SHARED_LIBS OFF"
    "CMAKE_POSITION_INDEPENDENT_CODE ON"
)

CPMAddPackage("gh:NVIDIA/nvbench#main")

include(cmake/TbmUtilities.cmake)

# Where to put build outputs. Use CMAKE_BINARY_DIR so they'll show up in the
# top-level project's dir when building Thrust via add_subdirectory.
set(TBM_LIBRARY_OUTPUT_DIR "${CMAKE_BINARY_DIR}/lib")
set(TBM_EXECUTABLE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/bin")

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(ThrustCUDA)  
set(THRUST_TARGETS ThrustCUDA)   

add_subdirectory(benches)
