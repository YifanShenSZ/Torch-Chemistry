cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_language(Fortran)

set(CMAKE_BUILD_TYPE Release)

# libtorch
set(CMAKE_PREFIX_PATH ~/Software/Programming/libtorch-cuda10.1-1.7.1)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Cpp-Library
set(CMAKE_PREFIX_PATH ~/Library/Cpp-Library)
find_package(CL REQUIRED)

include_directories(include)

add_library(tchem STATIC
    source/FORTRAN/My_dsygv.f90
    source/utility.cpp
    source/linalg.cpp
    source/polynomial.cpp
    source/SApolynomial.cpp
    source/gaussian.cpp
    source/intcoord.cpp
    source/SASintcoord.cpp
    source/chem/chemistry.cpp
    source/chem/phaser.cpp
    source/chem/normal_mode.cpp
)

target_link_libraries(tchem
    ifcore iomp5 pthread m dl mkl_intel_lp64 mkl_intel_thread mkl_core
    "${CL_LIBRARIES}" "${TORCH_LIBRARIES}"
)