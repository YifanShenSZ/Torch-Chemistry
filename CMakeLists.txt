cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# libtorch
set(CMAKE_PREFIX_PATH ~/Software/Programming/libtorch-cuda10.1-1.7.1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(tchem STATIC
    source/utility.cpp
    source/LinearAlgebra.cpp
    source/TorchSupport/utility.cpp
    source/TorchSupport/LinearAlgebra.cpp
    source/TorchSupport/InternalCoordinate.cpp
    source/TorchSupport/chemistry.cpp
)

if (CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -qopenmp -static-intel -parallel -ipo")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")
endif ()

target_link_libraries(tchem "${TORCH_LIBRARIES}")