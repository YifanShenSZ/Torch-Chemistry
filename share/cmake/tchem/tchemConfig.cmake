# Findtchem
# -------
#
# Finds Torch-Chemistry (tchem)
#
# This will define the following variables:
#
#   tchem_FOUND        -- True if the system has tchem
#   tchem_INCLUDE_DIRS -- The include directories for tchem
#   tchem_LIBRARIES    -- Libraries to link against
#   tchem_CXX_FLAGS    -- Additional (required) compiler flags
#
# and the following imported targets:
#
#   tchem

# Find tchem root
# Assume we are in ${tchemROOT}/share/cmake/tchem/tchemConfig.cmake
get_filename_component(CMAKE_CURRENT_LIST_DIR "${CMAKE_CURRENT_LIST_FILE}" PATH)
get_filename_component(tchemROOT "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

# include directory
set(tchem_INCLUDE_DIRS ${tchemROOT}/include)

# library
add_library(tchem STATIC IMPORTED)
set(tchem_LIBRARIES tchem)

# dependency 1: libtorch
if(NOT TORCH_FOUND)
    find_package(Torch REQUIRED PATHS ~/Software/Programming/libtorch-cuda10.1-1.7.1) 
    list(APPEND tchem_INCLUDE_DIRS ${TORCH_INCLUDE_DIRS})
    list(APPEND tchem_LIBRARIES ${TORCH_LIBRARIES})
    set(tchem_CXX_FLAGS "${TORCH_CXX_FLAGS}")
endif()

# dependency 2: Cpp-Library
if(NOT CL_FOUND)
    find_package(CL REQUIRED PATHS ~/Library/Cpp-Library)
    list(APPEND tchem_INCLUDE_DIRS ${CL_INCLUDE_DIRS})
    list(APPEND tchem_LIBRARIES ${CL_LIBRARIES})
endif()

# import location
find_library(tchem_LIBRARY tchem PATHS "${tchemROOT}/lib")
set_target_properties(tchem PROPERTIES
    IMPORTED_LOCATION "${tchem_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${tchem_INCLUDE_DIRS}"
    CXX_STANDARD 14
)