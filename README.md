# Torch-Chemistry
Chemical routines based on libtorch

## Featured utilities
Internal coordinate:
* Cartesian -> internal coordinate
* Jacobian of internal coordinate over Cartesian coordinate

Chemistry:
* Alternative representation near electronic degeneracy
* Phase fixing

Linear algebra:
* Triple product
* 3rd-order tensor operation

Utility:
* Some general basic routine

## Installation
1. `mkdir build lib`
2. `cd build`
3. `cmake ..`
4. `cmake --build .`
5. `mv lib* ../lib`
6. add `include` and `lib` to your path

## Usage
`#include <tchem/tchem.hpp>`

## Dependency
1. libtorch
2. My Cpp-Library