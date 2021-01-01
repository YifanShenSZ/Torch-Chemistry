# Torch-Chemistry
Chemical routines based on libtorch

## Featured utilities
Gaussian:
* Merge of gaussian functions
* Gaussian integrals

Internal coordinate:
* Cartesian -> internal coordinate
* Jacobian of internal coordinate over Cartesian coordinate

Chemistry:
* Alternative representation near electronic degeneracy
* Phase fixing

Utility:
* Some general basic routine

Polynomial:
* Polynomial operation

Linear algebra:
* Triple product
* 3rd-order tensor operation

## Installation
1. `mkdir build lib`
2. `cd build`
3. `cmake ..`
4. `cmake --build .`
5. `mv lib* ../lib`
6. add `include` and `lib` to your path

## Usage
`#include <tchem/tchem.hpp>`

## Source
Source code level from bottom to top:
1. utility, polynomial, linalg
2. gaussian, intcoord, chemistry

## Dependency
1. libtorch
2. My Cpp-Library