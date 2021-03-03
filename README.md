# Torch-Chemistry
Chemical routines based on libtorch

## Featured utilities
Internal coordinate:
* Cartesian -> internal coordinate
* Jacobian of internal coordinate over Cartesian coordinate
* Cartesian coordinate gradient <-> internal coordinate gradient
* Cartesian coordinate Hessian <-> internal coordinate Hessian
* For more details see `intcoord.md`
* For symmetry adaptation and scale see `SASintcoord.md`

Gaussian:
* Merge gaussian functions
* Gaussian integrals

Chemistry:
* Alternative representation near electronic degeneracy
* Vibration analysis

Utility:
* Some general basic routine

Linear algebra:
* Triple product
* 3rd-order tensor operation

Polynomial:
* Polynomial operation
* symmetry adaptation

Phaser:
* Phase fixing

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
1. utility, linalg, polynomial, SApolynomial, phaser
2. intcoord, SASintcoord, gaussian, chemistry

## Dependency
1. libtorch
2. My Cpp-Library
