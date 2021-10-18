# Torch-Chemistry
Chemical routines based on libtorch

We develope this library based on the LTS pytorch build 1.8.2, so using Torch-Chemistry along with libtorch 1.8.2 is recommended

## Featured utilities
Gaussian:
* value of gaussian functions
* product of gaussian functions
* gaussian integrals
* multivariate gaussian random number

Polynomial:
* value of polynomials
* Jacobian of polynomials
* transformation of polynomials under coordinate rotation
* transformation of polynomials under coordinate trasnlation
* symmetry adaptation

Chemistry:
* alternative representation near electronic degeneracy
* phase fixing for quantum observable matrices
* normal mode analysis

Internal coordinate:
* Cartesian -> internal coordinate
* Jacobian of internal coordinate over Cartesian coordinate
* 2nd order Jacobian of internal coordinate over Cartesian coordinate
* Cartesian coordinate gradient <-> internal coordinate gradient
* Cartesian coordinate Hessian <-> internal coordinate Hessian
* for more details see `intcoord.md`
* for symmetry adaptation and scale see `SASintcoord.md`

Linear algebra:
* outer product for general tensors
* map a vector to a symmetric tensor
* matrix dot multiplication between 3rd-order tensors
* matrix dot multiplication between 4-th and 3rd-order tensors
* matrix outer multiplication for general tensors
* unitary transformation

FORTRAN:
* generalized eigenvalue problem solver (dsygv)

Utility:
* some general basic routines

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
2. [Cpp-Library](https://github.com/YifanShenSZ/Cpp-Library)
