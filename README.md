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

## Usage
Include headers
1. Add `include` to your include path
2. `#include <tchem/tchem.hpp>`

Compile with `CMake`
1. In your app main directory, create a symbolic link to here
2. `add_subdirectory(Torch-Chemistry)` in your `CMakeLists.txt`
3. A static library named `tchem` will be built

## Dependency
libtorch
