#ifndef tchem_chemistry_hpp
#define tchem_chemistry_hpp

#include <torch/torch.h>

#include <tchem/chem/phaser.hpp>
#include <tchem/chem/normal_mode.hpp>

namespace tchem { namespace chem {

// Check if two energy levels are closer than the threshold
bool check_degeneracy(const at::Tensor & energy, const double & thresh);

// Transform Hamiltonian (or energy) and gradient to composite representation
// which is defined by diagonalizing tchem::linalg::sy3matdotmul(dH, dH)
// Return composite Hamiltonian and gradient
// Only read the "upper triangle" (i <= j) of H and dH
// Only write the "upper triangle" (i <= j) of the output tensor
std::tuple<at::Tensor, at::Tensor> composite_representation(const at::Tensor & H, const at::Tensor & dH);
// Only read/write the "upper triangle" (i <= j) of H and dH
void composite_representation_(at::Tensor & H, at::Tensor & dH);
// dot product defined with a metric S
// S must be real symmetric positive definite
// Warning: All elements of S will be read because of torch::mv
//          Will fix it some day if pytorch introduces "symv" like BLAS
std::tuple<at::Tensor, at::Tensor> composite_representation(const at::Tensor & H, const at::Tensor & dH, const at::Tensor & S);
void composite_representation_(at::Tensor & H, at::Tensor & dH, const at::Tensor & S);

} // namespace chem
} // namespace tchem

#endif