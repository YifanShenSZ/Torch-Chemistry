#ifndef tchem_chemistry_hpp
#define tchem_chemistry_hpp

#include <torch/torch.h>

namespace tchem { namespace chemistry {

bool check_degeneracy(const double & threshold, const at::Tensor & energy);

// Transform adiabatic energy (H) and gradient (dH) to composite representation
void composite_representation(at::Tensor & H, at::Tensor & dH);

// Matrix off-diagonal elements do not have determinate phase, because
// the eigenvectors defining a representation have indeterminate phase difference
void initialize_phase_fixing(const size_t & NStates_);
// Fix M by minimizing || M - ref ||_F^2
void fix(at::Tensor & M, const at::Tensor & ref);
// Fix M1 and M2 by minimizing weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
void fix(at::Tensor & M1, at::Tensor & M2, const at::Tensor & ref1, const at::Tensor & ref2, const double & weight);

} // namespace chemistry
} // namespace tchem

#endif