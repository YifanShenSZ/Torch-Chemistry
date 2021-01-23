/*
Additional linear algebra for libtorch tensor

Nomenclature (following LAPACK):
    ge  = general
    sy  = real symmetric
    asy = anti symmetric
    po  = real symmetric positive definite
Only use upper triangle of sy & po, strictly upper triangle of asy, otherwise specified

Symmetric high order tensor definition:
    3rd-order tensor: A_ijk = A_jik
*/

#ifndef tchem_linalg_hpp
#define tchem_linalg_hpp

#include <torch/torch.h>

namespace tchem { namespace LA {

at::Tensor triple_product(const at::Tensor & a, const at::Tensor & b, const at::Tensor & c);

// a.outer(b) only works for vectors a & b
// This function is meant for general a & b with
// result_i1,i2,...,im,j1,j2,...,jn = a_i1,i2,...,im b_j1,j2,...,jn
at::Tensor outer_product(const at::Tensor & a, const at::Tensor & b);

// Convert a vector `x` to an `order`-th order symmetric tensor
// Here a symmetric tensor `A` is defined to satisfy
// A[i1][i2]...[in] = A[i2][i1]...[in] = ... = A[i2]...[in][i1] = ...
// i.e. the permutation of indices all give a same element
// only the "upper triangle" (i1 <= i2 <= ... <= in) of the output tensor is filled
at::Tensor vec2sytensor(const at::Tensor & x, const size_t & order, const size_t & dimension);

// Matrix dot multiplication for 3rd-order tensor A and B
// A.size(2) == B.size(2), A.size(1) == B.size(0)
// result_ij = A_ikm * B_kjm
at::Tensor ge3matdotmul(const at::Tensor & A, const at::Tensor & B);
void ge3matdotmul(const at::Tensor & A, const at::Tensor & B, at::Tensor & result);
// For symmetric A and B
// Here a symmetric 3rd-order tensor `A` means A[:][i][j] = A[:][j][i]
at::Tensor sy3matdotmul(const at::Tensor & A, const at::Tensor & B);
void sy3matdotmul(const at::Tensor & A, const at::Tensor & B, at::Tensor & result);

// Unitary transformation for symmetric 3rd-order tensor A
// Here a symmetric 3rd-order tensor `A` means A[:][i][j] = A[:][j][i]
// result_ijm = U^T_ia * A_abm * U_bj
at::Tensor UT_A3_U(const at::Tensor & A, const at::Tensor & UT);
// On exit A harvests the result
void UT_A3_U_(at::Tensor & A, const at::Tensor & UT);

} // namespace LA
} // namespace tchem

#endif