/*
Additional linear algebra for libtorch tensor

Nomenclature (following LAPACK):
    ge  = general
    sy  = real symmetric
    asy = anti symmetric
    po  = real symmetric positive definite
*/

#ifndef tchem_linalg_hpp
#define tchem_linalg_hpp

#include <torch/torch.h>

namespace tchem { namespace linalg {

at::Tensor triple_product(const at::Tensor & a, const at::Tensor & b, const at::Tensor & c);

// a.outer(b) only works for vectors a & b
// This function is meant for general a & b with
// result_i1,i2,...,im,j1,j2,...,jn = a_i1,i2,...,im b_j1,j2,...,jn
at::Tensor outer_product(const at::Tensor & a, const at::Tensor & b);

// Convert a vector `x` to an `order`-th order symmetric tensor
// Here a symmetric tensor `A` is defined to satisfy
// A[i1][i2]...[in] = A[i2][i1]...[in] = ... = A[i2]...[in][i1] = ...
// i.e. the permutation of indices all give a same element
// Only write the "upper triangle" (i1 <= i2 <= ... <= in) of the output tensor
at::Tensor vec2sytensor(const at::Tensor & x, const size_t & order, const size_t & dimension);

// Matrix dot multiplication for 3rd-order tensors A and B
// A.size(2) == B.size(2), A.size(1) == B.size(0)
// result_ij = A_ikm * B_kjm
at::Tensor ge3matdotmul(const at::Tensor & A, const at::Tensor & B);
// For symmetric A and B
// Here a symmetric 3rd-order tensor `A` means A[i][j] = A[j][i]
// Only read the "upper triangle" (i <= j) of A and B
at::Tensor sy3matdotmul(const at::Tensor & A, const at::Tensor & B);
// dot product defined with a metric S
// S must be real symmetric positive definite
// Warning: All elements of S will be read because of torch::mv
//          Will fix it some day if pytorch introduces "symv" like BLAS
at::Tensor ge3matdotmul(const at::Tensor & A, const at::Tensor & B, const at::Tensor & S);
at::Tensor sy3matdotmul(const at::Tensor & A, const at::Tensor & B, const at::Tensor & S);

// Matrix dot multiplication for 4rh-order tensor A and 3rd-order tensor B
// A.size(-1) == B.size(-1), A.size(1) == B.size(0)
// result_ij = A_ik . B_kj
at::Tensor ge4matmvmulge3(const at::Tensor & A, const at::Tensor & B);
// For symmetric A and B
// Here a symmetric 3rd or 4th order tensor `A` means A[i][j] = A[j][i]
// Only read the "upper triangle" (i <= j) of A and B
at::Tensor sy4matmvmulsy3(const at::Tensor & A, const at::Tensor & B);
// matrix-vector product defined with a metric S
// S must be real symmetric positive definite
// Warning: All elements of S will be read because of torch::mv
//          Will fix it some day if pytorch introduces "symv" like BLAS
at::Tensor ge4matmvmulge3(const at::Tensor & A, const at::Tensor & B, const at::Tensor & S);
at::Tensor sy4matmvmulsy3(const at::Tensor & A, const at::Tensor & B, const at::Tensor & S);

// Matrix outer multiplication for matrices or higher order tensors A and B
// A.size(2) == B.size(2), A.size(1) == B.size(0)
// result_ij = outer_product(A_ik, B_kj)
at::Tensor gematoutermul(const at::Tensor & A, const at::Tensor & B);
// For symmetric A and B
// Here a symmetric 3rd-order tensor `A` means A[i][j] = A[j][i]
// Only read the "upper triangle" (i <= j) of A and B
at::Tensor symatoutermul(const at::Tensor & A, const at::Tensor & B);

// Unitary transformation for matrix or higher order tensor A
// result_ij = U^T_ia * A_ab * U_bj
at::Tensor UT_ge_U(const at::Tensor & A, const at::Tensor & U);
void UT_ge_U_(at::Tensor & A, const at::Tensor & U);
// For symmetric A
// Here a symmetric higher order tensor `A` means A[i][j] = A[j][i]
// Only read the "upper triangle" (i <= j) of A
// Only write the "upper triangle" (i <= j) of the output tensor
// Warning: this routine is known to deteriorate backward propagation
//          Probably because it explicitly loops over the matrix elements
//          Will fix it some day if pytorch introduces "symm" like BLAS
at::Tensor UT_sy_U(const at::Tensor & A, const at::Tensor & U);
void UT_sy_U_(at::Tensor & A, const at::Tensor & U);

} // namespace linalg
} // namespace tchem

#endif