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

#include <torch/torch.h>

namespace {
    // Support vec2sytensor
    // Recursively fill vector `x` into the "upper triangle" of symmetric tensor `A`
    // `xstart` and `Astart` indicates where to extract and fill data
    void vec2sytensor_support(const at::Tensor & x, at::Tensor A, size_t & xstart, const size_t & Astart) {
        if (A.sizes().size() == 1) {
            at::Tensor Aslice = A.slice(0, Astart);
            size_t stop = xstart + Aslice.size(0);
            Aslice.copy_(x.slice(0, xstart, stop));
            xstart = stop;
        }
        else {
            for (size_t i = Astart; i < A.size(0); i++) 
            vec2sytensor_support(x, A[i], xstart, i);
        }
    }
}

namespace tchem { namespace LA {

at::Tensor triple_product(const at::Tensor & a, const at::Tensor & b, const at::Tensor & c) {
    return ( c[0] * (a[1] * b[2] - a[2] * b[1])
           - c[1] * (a[0] * b[2] - a[2] * b[0])
           + c[2] * (a[0] * b[1] - a[1] * b[0]));
}

// a.outer(b) only works for vectors a & b
// This function is meant for general a & b with
// result_i1,i2,...,im,j1,j2,...,jn = a_i1,i2,...,im b_j1,j2,...,jn
at::Tensor outer_product(const at::Tensor & a, const at::Tensor & b) {
    std::vector<int64_t> dims(a.sizes().size() + b.sizes().size());
    for (size_t i = 0; i < a.sizes().size(); i++) dims[i] = a.size(i);
    for (size_t i = 0; i < b.sizes().size(); i++) dims[i + a.sizes().size()] = b.size(i);
    at::Tensor a_view = a.view(a.numel()),
               b_view = b.view(b.numel());
    c10::IntArrayRef sizes(dims.data(), dims.size());
    at::Tensor result = a_view.outer(b_view);
    return result.view(sizes);
}

// Convert a vector `x` to an `order`-th order symmetric tensor
// Here a symmetric tensor `A` is defined to satisfy
// A[i1][i2]...[in] = A[i2][i1]...[in] = ... = A[i2]...[in][i1] = ...
// i.e. the permutation of indices all give a same element
// only the "upper triangle" (i1 <= i2 <= ... <= in) of the output tensor is filled
at::Tensor vec2sytensor(const at::Tensor & x, const size_t & order, const size_t & dimension) {
    std::vector<int64_t> sizes_vector(order, dimension);
    c10::IntArrayRef sizes(sizes_vector.data(), order);
    at::Tensor result = x.new_empty(sizes);
    size_t xstart = 0, Astart = 0;
    vec2sytensor_support(x, result, xstart, Astart);
    return result;
}

// Matrix dot multiplication for 3rd-order tensor A and B
// A.size(2) == B.size(2), A.size(1) == B.size(0)
// result_ij = A_ikm * B_kjm
at::Tensor ge3matdotmul(const at::Tensor & A, const at::Tensor & B) {
    at::Tensor result = A.new_zeros({A.size(0), B.size(1)});
    for (int i = 0; i < result.size(0); i++)
    for (int j = 0; j < result.size(1); j++)
    for (int k = 0; k < B.size(0); k++)
    result[i][j] += A[i][k].dot(B[k][j]);
    return result;
}
void ge3matdotmul(const at::Tensor & A, const at::Tensor & B, at::Tensor & result) {
    result.fill_(0.0);
    for (int i = 0; i < result.size(0); i++)
    for (int j = 0; j < result.size(1); j++)
    for (int k = 0; k < B.size(0); k++)
    result[i][j] += A[i][k].dot(B[k][j]);
}
// For symmetric A and B
// Here a symmetric 3rd-order tensor `A` means A[:][i][j] = A[:][j][i]
at::Tensor sy3matdotmul(const at::Tensor & A, const at::Tensor & B) {
    at::Tensor result = A.new_zeros({A.size(0), B.size(1)});
    for (int i = 0; i < result.size(0); i++) {
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < j; k++) result[i][j] += A[k][i].dot(B[k][j]);
            for (int k = j; k < i; k++) result[i][j] += A[k][i].dot(B[j][k]);
            for (int k = i; k < B.size(0); k++) result[i][j] += A[i][k].dot(B[j][k]);
        }
        for (int j = i; j < result.size(1); j++) {
            for (int k = 0; k < i; k++) result[i][j] += A[k][i].dot(B[k][j]);
            for (int k = i; k < j; k++) result[i][j] += A[i][k].dot(B[k][j]);
            for (int k = j; k < B.size(0); k++) result[i][j] += A[i][k].dot(B[j][k]);
        }
    }
    return result;
}
void sy3matdotmul(const at::Tensor & A, const at::Tensor & B, at::Tensor & result) {
    result.fill_(0.0);
    for (int i = 0; i < result.size(0); i++) {
        for (int j = 0; j < i; j++) {
            for (int k = 0; k < j; k++) result[i][j] += A[k][i].dot(B[k][j]);
            for (int k = j; k < i; k++) result[i][j] += A[k][i].dot(B[j][k]);
            for (int k = i; k < B.size(0); k++) result[i][j] += A[i][k].dot(B[j][k]);
        }
        for (int j = i; j < result.size(1); j++) {
            for (int k = 0; k < i; k++) result[i][j] += A[k][i].dot(B[k][j]);
            for (int k = i; k < j; k++) result[i][j] += A[i][k].dot(B[k][j]);
            for (int k = j; k < B.size(0); k++) result[i][j] += A[i][k].dot(B[j][k]);
        }
    }
}

// Unitary transformation for symmetric 3rd-order tensor A
// Here a symmetric 3rd-order tensor `A` means A[:][i][j] = A[:][j][i]
// result_ijm = U^T_ia * A_abm * U_bj
at::Tensor UT_A3_U(const at::Tensor & A, const at::Tensor & UT) {
    int N = UT.size(0);
    // work_ibm = U^T_ia * A_abm
    at::Tensor work = A.new_zeros(A.sizes());
    for (int i = 0; i < N; i++)
    for (int b = 0; b < N; b++) {
        for (int a = 0; a < b; a++) work[i][b] += UT[i][a] * A[a][b];
        for (int a = b; a < N; a++) work[i][b] += UT[i][a] * A[b][a];
    }
    // result_ijm = work_ibm * U_bj = work_ibm * U^T_jb
    at::Tensor result = A.new_zeros(A.sizes());
    for (int i = 0; i < N; i++)
    for (int j = i; j < N; j++)
    for (int b = 0; b < N; b++)
    result[i][j] += work[i][b] * UT[j][b];
    return result;
}
// On exit A harvests the result
void UT_A3_U_(at::Tensor & A, const at::Tensor & UT) {
    int N = UT.size(0);
    // work_ibm = U^T_ia * A_abm
    at::Tensor work = A.new_zeros(A.sizes());
    for (int i = 0; i < N; i++)
    for (int b = 0; b < N; b++) {
        for (int a = 0; a < b; a++) work[i][b] += UT[i][a] * A[a][b];
        for (int a = b; a < N; a++) work[i][b] += UT[i][a] * A[b][a];
    }
    // result_ijm = work_ibm * U_bj = work_ibm * U^T_jb
    A.fill_(0.0);
    for (int i = 0; i < N; i++)
    for (int j = i; j < N; j++)
    for (int b = 0; b < N; b++)
    A[i][j] += work[i][b] * UT[j][b];
}

} // namespace LA
} // namespace tchem
