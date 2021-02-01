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
    assert(("a must be a 3D vector", a.sizes().size() == 1 && a.size(0) == 3));
    assert(("b must be a 3D vector", b.sizes().size() == 1 && b.size(0) == 3));
    assert(("c must be a 3D vector", c.sizes().size() == 1 && c.size(0) == 3));
    return ( c[0] * (a[1] * b[2] - a[2] * b[1])
           - c[1] * (a[0] * b[2] - a[2] * b[0])
           + c[2] * (a[0] * b[1] - a[1] * b[0]));
}

// a.outer(b) only works for vectors a & b
// This function is meant for general a & b with
// result_i1,i2,...,im,j1,j2,...,jn = a_i1,i2,...,im b_j1,j2,...,jn
at::Tensor outer_product(const at::Tensor & a, const at::Tensor & b) {
    // Normal vector outer product
    if (a.sizes().size() == 1 && b.sizes().size() == 1) {
        return a.outer(b);
    }
    // View as vector then outer product
    else {
        std::vector<int64_t> dims(a.sizes().size() + b.sizes().size());
        for (size_t i = 0; i < a.sizes().size(); i++) dims[i] = a.size(i);
        for (size_t i = 0; i < b.sizes().size(); i++) dims[i + a.sizes().size()] = b.size(i);
        at::Tensor a_view = a.view(a.numel()),
                   b_view = b.view(b.numel());
        c10::IntArrayRef sizes(dims.data(), dims.size());
        at::Tensor result = a_view.outer(b_view);
        return result.view(sizes);
    }
}

// Convert a vector `x` to an `order`-th order symmetric tensor
// Here a symmetric tensor `A` is defined to satisfy
// A[i1][i2]...[in] = A[i2][i1]...[in] = ... = A[i2]...[in][i1] = ...
// i.e. the permutation of indices all give a same element
// Only write the "upper triangle" (i1 <= i2 <= ... <= in) of the output tensor
at::Tensor vec2sytensor(const at::Tensor & x, const size_t & order, const size_t & dimension) {
    assert(("x must be a vector", x.sizes().size() == 1));
    std::vector<int64_t> sizes_vector(order, dimension);
    c10::IntArrayRef sizes(sizes_vector.data(), order);
    at::Tensor result = x.new_empty(sizes);
    size_t xstart = 0, Astart = 0;
    vec2sytensor_support(x, result, xstart, Astart);
    return result;
}

// Matrix dot multiplication for 3rd-order tensors A and B
// A.size(-1) == B.size(-1), A.size(1) == B.size(0)
// result_ij = A_ik . B_kj
at::Tensor ge3matdotmul(const at::Tensor & A, const at::Tensor & B) {
    assert(("A must be a 3rd-order tensor", A.sizes().size() == 3));
    assert(("B must be a 3rd-order tensor", B.sizes().size() == 3));
    assert(("A & B must share a same vector dimension", A.size(-1) == B.size(-1)));
    assert(("A & B must be matrix mutiplicable", A.size(1) == B.size(0)));
    at::Tensor result = A.new_zeros({A.size(0), B.size(1)});
    for (size_t i = 0; i < result.size(0); i++)
    for (size_t j = 0; j < result.size(1); j++)
    for (size_t k = 0; k < B.size(0); k++)
    result[i][j] += A[i][k].dot(B[k][j]);
    return result;
}
// For symmetric A and B
// Here a symmetric 3rd-order tensor `A` means A[i][j] = A[j][i]
// Only read the "upper triangle" (i <= j) of A and B
at::Tensor sy3matdotmul(const at::Tensor & A, const at::Tensor & B) {
    assert(("A must be a 3rd-order tensor", A.sizes().size() == 3));
    assert(("The matrix part of A must be square", A.size(0) == A.size(1)));
    assert(("B must be a 3rd-order tensor", B.sizes().size() == 3));
    assert(("The matrix part of B must be square", B.size(0) == B.size(1)));
    assert(("A & B must share a same vector dimension", A.size(-1) == B.size(-1)));
    assert(("A & B must be matrix mutiplicable", A.size(1) == B.size(0)));
    at::Tensor result = A.new_zeros({A.size(0), B.size(1)});
    for (size_t i = 0; i < result.size(0); i++) {
        for (size_t j = 0; j < i; j++) {
            for (size_t k = 0; k < j; k++)         result[i][j] += A[k][i].dot(B[k][j]);
            for (size_t k = j; k < i; k++)         result[i][j] += A[k][i].dot(B[j][k]);
            for (size_t k = i; k < B.size(0); k++) result[i][j] += A[i][k].dot(B[j][k]);
        }
        for (size_t j = i; j < result.size(1); j++) {
            for (size_t k = 0; k < i; k++)         result[i][j] += A[k][i].dot(B[k][j]);
            for (size_t k = i; k < j; k++)         result[i][j] += A[i][k].dot(B[k][j]);
            for (size_t k = j; k < B.size(0); k++) result[i][j] += A[i][k].dot(B[j][k]);
        }
    }
    return result;
}

// Matrix dot multiplication for 4rh-order tensor A and 3rd-order tensor B
// A.size(-1) == B.size(-1), A.size(1) == B.size(0)
// result_ij = A_ik . B_kj
at::Tensor ge4matmvmulge3(const at::Tensor & A, const at::Tensor & B) {
    assert(("A must be a 4th-order tensor", A.sizes().size() == 4));
    assert(("B must be a 3rd-order tensor", B.sizes().size() == 3));
    assert(("The matrix-vector part of A & B must be matrix-vector mutiplicable", A.size(-1) == B.size(-1)));
    assert(("A & B must be matrix mutiplicable", A.size(1) == B.size(0)));
    at::Tensor result = A.new_zeros({A.size(0), B.size(1), A.size(-2)});
    for (size_t i = 0; i < result.size(0); i++)
    for (size_t j = 0; j < result.size(1); j++)
    for (size_t k = 0; k < B.size(0); k++)
    result[i][j] += A[i][k].mv(B[k][j]);
    return result;
}
// For symmetric A and B
// Here a symmetric 3rd or 4th order tensor `A` means A[i][j] = A[j][i]
// Only read the "upper triangle" (i <= j) of A and B
at::Tensor sy4matmvmulsy3(const at::Tensor & A, const at::Tensor & B) {
    assert(("A must be a 4th-order tensor", A.sizes().size() == 4));
    assert(("The matrix part of A must be square", A.size(0) == A.size(1)));
    assert(("B must be a 3rd-order tensor", B.sizes().size() == 3));
    assert(("The matrix part of B must be square", B.size(0) == B.size(1)));
    assert(("The matrix-vector part of A & B must be matrix-vector mutiplicable", A.size(-1) == B.size(-1)));
    assert(("A & B must be matrix mutiplicable", A.size(1) == B.size(0)));
    at::Tensor result = A.new_zeros({A.size(0), B.size(1), A.size(-2)});
    for (size_t i = 0; i < result.size(0); i++) {
        for (size_t j = 0; j < i; j++) {
            for (size_t k = 0; k < j; k++)         result[i][j] += A[k][i].mv(B[k][j]);
            for (size_t k = j; k < i; k++)         result[i][j] += A[k][i].mv(B[j][k]);
            for (size_t k = i; k < B.size(0); k++) result[i][j] += A[i][k].mv(B[j][k]);
        }
        for (size_t j = i; j < result.size(1); j++) {
            for (size_t k = 0; k < i; k++)         result[i][j] += A[k][i].mv(B[k][j]);
            for (size_t k = i; k < j; k++)         result[i][j] += A[i][k].mv(B[k][j]);
            for (size_t k = j; k < B.size(0); k++) result[i][j] += A[i][k].mv(B[j][k]);
        }
    }
    return result;
}

// Matrix outer multiplication for matrices or higher order tensors A and B
// A.size(2) == B.size(2), A.size(1) == B.size(0)
// result_ij = outer_product(A_ik, B_kj)
at::Tensor gematoutermul(const at::Tensor & A, const at::Tensor & B) {
    assert(("A must be a matrix or higher order tensor", A.sizes().size() >= 2));
    assert(("B must be a matrix or higher order tensor", B.sizes().size() >= 2));
    assert(("A & B must be matrix mutiplicable", A.size(1) == B.size(0)));
    std::vector<int64_t> dims(A.sizes().size() + B.sizes().size() - 2);
    dims[0] = A.size(0);
    dims[1] = B.size(1);
    for (size_t i = 2; i < A.sizes().size(); i++) dims[i] = A.size(i);
    for (size_t i = 2; i < B.sizes().size(); i++) dims[i - 2 + A.sizes().size()] = B.size(i);
    c10::IntArrayRef sizes(dims.data(), dims.size());
    at::Tensor result = A.new_zeros(sizes);
    for (size_t i = 0; i < result.size(0); i++)
    for (size_t j = 0; j < result.size(1); j++)
    for (size_t k = 0; k < B.size(0); k++)
    result[i][j] += outer_product(A[i][k], B[k][j]);
    return result;
}
// For symmetric A and B
// Here a symmetric 3rd-order tensor `A` means A[i][j] = A[j][i]
// Only read the "upper triangle" (i <= j) of A and B
at::Tensor symatoutermul(const at::Tensor & A, const at::Tensor & B) {
    assert(("A must be a matrix or higher order tensor", A.sizes().size() >= 2));
    assert(("B must be a matrix or higher order tensor", B.sizes().size() >= 2));
    assert(("The matrix part of A must be square", A.size(0) == A.size(1)));
    assert(("The matrix part of B must be square", B.size(0) == B.size(1)));
    assert(("A & B must be matrix mutiplicable", A.size(1) == B.size(0)));
    std::vector<int64_t> dims(A.sizes().size() + B.sizes().size() - 2);
    dims[0] = A.size(0);
    dims[1] = B.size(1);
    for (size_t i = 2; i < A.sizes().size(); i++) dims[i] = A.size(i);
    for (size_t i = 2; i < B.sizes().size(); i++) dims[i - 2 + A.sizes().size()] = B.size(i);
    c10::IntArrayRef sizes(dims.data(), dims.size());
    at::Tensor result = A.new_zeros(sizes);
    for (size_t i = 0; i < result.size(0); i++) {
        for (size_t j = 0; j < i; j++) {
            for (size_t k = 0; k < j; k++)         result[i][j] += outer_product(A[k][i], B[k][j]);
            for (size_t k = j; k < i; k++)         result[i][j] += outer_product(A[k][i], B[j][k]);
            for (size_t k = i; k < B.size(0); k++) result[i][j] += outer_product(A[i][k], B[j][k]);
        }
        for (size_t j = i; j < result.size(1); j++) {
            for (size_t k = 0; k < i; k++)         result[i][j] += outer_product(A[k][i], B[k][j]);
            for (size_t k = i; k < j; k++)         result[i][j] += outer_product(A[i][k], B[k][j]);
            for (size_t k = j; k < B.size(0); k++) result[i][j] += outer_product(A[i][k], B[j][k]);
        }
    }
    return result;
}

// Unitary transformation for matrix or higher order tensor A
// result_ij = U^T_ia * A_ab * U_bj
at::Tensor UT_ge_U(const at::Tensor & A, const at::Tensor & U) {
    assert(("A must be a matrix or higher order tensor", A.sizes().size() >= 2));
    assert(("The matrix part of A must be square", A.size(0) == A.size(1)));
    assert(("U must be a matrix", U.sizes().size() == 2));
    assert(("U must be a square matrix", U.size(0) == U.size(1)));
    assert(("U & A must be matrix mutiplicable", U.size(0) == A.size(0)));
    // Normal matrix multiplication
    if (A.sizes().size() == 2) {
        return U.transpose(0, 1).mm(A.mm(U));
    }
    // Batched matmul with special reindexing
    else if (A.sizes().size() == 3) {
        at::Tensor A_batch = A.transpose(1, 2).transpose(0, 1);
        at::Tensor result_batch = torch::matmul(U.transpose(0, 1), torch::matmul(A_batch, U));
        return result_batch.transpose(0, 1).transpose(1, 2);
    }
    // Batched matmul with reindexing
    else {
        at::Tensor A_batch = A.transpose(0, -2).transpose(1, -1);
        at::Tensor result_batch = torch::matmul(U.transpose(0, 1), torch::matmul(A_batch, U));
        return result_batch.transpose(0, -2).transpose(1, -1);
    }
}
void UT_ge_U_(at::Tensor & A, const at::Tensor & U) {
    assert(("A must be a matrix or higher order tensor", A.sizes().size() >= 2));
    assert(("The matrix part of A must be square", A.size(0) == A.size(1)));
    assert(("U must be a matrix", U.sizes().size() == 2));
    assert(("U must be a square matrix", U.size(0) == U.size(1)));
    assert(("U & A must be matrix mutiplicable", U.size(0) == A.size(0)));
    // Normal matrix multiplication
    if (A.sizes().size() == 2) {
        A = U.transpose(0, 1).mm(A.mm(U));
    }
    // Batched matmul with special reindexing
    else if (A.sizes().size() == 3) {
        A.transpose_(1, 2).transpose_(0, 1);
        A = torch::matmul(U.transpose(0, 1), torch::matmul(A, U));
        A.transpose_(0, 1).transpose_(1, 2);
    }
    // Batched matmul with reindexing
    else {
        A.transpose_(0, -2).transpose_(1, -1);
        A = torch::matmul(U.transpose(0, 1), torch::matmul(A, U));
        A.transpose_(0, -2).transpose_(1, -1);
    }
}
// For symmetric A
// Here a symmetric higher order tensor `A` means A[i][j] = A[j][i]
// Only read the "upper triangle" (i <= j) of A
// Only write the "upper triangle" (i <= j) of the output tensor
// Warning: this routine is known to deteriorate backward propagation
//          Probably because it explicitly loops over the matrix elements
//          Will fix it some day if pytorch introduces "symm" like BLAS
at::Tensor UT_sy_U(const at::Tensor & A, const at::Tensor & U) {
    assert(("A must be a matrix or higher order tensor", A.sizes().size() >= 2));
    assert(("The matrix part of A must be square", A.size(0) == A.size(1)));
    assert(("U must be a matrix", U.sizes().size() == 2));
    assert(("U must be a square matrix", U.size(0) == U.size(1)));
    assert(("U & A must be matrix mutiplicable", U.size(0) == A.size(0)));
    size_t N = U.size(0);
    // work_ib = U^T_ia * A_ab = U_ai * A_ab:
    at::Tensor work = A.new_zeros(A.sizes());
    for (size_t i = 0; i < N; i++)
    for (size_t b = 0; b < N; b++) {
        for (size_t a = 0; a < b; a++) work[i][b] = work[i][b] + U[a][i] * A[a][b];
        for (size_t a = b; a < N; a++) work[i][b] = work[i][b] + U[a][i] * A[b][a];
    }
    // result_ij = work_ib * U_bj
    at::Tensor result = A.new_zeros(A.sizes());
    for (size_t i = 0; i < N; i++)
    for (size_t j = i; j < N; j++)
    for (size_t b = 0; b < N; b++)
    result[i][j] = result[i][j] + work[i][b] * U[b][j];
    return result;
}
void UT_sy_U_(at::Tensor & A, const at::Tensor & U) {
    assert(("A must be a matrix or higher order tensor", A.sizes().size() >= 2));
    assert(("The matrix part of A must be square", A.size(0) == A.size(1)));
    assert(("U must be a matrix", U.sizes().size() == 2));
    assert(("U must be a square matrix", U.size(0) == U.size(1)));
    assert(("U & A must be matrix mutiplicable", U.size(0) == A.size(0)));
    size_t N = U.size(0);
    // work_ibm = U^T_ia * A_abm = U_ai * A_abm
    at::Tensor work = A.new_zeros(A.sizes());
    for (size_t i = 0; i < N; i++)
    for (size_t b = 0; b < N; b++) {
        for (size_t a = 0; a < b; a++) work[i][b] += U[a][i] * A[a][b];
        for (size_t a = b; a < N; a++) work[i][b] += U[a][i] * A[b][a];
    }
    // result_ijm = work_ibm * U_bj
    A.fill_(0.0);
    for (size_t i = 0; i < N; i++)
    for (size_t j = i; j < N; j++)
    for (size_t b = 0; b < N; b++)
    A[i][j] += work[i][b] * U[b][j];
}

} // namespace LA
} // namespace tchem
