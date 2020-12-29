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

namespace tchem { namespace LA {

double triple_product(const at::Tensor & a, const at::Tensor & b, const at::Tensor & c) {
    return (c[0]*(a[1]*b[2]-a[2]*b[1])-c[1]*(a[0]*b[2]-a[2]*b[0])+c[2]*(a[0]*b[1]-a[1]*b[0])).item<double>();
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
