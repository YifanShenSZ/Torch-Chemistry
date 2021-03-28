#ifndef tchem_FORTRAN_hpp
#define tchem_FORTRAN_hpp

#include <torch/torch.h>

namespace { extern "C" {
    int32_t my_dsygv_(
        double * A, double * S, double * eigval, const int32_t & N,
        const int32_t & type, const int32_t & eigenvectors
    );
} }

namespace at {

// type of generalized eigenvalue problem
enum gv_type {
    type1 = 1, // A . eigvec = eigval * S . eigvec
    type2 = 2, // A . S . eigvec = eigval * eigvec
    type3 = 3  // S . A . eigvec = eigval * eigvec
};

inline std::tuple<at::Tensor, at::Tensor> dsygv(
const at::Tensor & A, const at::Tensor & S,
const gv_type & type = type1, const bool & eigenvectors = false) {
    if (A.sizes().size() != 2) throw std::invalid_argument(
    "at::dsygv: A must be a matrix");
    if (A.size(0) != A.size(1)) throw std::invalid_argument(
    "at::dsygv: A must be a square matrix");
    if (S.sizes().size() != 2) throw std::invalid_argument(
    "at::dsygv: S must be a matrix");
    if (S.size(0) != S.size(1)) throw std::invalid_argument(
    "at::dsygv: S must be a square matrix");
    if (A.size(0) != S.size(0)) throw std::invalid_argument(
    "at::dsygv: inconsistent size between A and S");
    int32_t N = A.size(0);
    at::Tensor eigval = A.new_empty(N),
               eigvec = A.clone(),
               metric = S.clone();
    int32_t F_eigenvectors;
    if (eigenvectors) F_eigenvectors = -1;
    else              F_eigenvectors = 0;
    int32_t info = my_dsygv_(
        eigvec.data_ptr<double>(), metric.data_ptr<double>(), eigval.data_ptr<double>(),
        N, type, F_eigenvectors
    );
    if (info != 0) throw std::runtime_error("at::dsygv: info = " + std::to_string(info));
    // Due to FORTRAN-C memory format difference eigvec stores eigenvectors in each row
    eigvec.transpose_(0, 1);
    return std::make_tuple(eigval, eigvec);
}

}

#endif