#include <torch/torch.h>

#include <tchem/linalg.hpp>

#include <tchem/chemistry.hpp>

namespace tchem { namespace chem {

// Check if two energy levels are closer than the threshold
bool check_degeneracy(const at::Tensor & energy, const double & thresh) {
    if (energy.sizes().size() != 1) throw std::invalid_argument(
    "tchem::chem::check_degeneracy: energy must be a vector");
    bool deg = false;
    for (size_t i = 0; i < energy.size(0) - 1; i++) {
        if (energy[i + 1].item<double>() - energy[i].item<double>() < thresh) {
            deg = true;
            break;
        }
    }
    return deg;
}

// Transform Hamiltonian (or energy) and gradient to composite representation
// which is defined by diagonalizing tchem::linalg::sy3matdotmul(▽H, ▽H)
// Return composite Hamiltonian and gradient
// Only read the "upper triangle" (i <= j) of H and ▽H
// Only write the "upper triangle" (i <= j) of the output tensor
std::tuple<at::Tensor, at::Tensor> composite_representation(const at::Tensor & H, const at::Tensor & dH) {
    if (H.sizes().size() != 2 && H.sizes().size() != 1) throw std::invalid_argument(
    "tchem::chem::composite_representation: H must be a matrix or an energy vector");
    if (dH.sizes().size() != 3) throw std::invalid_argument(
    "tchem::chem::composite_representation: ▽H must be a 3rd-order tensor");
    if (dH.size(0) != dH.size(1)) throw std::invalid_argument(
    "tchem::chem::composite_representation: The matrix part of ▽H must be square");
    at::Tensor dHdH = tchem::linalg::sy3matdotmul(dH, dH);
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = dHdH.symeig(true);
    at::Tensor H_c;
    if (H.sizes().size() == 2) H_c = tchem::linalg::UT_sy_U(H       , eigvec);
    else                       H_c = tchem::linalg::UT_sy_U(H.diag(), eigvec);
    at::Tensor dH_c = tchem::linalg::UT_sy_U(dH, eigvec);
    return std::make_tuple(H_c, dH_c);
}
// Only read/write the "upper triangle" (i <= j) of H and ▽H
void composite_representation_(at::Tensor & H, at::Tensor & dH) {
    if (H.sizes().size() != 2 && H.sizes().size() != 1) throw std::invalid_argument(
    "tchem::chem::composite_representation_: H must be a matrix or an energy vector");
    if (dH.sizes().size() != 3) throw std::invalid_argument(
    "tchem::chem::composite_representation_: ▽H must be a 3rd-order tensor");
    if (dH.size(0) != dH.size(1)) throw std::invalid_argument(
    "tchem::chem::composite_representation_: The matrix part of ▽H must be square");
    at::Tensor dHdH = tchem::linalg::sy3matdotmul(dH, dH);
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = dHdH.symeig(true);
    if (H.sizes().size() == 2) tchem::linalg::UT_sy_U_(H, eigvec);
    else                       H = tchem::linalg::UT_sy_U(H.diag(), eigvec);
    tchem::linalg::UT_sy_U_(dH, eigvec);
}
// dot product defined with a metric S
// S must be real symmetric positive definite
// Warning: All elements of S will be read because of torch::mv
//          Will fix it some day if pytorch introduces "symv" like BLAS
std::tuple<at::Tensor, at::Tensor> composite_representation(const at::Tensor & H, const at::Tensor & dH, const at::Tensor & S) {
    if (H.sizes().size() != 2 && H.sizes().size() != 1) throw std::invalid_argument(
    "tchem::chem::composite_representation: H must be a matrix or an energy vector");
    if (dH.sizes().size() != 3) throw std::invalid_argument(
    "tchem::chem::composite_representation: ▽H must be a 3rd-order tensor");
    if (dH.size(0) != dH.size(1)) throw std::invalid_argument(
    "tchem::chem::composite_representation: The matrix part of ▽H must be square");
    if (S.sizes().size() != 2) throw std::invalid_argument(
    "tchem::chem::composite_representation: S must be a matrix");
    if (S.size(0) != S.size(1)) throw std::invalid_argument(
    "tchem::chem::composite_representation: S must be a square matrix");   
    at::Tensor dHdH = tchem::linalg::sy3matdotmul(dH, dH, S);
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = dHdH.symeig(true);
    at::Tensor H_c;
    if (H.sizes().size() == 2) H_c = tchem::linalg::UT_sy_U(H       , eigvec);
    else                       H_c = tchem::linalg::UT_sy_U(H.diag(), eigvec);
    at::Tensor dH_c = tchem::linalg::UT_sy_U(dH, eigvec);
    return std::make_tuple(H_c, dH_c);
}
void composite_representation_(at::Tensor & H, at::Tensor & dH, const at::Tensor & S) {
    if (H.sizes().size() != 2 && H.sizes().size() != 1) throw std::invalid_argument(
    "tchem::chem::composite_representation_: H must be a matrix or an energy vector");
    if (dH.sizes().size() != 3) throw std::invalid_argument(
    "tchem::chem::composite_representation_: ▽H must be a 3rd-order tensor");
    if (dH.size(0) != dH.size(1)) throw std::invalid_argument(
    "tchem::chem::composite_representation_: The matrix part of ▽H must be square");
    if (S.sizes().size() != 2) throw std::invalid_argument(
    "tchem::chem::composite_representation_: S must be a matrix");
    if (S.size(0) != S.size(1)) throw std::invalid_argument(
    "tchem::chem::composite_representation_: S must be a square matrix");
    at::Tensor dHdH = tchem::linalg::sy3matdotmul(dH, dH, S);
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = dHdH.symeig(true);
    if (H.sizes().size() == 2) tchem::linalg::UT_sy_U_(H, eigvec);
    else                       H = tchem::linalg::UT_sy_U(H.diag(), eigvec);
    tchem::linalg::UT_sy_U_(dH, eigvec);
}






} // namespace chem
} // namespace tchem