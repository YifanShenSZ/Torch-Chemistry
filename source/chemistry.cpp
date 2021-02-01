#include <torch/torch.h>

#include <tchem/linalg.hpp>

#include <tchem/chemistry.hpp>

namespace tchem { namespace chem {

// Check if two energy levels are closer than the threshold
bool check_degeneracy(const at::Tensor & energy, const double & thresh) {
    assert(("energy must be a vector", energy.sizes().size() == 1));
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
// Return composite Hamiltonian and gradient
// Only read the "upper triangle" (i <= j) of H and dH
// Only write the "upper triangle" (i <= j) of the output tensor
std::tuple<at::Tensor, at::Tensor> composite_representation(const at::Tensor & H, const at::Tensor & dH) {
    assert(("Hamiltonian must be a matrix or an energy vector", H.sizes().size() == 2 || H.sizes().size() == 1));
    assert(("gradient must be a 3rd-order tensor", dH.sizes().size() == 3));
    assert(("The matrix part of gradient must be square", dH.size(0) == dH.size(1)));
    at::Tensor dHdH = tchem::LA::sy3matdotmul(dH, dH);
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = dHdH.symeig(true);
    at::Tensor H_c;
    if (H.sizes().size() == 2) H_c = tchem::LA::UT_sy_U(H       , eigvec);
    else                       H_c = tchem::LA::UT_sy_U(H.diag(), eigvec);
    at::Tensor dH_c = tchem::LA::UT_sy_U(dH, eigvec);
    return std::make_tuple(H_c, dH_c);
}
// Only read/write the "upper triangle" (i <= j) of H and dH
void composite_representation_(at::Tensor & H, at::Tensor & dH) {
    assert(("Hamiltonian must be a matrix or an energy vector", H.sizes().size() == 2 || H.sizes().size() == 1));
    assert(("gradient must be a 3rd-order tensor", dH.sizes().size() == 3));
    assert(("The matrix part of gradient must be square", dH.size(0) == dH.size(1)));
    at::Tensor dHdH = tchem::LA::sy3matdotmul(dH, dH);
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = dHdH.symeig(true);
    if (H.sizes().size() == 2) tchem::LA::UT_sy_U_(H, eigvec);
    else                       H = tchem::LA::UT_sy_U(H.diag(), eigvec);
    tchem::LA::UT_sy_U_(dH, eigvec);
}    





Phaser::Phaser(const size_t & _NStates) : NStates_(_NStates) {
    possible_phases_.resize((1 << (NStates_ - 1)) - 1);
    for (std::vector<bool> & phase : possible_phases_) phase.resize(NStates_ - 1);
    std::vector<bool> & phase = possible_phases_[0];
    phase[0] = true;
    for (size_t j = 1; j < phase.size(); j++) phase[j] = false;
    for (size_t i = 1; i < possible_phases_.size(); i++) {
        std::vector<bool> & phase = possible_phases_[i];
        // Q: Why not std::memcpy?
        // A: bool is a special type who does not really have a pointer
        for (size_t j = 0; j < phase.size(); j++) phase[j] = possible_phases_[i - 1][j];
        size_t index = 0;
        while(phase[index]) {
            phase[index] = false;
            index++;
        }
        phase[index] = true;
    }
}

// Alter the phase of M to the index-th possible phase
at::Tensor Phaser::alter(const at::Tensor & M, const size_t & index) const {
    if (index >= possible_phases_.size()) return M;
    at::Tensor result = M.new_empty(M.sizes());
    const std::vector<bool> & phase = possible_phases_[index];
    for (size_t i = 0; i < NStates_; i++) {
        result[i][i] = M[i][i];
        // From i,i+1 to i,NStates-2: phase = phase[i] ^ phase[j]
        for (size_t j = i + 1; j < NStates_ - 1; j++)
        if (phase[i] ^ phase[j]) result[i][j] = -M[i][j];
        else                     result[i][j] =  M[i][j];
        // i,NStates-1, phase = phase[i], since phase[NStates_ - 1] = false
        size_t j = NStates_ - 1;
        if (phase[i]) result[i][j] = -M[i][j];
        else          result[i][j] =  M[i][j];
    }
    return result;
}
void Phaser::alter_(at::Tensor & M, const size_t & index) const {
    if (index >= possible_phases_.size()) return;
    const std::vector<bool> & phase = possible_phases_[index];
    for (size_t i = 0; i < NStates_ - 1; i++) {
        // From i,i+1 to i,NStates-2: phase = phase[i] ^ phase[j]
        for (size_t j = i + 1; j < NStates_ - 1; j++)
        if (phase[i] ^ phase[j]) M[i][j].neg_();
        // i,NStates-1, phase = phase[i], since phase[NStates_ - 1] = false
        size_t j = NStates_ - 1;
        if (phase[i]) M[i][j].neg_();
    }
}

// Return the index of the possible phase who minimizes || M - ref ||_F^2
// Return -1 if no need to change phase
size_t Phaser::iphase_min(const at::Tensor & M, const at::Tensor & ref) const {
    assert(("M must be a matrix or higher order tensor", M.sizes().size() >= 2));
    assert(("The matrix part of M must be square", M.size(0) == M.size(1)));
    assert(("The matrix dimension must be the number of electronic states", M.size(0) == NStates_));
    assert(("M and ref must share a same shape", at::tensor(M.sizes()).equal(at::tensor(ref.sizes()))));
    // Calculate the initial difference of each matrix element
    at::Tensor diff_mat = (M - ref).pow_(2);
    if (M.sizes().size() == 3) {
        diff_mat.transpose_(1, 2).transpose_(0, 1);
        diff_mat = diff_mat.sum_to_size({(int64_t)NStates_, (int64_t)NStates_});
    }
    else if (M.sizes().size() > 3) {
        diff_mat.transpose_(0, -2).transpose_(1, -1);
        diff_mat = diff_mat.sum_to_size({(int64_t)NStates_, (int64_t)NStates_});
    }
    // Try out phase possibilities
    double change_min = 0.0;
    size_t iphase_min = -1;
    for (size_t iphase = 0; iphase < possible_phases_.size(); iphase++) {
        const std::vector<bool> & phase = possible_phases_[iphase];
        at::Tensor change = M.new_zeros({});
        for (size_t i = 0; i < NStates_ - 1; i++) {
            // From i,i+1 to i,NStates-2: phase = phase[i] ^ phase[j]
            for (size_t j = i + 1; j < NStates_ - 1; j++)
            if (phase[i] ^ phase[j])
            change += (-M[i][j] - ref[i][j]).pow_(2).sum() - diff_mat[i][j];
            // i,NStates-1, phase = phase[i], since phase[NStates_ - 1] = false
            size_t j = NStates_ - 1;
            if (phase[i])
            change += (-M[i][j] - ref[i][j]).pow_(2).sum() - diff_mat[i][j];
        }
        if (change.item<double>() < change_min) {
            change_min = change.item<double>();
            iphase_min = iphase;
        }
    }
    return iphase_min;
}
// Return the index of the possible phase who minimizes weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
// Return -1 if no need to change phase
size_t Phaser::iphase_min(const at::Tensor & M1, const at::Tensor & M2, const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const {
    assert(("M1 must be a matrix or higher order tensor", M1.sizes().size() >= 2));
    assert(("The matrix part of M1 must be square", M1.size(0) == M1.size(1)));
    assert(("M2 must be a matrix or higher order tensor", M2.sizes().size() >= 2));
    assert(("The matrix part of M2 must be square", M2.size(0) == M2.size(1)));
    assert(("M1 and M2 must share a same matrix dimension", M1.size(0) == M2.size(0)));
    assert(("The matrix dimension must be the number of electronic states", M1.size(0) == NStates_));
    assert(("M1 and ref1 must share a same shape", at::tensor(M1.sizes()).equal(at::tensor(ref1.sizes()))));
    assert(("M2 and ref2 must share a same shape", at::tensor(M2.sizes()).equal(at::tensor(ref2.sizes()))));
    // Calculate the initial difference of each matrix element
    at::Tensor diff_mat1 = (M1 - ref1).pow_(2);
    if (M1.sizes().size() == 3) {
        diff_mat1.transpose_(1, 2).transpose_(0, 1);
        diff_mat1 = diff_mat1.sum_to_size({(int64_t)NStates_, (int64_t)NStates_});
    }
    else if (M1.sizes().size() > 3) {
        diff_mat1.transpose_(0, -2).transpose_(1, -1);
        diff_mat1 = diff_mat1.sum_to_size({(int64_t)NStates_, (int64_t)NStates_});
    }
    at::Tensor diff_mat2 = (M2 - ref2).pow_(2);
    if (M2.sizes().size() == 3) {
        diff_mat2.transpose_(1, 2).transpose_(0, 1);
        diff_mat2 = diff_mat2.sum_to_size({(int64_t)NStates_, (int64_t)NStates_});
    }
    else if (M2.sizes().size() > 3) {
        diff_mat2.transpose_(0, -2).transpose_(1, -1);
        diff_mat2 = diff_mat2.sum_to_size({(int64_t)NStates_, (int64_t)NStates_});
    }
    at::Tensor diff_mat = weight * diff_mat1 + diff_mat2;
    // Try out phase possibilities
    double change_min = 0.0;
    size_t iphase_min = -1;
    for (size_t iphase = 0; iphase < possible_phases_.size(); iphase++) {
        const std::vector<bool> & phase = possible_phases_[iphase];
        at::Tensor change = M1.new_zeros({});
        for (size_t i = 0; i < NStates_ - 1; i++) {
            // From i,i+1 to i,NStates-2: phase = phase[i] ^ phase[j]
            for (size_t j = i + 1; j < NStates_ - 1; j++)
            if (phase[i] ^ phase[j])
            change += weight * (-M1[i][j] - ref1[i][j]).pow_(2).sum()
                    + (-M2[i][j] - ref2[i][j]).pow_(2).sum()
                    - diff_mat[i][j];
            // i,NStates-1, phase = phase[i], since phase[NStates_ - 1] = false
            size_t j = NStates_ - 1;
            if (phase[i])
            change += weight * (-M1[i][j] - ref1[i][j]).pow_(2).sum()
                    + (-M2[i][j] - ref2[i][j]).pow_(2).sum()
                    - diff_mat[i][j];
        }
        if (change.item<double>() < change_min) {
            change_min = change.item<double>();
            iphase_min = iphase;
        }
    }
    return iphase_min;
}

// Fix M by minimizing || M - ref ||_F^2
at::Tensor Phaser::fix(const at::Tensor & M, const at::Tensor & ref) const {
    size_t iphase = iphase_min(M, ref);
    return alter(M, iphase);
}
void Phaser::fix_(at::Tensor & M, const at::Tensor & ref) const {
    size_t iphase = iphase_min(M, ref);
    alter_(M, iphase);
}
// Fix M1 and M2 by minimizing weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
std::tuple<at::Tensor, at::Tensor> Phaser::fix(const at::Tensor & M1, const at::Tensor & M2, const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const {
    size_t iphase = iphase_min(M1, M2, ref1, ref2, weight);
    return std::make_tuple(alter(M1, iphase), alter(M2, iphase));
}
void Phaser::fix_(at::Tensor & M1, at::Tensor & M2, const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const {
    size_t iphase = iphase_min(M1, M2, ref1, ref2, weight);
    alter_(M1, iphase);
    alter_(M2, iphase);
}

} // namespace chem
} // namespace tchem