#include <tchem/chem/phaser.hpp>

namespace tchem { namespace chem {

Phaser::Phaser() {}
Phaser::Phaser(const size_t & _NStates) : NStates_(_NStates) {
    // There must be at least 2 states to have 'phase difference'
    if (_NStates < 2) return;
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
Phaser::~Phaser() {}

size_t Phaser::NStates() const {return NStates_;}
std::vector<std::vector<bool>> Phaser::possible_phases() const {return possible_phases_;}

// Alter the phase of eigenstates `U` to the `index`-th possible phase
at::Tensor Phaser::alter_states(const at::Tensor & U, const size_t & index) const {
    if (U.sizes().size() != 2) throw std::invalid_argument(
    "tchem::Phaser::alter_states: U must be a matrix");
    if (U.size(1) != NStates_) throw std::invalid_argument(
    "tchem::Phaser::alter_states: The number of columns must be the number of electronic states");
    if (index >= possible_phases_.size()) return U;
    at::Tensor result = U.clone().transpose_(0, 1);
    const std::vector<bool> & phase = possible_phases_[index];
    // The phase of the last state is always arbitrarily assigned to +
    for (size_t i = 0; i < NStates_ - 1; i++) if (phase[i]) result[i].neg_();
    result.transpose_(0, 1);
    return result;
}
void Phaser::alter_states_(at::Tensor & U, const size_t & index) const {
    if (U.sizes().size() != 2) throw std::invalid_argument(
    "tchem::Phaser::alter_states_: U must be a matrix");
    if (U.size(1) != NStates_) throw std::invalid_argument(
    "tchem::Phaser::alter_states_: The number of columns must be the number of electronic states");
    if (index >= possible_phases_.size()) return;
    U.transpose_(0, 1);
    const std::vector<bool> & phase = possible_phases_[index];
    // The phase of the last state is always arbitrarily assigned to +
    for (size_t i = 0; i < NStates_ - 1; i++) if (phase[i]) U[i].neg_();
    U.transpose_(0, 1);
}
// Alter the phase of observable `M` to the `index`-th possible phase
at::Tensor Phaser::alter_ob(const at::Tensor & M, const size_t & index) const {
    if (M.sizes().size() < 2) throw std::invalid_argument(
    "tchem::Phaser::alter_ob: M must be a matrix or higher order tensor");
    if (M.size(0) != M.size(1)) throw std::invalid_argument(
    "tchem::Phaser::alter_ob: The matrix part of M must be square");
    if (M.size(0) != NStates_) throw std::invalid_argument(
    "tchem::Phaser::alter_ob: The matrix dimension must be the number of electronic states");
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
void Phaser::alter_ob_(at::Tensor & M, const size_t & index) const {
    if (M.sizes().size() < 2) throw std::invalid_argument(
    "tchem::Phaser::alter_ob_: M must be a matrix or higher order tensor");
    if (M.size(0) != M.size(1)) throw std::invalid_argument(
    "tchem::Phaser::alter_ob_: The matrix part of M must be square");
    if (M.size(0) != NStates_) throw std::invalid_argument(
    "tchem::Phaser::alter_ob_: The matrix dimension must be the number of electronic states");
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
    if (M.sizes().size() < 2) throw std::invalid_argument(
    "tchem::Phaser::alter_ob: M must be a matrix or higher order tensor");
    if (M.size(0) != M.size(1)) throw std::invalid_argument(
    "tchem::Phaser::alter_ob: The matrix part of M must be square");
    if (M.size(0) != NStates_) throw std::invalid_argument(
    "tchem::Phaser::alter_ob: The matrix dimension must be the number of electronic states");
    if (! at::tensor(M.sizes()).equal(at::tensor(ref.sizes()))) throw std::invalid_argument(
    "tchem::Phaser::alter_ob: M & ref must share a same shape");
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
    if (M1.sizes().size() < 2) throw std::invalid_argument(
    "tchem::Phaser::alter_ob: M1 must be a matrix or higher order tensor");
    if (M1.size(0) != M1.size(1)) throw std::invalid_argument(
    "tchem::Phaser::alter_ob: The matrix part of M1 must be square");
    if (M2.sizes().size() < 2) throw std::invalid_argument(
    "tchem::Phaser::alter_ob: M2 must be a matrix or higher order tensor");
    if (M2.size(0) != M2.size(1)) throw std::invalid_argument(
    "tchem::Phaser::alter_ob: The matrix part of M2 must be square");
    if (M1.size(0) != NStates_ || M2.size(0) != NStates_) throw std::invalid_argument(
    "tchem::Phaser::alter_ob: The matrix dimension must be the number of electronic states");
    if (! (at::tensor(M1.sizes()).equal(at::tensor(ref1.sizes()))
        && at::tensor(M2.sizes()).equal(at::tensor(ref2.sizes())))
    ) throw std::invalid_argument(
    "tchem::Phaser::alter_ob: M & ref must share a same shape");
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

// Fix observable `M` by minimizing || M - ref ||_F^2
at::Tensor Phaser::fix_ob(const at::Tensor & M, const at::Tensor & ref) const {
    size_t iphase = iphase_min(M, ref);
    return alter_ob(M, iphase);
}
void Phaser::fix_ob_(at::Tensor & M, const at::Tensor & ref) const {
    size_t iphase = iphase_min(M, ref);
    alter_ob_(M, iphase);
}
// Fix observables `M1` and `M2` by minimizing weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
std::tuple<at::Tensor, at::Tensor> Phaser::fix_ob(const at::Tensor & M1, const at::Tensor & M2,
const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const {
    size_t iphase = iphase_min(M1, M2, ref1, ref2, weight);
    return std::make_tuple(alter_ob(M1, iphase), alter_ob(M2, iphase));
}
void Phaser::fix_ob_(at::Tensor & M1, at::Tensor & M2,
const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const {
    size_t iphase = iphase_min(M1, M2, ref1, ref2, weight);
    alter_ob_(M1, iphase);
    alter_ob_(M2, iphase);
}

} // namespace chem
} // namespace tchem