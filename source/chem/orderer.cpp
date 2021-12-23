#include <tchem/chem/orderer.hpp>

namespace tchem { namespace chem {

Orderer::Orderer() {}
Orderer::Orderer(const size_t & _NStates) : NStates_(_NStates) {
    // there must be at least 2 states to have 'ordering'
    if (_NStates < 2) return;
    // generate possible permutations
    std::vector<size_t> base(_NStates);
    std::iota(base.begin(), base.end(), 0);
    while (std::next_permutation(base.begin(), base.end())) permutations_.push_back(base);
    // construct necessary phasers
    phasers_.resize(_NStates + 1);
    for (size_t i = 0; i < phasers_.size(); i++) phasers_[i] = std::make_shared<tchem::chem::Phaser>(i);
}
Orderer::~Orderer() {}

// alter the ordering of eigenvalues `eigvals` to the `ipermutation`-th possible permutation
at::Tensor Orderer::alter_eigvals(const at::Tensor & eigvals, const size_t & ipermutation) const {
    if (eigvals.sizes().size() != 1) throw std::invalid_argument(
    "tchem::Orderer::alter_eigvals: eigvals must be a vector");
    if (eigvals.size(0) != NStates_) throw std::invalid_argument(
    "tchem::Orderer::alter_eigvals: The number of eigenvalues must be the number of electronic states");
    if (ipermutation >= permutations_.size()) return eigvals;

    const auto & permutation = permutations_[ipermutation];
    at::Tensor result = eigvals.new_empty(eigvals.sizes());
    for (size_t i = 0; i < NStates_; i++) result[i] = eigvals[permutation[i]];
    return result;
}
void Orderer::alter_eigvals_(at::Tensor & eigvals, const size_t & ipermutation) const {
    if (eigvals.sizes().size() != 1) throw std::invalid_argument(
    "tchem::Orderer::alter_eigvals_: eigvals must be a vector");
    if (eigvals.size(0) != NStates_) throw std::invalid_argument(
    "tchem::Orderer::alter_eigvals_: The number of eigenvalues must be the number of electronic states");
    if (ipermutation >= permutations_.size()) return;

    const auto & permutation = permutations_[ipermutation];
    at::Tensor eigvalssave = eigvals.clone();
    for (size_t i = 0; i < NStates_; i++) eigvals[i].copy_(eigvalssave[permutation[i]]);
}

// alter the ordering of eigenstates `U` to the `ipermutation`-th possible permutation,
at::Tensor Orderer::alter_states(const at::Tensor & U, const size_t & ipermutation) const {
    if (U.sizes().size() != 2) throw std::invalid_argument(
    "tchem::Orderer::alter_states: U must be a matrix");
    if (U.size(1) != NStates_) throw std::invalid_argument(
    "tchem::Orderer::alter_states: The number of columns must be the number of electronic states");
    if (ipermutation >= permutations_.size()) return U;

    const auto & permutation = permutations_[ipermutation];
    at::Tensor result = U.new_empty(U.sizes());
    for (size_t i = 0; i < NStates_; i++) result.select(1, i) = U.select(1, permutation[i]);
    return result;
}
void Orderer::alter_states_(at::Tensor & U, const size_t & ipermutation) const {
    if (U.sizes().size() != 2) throw std::invalid_argument(
    "tchem::Orderer::alter_states_: U must be a matrix");
    if (U.size(1) != NStates_) throw std::invalid_argument(
    "tchem::Orderer::alter_states_: The number of columns must be the number of electronic states");
    if (ipermutation >= permutations_.size()) return;

    const auto & permutation = permutations_[ipermutation];
    at::Tensor Usave = U.clone();
    for (size_t i = 0; i < NStates_; i++) U.select(1, i).copy_(Usave.select(1, permutation[i]));
}
// also alter the phase to `iphase`-th possible phase based on an `NRefStates`-state reference
at::Tensor Orderer::alter_states(const at::Tensor & U, const size_t & ipermutation, const size_t & NRefStates, const size_t & iphase) const {
    at::Tensor step1 = alter_states(U, ipermutation);
    at::Tensor step2 = phasers_[NRefStates]->alter_states(step1, iphase);
    return step2;
}
void Orderer::alter_states_(at::Tensor & U, const size_t & ipermutation, const size_t & NRefStates, const size_t & iphase) const {
    alter_states_(U, ipermutation);
    phasers_[NRefStates]->alter_states_(U, iphase);
}

// alter the ordering of observable `M` to the `ipermutation`-th possible permutation
at::Tensor Orderer::alter_ob(const at::Tensor & M, const size_t & ipermutation) const {
    if (M.sizes().size() < 2) throw std::invalid_argument(
    "tchem::Orderer::alter_ob: M must be a matrix or higher order tensor");
    if (M.size(0) != M.size(1)) throw std::invalid_argument(
    "tchem::Orderer::alter_ob: The matrix part of M must be square");
    if (M.size(0) != NStates_) throw std::invalid_argument(
    "tchem::Orderer::alter_ob: The matrix dimension must be the number of electronic states");
    if (ipermutation >= permutations_.size()) return M;

    const auto & permutation = permutations_[ipermutation];
    at::Tensor result = M.new_empty(M.sizes());
    for (size_t i = 0; i < NStates_; i++) {
        size_t oldi = permutation[i];
        result[i][i] = M[oldi][oldi];
        for (size_t j = i + 1; j < NStates_; j++) {
            size_t oldj = permutation[j];
            size_t row, col;
            if (oldi < oldj) {row = oldi; col = oldj;}
            else             {row = oldj; col = oldi;}
            result[i][j] = M[row][col];
        }
    }
    return result;
}
void Orderer::alter_ob_(at::Tensor & M, const size_t & ipermutation) const {
    if (M.sizes().size() < 2) throw std::invalid_argument(
    "tchem::Orderer::alter_ob_: M must be a matrix or higher order tensor");
    if (M.size(0) != M.size(1)) throw std::invalid_argument(
    "tchem::Orderer::alter_ob_: The matrix part of M must be square");
    if (M.size(0) != NStates_) throw std::invalid_argument(
    "tchem::Orderer::alter_ob_: The matrix dimension must be the number of electronic states");
    if (ipermutation >= permutations_.size()) return;

    const auto & permutation = permutations_[ipermutation];
    at::Tensor Msave = M.clone();
    for (size_t i = 0; i < NStates_; i++) {
        size_t oldi = permutation[i];
        M[i][i].copy_(Msave[oldi][oldi]);
        for (size_t j = i + 1; j < NStates_; j++) {
            size_t oldj = permutation[j];
            size_t row, col;
            if (oldi < oldj) {row = oldi; col = oldj;}
            else             {row = oldj; col = oldi;}
            M[i][j].copy_(Msave[row][col]);
        }
    }
}
// also alter the phase to `iphase`-th possible phase based on an `NRefStates`-state reference
at::Tensor Orderer::alter_ob(const at::Tensor & M, const size_t & ipermutation, const size_t & NRefStates, const size_t & iphase) const {
    at::Tensor step1 = alter_ob(M, ipermutation);
    at::Tensor step2 = phasers_[NRefStates]->alter_ob(step1, iphase);
    return step2;
}
void Orderer::alter_ob_(at::Tensor & M, const size_t & ipermutation, const size_t & NRefStates, const size_t & iphase) const {
    alter_ob_(M, ipermutation);
    phasers_[NRefStates]->alter_ob_(M, iphase);
}

// return the index of the possible permutation and phase who minimizes || M - ref ||_F^2
// return -1 if no need to change permutation or phase
std::tuple<size_t, size_t> Orderer::ipermutation_iphase_min(const at::Tensor & M, const at::Tensor & ref) const {
    if (M.sizes().size() < 2) throw std::invalid_argument(
    "tchem::Orderer::ipermutation_iphase_min: M must be a matrix or higher order tensor");
    if (M.size(0) != M.size(1)) throw std::invalid_argument(
    "tchem::Orderer::ipermutation_iphase_min: The matrix part of M must be square");
    if (M.size(0) != NStates_) throw std::invalid_argument(
    "tchem::Orderer::ipermutation_iphase_min: The matrix dimension must be the number of electronic states");

    size_t NRefStates = ref.size(0);
    const auto & phaser = phasers_[NRefStates];
    // for user input permutation, check its phase
    at::Tensor Mslice = M.slice(0, 0, NRefStates).slice(1, 0, NRefStates);
    if (! at::tensor(Mslice.sizes()).equal(at::tensor(ref.sizes()))) throw std::invalid_argument(
    "tchem::Orderer::ipermutation_iphase_min: M & ref must share a same non-electronic shape");
    size_t ipermutation_min = -1;
    double diff_min;
    size_t iphase_min = phaser->iphase_min(diff_min, Mslice, ref);
    // try out permutation possibilities
    for (size_t ipermutation = 0; ipermutation < permutations_.size(); ipermutation++) {
        at::Tensor Mperm = alter_ob(M, ipermutation);
        at::Tensor Mslice = Mperm.slice(0, 0, NRefStates).slice(1, 0, NRefStates);
        double diff;
        size_t iphase = phaser->iphase_min(diff, Mslice, ref);
        // prefer not to change unless considerable difference
        if (diff < diff_min - 1e-2) {
            diff_min = diff;
            ipermutation_min = ipermutation;
            iphase_min = iphase;
        }
    }
    return std::make_tuple(ipermutation_min, iphase_min);
}
// return the index of the possible permutation and phase who minimizes weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
// return -1 if no need to change permutation or phase
std::tuple<size_t, size_t> Orderer::ipermutation_iphase_min(const at::Tensor & M1, const at::Tensor & M2, const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const {
    if (M1.sizes().size() < 2) throw std::invalid_argument(
    "tchem::Orderer::ipermutation_iphase_min: M1 must be a matrix or higher order tensor");
    if (M1.size(0) != M1.size(1)) throw std::invalid_argument(
    "tchem::Orderer::ipermutation_iphase_min: The matrix part of M1 must be square");
    if (M2.sizes().size() < 2) throw std::invalid_argument(
    "tchem::Orderer::ipermutation_iphase_min: M2 must be a matrix or higher order tensor");
    if (M2.size(0) != M2.size(1)) throw std::invalid_argument(
    "tchem::Orderer::ipermutation_iphase_min: The matrix part of M2 must be square");
    if (M1.size(0) != M2.size(0) || M1.size(0) != NStates_) throw std::invalid_argument(
    "tchem::Orderer::ipermutation_iphase_min: The matrix dimension must be the number of electronic states");

    size_t NRefStates = ref1.size(0);
    if (ref1.size(0) != ref2.size(0)) throw std::invalid_argument(
    "tchem::Orderer::ipermutation_iphase_min: The references must share a same number of electronic states");
    const auto & phaser = phasers_[NRefStates];
    // for user input permutation, check its phase
    at::Tensor M1slice = M1.slice(0, 0, NRefStates).slice(1, 0, NRefStates),
               M2slice = M2.slice(0, 0, NRefStates).slice(1, 0, NRefStates);
    if (! (at::tensor(M1slice.sizes()).equal(at::tensor(ref1.sizes()))
        && at::tensor(M2slice.sizes()).equal(at::tensor(ref2.sizes())))
    ) throw std::invalid_argument(
    "tchem::Orderer::alter_ob: M & ref must share a same non-electronic shape");
    size_t ipermutation_min = -1;
    double diff_min;
    size_t iphase_min = phaser->iphase_min(diff_min, M1slice, M2slice, ref1, ref2, weight);
    // try out permutation possibilities
    for (size_t ipermutation = 0; ipermutation < permutations_.size(); ipermutation++) {
        at::Tensor M1perm = alter_ob(M1, ipermutation),
                   M2perm = alter_ob(M2, ipermutation);
        at::Tensor M1slice = M1perm.slice(0, 0, NRefStates).slice(1, 0, NRefStates),
                   M2slice = M2perm.slice(0, 0, NRefStates).slice(1, 0, NRefStates);
        double diff;
        size_t iphase = phaser->iphase_min(diff, M1slice, M2slice, ref1, ref2, weight);
        // prefer not to change unless considerable difference
        if (diff < diff_min - 1e-2) {
            diff_min = diff;
            ipermutation_min = ipermutation;
            iphase_min = iphase;
        }
    }
    return std::make_tuple(ipermutation_min, iphase_min);
}

// fix observable `M` by minimizing || M - ref ||_F^2
at::Tensor Orderer::fix_ob(const at::Tensor & M, const at::Tensor & ref) const {
    size_t ipermutation, iphase;
    std::tie(ipermutation, iphase) = ipermutation_iphase_min(M, ref);
    return alter_ob(M, ipermutation, ref.size(0), iphase);
}
void Orderer::fix_ob_(at::Tensor & M, const at::Tensor & ref) const {
    size_t ipermutation, iphase;
    std::tie(ipermutation, iphase) = ipermutation_iphase_min(M, ref);
    alter_ob_(M, ipermutation, ref.size(0), iphase);
}
// fix observables `M1` and `M2` by minimizing weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
std::tuple<at::Tensor, at::Tensor> Orderer::fix_ob(const at::Tensor & M1, const at::Tensor & M2,
const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const {
    size_t ipermutation, iphase;
    std::tie(ipermutation, iphase) = ipermutation_iphase_min(M1, M2, ref1, ref2, weight);
    return std::make_tuple(alter_ob(M1, ipermutation, ref1.size(0), iphase),
                           alter_ob(M2, ipermutation, ref2.size(0), iphase));
}
void Orderer::fix_ob_(at::Tensor & M1, at::Tensor & M2,
const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const {
    size_t ipermutation, iphase;
    std::tie(ipermutation, iphase) = ipermutation_iphase_min(M1, M2, ref1, ref2, weight);
    alter_ob_(M1, ipermutation, ref1.size(0), iphase);
    alter_ob_(M2, ipermutation, ref2.size(0), iphase);
}

} // namespace chem
} // namespace tchem