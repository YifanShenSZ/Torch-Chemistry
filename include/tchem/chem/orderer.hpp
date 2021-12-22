#ifndef tchem_chem_orderer_hpp
#define tchem_chem_orderer_hpp

#include <tchem/chem/phaser.hpp>

namespace tchem { namespace chem {

// observable matrix may need to switch rows and columns
// because the basis eigenstates may not be ordered properly,
// e.g. different calculations may order near degenerate states differently 
class Orderer {
    protected:
        // number of electronic states
        size_t NStates_;
        // There are NStates! permutations of NStates electronic states
        // The user input matrix serves as the base case and is excluded from trying,
        // so we try out NStates! - 1 possible permutations
        std::vector<std::vector<size_t>> permutations_;
        // it is possible to have a reference with < NStates states,
        // so we prepare all necessary phasers
        std::vector<std::shared_ptr<tchem::chem::Phaser>> phasers_;
    public:
        Orderer();
        Orderer(const size_t & _NStates);
        ~Orderer();

        // alter the ordering of eigenstates `U` to the `ipermutation`-th possible permutation
        at::Tensor alter_states(const at::Tensor & U, const size_t & ipermutation) const;
        void alter_states_(at::Tensor & U, const size_t & ipermutation) const;
        // also alter the phase to `iphase`-th possible phase based on an `NRefStates`-state reference
        at::Tensor alter_states(const at::Tensor & U, const size_t & ipermutation, const size_t & NRefStates, const size_t & iphase) const;
        void alter_states_(at::Tensor & U, const size_t & ipermutation, const size_t & NRefStates, const size_t & iphase) const;

        // alter the ordering of observable `M` to the `ipermutation`-th possible permutation
        at::Tensor alter_ob(const at::Tensor & M, const size_t & ipermutation) const;
        void alter_ob_(at::Tensor & M, const size_t & ipermutation) const;
        // also alter the phase to `iphase`-th possible phase based on an `NRefStates`-state reference
        at::Tensor alter_ob(const at::Tensor & M, const size_t & ipermutation, const size_t & NRefStates, const size_t & iphase) const;
        void alter_ob_(at::Tensor & M, const size_t & ipermutation, const size_t & NRefStates, const size_t & iphase) const;

        // return the index of the possible permutation and phase who minimizes || M - ref ||_F^2
        // return -1 if no need to change permutation or phase
        std::tuple<size_t, size_t> ipermutation_iphase_min(const at::Tensor & M, const at::Tensor & ref) const;
        // return the index of the possible permutation and phase who minimizes weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
        // return -1 if no need to change permutation or phase
        std::tuple<size_t, size_t> ipermutation_iphase_min(const at::Tensor & M1, const at::Tensor & M2,
                                                           const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const;

        // fix observable `M` by minimizing || M - ref ||_F^2
        at::Tensor fix_ob(const at::Tensor & M, const at::Tensor & ref) const;
        void fix_ob_(at::Tensor & M, const at::Tensor & ref) const;
        // fix observables `M1` and `M2` by minimizing weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
        std::tuple<at::Tensor, at::Tensor> fix_ob(const at::Tensor & M1, const at::Tensor & M2,
                                                  const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const;
        void fix_ob_(at::Tensor & M1, at::Tensor & M2,
                     const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const;
};

} // namespace chem
} // namespace tchem

#endif