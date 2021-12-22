#ifndef tchem_chem_phaser_hpp
#define tchem_chem_phaser_hpp

#include <torch/torch.h>

namespace tchem { namespace chem {

// observable matrix off-diagonal elements do not have determinate phase
// because the basis eigenstates have indeterminate phase
class Phaser {
    // since obervable matrix is real-symmetric, we:
    // * only use the upper triangle
    // * exclude strictly lower triangle from the frobenius norm
    protected:
        // number of electronic states to fix phase
        // for an observable with > NStates states, fix its lowest NStates states
        size_t NStates_;
        // Since the phase of observable matrix off-diagonal elements
        // depends only the phase difference between the basis eigenstates,
        // there are 2^(NStates - 1) possibilities in total
        // The user input matrix serves as the base case and is excluded from trying,
        // so we try out 2^(NStates - 1) - 1 possible phases
        // possible_phases[i] contains one of the phases of NStates electronic states
        // where true means - (flip sign), false means +
        // The phase of the last state is always arbitrarily assigned to +,
        // so possible_phases[i].size() = NStates - 1
        std::vector<std::vector<bool>> possible_phases_;
    public:
        Phaser();
        Phaser(const size_t & _NStates);
        ~Phaser();

        const size_t & NStates() const;

        // alter the phase of eigenstates `U` to the `index`-th possible phase
        at::Tensor alter_states(const at::Tensor & U, const size_t & index) const;
        void alter_states_(at::Tensor & U, const size_t & index) const;

        // alter the phase of observable `M` to the `index`-th possible phase
        at::Tensor alter_ob(const at::Tensor & M, const size_t & index) const;
        void alter_ob_(at::Tensor & M, const size_t & index) const;

        // return the index of the possible phase who minimizes || M - ref ||_F^2
        // return -1 if no need to change phase
        size_t iphase_min(const at::Tensor & M, const at::Tensor & ref) const;
        // `diff` harvests the minimal || M - ref ||_F^2
        size_t iphase_min(double & diff, const at::Tensor & M, const at::Tensor & ref) const;
        // return the index of the possible phase who minimizes weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
        // return -1 if no need to change phase
        size_t iphase_min(const at::Tensor & M1, const at::Tensor & M2,
                          const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const;
        // `diff` harvests the minimal weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
        size_t iphase_min(double & diff, const at::Tensor & M1, const at::Tensor & M2,
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