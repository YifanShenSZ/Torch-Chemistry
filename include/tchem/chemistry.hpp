#ifndef tchem_chemistry_hpp
#define tchem_chemistry_hpp

#include <torch/torch.h>

namespace tchem { namespace chem {

// Check if two energy levels are closer than the threshold
bool check_degeneracy(const at::Tensor & energy, const double & thresh);

// Transform Hamiltonian (or energy) and gradient to composite representation
// which is defined by diagonalizing tchem::linalg::sy3matdotmul(dH, dH)
// Return composite Hamiltonian and gradient
// Only read the "upper triangle" (i <= j) of H and dH
// Only write the "upper triangle" (i <= j) of the output tensor
std::tuple<at::Tensor, at::Tensor> composite_representation(const at::Tensor & H, const at::Tensor & dH);
// Only read/write the "upper triangle" (i <= j) of H and dH
void composite_representation_(at::Tensor & H, at::Tensor & dH);
// dot product defined with a metric S
// S must be real symmetric positive definite
// Warning: All elements of S will be read because of torch::mv
//          Will fix it some day if pytorch introduces "symv" like BLAS
std::tuple<at::Tensor, at::Tensor> composite_representation(const at::Tensor & H, const at::Tensor & dH, const at::Tensor & S);
void composite_representation_(at::Tensor & H, at::Tensor & dH, const at::Tensor & S);

// Observable matrix off-diagonal elements do not have determinate phase
// because the basis eigenstates have indeterminate phase
class Phaser {
    private:
        // number of electronic states
        size_t NStates_;
        // Since the phase of observable matrix off-diagonal elements
        // depends only the phase difference between the basis eigenstates,
        // there are 2^(NStates - 1) possibilities in total
        // The user input matrix serves as the base case and is excluded from trying,
        // so possible_phases.size() = 2^(NStates - 1) - 1
        // possible_phases[i] contains one of the phases of NStates electronic states
        // where true means - (flip sign), false means +
        // The phase of the last state is always arbitrarily assigned to +,
        // so possible_phases[i].size() = NStates - 1
        std::vector<std::vector<bool>> possible_phases_;
    public:
        Phaser();
        Phaser(const size_t & _NStates);
        ~Phaser();

        size_t NStates() const;
        std::vector<std::vector<bool>> possible_phases() const;

        // Alter the phase of eigenstates `U` to the `index`-th possible phase
        at::Tensor alter_states (const at::Tensor & U, const size_t & index) const;
        void       alter_states_(      at::Tensor & U, const size_t & index) const;
        // Alter the phase of observable `M` to the `index`-th possible phase
        at::Tensor alter_ob (const at::Tensor & M, const size_t & index) const;
        void       alter_ob_(      at::Tensor & M, const size_t & index) const;

        // Return the index of the possible phase who minimizes || M - ref ||_F^2
        // Return -1 if no need to change phase
        size_t iphase_min(const at::Tensor & M, const at::Tensor & ref) const;
        // Return the index of the possible phase who minimizes weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
        // Return -1 if no need to change phase
        size_t iphase_min(const at::Tensor & M1, const at::Tensor & M2,
                          const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const;

        // Fix observable `M` by minimizing || M - ref ||_F^2
        at::Tensor fix_ob (const at::Tensor & M, const at::Tensor & ref) const;
        void       fix_ob_(      at::Tensor & M, const at::Tensor & ref) const;
        // Fix observables `M1` and `M2` by minimizing weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
        std::tuple<at::Tensor, at::Tensor> fix_ob (const at::Tensor & M1, const at::Tensor & M2,
                                                   const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const;
        void                               fix_ob_(      at::Tensor & M1,       at::Tensor & M2,
                                                   const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const;
};

} // namespace chem
} // namespace tchem

#endif