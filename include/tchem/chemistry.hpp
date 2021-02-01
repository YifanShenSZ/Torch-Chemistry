#ifndef tchem_chemistry_hpp
#define tchem_chemistry_hpp

#include <torch/torch.h>

namespace tchem { namespace chem {

// Check if two energy levels are closer than the threshold
bool check_degeneracy(const at::Tensor & energy, const double & thresh);

// Transform Hamiltonian (or energy) and gradient to composite representation
// Return composite Hamiltonian and gradient
// Only read the "upper triangle" (i <= j) of H and dH
// Only write the "upper triangle" (i <= j) of the output tensor
std::tuple<at::Tensor, at::Tensor> composite_representation(const at::Tensor & H, const at::Tensor & dH);
// Only read/write the "upper triangle" (i <= j) of H and dH
void composite_representation_(at::Tensor & H, at::Tensor & dH);

// Observable matrix off-diagonal elements do not have determinate phase
// because the basis eigenvectors have indeterminate phase
class Phaser {
    private:
        // number of electronic states
        size_t NStates_;
        // Since the phase of observable matrix off-diagonal elements
        // depends only the phase difference between the basis eigenvectors,
        // there are 2^(NStates - 1) possibilities in total
        // The user input matrix serves as the base case and is excluded from trying,
        // so possible_phases.size() = 2^(NStates - 1) - 1
        // possible_phases[i] contains one of the phases of NStates electronic states
        // where true means - (flip sign), false means +
        // The phase of the last state is always arbitrarily assigned to +,
        // so possible_phases[i].size() = NStates - 1
        std::vector<std::vector<bool>> possible_phases_;
    public:
        inline Phaser() {}
        Phaser(const size_t & _NStates);
        inline ~Phaser() {}

        inline size_t NStates() const {return NStates_;}
        inline std::vector<std::vector<bool>> possible_phases() const {return possible_phases_;}

        // Alter the phase of M to the index-th possible phase
        at::Tensor alter(const at::Tensor & M, const size_t & index) const;
        void alter_(at::Tensor & M, const size_t & index) const;

        // Return the index of the possible phase who minimizes || M - ref ||_F^2
        // Return -1 if no need to change phase
        size_t iphase_min(const at::Tensor & M, const at::Tensor & ref) const;
        // Return the index of the possible phase who minimizes weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
        // Return -1 if no need to change phase
        size_t iphase_min(const at::Tensor & M1, const at::Tensor & M2, const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const;

        // Fix M by minimizing || M - ref ||_F^2
        at::Tensor fix(const at::Tensor & M, const at::Tensor & ref) const;
        void fix_(at::Tensor & M, const at::Tensor & ref) const;
        // Fix M1 and M2 by minimizing weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
        std::tuple<at::Tensor, at::Tensor> fix(const at::Tensor & M1, const at::Tensor & M2, const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const;
        void fix_(at::Tensor & M1, at::Tensor & M2, const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) const;
};

} // namespace chem
} // namespace tchem

#endif