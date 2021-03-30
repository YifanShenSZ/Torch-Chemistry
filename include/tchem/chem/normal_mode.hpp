#ifndef tchem_chem_normal_mode_hpp
#define tchem_chem_normal_mode_hpp

#include <torch/torch.h>

namespace tchem { namespace chem {

// Perform normal mode analysis given mass and Hessian in Cartesian coordiante
// Normal modes are the eigenvectors of mass weighed Hessian
// Lowest 6 modes are considered as translations and rotations thus ruled out
class CartNormalMode {
    protected:
        std::vector<double> masses_;
        at::Tensor Hessian_;

        // whether observables are ready for fetching
        bool ready_ = false;
        // Harmonic frequencies (negative if imaginary)
        at::Tensor frequency_;
        // Cartesian coordinate normal modes in each row (normalized by mass metric)
        at::Tensor cartmode_;
    public:
        CartNormalMode();
        CartNormalMode(const std::vector<double> & _masses, const at::Tensor & _Hessian);
        ~CartNormalMode();

        // Perform normal mode analysis, then observables are ready for fetching
        void kernel();

        // Harmonic frequencies (negative if imaginary)
        const at::Tensor & frequency() const;
        // Cartesian coordinate normal modes (normalized by mass metric)
        const at::Tensor & cartmode () const;
};

// Perform normal mode analysis given mass, Jacobian of internal coordinate over Cartesian coordinate, Hessian in internal coordinate
// Normal modes are the generalized eigenvectors of Hessian under (J . M^-1 . J^T)^-1 metric
class IntNormalMode : public CartNormalMode {
    private:
        at::Tensor Jacobian_;

        // Internal coordinate normal modes in each row (normalized by (J . M^-1 . J^T)^-1 metric)
        at::Tensor intmode_;
        // L^-1 = (intmode_^T)^-1
        at::Tensor Linv_;
    public:
        IntNormalMode();
        IntNormalMode(const std::vector<double> & _masses, const at::Tensor & _Jacobian, const at::Tensor & _Hessian);
        ~IntNormalMode();

        // Perform normal mode analysis, then observables are ready for fetching
        void kernel();

        // Internal coordinate normal modes (normalized by G^-1 metric)
        const at::Tensor & intmode() const;
        // intmode_^-1
        const at::Tensor & Linv   () const;
};

// Perform normal mode analysis given mass, Jacobian of symmetry adapted internal coordinate over Cartesian coordinate, Hessian in symmetry adapted internal coordinate
// Normal modes are the generalized eigenvectors of Hessian under (J . M^-1 . J^T)^-1 metric
class SANormalMode {
    private:
        std::vector<double> masses_;
        std::vector<at::Tensor> Jacobians_, Hessians_;

        // whether observables are ready for fetching
        bool ready_ = false;
        // Harmonic frequencies (negative if imaginary)
        std::vector<at::Tensor> frequencies_;
        // Internal coordinate normal modes in each row (normalized by (J . M^-1 . J^T)^-1 metric)
        std::vector<at::Tensor> intmodes_;
        // L^-1 = (intmode_^T)^-1
        std::vector<at::Tensor> Linvs_;
        // Cartesian coordinate normal modes in each row
        std::vector<at::Tensor> cartmodes_;
    public:
        SANormalMode();
        SANormalMode(const std::vector<double> & _masses, const std::vector<at::Tensor> & _Jacobians, const std::vector<at::Tensor> & _Hessians);
        ~SANormalMode();

        // Perform normal mode analysis, then observables are ready for fetching
        void kernel();

        // Harmonic frequencies (negative if imaginary)
        const std::vector<at::Tensor> & frequencies() const;
        // Internal coordinate normal modes (normalized by (J . M^-1 . J^T)^-1 metric)
        const std::vector<at::Tensor> & intmodes   () const;
        // intmode_^-1
        const std::vector<at::Tensor> & Linvs      () const;
        // Cartesian coordinate normal modes
        const std::vector<at::Tensor> & cartmodes  () const;
};

} // namespace chem
} // namespace tchem

#endif