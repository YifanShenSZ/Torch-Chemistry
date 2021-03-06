#ifndef tchem_gaussian_hpp
#define tchem_gaussian_hpp

#include <random>

#include <torch/torch.h>

#include <tchem/polynomial.hpp>

namespace tchem {

// Gaussian function g(r; miu, var) = (2pi)^(-dim/2) |var|^(-1/2) exp[-1/2 (r-miu)^T.var^-1.(r-miu)]
// where `dim` is the dimension of `r`
class Gaussian {
    private:
        // mean, variance
        at::Tensor miu_, var_;

        // For gaussian random tensor generation
        bool random_ready_ = false;
        // Eigenvalues and eigenvectors of variance
        at::Tensor eigvals_, eigvecs_;
        // Independent 1-dimensional gaussian distributions
        std::vector<std::normal_distribution<double>> independent_1Dgaussians_;
    public:
        Gaussian();
        // miu_ & var_ are deep copies of _miu & _var
        Gaussian(const at::Tensor & _miu, const at::Tensor & _var);
        ~Gaussian();

        at::Tensor miu() const;
        at::Tensor var() const;
        bool random_ready() const;

        // g(r; miu, var) = (2pi)^(-dim/2) |var|^(-1/2) exp[-1/2 (r-miu)^T.var^-1.(r-miu)]
        at::Tensor operator()(const at::Tensor & r) const;
        // g1(r; miu1, var1) * g2(r; miu2, var2) = c * g3(r; miu3, var3)
        // Return c and g3
        std::tuple<at::Tensor, Gaussian> operator*(const Gaussian & g2) const;

        Gaussian clone() const;

        // Intgerate[g(r; miu, var) * {P(r)}, {r, -Infinity, Infinity}]
        // {P(r)} is specified by `set`
        // The evaluation procedure for integrals is:
        // 1. diagonalize `var`
        // 2. transform `r` to miu-centred && `var`-diagonalized (normal) coordinate
        // 3. evaluate integrals
        // 4. transform integrals back to original coordinate
        // The necessary integrals in normal coordinate are specified in `normal_set`
        at::Tensor integral(const polynomial::PolynomialSet & set, const polynomial::PolynomialSet & normal_set) const;
        // Assuming terms are the same under transformation
        at::Tensor integral(const polynomial::PolynomialSet & set) const;

        // Initialize gaussian random tensor generation
        void rand_init();
        // Return a gaussian random tensor
        at::Tensor rand(std::default_random_engine & generator);
};

} // namespace tchem

#endif