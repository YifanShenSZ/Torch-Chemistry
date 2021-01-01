#ifndef tchem_gaussian_hpp
#define tchem_gaussian_hpp

#include <torch/torch.h>

#include <tchem/polynomial.hpp>

namespace tchem { namespace gaussian {

// Gaussian function g(r; miu, var) = (2pi)^(-dim/2) |var|^(-1/2) exp[-1/2 (r-miu)^T.var^-1.(r-miu)]
// where `dim` is the dimension of `r`
class Gaussian {
    private:
        // mean, variance
        at::Tensor miu_, var_;
    public:
        Gaussian();
        Gaussian(const at::Tensor & _miu, const at::Tensor & _var);
        ~Gaussian();

        inline at::Tensor miu() const {return miu_;}
        inline at::Tensor var() const {return var_;}

        // g(r; miu, var) = (2pi)^(-dim/2) |var|^(-1/2) exp[-1/2 (r-miu)^T.var^-1.(r-miu)]
        at::Tensor operator()(const at::Tensor & r) const;
        // g1(r; miu1, var1) * g2(r; miu2, var2) = c * g3(r; miu3, var3)
        // Return c and g3
        std::tuple<at::Tensor, Gaussian> operator*(const Gaussian & g2) const;

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
};

} // namespace gaussian
} // namespace tchem

#endif