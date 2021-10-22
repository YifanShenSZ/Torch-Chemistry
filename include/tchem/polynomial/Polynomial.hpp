#ifndef tchem_polynomial_Polynomial_hpp
#define tchem_polynomial_Polynomial_hpp

#include <torch/torch.h>

namespace tchem { namespace polynomial {

// polynomial P(x) = x[coords_[0]] * x[coords_[1]] * ... * x[coords_.back()]
class Polynomial {
    private:
        // the indices of the coordinates constituting the polynomial, sorted descendingly
        std::vector<size_t> coords_;

        // the unique coordinates and their orders
        std::vector<std::pair<size_t, size_t>> uniques_orders_;
    public:
        Polynomial();
        Polynomial(const std::vector<size_t> & _coords);
        Polynomial(const std::vector<std::pair<size_t, size_t>> _uniques_orders);
        ~Polynomial();

        const std::vector<size_t> & coords() const;
        const std::vector<std::pair<size_t, size_t>> & uniques_orders() const;

        size_t order() const;

        // Return the polynomial value P(x) given x
        at::Tensor operator()(const at::Tensor & x) const;
        // Return dP(x) / dx given x
        at::Tensor gradient(const at::Tensor & x) const;
        at::Tensor gradient_(const at::Tensor & x) const;
        // Return ddP(x) / dx^2 given x
        at::Tensor Hessian(const at::Tensor & x) const;
        at::Tensor Hessian_(const at::Tensor & x) const;
};

} // namespace polynomial
} // namespace tchem

#endif