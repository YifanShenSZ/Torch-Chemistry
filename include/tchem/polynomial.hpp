#ifndef tchem_polynomial_hpp
#define tchem_polynomial_hpp

#include <torch/torch.h>

namespace tchem { namespace polynomial {

// Polynomial P(x) = x[coords_[0]] * x[coords_[1]] * ... * x[coords_.back()]
class Polynomial {
    private:
        // The indices of the coordinates constituting the polynomial, sorted descendingly
        std::vector<size_t> coords_;
    public:
        Polynomial();
        Polynomial(const std::vector<size_t> & _coords, const bool & sorted = false);
        ~Polynomial();

        std::vector<size_t> coords() const;

        size_t order() const;

        // Return the unique coordinates and their orders
        std::tuple<std::vector<size_t>, std::vector<size_t>> uniques_orders() const;

        // Return the polynomial value P(x) given x
        at::Tensor operator()(const at::Tensor & x) const;
        // Return dP(x) / dx given x
        at::Tensor gradient(const at::Tensor & x) const;
};

// Polynomial set {P(x)}
class PolynomialSet {
    private:
        // Polynomials constituting the set, requirements:
        //     1. orders are sorted ascendingly
        //     2. same order terms are sorted ascendingly, 
        //        where the comparison is made from the last coordinate to the first
        // e.g. 2-dimensional 2nd-order: 1, x0, x1, x0 x0, x1 x0, x1 x1
        std::vector<Polynomial> polynomials_;
        // Dimension of the coordinate system constituting the polynomial set
        size_t dimension_;

        // Highest order among the polynomials
        size_t max_order_;
        // A view to `polynomials_` grouped by order
        std::vector<std::vector<const Polynomial *>> orders_;

        // Construct `max_order_` and `orders_` based on constructed `polynomials_`
        void construct_orders_();

        // Given a set of coordiantes constituting a polynomial,
        // try to locate its index within [lower, upper]
        void bisect_(const std::vector<size_t> coords, const size_t & lower, const size_t & upper, int64_t & index) const;
        // Given a set of coordiantes constituting a polynomial,
        // find its index in this polynomial set
        // If not found, return -1
        int64_t index_polynomial_(const std::vector<size_t> coords) const;
    public:
        PolynomialSet();
        // `_polynomials` must meet the requirements of `polynomials_`
        PolynomialSet(const std::vector<Polynomial> & _polynomials, const size_t & _dimension);
        // Generate all possible terms up to `order`-th order constituting of all `dimension` coordinates
        PolynomialSet(const size_t & _dimension, const size_t & _order);
        ~PolynomialSet();

        const std::vector<Polynomial> & polynomials() const;
        const size_t & dimension() const;
        const size_t & max_order() const;
        const std::vector<std::vector<const Polynomial *>> & orders() const;

        // Given `x`, the value of each term in {P(x)} as a vector
        // Return views to `x` grouped by order
        std::vector<at::Tensor> views(const at::Tensor & x) const;

        // Return the value of each term in {P(x)} given x as a vector
        at::Tensor operator()(const at::Tensor & x) const;
        // Return d{P(x)} / dx given x
        at::Tensor Jacobian(const at::Tensor & x) const;

        // Consider coordinate rotation y = U^-1 . x
        // so the polynomial set transforms as {P(x)} = T . {P(y)}
        // Assuming:
        //     1. All 0th and 1st order terms are present
        //     2. Polynomial.coords are sorted
        // Return transformation matrix T
        at::Tensor rotation(const at::Tensor & U, const PolynomialSet & q_set) const;
        // Assuming terms are the same under rotation
        at::Tensor rotation(const at::Tensor & U) const;

        // Consider coordinate translation y = x - a
        // so the polynomial set transforms as {P(x)} = T . {P(y)}
        // Assuming:
        //     1. All 0th and 1st order terms are present
        // Return transformation matrix T
        at::Tensor translation(const at::Tensor & a, const PolynomialSet & q_set) const;
        // Assuming terms are the same under translation
        at::Tensor translation(const at::Tensor & a) const;
};

} // namespace polynomial
} // namespace tchem

#endif