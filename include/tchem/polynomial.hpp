// A polynomial operation library

#ifndef tchem_polynomial_hpp
#define tchem_polynomial_hpp

#include <torch/torch.h>

namespace tchem { namespace polynomial {

// Polynomial P(r) = r[coords_[0]] * r[coords_[1]] * ... * r[coords_.back()]
class Polynomial {
    private:
        // Coordinates constituting the polynomial, sorted descendingly
        std::vector<size_t> coords_;
    public:
        Polynomial();
        Polynomial(const std::vector<size_t> & coords, const bool & sorted = false);
        ~Polynomial();

        inline std::vector<size_t> coords() const {return coords_;}

        // Return the polynomial value P(r) given r
        at::Tensor operator()(const at::Tensor & r) const;

        // Return the unique coordinates and their orders
        std::tuple<std::vector<size_t>, std::vector<size_t>> uniques_orders() const;
};

// Polynomial set {P(r)}
class PolynomialSet {
    private:
        // Dimension of the coordinate constituting the polynomial set
        size_t dimension_;
        // Highest order among the polynomials
        size_t order_;

        // Polynomials constituting the set, requirements:
        //     1. orders are sorted ascendingly
        //     2. same order terms are sorted ascendingly
        // e.g. 2-dimensional 2nd-order: 1, r0, r1, r0 r0, r1 r0, r1 r1
        std::vector<Polynomial> polynomials_;
        // A view to `polynomials_` grouped by order
        std::vector<std::vector<Polynomial *>> orders_;

        // Construct `orders_` after `polynomials_` has been constructed
        void create_orders();

        // Given a set of coordiantes constituting a polynomial,
        // try to locate its index within [lower, upper)
        void bisect(const std::vector<size_t> coords, const size_t & lower, const size_t & upper, int & index) const;
        // Given a set of coordiantes constituting a polynomial,
        // find its index in this polynomial set
        // If not found, return -1
        int index_polynomial(const std::vector<size_t> coords) const;
    public:
        PolynomialSet();
        // Generate all possible terms up to `order`-th order constituting of all `dimension` coordinates
        PolynomialSet(const size_t & _dimension, const size_t & _order);
        ~PolynomialSet();

        inline size_t dimension() const {return dimension_;}
        inline size_t order() const {return order_;}
        inline std::vector<Polynomial> polynomials() const {return polynomials_;}
        inline std::vector<std::vector<Polynomial *>> orders() const {return orders_;}

        // Return the value of each term in {P(r)} as a vector
        at::Tensor operator()(const at::Tensor & r) const;

        // Given `x`, the value of each term in {P(r)} as a vector
        // Return views to `x` grouped by order
        std::vector<at::Tensor> views(const at::Tensor & x) const;

        // Consider coordinate rotation q = U^T . r
        // so the polynomial set transforms as {P(r)} = T . {P(q)}
        // Assuming:
        //     1. All 0th and 1st order terms are present
        //     2. Polynomial.coords are sorted
        // Return transformation matrix T
        at::Tensor rotation(const at::Tensor & U, const PolynomialSet & q_set) const;
        // Assuming terms are the same under rotation
        at::Tensor rotation(const at::Tensor & U) const;

        // Consider coordinate translation q = r - a
        // so the polynomial set transforms as {P(r)} = T . {P(q)}
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