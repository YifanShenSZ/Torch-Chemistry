#ifndef tchem_polynomial_SAPSet_hpp
#define tchem_polynomial_SAPSet_hpp

#include <tchem/polynomial/SAP.hpp>

namespace tchem { namespace polynomial {

// symmetry adapted polynomial set {SAP(x)}
class SAPSet {
    private:
        // symmetry adapted polynomials constituting the set
        // If only desire the basic evaluations (value, Jacobian, ...), the SAPs can be stored in any order;
        // otherwise, the advanced functionalities (rotation, translation, ...) requires:
        // 1. orders are sorted ascendingly
        // 2. same order terms are sorted ascendingly
        //    where the comparison is made from the last coordinate to the first
        // e.g. 2-irreducible 2-dimensional 2nd-order: 1, x00, x01, x00 x00, x01 x00, x01 x01, x10 x10, x11 x10, x11 x11
        std::vector<SAP> SAPs_;
        // irreducible of this symmetry adapted polynomial set
        size_t irreducible_;
        // dimensions per irreducible of the coordinate system constituting the polynomial set
        std::vector<size_t> dimensions_;

        // highest order among the polynomials
        size_t max_order_;
        // a view to `polynomials_` grouped by order
        std::vector<std::vector<const SAP *>> orders_;

        // Construct `max_order_` and `orders_` based on constructed `polynomials_`
        void construct_orders_();

        // Given a set of coordiantes constituting a SAP, return its index in this SAP set
        // Return -1 if not found
        int64_t index_SAP_(const std::vector<std::pair<size_t, size_t>> coords) const;
    public:
        SAPSet();
        // `sapoly_file` contains one SAP per line, who must meet the requirements of `SAPs_`
        SAPSet(const std::string & sapoly_file, const size_t & _irreducible, const std::vector<size_t> & _dimensions);
        ~SAPSet();

        // Insert a 0th order (const) term if the set is totally symmetic and does not have const yet
        void insert_const();

        const std::vector<SAP> & SAPs() const;

        // Read-only reference to a symmetry adapted polynomial
        const SAP & operator[](const size_t & index) const;

        void pretty_print(std::ostream & stream) const;

        // Return the value of each term in {P(x)} as a vector
        at::Tensor operator()(const std::vector<at::Tensor> & xs) const;
        // Return d{P(x)} / dx given x
        std::vector<at::Tensor> Jacobian(const std::vector<at::Tensor> & xs) const;
        std::vector<at::Tensor> Jacobian_(const std::vector<at::Tensor> & xs) const;
        // Return d{P(x)} / dx given x
        // `J` harvests the concatenated symmetry adapted Jacobians
        std::vector<at::Tensor> Jacobian_(const std::vector<at::Tensor> & xs, at::Tensor & J) const;
        // Return dd{P(x)} / dx^2 given x
        CL::utility::matrix<at::Tensor> Jacobian2nd(const std::vector<at::Tensor> & xs) const;
        CL::utility::matrix<at::Tensor> Jacobian2nd_(const std::vector<at::Tensor> & xs) const;
        // Return dd{P(x)} / dx^2 given x
        // `K` harvests the concatenated symmetry adapted 2nd-order Jacobians
        CL::utility::matrix<at::Tensor> Jacobian2nd_(const std::vector<at::Tensor> & xs, at::Tensor & K) const;

        // Consider coordinate rotation y[irred] = U[irred]^-1 . x[irred]
        // so the SAP set rotates as {SAP(x)} = T . {SAP(y)}
        // Assuming:
        // 1. If there are 1st order terms, all are present
        // 2. SAP.coords are sorted
        // Return rotation matrix T
        at::Tensor rotation(const std::vector<at::Tensor> & U, const SAPSet & y_set) const;
        // Assuming terms are the same under rotation
        at::Tensor rotation(const std::vector<at::Tensor> & U) const;

        // Consider coordinate translation y[0] = x[0] - a
        // i.e. only the totally symmetric irreducible translates so that symmetry preserves
        // so the SAP set translates as {SAP(x)} = T . {SAP(y)}
        // Assuming:
        // 1. The totally symmetric irreducible must have the 0th order term
        // 2. If the totally symmetric irreducible has 1st order terms, all are present
        // Return translation matrix T
        at::Tensor translation(const at::Tensor & a, const SAPSet & y_set) const;
        // Assuming terms are the same under translation
        at::Tensor translation(const at::Tensor & a) const;
};

} // namespace polynomial
} // namespace tchem

#endif