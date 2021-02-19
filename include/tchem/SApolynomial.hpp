#ifndef sapoly_sapoly_hpp
#define sapoly_sapoly_hpp

#include <tchem/polynomial.hpp>

namespace tchem { namespace polynomial {

// Symmetry adapted polynomial SAP(x) =
// x[coords_[0].first][coords_[0].second] * x[coords_[1].first][coords_[1].second] * ... * x[coords_.back().first][coords_.back().second]
// where std::vector<at::Tensor> x is a symmetry adapted monomial set
class SAP {
    private:
        // The irreducibles and the indices of the coordinates constituting the polynomial, sorted descendingly
        std::vector<std::pair<size_t, size_t>> coords_;
    public:
        SAP();
        SAP(const std::vector<std::pair<size_t, size_t>> & _coords, const bool & sorted = false);
        // For example, the input line of a 2nd order term made up by
        // the 3rd coordinate in the 4th irreducible and
        // the 1st coordinate in the 2nd irreducible is:
        //     2    4,3    2,1
        // The splitted input line is taken in as `strs`
        SAP(const std::vector<std::string> & strs, const bool & sorted = false);
        ~SAP();

        std::vector<std::pair<size_t, size_t>> coords() const;

        size_t order() const;
        void pretty_print(std::ostream & stream) const;

        // Return the unique coordinates and their orders
        std::tuple<std::vector<std::pair<size_t, size_t>>, std::vector<size_t>> uniques_orders() const;

        // Return the symmetry adapted polynomial value SAP(x) given x
        at::Tensor operator()(const std::vector<at::Tensor> & xs) const;
        // Return dP(x) / dx given x
        std::vector<at::Tensor> gradient(const std::vector<at::Tensor> & xs) const;
};

// Symmetry adapted polynomial set {SAP(x)}
class SAPSet {
    private:
        // Symmetry adapted polynomials constituting the set, requirements:
        //     1. orders are sorted ascendingly
        //     2. same order terms are sorted ascendingly
        //        where the comparison is made from the last coordinate to the first
        // e.g. 2-irreducible 2-dimensional 2nd-order: 1, x00, x01, x00 x00, x01 x00, x01 x01, x10 x10, x11 x10, x11 x11
        std::vector<SAP> SAPs_;
        // Dimensions per irreducible of the coordinate system constituting the polynomial set
        std::vector<size_t> dimensions_;
    public:
        SAPSet();
        // `sapoly_file` contains one SAP per line, who must meet the requirements of `SAPs_`
        SAPSet(const std::string & sapoly_file, const std::vector<size_t> & _dimensions);
        ~SAPSet();

        std::vector<SAP> SAPs() const;

        void pretty_print(std::ostream & stream) const;

        // Return the value of each term in {P(x)} as a vector
        at::Tensor operator()(const std::vector<at::Tensor> & xs) const;
        // Return d{P(x)} / dx given x
        std::vector<at::Tensor> Jacobian(const std::vector<at::Tensor> & xs) const;
};

} // namespace polynomial
} // namespace tchem

#endif