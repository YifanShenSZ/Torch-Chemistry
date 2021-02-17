#ifndef sapoly_sapoly_hpp
#define sapoly_sapoly_hpp

#include <tchem/polynomial.hpp>

namespace tchem { namespace polynomial {

// Symmetry adapted polynomial SAP(x) =
// x[irreds_[0]][coords_[0]] * x[irreds_[1]][coords_[1]] * ... * x[irreds_.back()][coords_.back()]
// where x is a symmetry adapted monomial set
class SAP {
    private:
        // The irreducibles of the coordinates
        std::vector<size_t> irreds_;
        // Coordinates constituting the polynomial
        std::vector<size_t> coords_;
    public:
        SAP();
        // For example, the input line of a 2nd order term made up by
        // the 1st coordinate in the 2nd irreducible and
        // the 3rd coordinate in the 4th irreducible is:
        //     2    2,1    4,3
        // The splitted input line is taken in as `strs`
        SAP(const std::vector<std::string> & strs);
        ~SAP();

        std::vector<size_t> irreds() const;
        std::vector<size_t> coords() const;

        void pretty_print(std::ostream & stream) const;

        // Return the symmetry adapted polynomial SAP(x) given x
        at::Tensor operator()(const std::vector<at::Tensor> & xs) const;
};

// Symmetry adapted polynomial set {SAP(x)}
class SAPSet {
    private:
        std::vector<SAP> SAPs_;
    public:
        SAPSet();
        // `sapoly_file` contains one SAP per line
        SAPSet(const std::string sapoly_file);
        ~SAPSet();

        std::vector<SAP> SAPs() const;

        void pretty_print(std::ostream & stream) const;

        // Return the value of each term in {P(x)} as a vector
        at::Tensor operator()(const std::vector<at::Tensor> & xs) const;
};

} // namespace polynomial
} // namespace tchem

#endif