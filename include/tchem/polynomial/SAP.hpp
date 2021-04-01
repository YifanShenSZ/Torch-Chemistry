#ifndef tchem_polynomial_SAP_hpp
#define tchem_polynomial_SAP_hpp

#include <torch/torch.h>

namespace tchem { namespace polynomial {

// symmetry adapted polynomial SAP(x) =
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

        const std::vector<std::pair<size_t, size_t>> & coords() const;

        const std::pair<size_t, size_t> & operator[](const size_t & index) const;

        size_t order() const;
        void pretty_print(std::ostream & stream) const;

        // Return the unique coordinates and their orders
        std::tuple<std::vector<std::pair<size_t, size_t>>, std::vector<size_t>> uniques_orders() const;

        // Return the symmetry adapted polynomial value SAP(x) given x
        at::Tensor operator()(const std::vector<at::Tensor> & xs) const;
        // Return dP(x) / dx given x
        std::vector<at::Tensor> gradient(const std::vector<at::Tensor> & xs) const;
};

} // namespace polynomial
} // namespace tchem

#endif