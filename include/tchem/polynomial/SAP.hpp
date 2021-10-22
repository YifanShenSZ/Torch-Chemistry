#ifndef tchem_polynomial_SAP_hpp
#define tchem_polynomial_SAP_hpp

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace tchem { namespace polynomial {

// symmetry adapted polynomial SAP(x) =
// x[coords_[0].first][coords_[0].second] * x[coords_[1].first][coords_[1].second] * ... * x[coords_.back().first][coords_.back().second]
// where std::vector<at::Tensor> x is a symmetry adapted monomial set
class SAP {
    private:
        // the irreducibles and the indices of the coordinates constituting the polynomial, sorted descendingly
        std::vector<std::pair<size_t, size_t>> coords_;

        // the unique coordinates and their orders
        std::vector<std::pair<std::pair<size_t, size_t>, size_t>> uniques_orders_;
    public:
        SAP();
        SAP(const std::vector<std::pair<size_t, size_t>> & _coords);
        // For example, the input line of a 2nd order term made up by
        // the 3rd coordinate in the 4th irreducible and
        // the 1st coordinate in the 2nd irreducible is:
        //     2    4,3    2,1
        // The splitted input line is taken in as `strs`
        SAP(const std::vector<std::string> & strs);
        ~SAP();

        const std::vector<std::pair<size_t, size_t>> & coords() const;
        const std::vector<std::pair<std::pair<size_t, size_t>, size_t>> & uniques_orders() const;

        const std::pair<size_t, size_t> & operator[](const size_t & index) const;

        size_t order() const;
        void pretty_print(std::ostream & stream) const;

        // Return the symmetry adapted polynomial value SAP(x) given x
        at::Tensor operator()(const std::vector<at::Tensor> & xs) const;
        // Return dP(x) / dx given x
        std::vector<at::Tensor> gradient(const std::vector<at::Tensor> & xs) const;
        std::vector<at::Tensor> gradient_(const std::vector<at::Tensor> & xs) const;
        // Return dP(x) / dx given x
        // `grad` harvests the concatenated symmetry adapted gradients
        std::vector<at::Tensor> gradient_(const std::vector<at::Tensor> & xs, at::Tensor & grad) const;
        // Return ddP(x) / dx^2 given x
        CL::utility::matrix<at::Tensor> Hessian(const std::vector<at::Tensor> & xs) const;
        CL::utility::matrix<at::Tensor> Hessian_(const std::vector<at::Tensor> & xs) const;
        // Return ddP(x) / dx^2 given x
        // `hess` harvests the concatenated symmetry adapted Hessians
        CL::utility::matrix<at::Tensor> Hessian_(const std::vector<at::Tensor> & xs, at::Tensor & hess) const;
};

} // namespace polynomial
} // namespace tchem

#endif