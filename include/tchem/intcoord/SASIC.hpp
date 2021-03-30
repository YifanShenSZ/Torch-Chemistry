#ifndef tchem_IC_SASIC_hpp
#define tchem_IC_SASIC_hpp

#include <torch/torch.h>

namespace tchem { namespace IC {

// A symmetry adapted and scaled internal coordinate
class SASIC {
    private:
        // linear combination coefficient - index of scaled internal coordinate pairs
        std::vector<std::pair<double, size_t>> coeff_indices_;
    public:
        SASIC();
        ~SASIC();

        const std::vector<std::pair<double, size_t>> & coeff_indices() const;

        // Append a linear combination coefficient - index of scaled internal coordinate pair
        void append(const std::pair<double, size_t> & coeff_index);
        void append(const double & coeff, const size_t & index);

        // Normalize linear combination coefficients
        void normalize();

        // Return the symmetry adapted and scaled internal coordinate
        // given the scaled internal coordinate vector
        at::Tensor operator()(const at::Tensor & SIC) const;
};

} // namespace IC
} // namespace tchem

#endif