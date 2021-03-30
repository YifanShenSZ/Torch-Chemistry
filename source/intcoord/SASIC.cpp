#include <CppLibrary/linalg.hpp>

#include <tchem/intcoord/SASIC.hpp>

namespace tchem { namespace IC {

SASIC::SASIC() {}
SASIC::~SASIC() {}

const std::vector<std::pair<double, size_t>> & SASIC::coeff_indices() const {return coeff_indices_;}

// Append a linear combination coefficient - index of scaled internal coordinate pair
void SASIC::append(const std::pair<double, size_t> & coeff_index) {
    coeff_indices_.push_back(coeff_index);
}
void SASIC::append(const double & coeff, const size_t & index) {
    coeff_indices_.push_back(std::pair<double, size_t>(coeff, index));
}

// Normalize linear combination coefficients
void SASIC::normalize() {
    double norm2 = 0.0;
    for (const auto & coeff_index : coeff_indices_) norm2 += coeff_index.first * coeff_index.first;
    norm2 /= sqrt(norm2);
    for (auto & coeff_index : coeff_indices_) coeff_index.first /= norm2;
}

// Return the symmetry adapted and scaled internal coordinate
// given the scaled internal coordinate vector
at::Tensor SASIC::operator()(const at::Tensor & SIC) const {
    at::Tensor sasic = coeff_indices_[0].first * SIC[coeff_indices_[0].second];
    for (size_t i = 1; i < coeff_indices_.size(); i++)
    sasic = sasic + coeff_indices_[i].first * SIC[coeff_indices_[i].second];
    return sasic;
}

} // namespace IC
} // namespace tchem