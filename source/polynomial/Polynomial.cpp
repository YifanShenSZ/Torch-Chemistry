#include <tchem/polynomial/Polynomial.hpp>

namespace tchem { namespace polynomial {

Polynomial::Polynomial() {}
Polynomial::Polynomial(const std::vector<size_t> & _coords, const bool & sorted)
: coords_(_coords) {
    if (! sorted)
    std::sort(coords_.begin(), coords_.end(), std::greater<size_t>());
}
Polynomial::~Polynomial() {}

const std::vector<size_t> & Polynomial::coords() const {return coords_;}

size_t Polynomial::order() const {return coords_.size();}

// Return the unique coordinates and their orders
std::tuple<std::vector<size_t>, std::vector<size_t>> Polynomial::uniques_orders() const {
    std::vector<size_t> uniques, orders;
    size_t coord_old = -1;
    for (const size_t & coord : coords_)
    if (coord != coord_old) {
        uniques.push_back(coord);
        orders .push_back(1);
        coord_old = coord;
    }
    else {
        orders.back() += 1;
    }
    return std::make_tuple(uniques, orders);
}

// Return the polynomial value P(x)
at::Tensor Polynomial::operator()(const at::Tensor & x) const {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::Polynomial::operator(): x must be a vector");
    at::Tensor value = x.new_full({}, 1.0);
    for (auto & coord : coords_) value = value * x[coord];
    return value;
}
// Return dP(x) / dx given x
at::Tensor Polynomial::gradient(const at::Tensor & x) const {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::Polynomial::gradient: x must be a vector");
    std::vector<size_t> uniques, orders;
    std::tie(uniques, orders) = this->uniques_orders();
    at::Tensor grad = x.new_zeros(x.sizes());
    for (size_t i = 0; i < uniques.size(); i++) {
        grad[uniques[i]] = (double)orders[i] * at::pow(x[uniques[i]], (double)(orders[i] - 1));
        for (size_t j = 0; j < i; j++)
        grad[uniques[i]] = grad[uniques[i]] * at::pow(x[uniques[j]], (double)orders[j]);
        for (size_t j = i + 1; j < uniques.size(); j++)
        grad[uniques[i]] = grad[uniques[i]] * at::pow(x[uniques[j]], (double)orders[j]);
    }
    return grad;
}

} // namespace polynomial
} // namespace tchem