#include <unordered_map>

#include <tchem/polynomial/Polynomial.hpp>

namespace tchem { namespace polynomial {

Polynomial::Polynomial() {}
Polynomial::Polynomial(const std::vector<size_t> & _coords) : coords_(_coords) {
    std::sort(coords_.begin(), coords_.end(), std::greater<size_t>());
    // construct the unique coordinates and their orders
    std::unordered_map<size_t, size_t> unique2order;
    for (const size_t & coord : coords_)
    if (unique2order.count(coord) == 0) unique2order[coord] = 1;
    else                                unique2order[coord]++;
    uniques_orders_.reserve(unique2order.size());
    for (const auto & unique_order : unique2order) uniques_orders_.push_back(unique_order);
}
Polynomial::Polynomial(const std::vector<std::pair<size_t, size_t>> _uniques_orders) : uniques_orders_(_uniques_orders) {
    // construct the indices of the coordinates constituting the polynomial
    for (const auto & unique_order : uniques_orders_)
    for (size_t i = 0; i < unique_order.second; i++)
    coords_.push_back(unique_order.first);
    std::sort(coords_.begin(), coords_.end(), std::greater<size_t>());
}
Polynomial::~Polynomial() {}

const std::vector<size_t> & Polynomial::coords() const {return coords_;}
const std::vector<std::pair<size_t, size_t>> & Polynomial::uniques_orders() const {return uniques_orders_;}

size_t Polynomial::order() const {return coords_.size();}

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
    at::Tensor grad = x.new_zeros(x.sizes());
    size_t NUniques = uniques_orders_.size();
    for (size_t i = 0; i < NUniques; i++) {
        const size_t & unique = uniques_orders_[i].first ,
                     & order  = uniques_orders_[i].second;
        grad[unique] = (double)order * at::pow(x[unique], (double)(order - 1));
        for (size_t j = 0; j < i; j++)
        grad[unique] = grad[unique] * at::pow(x[uniques_orders_[j].first], (double)uniques_orders_[j].second);
        for (size_t j = i + 1; j < NUniques; j++)
        grad[unique] = grad[unique] * at::pow(x[uniques_orders_[j].first], (double)uniques_orders_[j].second);
    }
    return grad;
}
at::Tensor Polynomial::gradient_(const at::Tensor & x) const {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::Polynomial::gradient_: x must be a vector");
    at::Tensor grad = x.new_zeros(x.sizes());
    const double * px = x.data_ptr<double>();
    size_t NUniques = uniques_orders_.size();
    for (size_t i = 0; i < NUniques; i++) {
        const size_t & unique = uniques_orders_[i].first ,
                     & order  = uniques_orders_[i].second;
        const at::Tensor & el = grad[unique];
        el.fill_(order * pow(px[unique], order - 1));
        for (size_t j = 0; j < i; j++)
        el.mul_(pow(px[uniques_orders_[j].first], uniques_orders_[j].second));
        for (size_t j = i + 1; j < NUniques; j++)
        el.mul_(pow(px[uniques_orders_[j].first], uniques_orders_[j].second));
    }
    return grad;
}
// Return ddP(x) / dx^2 given x
at::Tensor Polynomial::Hessian(const at::Tensor & x) const {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::Polynomial::Hessian: x must be a vector");
    at::Tensor hess = x.new_zeros({x.size(0), x.size(0)});
    size_t NUniques = uniques_orders_.size();
    for (size_t i = 0; i < NUniques; i++) {
        const size_t & unique_i = uniques_orders_[i].first ,
                     &  order_i = uniques_orders_[i].second;
        // diagonal
        if (order_i < 2) hess[unique_i][unique_i] = 0.0;
        else {
            hess[unique_i][unique_i] = (double)(order_i * (order_i - 1)) * at::pow(x[unique_i], (double)(order_i - 2));
            for (size_t k = 0; k < i; k++)
            hess[unique_i][unique_i] = hess[unique_i][unique_i] * at::pow(x[uniques_orders_[k].first], (double)uniques_orders_[k].second);
            for (size_t k = i + 1; k < NUniques; k++)
            hess[unique_i][unique_i] = hess[unique_i][unique_i] * at::pow(x[uniques_orders_[k].first], (double)uniques_orders_[k].second);
        }
        // strict upper-triangle
        for (size_t j = i + 1; j < NUniques; j++) {
            const size_t & unique_j = uniques_orders_[j].first ,
                         &  order_j = uniques_orders_[j].second;
            hess[unique_i][unique_j] = (double)(order_i * order_j)
                                   * at::pow(x[unique_i], (double)(order_i - 1))
                                   * at::pow(x[unique_j], (double)(order_j - 1));
            for (size_t k = 0; k < i; k++)
            hess[unique_i][unique_j] = hess[unique_i][unique_j] * at::pow(x[uniques_orders_[k].first], (double)uniques_orders_[k].second);
            for (size_t k = i + 1; k < j; k++)
            hess[unique_i][unique_j] = hess[unique_i][unique_j] * at::pow(x[uniques_orders_[k].first], (double)uniques_orders_[k].second);
            for (size_t k = j + 1; k < NUniques; k++)
            hess[unique_i][unique_j] = hess[unique_i][unique_j] * at::pow(x[uniques_orders_[k].first], (double)uniques_orders_[k].second);
            // copy to strict lower-triangle
            hess[unique_j][unique_i] = hess[unique_i][unique_j];
        }
    }
    return hess;
}
at::Tensor Polynomial::Hessian_(const at::Tensor & x) const {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::Polynomial::Hessian_: x must be a vector");
    at::Tensor hess = x.new_zeros({x.size(0), x.size(0)});
    const double * px = x.data_ptr<double>();
    size_t NUniques = uniques_orders_.size();
    for (size_t i = 0; i < NUniques; i++) {
        const size_t & unique_i = uniques_orders_[i].first ,
                     &  order_i = uniques_orders_[i].second;
        // diagonal
        const at::Tensor & el = hess[unique_i][unique_i];
        if (order_i < 2) el.zero_();
        else {
            el.fill_((order_i * (order_i - 1)) * pow(x[unique_i].item<double>(), order_i - 2));
            for (size_t k = 0; k < i; k++)
            el.mul_(pow(px[uniques_orders_[k].first], uniques_orders_[k].second));
            for (size_t k = i + 1; k < NUniques; k++)
            el.mul_(pow(px[uniques_orders_[k].first], uniques_orders_[k].second));
        }
        // strict upper-triangle
        for (size_t j = i + 1; j < NUniques; j++) {
            const size_t & unique_j = uniques_orders_[j].first ,
                         &  order_j = uniques_orders_[j].second;
            const at::Tensor & el = hess[unique_i][unique_j];
            el.fill_((order_i * order_j)
                     * pow(x[unique_i].item<double>(), order_i - 1)
                     * pow(x[unique_j].item<double>(), order_j - 1));
            for (size_t k = 0; k < i; k++)
            el.mul_(pow(px[uniques_orders_[k].first], uniques_orders_[k].second));
            for (size_t k = i + 1; k < j; k++)
            el.mul_(pow(px[uniques_orders_[k].first], uniques_orders_[k].second));
            for (size_t k = j + 1; k < NUniques; k++)
            el.mul_(pow(px[uniques_orders_[k].first], uniques_orders_[k].second));
            // copy to strict lower-triangle
            hess[unique_j][unique_i].copy_(el);
        }
    }
    return hess;
}

} // namespace polynomial
} // namespace tchem