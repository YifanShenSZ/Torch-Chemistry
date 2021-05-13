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
at::Tensor Polynomial::gradient_(const at::Tensor & x) const {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::Polynomial::gradient_: x must be a vector");
    std::vector<size_t> uniques, orders;
    std::tie(uniques, orders) = this->uniques_orders();
    at::Tensor grad = x.new_zeros(x.sizes());
    const double * px = x.data_ptr<double>();
    for (size_t i = 0; i < uniques.size(); i++) {
        const at::Tensor & el = grad[uniques[i]];
        el.fill_(orders[i] * pow(px[uniques[i]], orders[i] - 1));
        for (size_t j = 0; j < i; j++)
        el.mul_(pow(px[uniques[j]], orders[j]));
        for (size_t j = i + 1; j < uniques.size(); j++)
        el.mul_(pow(px[uniques[j]], orders[j]));
    }
    return grad;
}
// Return ddP(x) / dx^2 given x
at::Tensor Polynomial::Hessian(const at::Tensor & x) const {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::Polynomial::Hessian: x must be a vector");
    std::vector<size_t> uniques, orders;
    std::tie(uniques, orders) = this->uniques_orders();
    at::Tensor hess = x.new_zeros({x.size(0), x.size(0)});
    for (size_t i = 0; i < uniques.size(); i++) {
        if (orders[i] < 2) hess[uniques[i]][uniques[i]] = 0.0;
        else {
            hess[uniques[i]][uniques[i]] = (double)(orders[i] * (orders[i] - 1)) * at::pow(x[uniques[i]], (double)(orders[i] - 2));
            for (size_t j = 0; j < i; j++)
            hess[uniques[i]][uniques[i]] = hess[uniques[i]][uniques[i]] * at::pow(x[uniques[j]], (double)orders[j]);
            for (size_t j = i + 1; j < uniques.size(); j++)
            hess[uniques[i]][uniques[i]] = hess[uniques[i]][uniques[i]] * at::pow(x[uniques[j]], (double)orders[j]);
        }
        for (size_t j = i + 1; j < uniques.size(); j++) {
            hess[uniques[j]][uniques[i]] = (double)(orders[i] * orders[j])
                                         * at::pow(x[uniques[i]], (double)(orders[i] - 1))
                                         * at::pow(x[uniques[j]], (double)(orders[j] - 1));
            for (size_t k = 0; k < i; k++)
            hess[uniques[j]][uniques[i]] = hess[uniques[j]][uniques[i]] * at::pow(x[uniques[k]], (double)orders[k]);
            for (size_t k = i + 1; k < j; k++)
            hess[uniques[j]][uniques[i]] = hess[uniques[j]][uniques[i]] * at::pow(x[uniques[k]], (double)orders[k]);
            for (size_t k = j + 1; k < uniques.size(); k++)
            hess[uniques[j]][uniques[i]] = hess[uniques[j]][uniques[i]] * at::pow(x[uniques[k]], (double)orders[k]);
            hess[uniques[i]][uniques[j]] = hess[uniques[j]][uniques[i]];
        }
    }
    return hess;
}
at::Tensor Polynomial::Hessian_(const at::Tensor & x) const {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::Polynomial::Hessian_: x must be a vector");
    std::vector<size_t> uniques, orders;
    std::tie(uniques, orders) = this->uniques_orders();
    at::Tensor hess = x.new_zeros({x.size(0), x.size(0)});
    for (size_t i = 0; i < uniques.size(); i++) {
        const at::Tensor & el = hess[uniques[i]][uniques[i]];
        if (orders[i] < 2) el.zero_();
        else {
            el.fill_((orders[i] * (orders[i] - 1)) * pow(x[uniques[i]].item<double>(), orders[i] - 2));
            for (size_t j = 0; j < i; j++)
            el.mul_(pow(x[uniques[j]].item<double>(), orders[j]));
            for (size_t j = i + 1; j < uniques.size(); j++)
            el.mul_(pow(x[uniques[j]].item<double>(), orders[j]));
        }
        for (size_t j = i + 1; j < uniques.size(); j++) {
            const at::Tensor & el = hess[uniques[j]][uniques[i]];
            el.fill_((orders[i] * orders[j])
                     * pow(x[uniques[i]].item<double>(), orders[i] - 1)
                     * pow(x[uniques[j]].item<double>(), orders[j] - 1));
            for (size_t k = 0; k < i; k++)
            el.mul_(pow(x[uniques[k]].item<double>(), orders[k]));
            for (size_t k = i + 1; k < j; k++)
            el.mul_(pow(x[uniques[k]].item<double>(), orders[k]));
            for (size_t k = j + 1; k < uniques.size(); k++)
            el.mul_(pow(x[uniques[k]].item<double>(), orders[k]));
            hess[uniques[i]][uniques[j]].copy_(el);
        }
    }
    return hess;
}

} // namespace polynomial
} // namespace tchem