#include <CppLibrary/utility.hpp>

#include <tchem/polynomial/SAP.hpp>

namespace tchem { namespace polynomial {

SAP::SAP() {}
SAP::SAP(const std::vector<std::pair<size_t, size_t>> & _coords, const bool & sorted)
: coords_(_coords) {
    if (! sorted) std::sort(coords_.begin(), coords_.end(), std::greater<std::pair<size_t, size_t>>());
}
// For example, the input line of a 2nd order term made up by
// the 3rd coordinate in the 4th irreducible and
// the 1st coordinate in the 2nd irreducible is:
//     2    4,3    2,1
// The splitted input line is taken in as `strs`
SAP::SAP(const std::vector<std::string> & strs, const bool & sorted) {
    size_t order = std::stoul(strs[0]);
    coords_.resize(order);
    for (size_t i = 0; i < order; i++) {
        std::vector<std::string> irred_coord = CL::utility::split(strs[i + 1], ',');
        coords_[i].first  = std::stoul(irred_coord[0]) - 1;
        coords_[i].second = std::stoul(irred_coord[1]) - 1;
    }
    if (! sorted) std::sort(coords_.begin(), coords_.end(), std::greater<std::pair<size_t, size_t>>());
}
SAP::~SAP() {}

const std::vector<std::pair<size_t, size_t>> & SAP::coords() const {return coords_;}

const std::pair<size_t, size_t> & SAP::operator[](const size_t & index) const {return coords_[index];}

size_t SAP::order() const {return coords_.size();}
void SAP::pretty_print(std::ostream & stream) const {
    stream << coords_.size() << "    ";
    for (size_t i = 0; i < coords_.size(); i++)
    stream << coords_[i].first + 1 << ',' << coords_[i].second + 1 << "    ";
    stream << '\n';
}

 // Return the unique coordinates and their orders
std::tuple<std::vector<std::pair<size_t, size_t>>, std::vector<size_t>> SAP::uniques_orders() const {
    std::vector<std::pair<size_t, size_t>> uniques;
    std::vector<size_t> orders;
    std::pair<size_t, size_t> coord_old(-1, -1);
    for (const std::pair<size_t, size_t> & coord : coords_)
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

// Return the symmetry adapted polynomial value SAP(x) given x
at::Tensor SAP::operator()(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::operator(): x must be a vector");
    at::Tensor value = xs[0].new_full({}, 1.0);
    for (size_t i = 0; i < coords_.size(); i++)
    value = value * xs[coords_[i].first][coords_[i].second];
    return value;
}
// Return dP(x) / dx given x
std::vector<at::Tensor> SAP::gradient(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::gradient: x must be a vector");
    std::vector<std::pair<size_t, size_t>> uniques;
    std::vector<size_t> orders;
    std::tie(uniques, orders) = this->uniques_orders();
    std::vector<at::Tensor> grads(xs.size());
    for (size_t i = 0; i < xs.size(); i++) grads[i] = xs[i].new_zeros(xs[i].sizes());
    for (size_t i = 0; i < uniques.size(); i++) {
        grads[uniques[i].first][uniques[i].second] = (double)orders[i] * at::pow(xs[uniques[i].first][uniques[i].second], (double)(orders[i] - 1));
        for (size_t j = 0; j < i; j++)
        grads[uniques[i].first][uniques[i].second] = grads[uniques[i].first][uniques[i].second]
                                                   * at::pow(xs[uniques[j].first][uniques[j].second], (double)orders[j]);
        for (size_t j = i + 1; j < uniques.size(); j++)
        grads[uniques[i].first][uniques[i].second] = grads[uniques[i].first][uniques[i].second]
                                                   * at::pow(xs[uniques[j].first][uniques[j].second], (double)orders[j]);
    }
    return grads;
}
std::vector<at::Tensor> SAP::gradient_(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::gradient_: x must be a vector");
    std::vector<std::pair<size_t, size_t>> uniques;
    std::vector<size_t> orders;
    std::tie(uniques, orders) = this->uniques_orders();
    std::vector<at::Tensor> grads(xs.size());
    std::vector<const double *> pxs(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        grads[i] = xs[i].new_zeros(xs[i].sizes());
        pxs[i] = xs[i].data_ptr<double>();
    }
    for (size_t i = 0; i < uniques.size(); i++) {
        const at::Tensor & el = grads[uniques[i].first][uniques[i].second];
        el.fill_(orders[i] * pow(pxs[uniques[i].first][uniques[i].second], orders[i] - 1));
        for (size_t j = 0; j < i; j++)
        el.mul_(pow(pxs[uniques[j].first][uniques[j].second], orders[j]));
        for (size_t j = i + 1; j < uniques.size(); j++)
        el.mul_(pow(pxs[uniques[j].first][uniques[j].second], orders[j]));
    }
    return grads;
}
// Return dP(x) / dx given x
// `result` harvests the concatenated symmetry adapted gradients
std::vector<at::Tensor> SAP::gradient_(const std::vector<at::Tensor> & xs, at::Tensor & grad) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::gradient_: x must be a vector");
    std::vector<std::pair<size_t, size_t>> uniques;
    std::vector<size_t> orders;
    std::tie(uniques, orders) = this->uniques_orders();
    int64_t dimension = 0;
    for (const at::Tensor & x : xs) dimension += x.size(0);
    grad = xs[0].new_zeros(dimension);
    int64_t start = 0, stop;
    std::vector<at::Tensor> grads(xs.size());
    std::vector<const double *> pxs(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        stop = start + xs[i].size(0);
        grads[i] = grad.slice(0, start, stop);
        start = stop;
        pxs[i] = xs[i].data_ptr<double>();
    }
    for (size_t i = 0; i < uniques.size(); i++) {
        const at::Tensor & el = grads[uniques[i].first][uniques[i].second];
        el.fill_(orders[i] * pow(pxs[uniques[i].first][uniques[i].second], orders[i] - 1));
        for (size_t j = 0; j < i; j++)
        el.mul_(pow(pxs[uniques[j].first][uniques[j].second], orders[j]));
        for (size_t j = i + 1; j < uniques.size(); j++)
        el.mul_(pow(pxs[uniques[j].first][uniques[j].second], orders[j]));
    }
    return grads;
}
// Return ddP(x) / dx^2 given x
CL::utility::matrix<at::Tensor> SAP::Hessian(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::Hessian: x must be a vector");
    std::vector<std::pair<size_t, size_t>> uniques;
    std::vector<size_t> orders;
    std::tie(uniques, orders) = this->uniques_orders();
    CL::utility::matrix<at::Tensor> hesses(xs.size());
    for (size_t i = 0; i < xs.size(); i++)
    for (size_t j = i; j < xs.size(); j++)
    hesses[i][j] = xs[i].new_zeros({xs[i].size(0), xs[j].size(0)});
    for (size_t i = 0; i < uniques.size(); i++) {
        at::Tensor & hess_ii = hesses[uniques[i].first][uniques[i].first];
        if (orders[i] < 2) hess_ii[uniques[i].second][uniques[i].second] = 0.0;
        else {
            hess_ii[uniques[i].second][uniques[i].second]
                = (double)(orders[i] * (orders[i] - 1))
                * at::pow(xs[uniques[i].first][uniques[i].second], (double)(orders[i] - 2));
            for (size_t k = 0; k < i; k++)
            hess_ii[uniques[i].second][uniques[i].second]
                = hess_ii[uniques[i].second][uniques[i].second]
                * at::pow(xs[uniques[k].first][uniques[k].second], (double)orders[k]);
            for (size_t k = i + 1; k < uniques.size(); k++)
            hess_ii[uniques[i].second][uniques[i].second]
                = hess_ii[uniques[i].second][uniques[i].second]
                * at::pow(xs[uniques[k].first][uniques[k].second], (double)orders[k]);
        }
        for (size_t j = i + 1; j < uniques.size(); j++) {
            at::Tensor & hess_ji = hesses[uniques[j].first][uniques[i].first];
            hess_ji[uniques[j].second][uniques[i].second]
                = (double)(orders[i] * orders[j])
                * at::pow(xs[uniques[i].first][uniques[i].second], (double)(orders[i] - 1))
                * at::pow(xs[uniques[j].first][uniques[j].second], (double)(orders[j] - 1));
            for (size_t k = 0; k < i; k++)
            hess_ji[uniques[j].second][uniques[i].second]
                = hess_ji[uniques[j].second][uniques[i].second]
                * at::pow(xs[uniques[k].first][uniques[k].second], (double)orders[k]);
            for (size_t k = i + 1; k < j; k++)
            hess_ji[uniques[j].second][uniques[i].second]
                = hess_ji[uniques[j].second][uniques[i].second]
                * at::pow(xs[uniques[k].first][uniques[k].second], (double)orders[k]);
            for (size_t k = j + 1; k < uniques.size(); k++)
            hess_ji[uniques[j].second][uniques[i].second]
                = hess_ji[uniques[j].second][uniques[i].second]
                * at::pow(xs[uniques[k].first][uniques[k].second], (double)orders[k]);
        }
    }
    return hesses;
}
CL::utility::matrix<at::Tensor> SAP::Hessian_(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::Hessian_: x must be a vector");
    std::vector<std::pair<size_t, size_t>> uniques;
    std::vector<size_t> orders;
    std::tie(uniques, orders) = this->uniques_orders();
    CL::utility::matrix<at::Tensor> hesses(xs.size());
    std::vector<const double *> pxs(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        pxs[i] = xs[i].data_ptr<double>();
        for (size_t j = i; j < xs.size(); j++) hesses[i][j] = xs[i].new_zeros({xs[i].size(0), xs[j].size(0)});
    }
    for (size_t i = 0; i < uniques.size(); i++) {
        const at::Tensor & el = hesses[uniques[i].first][uniques[i].first][uniques[i].second][uniques[i].second];
        if (orders[i] < 2) el.zero_();
        else {
            el.fill_((orders[i] * (orders[i] - 1))* pow(pxs[uniques[i].first][uniques[i].second], orders[i] - 2));
            for (size_t k = 0; k < i; k++)
            el.mul_(pow(pxs[uniques[k].first][uniques[k].second], orders[k]));
            for (size_t k = i + 1; k < uniques.size(); k++)
            el.mul_(pow(pxs[uniques[k].first][uniques[k].second], orders[k]));
        }
        for (size_t j = i + 1; j < uniques.size(); j++) {
            const at::Tensor & el = hesses[uniques[j].first][uniques[i].first][uniques[j].second][uniques[i].second];
            el.fill_((orders[i] * orders[j])
                     * pow(pxs[uniques[i].first][uniques[i].second], orders[i] - 1)
                     * pow(pxs[uniques[j].first][uniques[j].second], orders[j] - 1));
            for (size_t k = 0; k < i; k++)
            el.mul_(pow(pxs[uniques[k].first][uniques[k].second], orders[k]));
            for (size_t k = i + 1; k < j; k++)
            el.mul_(pow(pxs[uniques[k].first][uniques[k].second], orders[k]));
            for (size_t k = j + 1; k < uniques.size(); k++)
            el.mul_(pow(pxs[uniques[k].first][uniques[k].second], orders[k]));
        }
    }
    return hesses;
}
// Return ddP(x) / dx^2 given x
// `hess` harvests the concatenated symmetry adapted Hessians
CL::utility::matrix<at::Tensor> SAP::Hessian_(const std::vector<at::Tensor> & xs, at::Tensor & hess) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::Hessian_: x must be a vector");
    std::vector<std::pair<size_t, size_t>> uniques;
    std::vector<size_t> orders;
    std::tie(uniques, orders) = this->uniques_orders();
    int64_t dimension = 0;
    for (const at::Tensor & x : xs) dimension += x.size(0);
    hess = xs[0].new_zeros({dimension, dimension});
    int64_t start_row = 0, stop_row;
    CL::utility::matrix<at::Tensor> hesses(xs.size());
    std::vector<const double *> pxs(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        pxs[i] = xs[i].data_ptr<double>();
        stop_row = start_row + xs[i].size(0);
        int64_t start_col = start_row, stop_col;
        for (size_t j = i; j < xs.size(); j++) {
            stop_col = start_col + xs[j].size(0);
            hesses[i][j] = hess.slice(0, start_row, stop_row).slice(1, start_col, stop_col);
            start_col = stop_col;
        }
        start_row = stop_row;
    }
    for (size_t i = 0; i < uniques.size(); i++) {
        const at::Tensor & el = hesses[uniques[i].first][uniques[i].first][uniques[i].second][uniques[i].second];
        if (orders[i] < 2) el.zero_();
        else {
            el.fill_((orders[i] * (orders[i] - 1))* pow(pxs[uniques[i].first][uniques[i].second], orders[i] - 2));
            for (size_t k = 0; k < i; k++)
            el.mul_(pow(pxs[uniques[k].first][uniques[k].second], orders[k]));
            for (size_t k = i + 1; k < uniques.size(); k++)
            el.mul_(pow(pxs[uniques[k].first][uniques[k].second], orders[k]));
        }
        for (size_t j = i + 1; j < uniques.size(); j++) {
            const at::Tensor & el = hesses[uniques[j].first][uniques[i].first][uniques[j].second][uniques[i].second];
            el.fill_((orders[i] * orders[j])
                     * pow(pxs[uniques[i].first][uniques[i].second], orders[i] - 1)
                     * pow(pxs[uniques[j].first][uniques[j].second], orders[j] - 1));
            for (size_t k = 0; k < i; k++)
            el.mul_(pow(pxs[uniques[k].first][uniques[k].second], orders[k]));
            for (size_t k = i + 1; k < j; k++)
            el.mul_(pow(pxs[uniques[k].first][uniques[k].second], orders[k]));
            for (size_t k = j + 1; k < uniques.size(); k++)
            el.mul_(pow(pxs[uniques[k].first][uniques[k].second], orders[k]));
        }
    }
    return hesses;
}

} // namespace polynomial
} // namespace tchem