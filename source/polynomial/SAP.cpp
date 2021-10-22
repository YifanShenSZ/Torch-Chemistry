#include <unordered_map>

#include <CppLibrary/utility.hpp>

#include <tchem/polynomial/SAP.hpp>

namespace tchem { namespace polynomial {

SAP::SAP() {}
SAP::SAP(const std::vector<std::pair<size_t, size_t>> & _coords) : coords_(_coords) {
    std::sort(coords_.begin(), coords_.end(), std::greater<std::pair<size_t, size_t>>());
    // construct the unique coordinates and their orders
    std::unordered_map<size_t, std::unordered_map<size_t, size_t>> irred2index2order;
    size_t NUniques = 0;
    for (const auto & coord : coords_) {
        const size_t & irred = coord.first, & index = coord.second;
        if (irred2index2order.count(irred) == 0) {
            irred2index2order[irred] = {{index, 1}};
            NUniques++;
        }
        else {
            auto & index2order = irred2index2order[irred];
            if (index2order.count(index) == 0) {
                index2order[index] = 1;
                NUniques++;
            }
            else index2order[index]++;
        }
    }
    uniques_orders_.reserve(NUniques);
    for (const auto & irred_index2order : irred2index2order) {
        const size_t & irred = irred_index2order.first;
        for (const auto & index_order : irred_index2order.second)
        uniques_orders_.push_back({{irred, index_order.first}, index_order.second});
    }
}
// For example, the input line of a 2nd order term made up by
// the 3rd coordinate in the 4th irreducible and
// the 1st coordinate in the 2nd irreducible is:
//     2    4,3    2,1
// The splitted input line is taken in as `strs`
SAP::SAP(const std::vector<std::string> & strs) {
    // construct the irreducibles and the indices of the coordinates constituting the polynomial
    size_t order = std::stoul(strs[0]);
    coords_.resize(order);
    for (size_t i = 0; i < order; i++) {
        std::vector<std::string> irred_coord = CL::utility::split(strs[i + 1], ',');
        coords_[i].first  = std::stoul(irred_coord[0]) - 1;
        coords_[i].second = std::stoul(irred_coord[1]) - 1;
    }
    std::sort(coords_.begin(), coords_.end(), std::greater<std::pair<size_t, size_t>>());
    // construct the unique coordinates and their orders
    std::unordered_map<size_t, std::unordered_map<size_t, size_t>> irred2index2order;
    size_t NUniques = 0;
    for (const auto & coord : coords_) {
        const size_t & irred = coord.first, & index = coord.second;
        if (irred2index2order.count(irred) == 0) {
            irred2index2order[irred] = {{index, 1}};
            NUniques++;
        }
        else {
            auto & index2order = irred2index2order[irred];
            if (index2order.count(index) == 0) {
                index2order[index] = 1;
                NUniques++;
            }
            else index2order[index]++;
        }
    }
    uniques_orders_.reserve(NUniques);
    for (const auto & irred_index2order : irred2index2order) {
        const size_t & irred = irred_index2order.first;
        for (const auto & index_order : irred_index2order.second)
        uniques_orders_.push_back({{irred, index_order.first}, index_order.second});
    }
}
SAP::~SAP() {}

const std::vector<std::pair<size_t, size_t>> & SAP::coords() const {return coords_;}
const std::vector<std::pair<std::pair<size_t, size_t>, size_t>> & SAP::uniques_orders() const {return uniques_orders_;}

const std::pair<size_t, size_t> & SAP::operator[](const size_t & index) const {return coords_[index];}

size_t SAP::order() const {return coords_.size();}
void SAP::pretty_print(std::ostream & stream) const {
    stream << coords_.size() << "    ";
    for (size_t i = 0; i < coords_.size(); i++)
    stream << coords_[i].first + 1 << ',' << coords_[i].second + 1 << "    ";
    stream << '\n';
}

// Return the symmetry adapted polynomial value SAP(x) given x
at::Tensor SAP::operator()(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::operator(): x must be a vector");
    at::Tensor value = xs[0].new_full({}, 1.0);
    for (const auto & irred_index : coords_) value = value * xs[irred_index.first][irred_index.second];
    return value;
}
// Return dP(x) / dx given x
std::vector<at::Tensor> SAP::gradient(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::gradient: x must be a vector");
    std::vector<at::Tensor> grads(xs.size());
    for (size_t i = 0; i < xs.size(); i++) grads[i] = xs[i].new_zeros(xs[i].sizes());
    size_t NUniques = uniques_orders_.size();
    for (size_t i = 0; i < NUniques; i++) {
        const size_t & irred = uniques_orders_[i].first.first,
                     & index = uniques_orders_[i].first.second,
                     & order = uniques_orders_[i].second;
        at::Tensor & grad = grads[irred];
        grad[index] = (double)order * at::pow(xs[irred][index], (double)(order - 1));
        for (size_t j = 0; j < i; j++)
        grad[index] = grad[index]
                    * at::pow(xs[uniques_orders_[j].first.first][uniques_orders_[j].first.second], (double)uniques_orders_[j].second);
        for (size_t j = i + 1; j < NUniques; j++)
        grad[index] = grad[index]
                    * at::pow(xs[uniques_orders_[j].first.first][uniques_orders_[j].first.second], (double)uniques_orders_[j].second);
    }
    return grads;
}
std::vector<at::Tensor> SAP::gradient_(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::gradient_: x must be a vector");
    std::vector<at::Tensor> grads(xs.size());
    std::vector<const double *> pxs(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        grads[i] = xs[i].new_zeros(xs[i].sizes());
        pxs[i] = xs[i].data_ptr<double>();
    }
    size_t NUniques = uniques_orders_.size();
    for (size_t i = 0; i < NUniques; i++) {
        const size_t & irred = uniques_orders_[i].first.first,
                     & index = uniques_orders_[i].first.second,
                     & order = uniques_orders_[i].second;
        const at::Tensor & el = grads[irred][index];
        el.fill_(order * pow(pxs[irred][index], order - 1));
        for (size_t j = 0; j < i; j++)
        el.mul_(pow(pxs[uniques_orders_[j].first.first][uniques_orders_[j].first.second], uniques_orders_[j].second));
        for (size_t j = i + 1; j < NUniques; j++)
        el.mul_(pow(pxs[uniques_orders_[j].first.first][uniques_orders_[j].first.second], uniques_orders_[j].second));
    }
    return grads;
}
// Return dP(x) / dx given x
// `grad` harvests the concatenated symmetry adapted gradients
std::vector<at::Tensor> SAP::gradient_(const std::vector<at::Tensor> & xs, at::Tensor & grad) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::gradient_: x must be a vector");
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
    size_t NUniques = uniques_orders_.size();
    for (size_t i = 0; i < NUniques; i++) {
        const size_t & irred = uniques_orders_[i].first.first,
                     & index = uniques_orders_[i].first.second,
                     & order = uniques_orders_[i].second;
        const at::Tensor & el = grads[irred][index];
        el.fill_(order * pow(pxs[irred][index], order - 1));
        for (size_t j = 0; j < i; j++)
        el.mul_(pow(pxs[uniques_orders_[j].first.first][uniques_orders_[j].first.second], uniques_orders_[j].second));
        for (size_t j = i + 1; j < NUniques; j++)
        el.mul_(pow(pxs[uniques_orders_[j].first.first][uniques_orders_[j].first.second], uniques_orders_[j].second));
    }
    return grads;
}
// Return ddP(x) / dx^2 given x
CL::utility::matrix<at::Tensor> SAP::Hessian(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::Hessian: x must be a vector");
    CL::utility::matrix<at::Tensor> hesses(xs.size());
    for (size_t i = 0; i < xs.size(); i++)
    for (size_t j = i; j < xs.size(); j++)
    hesses[i][j] = xs[i].new_zeros({xs[i].size(0), xs[j].size(0)});
    size_t NUniques = uniques_orders_.size();
    for (size_t i = 0; i < NUniques; i++) {
        const size_t & irred_i = uniques_orders_[i].first.first,
                     & index_i = uniques_orders_[i].first.second,
                     & order_i = uniques_orders_[i].second;
        // diagonal
        at::Tensor & hess_ii = hesses[irred_i][irred_i];
        if (order_i < 2) hess_ii[index_i][index_i] = 0.0;
        else {
            hess_ii[index_i][index_i] = (double)(order_i * (order_i - 1))
                                      * at::pow(xs[irred_i][index_i], (double)(order_i - 2));
            for (size_t k = 0; k < i; k++)
            hess_ii[index_i][index_i] = hess_ii[index_i][index_i]
                                      * at::pow(xs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], (double)uniques_orders_[k].second);
            for (size_t k = i + 1; k < NUniques; k++)
            hess_ii[index_i][index_i] = hess_ii[index_i][index_i]
                                      * at::pow(xs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], (double)uniques_orders_[k].second);
        }
        // strict upper-triangle
        for (size_t j = i + 1; j < NUniques; j++) {
            const size_t & irred_j = uniques_orders_[j].first.first,
                         & index_j = uniques_orders_[j].first.second,
                         & order_j = uniques_orders_[j].second;
            at::Tensor & hess_ij = hesses[irred_i][irred_j];
            hess_ij[index_i][index_j] = (double)(order_i * order_j)
                                      * at::pow(xs[irred_i][index_i], (double)(order_i - 1))
                                      * at::pow(xs[irred_j][index_j], (double)(order_j - 1));
            for (size_t k = 0; k < i; k++)
            hess_ij[index_i][index_j] = hess_ij[index_i][index_j]
                                      * at::pow(xs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], (double)uniques_orders_[k].second);
            for (size_t k = i + 1; k < j; k++)
            hess_ij[index_i][index_j] = hess_ij[index_i][index_j]
                                      * at::pow(xs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], (double)uniques_orders_[k].second);
            for (size_t k = j + 1; k < NUniques; k++)
            hess_ij[index_i][index_j] = hess_ij[index_i][index_j]
                                      * at::pow(xs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], (double)uniques_orders_[k].second);
            // copy to strict lower-triangle
            // hesses[irred_j][irred_i][index_j][index_i] = hess_ij[index_i][index_j];
        }
    }
    return hesses;
}
CL::utility::matrix<at::Tensor> SAP::Hessian_(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::Hessian_: x must be a vector");
    CL::utility::matrix<at::Tensor> hesses(xs.size());
    std::vector<const double *> pxs(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        pxs[i] = xs[i].data_ptr<double>();
        for (size_t j = i; j < xs.size(); j++) hesses[i][j] = xs[i].new_zeros({xs[i].size(0), xs[j].size(0)});
    }
    size_t NUniques = uniques_orders_.size();
    for (size_t i = 0; i < NUniques; i++) {
        const size_t & irred_i = uniques_orders_[i].first.first,
                     & index_i = uniques_orders_[i].first.second,
                     & order_i = uniques_orders_[i].second;
        // diagonal
        const at::Tensor & el = hesses[irred_i][irred_i][index_i][index_i];
        if (order_i < 2) el.zero_();
        else {
            el.fill_((order_i * (order_i - 1))* pow(pxs[irred_i][index_i], order_i - 2));
            for (size_t k = 0; k < i; k++)
            el.mul_(pow(pxs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], uniques_orders_[k].second));
            for (size_t k = i + 1; k < NUniques; k++)
            el.mul_(pow(pxs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], uniques_orders_[k].second));
        }
        for (size_t j = i + 1; j < NUniques; j++) {
            const size_t & irred_j = uniques_orders_[j].first.first,
                         & index_j = uniques_orders_[j].first.second,
                         & order_j = uniques_orders_[j].second;
            const at::Tensor & el = hesses[irred_i][irred_j][index_i][index_j];
            el.fill_((order_i * order_j)
                     * pow(pxs[irred_i][index_i], order_i - 1)
                     * pow(pxs[irred_j][index_j], order_j - 1));
            for (size_t k = 0; k < i; k++)
            el.mul_(pow(pxs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], uniques_orders_[k].second));
            for (size_t k = i + 1; k < j; k++)
            el.mul_(pow(pxs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], uniques_orders_[k].second));
            for (size_t k = j + 1; k < NUniques; k++)
            el.mul_(pow(pxs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], uniques_orders_[k].second));
        }
    }
    return hesses;
}
// Return ddP(x) / dx^2 given x
// `hess` harvests the concatenated symmetry adapted Hessians
CL::utility::matrix<at::Tensor> SAP::Hessian_(const std::vector<at::Tensor> & xs, at::Tensor & hess) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::Hessian_: x must be a vector");
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
    size_t NUniques = uniques_orders_.size();
    for (size_t i = 0; i < NUniques; i++) {
        const size_t & irred_i = uniques_orders_[i].first.first,
                     & index_i = uniques_orders_[i].first.second,
                     & order_i = uniques_orders_[i].second;
        // diagonal
        const at::Tensor & el = hesses[irred_i][irred_i][index_i][index_i];
        if (order_i < 2) el.zero_();
        else {
            el.fill_((order_i * (order_i - 1))* pow(pxs[irred_i][index_i], order_i - 2));
            for (size_t k = 0; k < i; k++)
            el.mul_(pow(pxs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], uniques_orders_[k].second));
            for (size_t k = i + 1; k < NUniques; k++)
            el.mul_(pow(pxs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], uniques_orders_[k].second));
        }
        for (size_t j = i + 1; j < NUniques; j++) {
            const size_t & irred_j = uniques_orders_[j].first.first,
                         & index_j = uniques_orders_[j].first.second,
                         & order_j = uniques_orders_[j].second;
            const at::Tensor & el = hesses[irred_i][irred_j][index_i][index_j];
            el.fill_((order_i * order_j)
                     * pow(pxs[irred_i][index_i], order_i - 1)
                     * pow(pxs[irred_j][index_j], order_j - 1));
            for (size_t k = 0; k < i; k++)
            el.mul_(pow(pxs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], uniques_orders_[k].second));
            for (size_t k = i + 1; k < j; k++)
            el.mul_(pow(pxs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], uniques_orders_[k].second));
            for (size_t k = j + 1; k < NUniques; k++)
            el.mul_(pow(pxs[uniques_orders_[k].first.first][uniques_orders_[k].first.second], uniques_orders_[k].second));
        }
    }
    return hesses;
}

} // namespace polynomial
} // namespace tchem