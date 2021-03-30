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
    for (const at::Tensor & x : xs)
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::operator(): x must be a vector");
    at::Tensor value = xs[0].new_full({}, 1.0);
    for (size_t i = 0; i < coords_.size(); i++)
    value = value * xs[coords_[i].first][coords_[i].second];
    return value;
}
// Return dP(x) / dx given x
std::vector<at::Tensor> SAP::gradient(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs)
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::gradient: x must be a vector");
    std::vector<std::pair<size_t, size_t>> uniques;
    std::vector<size_t> orders;
    std::tie(uniques, orders) = this->uniques_orders();
    std::vector<at::Tensor> grad(xs.size());
    for (size_t i = 0; i < xs.size(); i++) grad[i] = xs[i].new_zeros(xs[i].sizes());
    for (size_t i = 0; i < uniques.size(); i++) {
        grad[uniques[i].first][uniques[i].second] = (double)orders[i] * at::pow(xs[uniques[i].first][uniques[i].second], (double)(orders[i] - 1));
        for (size_t j = 0; j < i; j++)
        grad[uniques[i].first][uniques[i].second] = grad[uniques[i].first][uniques[i].second]
                                                  * at::pow(xs[uniques[j].first][uniques[j].second], (double)orders[j]);
        for (size_t j = i + 1; j < uniques.size(); j++)
        grad[uniques[i].first][uniques[i].second] = grad[uniques[i].first][uniques[i].second]
                                                  * at::pow(xs[uniques[j].first][uniques[j].second], (double)orders[j]);
    }
    return grad;
}

} // namespace polynomial
} // namespace tchem