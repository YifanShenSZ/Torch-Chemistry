#include <regex>

#include <CppLibrary/utility.hpp>

#include <tchem/SApolynomial.hpp>

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

std::vector<std::pair<size_t, size_t>> SAP::coords() const {return coords_;}

size_t SAP::order() const {return coords_.size();}
void SAP::pretty_print(std::ostream & stream) const {
    stream << coords_.size() << "    ";
    for (size_t i = 0; i < coords_.size(); i++)
    stream << coords_[i].first << ',' << coords_[i].second << "    ";
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





SAPSet::SAPSet() {}
// `sapoly_file` contains one SAP per line
SAPSet::SAPSet(const std::string & sapoly_file, const std::vector<size_t> & _dimensions)
: dimensions_(_dimensions) {
    std::ifstream ifs; ifs.open(sapoly_file);
    if (! ifs.good()) throw CL::utility::file_error(sapoly_file);
    while (true) {
        std::string line;
        std::getline(ifs, line);
        if (! ifs.good()) break;
        std::vector<std::string> strs = CL::utility::split(line);
        SAPs_.push_back(SAP(strs));
    }
    ifs.close();
}
SAPSet::~SAPSet() {}

std::vector<SAP> SAPSet::SAPs() const {return SAPs_;}

void SAPSet::pretty_print(std::ostream & stream) const {
    for (const SAP & sap : SAPs_) sap.pretty_print(stream);
}

// Return the value of each term in {P(x)} as a vector of vectors
at::Tensor SAPSet::operator()(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs)
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::operator(): x must be a vector");
    if (xs.size() != dimensions_.size()) throw std::invalid_argument(
    "tchem::polynomial::SAP::operator(): x must share a same number of irreducibles as the coordinate system");
    for (size_t i = 0; i < xs.size(); i++)
    if (xs[i].size(0) != dimensions_[i]) throw std::invalid_argument(
    "tchem::polynomial::SAP::operator(): x must have a same dimension as the coordinates");
    at::Tensor y = xs[0].new_empty(SAPs_.size());
    for (size_t i = 0; i < SAPs_.size(); i++) y[i] = SAPs_[i](xs);
    return y;
}
// Return d{P(x)} / dx given x
std::vector<at::Tensor> SAPSet::Jacobian(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs)
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAP::Jacobian: x must be a vector");
    if (xs.size() != dimensions_.size()) throw std::invalid_argument(
    "tchem::polynomial::SAP::Jacobian: x must share a same number of irreducibles as the coordinate system");
    for (size_t i = 0; i < xs.size(); i++)
    if (xs[i].size(0) != dimensions_[i]) throw std::invalid_argument(
    "tchem::polynomial::SAP::Jacobian: x must have a same dimension as the coordinates");
    std::vector<at::Tensor> Js(xs.size());
    for (size_t i = 0; i < xs.size(); i++) Js[i] = xs[i].new_empty({(int64_t)SAPs_.size(), xs[i].size(0)});
    for (size_t i = 0; i < SAPs_.size(); i++) {
        std::vector<at::Tensor> rows = SAPs_[i].gradient(xs);
        for (size_t j = 0; j < xs.size(); j++) Js[j][i] = rows[j];
    }
    return Js;
}

} // namespace polynomial
} // namespace tchem