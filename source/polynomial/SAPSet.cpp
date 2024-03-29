#include <regex>

#include <CppLibrary/utility.hpp>

#include <tchem/polynomial/SAPSet.hpp>

namespace {
    // Support SAPSet::rotation
    bool match_irred(const std::vector<std::pair<size_t, size_t>> & x, const std::vector<std::pair<size_t, size_t>> & y) {
        assert(("x and y have a same size", x.size() == y.size()));
        bool match = true;
        for (size_t i = 0; i < x.size(); i++)
        if (x[i].first != y[i].first) {
            match = false;
            break;
        }
        return match;
    }

    // Support SAPSet::translation
    bool all_zero(const std::vector<std::pair<size_t, size_t>> & x) {
        bool all_zero = true;
        for (const auto & el : x) if (el.first != 0) {
            all_zero = false;
            break;
        }
        return all_zero;
    }
}

namespace tchem { namespace polynomial {

// Construct `orders_` based on constructed `SAP_`
void SAPSet::construct_orders_() {
    assert(("`SAPs_` must have been constructed", ! SAPs_.empty()));
    // find out the highest order among the SAPs
    max_order_ = 0;
    for (const SAP & sap : SAPs_)
    if (sap.order() > max_order_)
    max_order_ = sap.order();
    // construct a view to `SAPs_` grouped by order
    orders_.clear();
    orders_.resize(max_order_ + 1);
    for (const SAP & sap : SAPs_)
    orders_[sap.order()].push_back(& sap);
    // sanity check
    if (irreducible_ != 0 && (! orders_[0].empty())) throw std::invalid_argument(
    "tchem::SAPSet::construct_orders_: Only the totally symmetric irreducible can have 0th order term");
}

// Given a set of coordiantes constituting a SAP, return its index in this SAP set
// Return -1 if not found
int64_t SAPSet::index_SAP_(const std::vector<std::pair<size_t, size_t>> coords) const {
    size_t order = coords.size();
    // [) bisection search
    size_t lower = 0;
    for (size_t i = 0; i < order; i++) lower += orders_[i].size();
    size_t upper = lower + orders_[order].size();
    while (lower < upper) {
        size_t mid = (lower + upper) / 2;
        bool match = true;
        const std::vector<std::pair<size_t, size_t>> & ref_coords = SAPs_[mid].coords();
        int64_t i;
        for (i = coords.size() - 1; i > -1 ; i--)
        if (coords[i] != ref_coords[i]) {
            match = false;
            break;
        }
        if (match) return mid;
        // next range
        if (coords[i] > ref_coords[i]) lower = mid + 1;
        else                           upper = mid;
    }
    // bisection search is terminated by lower == upper rather than found, so try lower as the final round
    bool match = true;
    const std::vector<std::pair<size_t, size_t>> & ref_coords = SAPs_[lower].coords();
    for (int64_t i = coords.size() - 1; i > -1 ; i--)
    if (coords[i] != ref_coords[i]) {
        match = false;
        break;
    }
    if (match) return lower;
    else       return -1;
}

SAPSet::SAPSet() {}
// `sapoly_file` contains one SAP per line, who must meet the requirements of `SAPs_`
// At the end of each line, anything after # is considered as comment
SAPSet::SAPSet(const std::string & sapoly_file, const size_t & _irreducible, const std::vector<size_t> & _dimensions)
: irreducible_(_irreducible), dimensions_(_dimensions) {
    std::ifstream ifs; ifs.open(sapoly_file);
    if (! ifs.good()) throw CL::utility::file_error(sapoly_file);
    while (true) {
        std::string line;
        std::getline(ifs, line);
        if (! ifs.good()) break;
        std::vector<std::string> strs = CL::utility::split(line);
        // trim comments
        for (size_t i = 0; i < strs.size(); i++)
        if (strs[i] == "#") {
            strs.resize(i);
            break;
        }
        // create SAP from strings
        SAPs_.push_back(SAP(strs));
    }
    ifs.close();
    this->construct_orders_();
}
SAPSet::~SAPSet() {}

// insert a 0th order (const) term if the set is totally symmetic and does not have const yet
void SAPSet::insert_const() {
    if (irreducible_ == 0 & orders_[0].empty()) {
        auto copy = SAPs_;
        SAPs_.clear();
        SAPs_.resize(1 + copy.size());
        SAPs_[0] = SAP();
        for (size_t i = 0; i < copy.size(); i++) SAPs_[i + 1] = copy[i];
        this->construct_orders_();
    }
}

const std::vector<SAP> & SAPSet::SAPs() const {return SAPs_;}

// read-only reference to a symmetry adapted polynomial
const SAP & SAPSet::operator[](const size_t & index) const {return SAPs_[index];}

void SAPSet::pretty_print(std::ostream & stream) const {
    for (const SAP & sap : SAPs_) sap.pretty_print(stream);
}

// Return the value of each term in {P(x)} as a vector of vectors
at::Tensor SAPSet::operator()(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::operator(): x must be a vector");
    if (xs.size() != dimensions_.size()) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::operator(): x must share a same number of irreducibles as the coordinate system");
    for (size_t i = 0; i < xs.size(); i++) if (xs[i].size(0) != dimensions_[i]) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::operator(): x must have a same dimension as the coordinates");
    at::Tensor y = xs[0].new_empty(SAPs_.size());
    for (size_t i = 0; i < SAPs_.size(); i++) y[i] = SAPs_[i](xs);
    return y;
}
// Return d{P(x)} / dx given x
std::vector<at::Tensor> SAPSet::Jacobian(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian: x must be a vector");
    if (xs.size() != dimensions_.size()) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian: x must share a same number of irreducibles as the coordinate system");
    for (size_t i = 0; i < xs.size(); i++) if (xs[i].size(0) != dimensions_[i]) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian: x must have a same dimension as the coordinates");
    std::vector<at::Tensor> Js(xs.size());
    for (size_t i = 0; i < xs.size(); i++) Js[i] = xs[i].new_empty({(int64_t)SAPs_.size(), xs[i].size(0)});
    for (size_t i = 0; i < SAPs_.size(); i++) {
        std::vector<at::Tensor> rows = SAPs_[i].gradient(xs);
        for (size_t j = 0; j < xs.size(); j++) Js[j][i] = rows[j];
    }
    return Js;
}
std::vector<at::Tensor> SAPSet::Jacobian_(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian_: x must be a vector");
    if (xs.size() != dimensions_.size()) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian_: x must share a same number of irreducibles as the coordinate system");
    for (size_t i = 0; i < xs.size(); i++) if (xs[i].size(0) != dimensions_[i]) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian_: x must have a same dimension as the coordinates");
    std::vector<at::Tensor> Js(xs.size());
    for (size_t i = 0; i < xs.size(); i++) Js[i] = xs[i].new_empty({(int64_t)SAPs_.size(), xs[i].size(0)});
    for (size_t j = 0; j < SAPs_.size(); j++) {
        std::vector<at::Tensor> rows = SAPs_[j].gradient_(xs);
        for (size_t i = 0; i < xs.size(); i++) Js[i][j].copy_(rows[i]);
    }
    return Js;
}
// Return d{P(x)} / dx given x
// `J` harvests the concatenated symmetry adapted gradients
std::vector<at::Tensor> SAPSet::Jacobian_(const std::vector<at::Tensor> & xs, at::Tensor & J) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian_: x must be a vector");
    if (xs.size() != dimensions_.size()) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian_: x must share a same number of irreducibles as the coordinate system");
    for (size_t i = 0; i < xs.size(); i++) if (xs[i].size(0) != dimensions_[i]) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian_: x must have a same dimension as the coordinates");
    int64_t dimension = 0;
    for (const at::Tensor & x : xs) dimension += x.size(0);
    J = xs[0].new_empty({(int64_t)SAPs_.size(), dimension});
    int64_t start = 0, stop;
    std::vector<at::Tensor> Js(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        stop = start + xs[i].size(0);
        Js[i] = J.slice(1, start, stop);
        start = stop;
    }
    for (size_t i = 0; i < SAPs_.size(); i++) {
        at::Tensor grad;
        std::vector<at::Tensor> rows = SAPs_[i].gradient_(xs, grad);
        J[i].copy_(grad);
    }
    return Js;
}
// Return dd{P(x)} / dx^2 given x
CL::utility::matrix<at::Tensor> SAPSet::Jacobian2nd(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian2nd: x must be a vector");
    if (xs.size() != dimensions_.size()) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian2nd: x must share a same number of irreducibles as the coordinate system");
    for (size_t i = 0; i < xs.size(); i++) if (xs[i].size(0) != dimensions_[i]) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian2nd: x must have a same dimension as the coordinates");
    CL::utility::matrix<at::Tensor> Ks(xs.size());
    for (size_t i = 0; i < xs.size(); i++)
    for (size_t j = i; j < xs.size(); j++)
    Ks[i][j] = xs[i].new_zeros({(int64_t)SAPs_.size(), xs[i].size(0), xs[j].size(0)});
    for (size_t k = 0; k < SAPs_.size(); k++) {
        CL::utility::matrix<at::Tensor> rows = SAPs_[k].Hessian(xs);
        for (size_t i = 0; i < xs.size(); i++)
        for (size_t j = i; j < xs.size(); j++)
        Ks[i][j][k] = rows[i][j];
    }
    return Ks;
}
CL::utility::matrix<at::Tensor> SAPSet::Jacobian2nd_(const std::vector<at::Tensor> & xs) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian2nd_: x must be a vector");
    if (xs.size() != dimensions_.size()) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian2nd_: x must share a same number of irreducibles as the coordinate system");
    for (size_t i = 0; i < xs.size(); i++) if (xs[i].size(0) != dimensions_[i]) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian2nd_: x must have a same dimension as the coordinates");
    CL::utility::matrix<at::Tensor> Ks(xs.size());
    for (size_t i = 0; i < xs.size(); i++)
    for (size_t j = i; j < xs.size(); j++)
    Ks[i][j] = xs[i].new_zeros({(int64_t)SAPs_.size(), xs[i].size(0), xs[j].size(0)});
    for (size_t k = 0; k < SAPs_.size(); k++) {
        CL::utility::matrix<at::Tensor> rows = SAPs_[k].Hessian_(xs);
        for (size_t i = 0; i < xs.size(); i++)
        for (size_t j = i; j < xs.size(); j++)
        Ks[i][j][k].copy_(rows[i][j]);
    }
    return Ks;
}
// Return dd{P(x)} / dx^2 given x
// `K` harvests the concatenated symmetry adapted 2nd-order Jacobians
CL::utility::matrix<at::Tensor> SAPSet::Jacobian2nd_(const std::vector<at::Tensor> & xs, at::Tensor & K) const {
    for (const at::Tensor & x : xs) if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian2nd_: x must be a vector");
    if (xs.size() != dimensions_.size()) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian2nd_: x must share a same number of irreducibles as the coordinate system");
    for (size_t i = 0; i < xs.size(); i++) if (xs[i].size(0) != dimensions_[i]) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::Jacobian2nd_: x must have a same dimension as the coordinates");
    int64_t dimension = 0;
    for (const at::Tensor & x : xs) dimension += x.size(0);
    K = xs[0].new_empty({(int64_t)SAPs_.size(), dimension, dimension});
    int64_t start_row = 0, stop_row;
    CL::utility::matrix<at::Tensor> Ks(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        stop_row = start_row + xs[i].size(0);
        int64_t start_col = start_row, stop_col;
        for (size_t j = i; j < xs.size(); j++) {
            stop_col = start_col + xs[j].size(0);
            Ks[i][j] = K.slice(1, start_row, stop_row).slice(2, start_col, stop_col);
            start_col = stop_col;
        }
        start_row = stop_row;
    }
    for (size_t k = 0; k < SAPs_.size(); k++) {
        CL::utility::matrix<at::Tensor> rows = SAPs_[k].Hessian_(xs);
        for (size_t i = 0; i < xs.size(); i++)
        for (size_t j = i; j < xs.size(); j++)
        Ks[i][j][k].copy_(rows[i][j]);
    }
    // Copy the upper triangle to the lower triangle
    for (size_t i = 0; i < dimension; i++)
    for (size_t j = i + 1; j < dimension; j++)
    K.select(1, j).select(-1, i).copy_(K.select(1, i).select(-1, j));
    return Ks;
}

// Consider coordinate rotation y[irred] = U[irred]^-1 . x[irred]
// so the SAP set rotates as {SAP(x)} = T . {SAP(y)}
// Assuming:
// 1. If there are 1st order terms, all are present
// 2. SAP.coords are sorted
// Return rotation matrix T
at::Tensor SAPSet::rotation(const std::vector<at::Tensor> & U, const SAPSet & y_set) const {
    if (dimensions_.size() != U.size()) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::rotation: inconsistent number of irreducibles between U and the coordinate system");
    for (size_t irred = 0; irred < U.size(); irred++) {
        if (U[irred].sizes().size() != 2) throw std::invalid_argument(
        "tchem::polynomial::SAPSet::rotation: U must be a matrix");
        if (U[irred].size(0) != U[irred].size(1)) throw std::invalid_argument(
        "tchem::polynomial::SAPSet::rotation: U must be a square matrix");
        if (U[irred].size(0) != dimensions_[irred]) throw std::invalid_argument(
        "tchem::polynomial::SAPSet::rotation: U must share a same dimension with the coordinates");
    }
    if (max_order_ != y_set.max_order_) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::rotation: The 2 polynomial sets must share a same order");
    // 0 and 1 dimensional coordinates do not rotate at all
    if (std::accumulate(dimensions_.begin(), dimensions_.end(), 0) < 2) return at::eye(SAPs_.size(), U[0].options());
    // Allocate memory
    at::Tensor T = U[0].new_zeros({(int64_t)SAPs_.size(), (int64_t)y_set.SAPs_.size()});
    // Start filling in T
    size_t start_x = 0, start_y = 0;
    // 0th order term does not rotate
    if (! orders_[0].empty()) {
        T[0][0] = 1.0;
        start_x++;
        start_y++;
    }
    // 1st order terms rotate as x[irred] = U[irred] . y[irred]
    if (max_order_ >= 1) if (! orders_[1].empty()) {
        T.slice(0, start_x, start_x + dimensions_[irreducible_])
         .slice(1, start_y, start_y + dimensions_[irreducible_])
         .copy_(U[irreducible_]);
        start_x += dimensions_[irreducible_];
        start_y += dimensions_[irreducible_];
    }
    // 2nd and higher order terms rotate as
    // x[irred1]i1 x[irred2]i2 ... x[irredn]in
    //     = (U[irred1]i1j1 y[irred1]j1) (U[irred2]i2j2 y[irred2]j2) ... (U[irredn]injn y[irredn]jn)
    //     = (U[irred1]i1j1 U[irred2]i2j2 ... U[irredn]injn) (y[irred1]j1 y[irred2]j2 ... y[irredn]jn)
    // equivalent (y[irred1]j1 y[irred2]j2 ... y[irredn]jn)s have their (U[irred1]i1j1 U[irred2]i2j2 ... U[irredn]injn)s merged
    for (size_t iorder = 2; iorder <= max_order_; iorder++) {
        size_t NTerms_x = orders_[iorder].size();
        size_t   stop_x = start_x + NTerms_x;
        size_t NTerms_y = y_set.orders_[iorder].size();
        size_t   stop_y = start_y + NTerms_y;
        at::Tensor T_block = T.slice(0, start_x, stop_x).slice(1, start_y, stop_y);
        for (size_t i = 0; i < NTerms_x; i++) {
            auto x_coords = orders_[iorder][i]->coords();
            for (size_t j = 0; j < NTerms_y; j++) {
                auto y_coords = y_set.orders_[iorder][j]->coords();
                if (! match_irred(x_coords, y_coords)) {
                    T_block[i][j] = 0.0;
                    continue;
                }
                // only 1 permutation when all coordinates are the same
                if (y_set.orders_[iorder][j]->uniques_orders().size() == 1) {
                    T_block[i][j] = U[x_coords[0].first][x_coords[0].second][y_coords[0].second];
                    for (size_t k = 1; k < iorder; k++) T_block[i][j] *= U[x_coords[k].first][x_coords[k].second][y_coords[k].second];
                }
                // sum over all permutations of the unique coordinates
                // reference: https://www.geeksforgeeks.org/print-all-permutations-of-a-string-with-duplicates-allowed-in-input-string
                else {
                    // the 1st permutation: all coordinates sorted ascendingly
                    std::sort(y_coords.begin(), y_coords.end());
                    // the following permutations
                    while (true) {
                        // sum the current permutation
                        if (match_irred(x_coords, y_coords)) {
                            at::Tensor current = U[0].new_empty({});
                            current.copy_(U[x_coords[0].first][x_coords[0].second][y_coords[0].second]);
                            for (size_t k = 1; k < iorder; k++) current *= U[x_coords[k].first][x_coords[k].second][y_coords[k].second];
                            T_block[i][j] += current;
                        }
                        // find the rightmost element which is smaller than its next
                        // let us call it "edge element"
                        int64_t edge_index;
                        for (edge_index = iorder - 2; edge_index > -1; edge_index--)
                        if (y_coords[edge_index] < y_coords[edge_index + 1]) break;
                        // no such element, all sorted descendingly, done
                        if (edge_index == -1) break;
                        // find the ceil of "edge element" in the right of it
                        // ceil of an element is the smallest element greater than it
                        size_t ceil_index = edge_index + 1;
                        for (size_t k = edge_index + 2; k < iorder; k++)
                        if (y_coords[k] > y_coords[edge_index]
                        &&  y_coords[k] < y_coords[ceil_index]) ceil_index = k;
                        // swap edge and ceil
                        auto save = y_coords[edge_index];
                        y_coords[edge_index] = y_coords[ceil_index];
                        y_coords[ceil_index] = save;
                        // sort the sub vector on the right of edge
                        std::sort(y_coords.begin() + edge_index + 1, y_coords.end());
                    }
                }
            }
        }
        start_x = stop_x;
        start_y = stop_y;
    }
    return T;
}
// Assuming terms are the same under rotation
at::Tensor SAPSet::rotation(const std::vector<at::Tensor> & U) const {return rotation(U, * this);}

// Consider coordinate translation y[0] = x[0] - a
// i.e. only the totally symmetric irreducible translates so that symmetry preserves
// so the SAP set translates as {SAP(x)} = T . {SAP(y)}
// Assuming:
// 1. The totally symmetric irreducible must have the 0th order term
// 2. If the totally symmetric irreducible has 1st order terms, all are present
// Return translation matrix T
at::Tensor SAPSet::translation(const at::Tensor & a, const SAPSet & y_set) const {
    if (a.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::translation: a must be a vector");
    if (a.size(0) != dimensions_[0]) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::translation: a must share a same dimension with the coordinates");
    if (max_order_ != y_set.max_order_) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::translation: The 2 polynomial sets must share a same order");
    // Allocate memory
    at::Tensor T = a.new_zeros({(int64_t)SAPs_.size(), (int64_t)y_set.SAPs_.size()});
    // Start filling in T
    size_t start_x = 0;
    // Totally symmetric irreducible must have 0th order term and may translate 1st order terms
    if (irreducible_ == 0) {
        if (orders_[0].empty()) throw std::invalid_argument(
        "tchem::polynomial::SAPSet::translation: The totally symmetric irreducible must have the 0th order term");
        // 0th order term does not shift
        T[0][0] = 1.0;
        start_x++;
        // 1st order terms shift as x[0] = y[0] + a
        if (max_order_ >= 1) if (! orders_[1].empty())
        for (size_t i = 0; i < dimensions_[0]; i++) {
            T[start_x][0      ] = a[i];
            T[start_x][start_x] = 1.0;
            start_x++;
        }
    }
    // Other irreducibles does not translate 1st order terms
    else if (max_order_ >= 1) if (! orders_[1].empty()) {
        start_x = dimensions_[irreducible_];
        T.slice(0, 0, start_x).slice(1, 0, start_x).fill_diagonal_(1.0);
    }
    // 2nd and higher order terms shift as
    // x[irred1]i1 x[irred2]i2 ... x[irredn]in
    //     = (y[irred1]i1 + a[irred1]i1) (y[irred2]i2 + a[irred2]i2) ... (y[irredn]in + a[irredn]in)
    //     = a[irred1]i1 a[irred2]i2 ... a[irredn]in
    //     + y[irred1]i1 a[irred2]i2 ... a[irredn]in
    //     + ...
    //     + y[irred1]i1 y[irred2]i2 ... y[irredn]in
    // if irred != 0: a[irred]i = 0
    for (size_t iorder = 2; iorder <= max_order_; iorder++) {
        size_t NTerms_x = orders_[iorder].size();
        size_t   stop_x = start_x + NTerms_x;
        at::Tensor T_block = T.slice(0, start_x, stop_x);
        for (size_t i = 0; i < NTerms_x; i++) {
            auto x_coords = orders_[iorder][i]->coords();
            // The 1st term: a[irred1]i1 a[irred2]i2 ... a[irredn]in
            if (all_zero(x_coords)) {
                T_block[i][0] = a[x_coords[0].second];
                for (size_t j = 1; j < iorder; j++) T_block[i][0] *= a[x_coords[j].second];
            }
            else T_block[i][0] = 0.0;
            // The other terms: as a binary counter
            // when a bit == 1, place y there
            std::vector<size_t> use_var(iorder, 0);
            while (true) {
                use_var[0] += 1;
                // Carry to latter digits
                for (size_t j = 0; j < iorder - 1; j++)
                if (use_var[j] == 2) {
                    use_var[j] = 0;
                    use_var[j + 1] += 1;
                }
                // Binary counter overflows, done
                if (use_var.back() == 2) break;
                // Build the coordinates for y and a
                size_t NVars = std::accumulate(use_var.begin(), use_var.end(), 0);
                size_t NCons = iorder - NVars;
                std::vector<std::pair<size_t, size_t>> y_coords(NVars), a_coords(NCons);
                size_t count_q = 0, count_a = 0;
                for (size_t j = 0; j < iorder; j++)
                if (use_var[j] == 1) {
                    y_coords[count_q] = x_coords[j];
                    count_q++;
                }
                else {
                    a_coords[count_a] = x_coords[j];
                    count_a++;
                }
                // Determine T block element
                if (all_zero(a_coords)) {
                    int64_t index = y_set.index_SAP_(y_coords);
                    if (index >= 0) {
                        at::Tensor current = a.new_empty({});
                        current.fill_(1.0);
                        for (auto & coord : a_coords) current *= a[coord.second];
                        T_block[i][index] += current;
                    }
                }
            }
        }
        start_x = stop_x;
    }
    return T;
}
// Assuming terms are the same under translation
at::Tensor SAPSet::translation(const at::Tensor & a) const {return translation(a, * this);}

} // namespace polynomial
} // namespace tchem