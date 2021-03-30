#include <regex>

#include <CppLibrary/utility.hpp>

#include <tchem/polynomial/SAPSet.hpp>

namespace {
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
}

namespace tchem { namespace polynomial {

// Construct `orders_` based on constructed `SAP_`
void SAPSet::construct_orders_() {
    assert(("`SAPs_` must have been constructed", ! SAPs_.empty()));
    // Find out the highest order among the SAPs
    max_order_ = 0;
    for (const SAP & sap : SAPs_)
    if (sap.order() > max_order_)
    max_order_ = sap.order();
    // Construct a view to `SAPs_` grouped by order
    orders_.clear();
    orders_.resize(max_order_ + 1);
    for (const SAP & sap : SAPs_)
    orders_[sap.order()].push_back(& sap);
}

// Given a set of coordiantes constituting a SAP, try to locate its index within [lower, upper]
void SAPSet::bisect_(const std::vector<std::pair<size_t, size_t>> coords, const size_t & lower, const size_t & upper, int64_t & index) const {
    // Final round
    if (upper - lower == 1) {
        // Try lower
        bool match = true;
        std::vector<std::pair<size_t, size_t>> ref_coords = SAPs_[lower].coords();
        for (size_t i = 0; i < coords.size(); i++)
        if (coords[i] != ref_coords[i]) {
            match = false;
            break;
        }
        if (match) {
            index = lower;
            return;
        }
        // Try upper
        match = true;
        ref_coords = SAPs_[upper].coords();
        for (size_t i = 0; i < coords.size(); i++)
        if (coords[i] != ref_coords[i]) {
            match = false;
            break;
        }
        if (match) {
            index = upper;
            return;
        }
        // Neither
        index = -1;
    }
    // Normal bisection process
    else {
        // Try bisection
        size_t bisection = (lower + upper) / 2;
        bool match = true;
        std::vector<std::pair<size_t, size_t>> ref_coords = SAPs_[bisection].coords();
        size_t i;
        for (i = 0; i < coords.size(); i++)
        if (coords[i] != ref_coords[i]) {
            match = false;
            break;
        }
        if (match) {
            index = bisection;
            return;
        }
        // Next range
        if (coords[i] > ref_coords[i]) bisect_(coords, bisection, upper, index);
        else                           bisect_(coords, lower, bisection, index);
    }
}
// Given a set of coordiantes constituting a SAP, return its index in this SAP set
// Return -1 if not found
int64_t SAPSet::index_SAP_(const std::vector<std::pair<size_t, size_t>> coords) const {
    size_t order = coords.size();
    size_t lower = 0;
    for (size_t i = 0; i < order; i++) lower += orders_[i].size();
    size_t upper = lower + orders_[order].size() - 1;
    int64_t index;
    bisect_(coords, lower, upper, index);
    return index;
}

SAPSet::SAPSet() {}
// `sapoly_file` contains one SAP per line, who must meet the requirements of `SAPs_`
SAPSet::SAPSet(const std::string & sapoly_file, const size_t & _irreducible, const std::vector<size_t> & _dimensions)
: irreducible_(_irreducible), dimensions_(_dimensions) {
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
    this->construct_orders_();
}
SAPSet::~SAPSet() {}

const std::vector<SAP> & SAPSet::SAPs() const {return SAPs_;}

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

// Consider coordinate rotation y[irred] = U[irred]^-1 . x[irred]
// so the SAP set rotates as {SAP(x)} = T . {SAP(y)}
// Assuming:
//     1. All 0th and 1st order terms are present
//     2. SAP.coords are sorted
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
    size_t start_x, start_y;
    // Totally symmetric irreducible has 0th order term
    if (irreducible_ == 0) {
        // 0th order term does not rotate
        T[0][0] = 1.0;
        // 1st order terms rotate as x[irred] = U[irred] . y[irred]
        if (max_order_ >= 1)
        T.slice(0, 1, dimensions_[irreducible_] + 1).slice(1, 1, dimensions_[irreducible_] + 1).copy_(U[irreducible_]);
        start_x = dimensions_[irreducible_] + 1;
        start_y = dimensions_[irreducible_] + 1;
    }
    // Other irreducibles starts from 1st order
    else {
        // 1st order terms rotate as x[irred] = U[irred] . y[irred]
        if (max_order_ >= 1)
        T.slice(0, 0, dimensions_[irreducible_]).slice(1, 0, dimensions_[irreducible_]).copy_(U[irreducible_]);
        start_x = dimensions_[irreducible_];
        start_y = dimensions_[irreducible_];
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
                // Get  the unique coordinates and their number of repeats
                // i.e. the unique coordinates and their orders
                std::vector<std::pair<size_t, size_t>> uniques;
                std::vector<size_t> repeats;
                std::tie(uniques, repeats) = y_set.orders_[iorder][j]->uniques_orders();
                // Only 1 permutation when all coordinates are the same
                if (uniques.size() == 1) {
                    T_block[i][j] = U[x_coords[0].first][x_coords[0].second][y_coords[0].second];
                    for (size_t k = 1; k < iorder; k++) T_block[i][j] *= U[x_coords[k].first][x_coords[k].second][y_coords[k].second];
                }
                // Sum over all permutations of the unique coordinates
                // Reference: https://www.geeksforgeeks.org/print-all-permutations-of-a-string-with-duplicates-allowed-in-input-string
                else {
                    // The 1st permutation: all coordinates sorted ascendingly
                    std::sort(y_coords.begin(), y_coords.end());
                    // The following permutations
                    while (true) {
                        // Sum the current permutation
                        if (match_irred(x_coords, y_coords)) {
                            at::Tensor current = U[0].new_empty({});
                            current.copy_(U[x_coords[0].first][x_coords[0].second][y_coords[0].second]);
                            for (size_t k = 1; k < iorder; k++) current *= U[x_coords[k].first][x_coords[k].second][y_coords[k].second];
                            T_block[i][j] += current;
                        }
                        // Find the rightmost element which is smaller than its next
                        // Let us call it "edge element"
                        int64_t edge_index;
                        for (edge_index = iorder - 2; edge_index > -1; edge_index--)
                        if (y_coords[edge_index] < y_coords[edge_index + 1]) break;
                        // No such element, all sorted descendingly, done
                        if (edge_index == -1) break;
                        // Find the ceil of "edge element" in the right of it
                        // Ceil of an element is the smallest element greater than it
                        size_t ceil_index = edge_index + 1;
                        for (size_t k = edge_index + 2; k < iorder; k++)
                        if (y_coords[k] > y_coords[edge_index]
                        &&  y_coords[k] < y_coords[ceil_index]) ceil_index = k;
                        // Swap edge and ceil
                        auto save = y_coords[edge_index];
                        y_coords[edge_index] = y_coords[ceil_index];
                        y_coords[ceil_index] = save;
                        // Sort the sub vector on the right of edge
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

// Consider coordinate translation y[irred] = x[irred] - a[irred]
// so the SAP set translates as {SAP(x)} = T . {SAP(y)}
// Assuming:
//     1. All 0th and 1st order terms are present
// Return translation matrix T
at::Tensor SAPSet::translation(const std::vector<at::Tensor> & a, const SAPSet & y_set) const {
    if (dimensions_.size() != a.size()) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::translation: inconsistent number of irreducibles between a and the coordinate system");
    for (size_t irred = 0; irred < a.size(); irred++) {
        if (a[irred].sizes().size() != 1) throw std::invalid_argument(
        "tchem::polynomial::SAPSet::translation: a must be a vector");
        if (a[irred].size(0) != dimensions_[irred]) throw std::invalid_argument(
        "tchem::polynomial::SAPSet::translation: a must share a same dimension with the coordinates");
    }
    if (max_order_ != y_set.max_order_) throw std::invalid_argument(
    "tchem::polynomial::SAPSet::translation: The 2 polynomial sets must share a same order");
    // Allocate memory
    at::Tensor T = a[0].new_zeros({(int64_t)SAPs_.size(), (int64_t)y_set.SAPs_.size()});
    // Start filling in T
    size_t start_x;
    // Totally symmetric irreducible has 0th order term
    if (irreducible_ == 0) {
        // 0th order term does not shift
        T[0][0] = 1.0;
        // 1st order terms shift as x[irred] = y[irred] + a[irred]
        if (max_order_ >= 1)
        for (size_t i = 0; i < dimensions_[irreducible_]; i++) {
            T[i + 1][0    ] = a[irreducible_][i];
            T[i + 1][i + 1] = 1.0;
        }
        start_x = dimensions_[irreducible_] + 1;
    }
    // Other irreducibles starts from 1st order
    else {
        // 1st order terms shift as x[irred] = y[irred] + a[irred]
        if (max_order_ >= 1)
        for (size_t i = 0; i < dimensions_[irreducible_]; i++) {
            T[i][0] = a[irreducible_][i];
            T[i][i] = 1.0;
        }
        start_x = dimensions_[irreducible_];
    }
    // 2nd and higher order terms shift as
    // x[irred1]i1 x[irred2]i2 ... x[irredn]in
    //     = (y[irred1]i1 + a[irred1]i1) (y[irred2]i2 + a[irred2]i2) ... (y[irredn]in + a[irredn]in)
    //     = a[irred1]i1 a[irred2]i2 ... a[irredn]in
    //     + y[irred1]i1 a[irred2]i2 ... a[irredn]in
    //     + ...
    //     + y[irred1]i1 y[irred2]i2 ... y[irredn]in
    for (size_t iorder = 2; iorder <= max_order_; iorder++) {
        size_t NTerms_x = orders_[iorder].size();
        size_t   stop_x = start_x + NTerms_x;
        at::Tensor T_block = T.slice(0, start_x, stop_x);
        for (size_t i = 0; i < NTerms_x; i++) {
            auto x_coords = orders_[iorder][i]->coords();
            // The 1st term: a[irred1]i1 a[irred2]i2 ... a[irredn]in
            T_block[i][0] = a[x_coords[0].first][x_coords[0].second];
            for (size_t j = 1; j < iorder; j++) T_block[i][0] *= a[x_coords[j].first][x_coords[j].second];
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
                int64_t index = y_set.index_SAP_(y_coords);
                if (index >= 0) {
                    at::Tensor current = a[0].new_empty({});
                    current.fill_(1.0);
                    for (auto & coord : a_coords) current *= a[coord.first][coord.second];
                    T_block[i][index] += current;
                }
            }
        }
        start_x = stop_x;
    }
    return T;
}
// Assuming terms are the same under translation
at::Tensor SAPSet::translation(const std::vector<at::Tensor> & a) const {return translation(a, * this);}

} // namespace polynomial
} // namespace tchem