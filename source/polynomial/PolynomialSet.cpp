#include <CppLibrary/math.hpp>

#include <tchem/polynomial/PolynomialSet.hpp>

namespace tchem { namespace polynomial {

// Construct `max_order_` and `orders_` based on constructed `polynomials_`
void PolynomialSet::construct_orders_() {
    assert(("`polynomials_` must have been constructed", ! polynomials_.empty()));
    // Find out the highest order among the polynomials
    max_order_ = 0;
    for (const Polynomial & polynomial : polynomials_)
    if (polynomial.order() > max_order_)
    max_order_ = polynomial.order();
    // Construct a view to `polynomials_` grouped by order
    orders_.resize(max_order_ + 1);
    for (const Polynomial & polynomial : polynomials_)
    orders_[polynomial.order()].push_back(& polynomial);
}

// Given a set of coordiantes constituting a polynomial, try to locate its index within [lower, upper]
void PolynomialSet::bisect_(const std::vector<size_t> coords, const size_t & lower, const size_t & upper, int64_t & index) const {
    // Final round
    if (upper - lower == 1) {
        // Try lower
        bool match = true;
        std::vector<size_t> ref_coords = polynomials_[lower].coords();
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
        ref_coords = polynomials_[upper].coords();
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
        std::vector<size_t> ref_coords = polynomials_[bisection].coords();
        size_t i;
        // Although the comparison should be made from the last coordinate to the first,
        // we can still start from the first since Polynomial has its coordinates sorted descendingly
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
// Given a set of coordiantes constituting a polynomial, return its index in this polynomial set
// Return -1 if not found
int64_t PolynomialSet::index_polynomial_(const std::vector<size_t> coords) const {
    size_t order = coords.size();
    size_t lower = 0;
    for (size_t i = 0; i < order; i++) lower += orders_[i].size();
    size_t upper = lower + orders_[order].size() - 1;
    int64_t index;
    bisect_(coords, lower, upper, index);
    return index;
}

PolynomialSet::PolynomialSet() {}
PolynomialSet::PolynomialSet(const std::vector<Polynomial> & _polynomials, const size_t & _dimension)
: polynomials_(_polynomials), dimension_(_dimension) {this->construct_orders_();}
// Generate all possible terms up to `order`-th order constituting of all `dimension` coordinates
PolynomialSet::PolynomialSet(const size_t & _dimension, const size_t & _order)
: dimension_(_dimension), max_order_(_order) {
    // 0th order term only
    if (dimension_ == 0) {
        polynomials_.resize(1);
        std::vector<size_t> coords;
        polynomials_[0] = Polynomial(coords);
        return;
    }
    // Count number of terms
    size_t count = 0;
    for (size_t i = 0; i <= max_order_; i++) count += CL::math::iCombination(dimension_ + i - 1, i);
    polynomials_.resize(count);
    // Generate 0th order term
    std::vector<size_t> coords;
    polynomials_[0] = Polynomial(coords);
    // Generate 1st and higher orders
    count = 1;
    for (size_t iorder = 1; iorder <= max_order_; iorder++) {
        // The 1st term: r0^iorder
        coords.resize(iorder);
        fill(coords.begin(), coords.end(), 0);
        polynomials_[count] = Polynomial(coords, true);
        count++;
        // The other terms: as a dimension-nary counter
        // with former digit >= latter digit to avoid double counting
        if (dimension_ > 1) while (true) {
            coords[0] += 1;
            // Carry to latter digits
            for (size_t i = 0; i < iorder - 1; i++)
            if (coords[i] >= dimension_) {
                coords[i] = 0;
                coords[i + 1] += 1;
            }
            // Guarantee former digit >= latter digit
            for (int64_t i = iorder - 2; i > -1; i--)
            if (coords[i] < coords[i + 1]) {
                coords[i] = coords[i + 1];
            }
            polynomials_[count] = Polynomial(coords, true);
            count++;
            if (coords.back() >= dimension_ - 1) break;
        }
    }
    // Sanity check
    if (count != polynomials_.size()) std::cerr << "Error in PolynomialSet construction";
    this->construct_orders_();
}
PolynomialSet::~PolynomialSet() {}

const std::vector<Polynomial> & PolynomialSet::polynomials() const {return polynomials_;}
const size_t & PolynomialSet::dimension() const {return dimension_;}
const size_t & PolynomialSet::max_order() const {return max_order_;}
const std::vector<std::vector<const Polynomial *>> & PolynomialSet::orders() const {return orders_;}

// Read-only reference to a polynomial
const Polynomial & PolynomialSet::operator[](const size_t & index) const {return polynomials_[index];}

// Given `x`, the value of each term in {P(x)} as a vector
// Return views to `x` grouped by order
std::vector<at::Tensor> PolynomialSet::views(const at::Tensor & x) const {
    assert(("x must be a vector", x.sizes().size() == 1));
    std::vector<at::Tensor> views(max_order_ + 1);
    size_t start = 0, stop;
    for (size_t i = 0; i < max_order_ + 1; i++) {
        stop = start + orders_[i].size();
        views[i] = x.slice(0, start, stop);
        start = stop;
    }
    assert(("The length of x must equal to the number of polynomial terms", x.size(0) == stop));
    return views;
}

// Return the value of each term in {P(x)} given x as a vector
at::Tensor PolynomialSet::operator()(const at::Tensor & x) const {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::PolynomialSet::operator(): x must be a vector");
    if (x.size(0) != dimension_) throw std::invalid_argument(
    "tchem::polynomial::PolynomialSet::operator(): x must have a same dimension as the coordinates");
    at::Tensor value = x.new_empty(polynomials_.size());
    for (size_t i = 0; i < polynomials_.size(); i++) value[i] = polynomials_[i](x);
    return value;
}
// Return d{P(x)} / dx given x
at::Tensor PolynomialSet::Jacobian(const at::Tensor & x) const {
    if (x.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::PolynomialSet::Jacobian: x must be a vector");
    if (x.size(0) != dimension_) throw std::invalid_argument(
    "tchem::polynomial::PolynomialSet::Jacobian: x must have a same dimension as the coordinates");
    at::Tensor J = x.new_empty({(int64_t)polynomials_.size(), x.size(0)});
    for (size_t i = 0; i < polynomials_.size(); i++) J[i] = polynomials_[i].gradient(x);
    return J;
}

// Consider coordinate rotation y = U^-1 . x
// so the polynomial set rotates as {P(x)} = T . {P(y)}
// Assuming:
//     1. All 0th and 1st order terms are present
//     2. Polynomial.coords are sorted
// Return rotation matrix T
at::Tensor PolynomialSet::rotation(const at::Tensor & U, const PolynomialSet & y_set) const {
    if (U.sizes().size() != 2) throw std::invalid_argument(
    "tchem::polynomial::PolynomialSet::rotation: U must be a matrix");
    if (U.size(0) != U.size(1)) throw std::invalid_argument(
    "tchem::polynomial::PolynomialSet::rotation: U must be a square matrix");
    if (U.size(0) != dimension_) throw std::invalid_argument(
    "tchem::polynomial::PolynomialSet::rotation: U must share a same dimension with the coordinates");
    if (max_order_ != y_set.max_order_) throw std::invalid_argument(
    "tchem::polynomial::PolynomialSet::rotation: The 2 polynomial sets must share a same order");
    // 0 and 1 dimensional coordinates do not rotate at all
    if (dimension_ < 2) return at::eye(polynomials_.size(), U.options());
    // Allocate memory
    at::Tensor T = U.new_zeros({(int64_t)polynomials_.size(), (int64_t)y_set.polynomials_.size()});
    // 0th order term does not rotate
    T[0][0] = 1.0;
    // 1st order terms rotate as x = U . y
    if (max_order_ >= 1)
    T.slice(0, 1, dimension_ + 1).slice(1, 1, dimension_ + 1).copy_(U);
    // 2nd and higher order terms rotate as
    // xi1 xi2 ... xin = (Ui1j1 yj1) (Ui2j2 yj2) ... (Uinjn yjn)
    //                 = (Ui1j1 Ui2j2 ... Uinjn) (yj1 yj2 ... yjn)
    // equivalent (yj1 yj2 ... yjn)s have their (Ui1j1 Ui2j2 ... Uinjn)s merged
    size_t start_x = dimension_ + 1;
    size_t start_y = dimension_ + 1;
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
                // Get  the unique coordinates and their number of repeats
                // i.e. the unique coordinates and their orders
                std::vector<size_t> uniques, repeats;
                std::tie(uniques, repeats) = y_set.orders_[iorder][j]->uniques_orders();
                // Only 1 permutation when all coordinates are the same
                if (uniques.size() == 1) {
                    T_block[i][j] = U[x_coords[0]][y_coords[0]];
                    for (size_t k = 1; k < iorder; k++) T_block[i][j] *= U[x_coords[k]][y_coords[k]];
                }
                // Sum over all permutations of the unique coordinates
                // Reference: https://www.geeksforgeeks.org/print-all-permutations-of-a-string-with-duplicates-allowed-in-input-string
                else {
                    // The 1st permutation: all coordinates sorted ascendingly
                    std::sort(y_coords.begin(), y_coords.end());
                    // The following permutations
                    while (true) {
                        // Sum the current permutation
                        at::Tensor current = U.new_empty({});
                        current.copy_(U[x_coords[0]][y_coords[0]]);
                        for (size_t k = 1; k < iorder; k++) current *= U[x_coords[k]][y_coords[k]];
                        T_block[i][j] += current;
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
                        size_t save = y_coords[edge_index];
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
at::Tensor PolynomialSet::rotation(const at::Tensor & U) const {return rotation(U, * this);}

// Consider coordinate translation y = x - a
// so the polynomial set translates as {P(x)} = T . {P(y)}
// Assuming:
//     1. All 0th and 1st order terms are present
// Return translation matrix T
at::Tensor PolynomialSet::translation(const at::Tensor & a, const PolynomialSet & y_set) const {
    if (a.sizes().size() != 1) throw std::invalid_argument(
    "tchem::polynomial::PolynomialSet::translation: a must be a vector");
    if (a.size(0) != dimension_) throw std::invalid_argument(
    "tchem::polynomial::PolynomialSet::translation: a must share a same dimension with the coordinates");
    if (max_order_ != y_set.max_order_) throw std::invalid_argument(
    "tchem::polynomial::PolynomialSet::translation: The 2 polynomial sets must share a same order");
    // Allocate memory
    at::Tensor T = a.new_zeros({(int64_t)polynomials_.size(), (int64_t)y_set.polynomials_.size()});
    // 0th order term does not shift
    T[0][0] = 1.0;
    // 1st order terms shift as x = y + a
    if (max_order_ >= 1) for (size_t i = 1; i < dimension_ + 1; i++) {
        T[i][0] = a[i - 1];
        T[i][i] = 1.0;
    }
    // 2nd and higher order terms shift as
    // xi1 xi2 ... xin = (yi1 + ai1) (yi2 + ai2) ... (yin + ain)
    //                 = ai1 ai2 ... ain + yi1 ai2 ... ain + ... + yi1 yi2 ... yin
    size_t start_x = dimension_ + 1;
    for (size_t iorder = 2; iorder <= max_order_; iorder++) {
        size_t NTerms_x = orders_[iorder].size();
        size_t   stop_x = start_x + NTerms_x;
        at::Tensor T_block = T.slice(0, start_x, stop_x);
        for (size_t i = 0; i < NTerms_x; i++) {
            auto x_coords = orders_[iorder][i]->coords();
            // The 1st term: ai1 ai2 ... ain
            T_block[i][0] = a[x_coords[0]];
            for (size_t j = 1; j < iorder; j++) T_block[i][0] *= a[x_coords[j]];
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
                std::vector<size_t> y_coords(NVars), a_coords(NCons);
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
                int64_t index = y_set.index_polynomial_(y_coords);
                if (index >= 0) {
                    at::Tensor current = a.new_empty({});
                    current.fill_(1.0);
                    for (size_t & coord : a_coords) current *= a[coord];
                    T_block[i][index] += current;
                }
            }
        }
        start_x = stop_x;
    }
    return T;
}
// Assuming terms are the same under translation
at::Tensor PolynomialSet::translation(const at::Tensor & a) const {return translation(a, * this);}

} // namespace polynomial
} // namespace tchem