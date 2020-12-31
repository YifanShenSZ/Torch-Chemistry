#include <torch/torch.h>

#include <CppLibrary/math.hpp>

#include <tchem/polynomial.hpp>

using namespace torch::indexing;

namespace tchem { namespace polynomial {

Polynomial::Polynomial() {}
Polynomial::Polynomial(const std::vector<size_t> & coords) : coords_(coords) {}
Polynomial::~Polynomial() {}

// Return the polynomial value
at::Tensor Polynomial::value(const at::Tensor & r) {
    at::Tensor value = r.new_empty(1);
    value[0] = 1.0;
    for (size_t & coord : coords_) value *= r[coord];
    return value[0];
}



// Construct `orders_` after `polynomials_` has been constructed
void PolynomialSet::create_orders() {
    orders_.clear();
    size_t order_old = -1;
    for (auto & polynomial : polynomials_) {
        size_t order = polynomial.coords().size();
        if (order != order_old) {
            orders_.push_back(std::vector<Polynomial *>());
            order_old = order;
        }
        orders_.back().push_back(& polynomial);
    }
}

// Given a set of coordiantes constituting a polynomial,
// try to locate its index within [lower, upper]
void PolynomialSet::bisect(const std::vector<size_t> coords, const size_t & lower, const size_t & upper, int & index) {
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
        if (coords[i] > ref_coords[i]) bisect(coords, bisection, upper, index);
        else                           bisect(coords, lower, bisection, index);
    }
}
// Given a set of coordiantes constituting a polynomial,
// find its index in this polynomial set
// If not found, return -1
int PolynomialSet::index_polynomial(const std::vector<size_t> coords) {
    size_t order = coords.size();
    size_t lower = 0;
    for (size_t i = 0; i < order; i++) lower += orders_[i].size();
    size_t upper = lower + orders_[order].size() - 1;
    int index;
    bisect(coords, lower, upper, index);
    return index;
}

PolynomialSet::PolynomialSet() {}
// Generate all possible terms up to `order`-th order constituting of all `dimension` coordinates
// Polynomial.coords are sorted descendingly, e.g. 2-dimensional 2nd-order terms: r0 r0, r1 r0, r1 r1
PolynomialSet::PolynomialSet(const size_t & dimension_, const size_t & order_)
: dimension(dimension_), order(order_) {
    // 0th order term only
    if (dimension == 0) {
        polynomials_.resize(1);
        std::vector<size_t> coords;
        polynomials_[0] = Polynomial(coords);
        return;
    }
    // Count number of terms
    size_t count = 0;
    for (size_t i = 0; i <= order; i++) count += CL::math::iCombination(dimension + i - 1, i);
    polynomials_.resize(count);
    // Generate 0th order term
    std::vector<size_t> coords;
    polynomials_[0] = Polynomial(coords);
    // Generate 1st and higher orders
    count = 1;
    for (size_t iorder = 1; iorder <=order; iorder++) {
        // The 1st term: r0^iorder
        coords.resize(iorder);
        fill(coords.begin(), coords.end(), 0);
        polynomials_[count] = Polynomial(coords);
        count++;
        // The other terms: as a dimension-nary counter
        // with former digit >= latter digit to avoid double counting
        if (dimension > 1) while (true) {
            coords[0] += 1;
            // Carry to latter digits
            for (size_t i = 0; i < iorder - 1; i++)
            if (coords[i] >= dimension) {
                coords[i] = 0;
                coords[i + 1] += 1;
            }
            // Guarantee former digit >= latter digit
            for (int i = iorder - 2; i > -1; i--)
            if (coords[i] < coords[i + 1]) {
                coords[i] = coords[i + 1];
            }
            polynomials_[count] = Polynomial(coords);
            count++;
            if (coords.back() >= dimension - 1) break;
        }
    }
    // Sanity check
    if (count != polynomials_.size()) std::cerr << "Error in PolynomialSet construction";
    this->create_orders();
}
PolynomialSet::~PolynomialSet() {}

// Return the value of each term as a vector
at::Tensor PolynomialSet::value(const at::Tensor & r) {
    at::Tensor value = r.new_empty(polynomials_.size());
    for (size_t i = 0; i < polynomials_.size(); i++) value[i] = polynomials_[i].value(r);
    return value;
}

// Consider coordinate rotation q = U^T . r
// so the polynomial set rotates as {r} = T . {q}
// Assuming:
//     1. All 0th and 1st order terms are present
//     2. Polynomial.coords are sorted
// Return rotation matrix T
at::Tensor PolynomialSet::rotation(const at::Tensor & U, const PolynomialSet * q_set) {
    assert(("U must be a matrix", U.sizes().size() == 2));
    assert(("U must be a square matrix", U.size(0) == U.size(1)));
    assert(("U must have a same dimension as the coordinates", U.size(0) == dimension));
    assert(("The 2 polynomial sets must share same order", order == q_set->order));
    // 0 and 1 dimensional coordinates do not rotate at all
    if (dimension < 2) return at::eye(polynomials_.size(), U.options());
    // Allocate memory
    at::Tensor T = U.new_zeros({(int)polynomials_.size(), (int)q_set->polynomials_.size()});
    // 0th order term does not rotate
    T[0][0] = 1.0;
    // 1st order terms rotate as r = U . q
    if (order >= 1) {
        at::Tensor T_block = T.slice(0, 1, dimension + 1);
        T_block.slice(1, 1, dimension + 1) = U;
    }
    // 2nd and higher order terms rotate as
    // ri1 ri2 ... rin = (Ui1j1 qj1) (Ui2j2 qj2) ... (Uinjn qjn)
    //                 = (Ui1j1 Ui2j2 ... Uinjn) (qj1 qj2 ... qjn)
    // equivalent (qj1 qj2 ... qjn)s have their (Ui1j1 Ui2j2 ... Uinjn)s merged
    size_t start_r = dimension + 1;
    size_t start_q = dimension + 1;
    for (size_t iorder = 2; iorder <= order; iorder++) {
        size_t NTerms_r = orders_[iorder].size();
        size_t stop_r   = start_r + NTerms_r;
        size_t NTerms_q = q_set->orders_[iorder].size();
        size_t stop_q   = start_q + NTerms_q;
        at::Tensor T_block = T.slice(0, start_r, stop_r);
        T_block = T_block.slice(1, start_q, stop_q);
        for (size_t i = 0; i < NTerms_r; i++) {
            auto r_coords = orders_[iorder][i]->coords();
            for (size_t j = 0; j < NTerms_q; j++) {
                auto q_coords = q_set->orders_[iorder][j]->coords();
                // Get the unique coordinates and their number of repeats
                std::vector<size_t> uniques, repeats;
                size_t coord_old = -1;
                for (size_t & coord : q_coords)
                if (coord != coord_old) {
                    uniques.push_back(coord);
                    repeats.push_back(1);
                    coord_old = coord;
                }
                else {
                    repeats.back() += 1;
                }
                // Only 1 permutation when all coordinates are the same
                if (uniques.size() == 1) {
                    T_block[i][j] = U[r_coords[0]][q_coords[0]];
                    for (size_t k = 1; k < iorder; k++) T_block[i][j] *= U[r_coords[k]][q_coords[k]];
                }
                // Sum over all permutations of the unique coordinates
                else {
                    // The 1st permutation: all coordinates sorted ascendingly
                    std::sort(q_coords.begin(), q_coords.end());
                    // The following permutations
                    bool done = false;
                    while (! done) {
                        // Sum the current permutation
                        at::Tensor current = U.new_zeros(1);
                        current[0] = U[r_coords[0]][q_coords[0]];
                        for (size_t k = 1; k < iorder; k++) current[0] *= U[r_coords[k]][q_coords[k]];
                        T_block[i][j] += current[0];
                        // Find the rightmost element which is smaller than its next
                        // Let us call it "edge element"
                        int edge_index;
                        for (edge_index = iorder - 2; edge_index > -1; edge_index--)
                        if (q_coords[edge_index] < q_coords[edge_index + 1]) break;
                        // No such element, all sorted descendingly, done
                        if (edge_index == -1) done = true; 
                        else {
                            // Find the ceil of "edge element" in the right of it
                            // Ceil of an element is the smallest element greater than it
                            size_t ceil_index = edge_index + 1;
                            for (size_t k = edge_index + 2; k < iorder; k++)
                            if (q_coords[k] > q_coords[edge_index]
                            &&  q_coords[k] < q_coords[ceil_index]) ceil_index = k;
                            // Swap edge and ceil
                            size_t save = q_coords[edge_index];
                            q_coords[edge_index] = q_coords[ceil_index];
                            q_coords[ceil_index] = save;
                            // Sort the sub vector on the right of edge
                            std::sort(q_coords.begin() + edge_index + 1, q_coords.end());
                        }
                    }
                }
            }
        }
        start_r = stop_r;
        start_q = stop_q;
    }
    return T;
}
// Assuming terms are the same under rotation
at::Tensor PolynomialSet::rotation(const at::Tensor & U) {return rotation(U, this);}

// Consider coordinate translation q = r - a
// so the polynomial set transforms as {r} = T . {q}
// Assuming:
//     1. All 0th and 1st order terms are present
// Return transformation matrix T
at::Tensor PolynomialSet::translation(const at::Tensor & a, const PolynomialSet * q_set) {
    assert(("a must be a vector", a.sizes().size() == 1));
    assert(("a must have a same dimension as the coordinates", a.size(0) == dimension));
    // Allocate memory
    at::Tensor T = a.new_zeros({(int)polynomials_.size(), (int)q_set->polynomials_.size()});
    // 0th order term does not shift
    T[0][0] = 1.0;
    // 1st order terms shift as r = q + a
    if (order >= 1) for (size_t i = 1; i < dimension + 1; i++) {
        T[i][0] = a[i - 1];
        T[i][i] = 1.0;
    }
    // 2nd and higher order terms shift as
    // ri1 ri2 ... rin = (qi1 + ai1) (qi2 + ai2) ... (qin + ain)
    //                 = ai1 ai2 ... ain + qi1 ai2 ... ain + ... + qi1 qi2 ... qin
    size_t start_r = dimension + 1;
    for (size_t iorder = 2; iorder <= order; iorder++) {
        size_t NTerms_r = orders_[iorder].size();
        size_t stop_r   = start_r + NTerms_r;
        at::Tensor T_block = T.slice(0, start_r, stop_r);
        for (size_t i = 0; i < NTerms_r; i++) {
            auto r_coords = orders_[iorder][i]->coords();
            // The 1st term: ai1 ai2 ... ain
            T_block[i][0] = a[r_coords[0]];
            for (size_t j = 1; j < iorder; j++) T_block[i][0] *= a[r_coords[j]];
            // The other terms: as a binary counter
            // when a bit == 1, place q there
            std::vector<size_t> use_var(iorder, 0);
            while (true) {
                use_var[0] += 1;
                // Carry to latter digits
                for (size_t j = 0; j < iorder - 1; j++)
                if (use_var[j] == 2) {
                    use_var[j] = 0;
                    use_var[j + 1] += 1;
                }
                if (use_var.back() == 2) break;
                // Build the coordinates for q and a
                size_t NVars = std::accumulate(use_var.begin(), use_var.end(), 0);
                size_t NCons = iorder - NVars;
                std::vector<size_t> q_coords(NVars), a_coords(NCons);
                size_t count_q = 0, count_a = 0;
                for (size_t j = 0; j < iorder; j++)
                if (use_var[j] == 1) {
                    q_coords[count_q] = r_coords[j];
                    count_q++;
                }
                else {
                    a_coords[count_a] = r_coords[j];
                    count_a++;
                }
                // Determine T block element
                int index = index_polynomial(q_coords);
                if (index >= 0) {
                    at::Tensor current = a.new_zeros(1);
                    current[0] = 1.0;
                    for (size_t & coord : a_coords) current[0] *= a[coord];
                    T_block[i][index] += current[0];
                }
            }
        }
        start_r = stop_r;
    }
    return T;
}
// Assuming terms are the same under translation
at::Tensor PolynomialSet::translation(const at::Tensor & a) {return translation(a, this);}

} // namespace polynomial
} // namespace tchem
