#include <CppLibrary/linalg.hpp>

#include <tchem/IC/IntCoord.hpp>

namespace tchem { namespace IC {

IntCoord::IntCoord() {}
IntCoord::IntCoord(const std::vector<std::pair<double, InvDisp>> & _coeff_invdisps)
: coeff_invdisps_(_coeff_invdisps) {}
IntCoord::~IntCoord() {}

const std::vector<std::pair<double, InvDisp>> & IntCoord::coeff_invdisps() const {return coeff_invdisps_;}

// Read-only reference to a linear combination coefficient - invariant displacement pair
const std::pair<double, InvDisp> & IntCoord::operator[](const size_t & index) const {return coeff_invdisps_[index];}

// Append a linear combination coefficient - invariant displacement pair
void IntCoord::append(const std::pair<double, InvDisp> & coeff_invdisp) {
    coeff_invdisps_.push_back(coeff_invdisp);
}
void IntCoord::append(const double & coeff, const InvDisp & invdisp) {
    coeff_invdisps_.push_back(std::pair<double, InvDisp>(coeff, invdisp));
}

// Normalize linear combination coefficients
void IntCoord::normalize() {
    double norm2 = 0.0;
    for (const auto & coeff_invdisp : coeff_invdisps_) norm2 += coeff_invdisp.first * coeff_invdisp.first;
    norm2 /= sqrt(norm2);
    for (auto & coeff_invdisp : coeff_invdisps_) coeff_invdisp.first /= norm2;
}

void IntCoord::print(std::ofstream & ofs, const std::string & format) const {
    if (format == "Columbus7") {
        ofs << "K   " << std::setw(12) << coeff_invdisps_[0].first;
        coeff_invdisps_[0].second.print(ofs, format);
        for (size_t i = 1; i < coeff_invdisps_.size(); i++) {
            ofs << "    " << std::setw(12) << coeff_invdisps_[i].first;
            coeff_invdisps_[i].second.print(ofs, format);
        }
    }
    else {
        ofs << "coord " << std::setw(12) << coeff_invdisps_[0].first;
        coeff_invdisps_[0].second.print(ofs, format);
        for (size_t i = 1; i < coeff_invdisps_.size(); i++) {
            ofs << "      " << std::setw(12) << coeff_invdisps_[i].first;
            coeff_invdisps_[i].second.print(ofs, format);
        }
    }
}

// Return the internal coordinate given r
at::Tensor IntCoord::operator()(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoord::operator(): r must be a vector");
    at::Tensor q = coeff_invdisps_[0].first * coeff_invdisps_[0].second(r);
    for (size_t i = 1; i < coeff_invdisps_.size(); i++)
    q = q + coeff_invdisps_[i].first * coeff_invdisps_[i].second(r);
    return q;
}
// Return the internal coordinate and its gradient over r given r
std::tuple<at::Tensor, at::Tensor> IntCoord::compute_IC_J(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoord::compute_IC_J: r must be a vector");
    at::Tensor q, J;
    std::tie(q, J) = coeff_invdisps_[0].second.compute_IC_J(r);
    q = coeff_invdisps_[0].first * q;
    J = coeff_invdisps_[0].first * J;
    for (size_t i = 1; i < coeff_invdisps_.size(); i++) {
        at::Tensor qi, Ji;
        std::tie(qi, Ji) = coeff_invdisps_[i].second.compute_IC_J(r);
        q = q + coeff_invdisps_[i].first * qi;
        J = J + coeff_invdisps_[i].first * Ji;
    }
    return std::make_tuple(q, J);
}
// Return the internal coordinate and its 1st and 2nd order gradient over r given r
std::tuple<at::Tensor, at::Tensor, at::Tensor> IntCoord::compute_IC_J_K(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoord::compute_IC_J_K: r must be a vector");
    at::Tensor q, J, K;
    std::tie(q, J, K) = coeff_invdisps_[0].second.compute_IC_J_K(r);
    q = coeff_invdisps_[0].first * q;
    J = coeff_invdisps_[0].first * J;
    K = coeff_invdisps_[0].first * K;
    for (size_t i = 1; i < coeff_invdisps_.size(); i++) {
        at::Tensor qi, Ji, Ki;
        std::tie(qi, Ji, Ki) = coeff_invdisps_[i].second.compute_IC_J_K(r);
        q = q + coeff_invdisps_[i].first * qi;
        J = J + coeff_invdisps_[i].first * Ji;
        K = K + coeff_invdisps_[i].first * Ki;
    }
    return std::make_tuple(q, J, K);
}

} // namespace IC
} // namespace tchem