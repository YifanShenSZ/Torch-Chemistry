#include <regex>
#include <torch/torch.h>

#include <CppLibrary/linalg.hpp>

#include <tchem/linalg.hpp>

#include <tchem/intcoord.hpp>

namespace tchem { namespace IC {

InvDisp::InvDisp() {}
InvDisp::InvDisp(const std::string & _type, const std::vector<size_t> & _atoms) : type_(_type), atoms_(_atoms) {}
InvDisp::InvDisp(const std::string & _type, const std::vector<size_t> & _atoms, const double & _min) : type_(_type), atoms_(_atoms), min_(_min) {}
InvDisp::~InvDisp() {}

std::string InvDisp::type() const {return type_;}
std::vector<size_t> InvDisp::atoms() const {return atoms_;}
double InvDisp::min() const {return min_;}

// Return the displacement given r
at::Tensor InvDisp::operator()(const at::Tensor & r) const {
    assert(("r must be a vector", r.sizes().size() == 1));
    if (type_ == "stretching") {
        at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                       - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
        return r12.norm();
    }
    else if (type_ == "bending") {
        at::Tensor r21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                       - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        r21 = r21 / r21.norm();
        at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                       - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        r23 = r23 / r23.norm();
        // Q: Why no fail safe for acos?
        // A: acos is problematic only at 0 and pi
        //    exactly where you should avoid bending
        return at::acos(r21.dot(r23));
    }
    else if (type_ == "torsion") {
        at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                       - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
        at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                       - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor r34 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                       - r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3);
        at::Tensor n123 = r12.cross(r23); n123 = n123 / n123.norm();
        at::Tensor n234 = r23.cross(r34); n234 = n234 / n234.norm();
        at::Tensor theta = n123.dot(n234);
        // Fail safe for 0 and pi, but that breaks backward propagation
        if (theta.item<double>() > 1.0) theta.fill_(0.0);
        else if (theta.item<double>() < -1.0) theta.fill_(M_PI);
        else theta = at::acos(theta);
        if(tchem::LA::triple_product(n123, n234, r23).item<double>() < 0.0) theta = -theta;
        if(theta.item<double>() < min_) theta = theta + 2.0 * M_PI;
        else if(theta.item<double>() > min_ + 2.0 * M_PI) theta = theta - 2.0 * M_PI;
        return theta;
    }
    else if (type_ == "OutOfPlane") {
        at::Tensor r21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                       - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        r21 = r21 / r21.norm();
        at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                       - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor r24 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                       - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor n234 = r23.cross(r24); n234 = n234 / n234.norm();
        // Q: Why no fail safe for asin?
        // A: asin is problematic only at +-pi/2
        //    exactly where you should avoid out of plane
        return at::asin(n234.dot(r21));
    }
    else {
        throw "Unsupported internal coordinate type: " + type_;
    }
}
// Return the displacement and its gradient over r given r
std::tuple<at::Tensor, at::Tensor> InvDisp::compute_IC_J(const at::Tensor & r) const {
    assert(("r must be a vector", r.sizes().size() == 1));
    if (type_ == "stretching") {
        // Prepare
        at::Tensor runit12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
        at::Tensor r12 = runit12.norm();
        runit12 = runit12 / r12;
        // Output
        at::Tensor q = r12;
        at::Tensor J = r.new_zeros(r.sizes());
        J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = -runit12;
        J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) =  runit12;
        return std::make_tuple(q, J);
    }
    else if (type_ == "bending") {
        // Prepare
        at::Tensor runit21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor r21 = runit21.norm();
        runit21 = runit21 / r21;
        at::Tensor runit23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor r23 = runit23.norm();
        runit23 = runit23 / r23;
        at::Tensor costheta = runit21.dot(runit23);
        at::Tensor sintheta = at::sqrt(1.0 - costheta * costheta);
        at::Tensor J0 = (costheta * runit21 - runit23) / (sintheta * r21);
        at::Tensor J2 = (costheta * runit23 - runit21) / (sintheta * r23);
        // Output
        at::Tensor q = at::acos(costheta);
        at::Tensor J = r.new_zeros(r.sizes());
        J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
        J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
        J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = (- J0 - J2);
        return std::make_tuple(q, J);
    }
    else if (type_ == "torsion") {
        // Prepare
        at::Tensor runit12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
        at::Tensor r12 = runit12.norm();
        runit12 = runit12 / r12;
        at::Tensor runit23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor r23 = runit23.norm();
        runit23 = runit23 / r23;
        at::Tensor runit34 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                           - r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3);
        at::Tensor r34 = runit34.norm();
        runit34 = runit34 / r34;
        at::Tensor cos123 = -(runit12.dot(runit23));
        at::Tensor sin123 = at::sqrt(1.0 - cos123 * cos123);
        at::Tensor cos234 = -(runit23.dot(runit34));
        at::Tensor sin234 = at::sqrt(1.0 - cos234 * cos234);
        at::Tensor n123 = runit12.cross(runit23) / sin123;
        at::Tensor n234 = runit23.cross(runit34) / sin234;
        at::Tensor theta = n123.dot(n234);
        if (theta.item<double>() > 1.0) theta.fill_(0.0);
        else if (theta.item<double>() < -1.0) theta.fill_(M_PI);
        else theta = at::acos(theta);
        if(tchem::LA::triple_product(n123, n234, runit23).item<double>() < 0.0) theta = -theta;
        if(theta.item<double>() < min_) theta = theta + 2.0 * M_PI;
        else if(theta.item<double>() > min_ + 2.0 * M_PI) theta = theta - 2.0 * M_PI;
        // Output
        at::Tensor q = theta;
        at::Tensor J = r.new_zeros(r.sizes());
        J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = (-n123 / (r12 * sin123));
        J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = ((r23 - r12 * cos123) / (r12 * r23 * sin123) * n123 - cos234 / (r23 * sin234) * n234);
        J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = ((r34 * cos234 - r23) / (r23 * r34 * sin234) * n234 + cos123 / (r23 * sin123) * n123);
        J.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3) = ( n234 / (r34 * sin234));
        return std::make_tuple(q, J);
    }
    else if (type_ == "OutOfPlane") {
        // Prepare
        at::Tensor runit21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor r21 = runit21.norm();
        runit21 = runit21 / r21;
        at::Tensor runit23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor r23 = runit23.norm();
        runit23 = runit23 / r23;
        at::Tensor runit24 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor r24 = runit24.norm();
        runit24 = runit24 / r24;
        at::Tensor cos324 = runit23.dot(runit24);
        at::Tensor sin324sq = 1.0 - cos324 * cos324;
        at::Tensor sin324 = at::sqrt(sin324sq);
        at::Tensor sintheta = tchem::LA::triple_product(runit23, runit24, runit21) / sin324;
        at::Tensor costheta = at::sqrt(1.0 - sintheta * sintheta);
        at::Tensor tantheta = sintheta/costheta;
        at::Tensor J0 = (runit23.cross(runit24) / costheta / sin324 - tantheta * runit21) / r21;
        at::Tensor J2 = (runit24.cross(runit21) / costheta / sin324 - tantheta / sin324sq * (runit23 - cos324 * runit24)) / r23;
        at::Tensor J3 = (runit21.cross(runit23) / costheta / sin324 - tantheta / sin324sq * (runit24 - cos324 * runit23)) / r24;
        // Output
        at::Tensor q = at::asin(sintheta);
        at::Tensor J = r.new_zeros(r.sizes());
        J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
        J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
        J.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3) = J3;
        J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = (- J0 - J2 - J3);
        return std::make_tuple(q, J);
    }
    else {
        throw "Unsupported internal coordinate type: " + type_;
    }
}





IntCoord::IntCoord() {}
IntCoord::~IntCoord() {}

std::vector<double> IntCoord::coeffs() const {return coeffs_;}
std::vector<InvDisp> IntCoord::invdisps() const {return invdisps_;}

// Append a linear combination coefficient - invariant displacement pair
void IntCoord::append(const double & coeff, const InvDisp & invdisp) {
    coeffs_.push_back(coeff);
    invdisps_.push_back(invdisp);
}
// Normalize linear combination coefficients
void IntCoord::normalize() {
    double norm2 = CL::LA::norm2(coeffs_);
    for (double & coeff : coeffs_) coeff /= norm2;
}

// Return the internal coordinate given r
at::Tensor IntCoord::operator()(const at::Tensor & r) const {
    assert(("r must be a vector", r.sizes().size() == 1));
    at::Tensor q = coeffs_[0] * invdisps_[0](r);
    for (size_t i = 1; i < coeffs_.size(); i++) q = q + coeffs_[i] * invdisps_[i](r);
    return q;
}
// Return the internal coordinate and its gradient over r given r
std::tuple<at::Tensor, at::Tensor> IntCoord::compute_IC_J(const at::Tensor & r) const {
    assert(("r must be a vector", r.sizes().size() == 1));
    at::Tensor q, J;
    std::tie(q, J) = invdisps_[0].compute_IC_J(r);
    q = q * coeffs_[0];
    J = J * coeffs_[0];
    for (size_t i = 1; i < coeffs_.size(); i++) {
        at::Tensor qi, Ji;
        std::tie(qi, Ji) = invdisps_[i].compute_IC_J(r);
        q = q + coeffs_[i] * qi;
        J = J + coeffs_[i] * Ji;
    }
    return std::make_tuple(q, J);
}





IntCoordSet::IntCoordSet() {}
// file format (Columbus7, default), internal coordinate definition file
IntCoordSet::IntCoordSet(const std::string & format, const std::string & file) {
    if (format == "Columbus7") {
        // First line is always "TEXAS"
        // New internal coordinate line starts with 'K'
        std::ifstream ifs; ifs.open(file);
        if (! ifs.good()) {ifs.close(); ifs.open("intcfl");}
        std::string line; std::getline(ifs, line);
        while (true) {
            std::getline(ifs, line);
            if (! ifs.good()) break;
            if (line[0] == 'K') intcoords_.push_back(IntCoord());
            double coeff = 1.0;
            if (line.substr(10, 10) != "          ") coeff = std::stod(line.substr(10, 10));
            std::string type;
            std::vector<size_t> atom;
            if (line.substr(20,4) == "STRE") {
                type = "stretching";
                atom = {std::stoul(line.substr(28, 5)) - 1,
                        std::stoul(line.substr(34, 9)) - 1};
            }
            else if (line.substr(20,4) == "BEND") {
                type = "bending";
                atom = {std::stoul(line.substr(28, 6)) - 1,
                        std::stoul(line.substr(45, 9)) - 1,
                        std::stoul(line.substr(35, 9)) - 1};
            }
            else if (line.substr(20,4) == "TORS") {
                type = "torsion";
                atom = {std::stoul(line.substr(28, 6)) - 1,
                        std::stoul(line.substr(35, 9)) - 1,
                        std::stoul(line.substr(45, 9)) - 1,
                        std::stoul(line.substr(55, 9)) - 1};
            }
            else if (line.substr(20,3) == "OUT") {
                type = "OutOfPlane";
                atom = {std::stoul(line.substr(28, 6)) - 1,
                        std::stoul(line.substr(55, 9)) - 1,
                        std::stoul(line.substr(35, 9)) - 1,
                        std::stoul(line.substr(45, 9)) - 1};
            }
            else break;
            intcoords_.back().append(coeff, InvDisp(type, atom));
        }
        ifs.close();
    }
    else {
        // First 6 spaces of a line are reserved to indicate the start of new internal coordinate
        // For a line defining torsion, an additional number at the end of the line defines min
        // At the end of each line, anything after # is considered as comment
        std::ifstream ifs; ifs.open(file);
        if (! ifs.good()) {ifs.close(); ifs.open("IntCoordDef");}
        while (true) {
            std::string line;
            std::getline(ifs, line);
            if (! ifs.good()) break;
            std::istringstream iss(line);
            std::forward_list<std::string> strs(std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>());
            if (line.substr(0, 6) != "      ") {
                intcoords_.push_back(IntCoord());
                strs.pop_front();
            }
            double coeff = std::stod(strs.front()); strs.pop_front();
            std::string type = strs.front(); strs.pop_front();
            std::vector<size_t> atom;
            if (type == "stretching") {
                atom.resize(2);
                atom[0] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[1] = std::stoul(strs.front()) - 1; strs.pop_front();
            }
            else if (type == "bending") {
                atom.resize(3);
                atom[0] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[1] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[2] = std::stoul(strs.front()) - 1; strs.pop_front();
            }
            else if (type == "torsion") {
                atom.resize(4);
                atom[0] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[1] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[2] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[3] = std::stoul(strs.front()) - 1; strs.pop_front();
            }
            else if (type == "OutOfPlane") {
                atom.resize(4);
                atom[0] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[1] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[2] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[3] = std::stoul(strs.front()) - 1; strs.pop_front();
            }
            else {
                throw "Error during reading internal coordinate definition: unsupported internal coordinate type: " + type;
            }
            double min = -M_PI;
            if (! strs.empty()) if (std::regex_match(strs.front(), std::regex("-?\\d+\\.?\\d*"))) min = std::stod(strs.front());
            intcoords_.back().append(coeff, InvDisp(type, atom, min));
        }
        ifs.close();
    }
    // Normalize linear combination coefficient
    for (IntCoord & intcoord : intcoords_) intcoord.normalize();
}
IntCoordSet::~IntCoordSet() {}

std::vector<IntCoord> IntCoordSet::intcoords() const {return intcoords_;}

size_t IntCoordSet::size() const {return intcoords_.size();}

// Return q given r
at::Tensor IntCoordSet::operator()(const at::Tensor & r) const {
    assert(("r must be a vector", r.sizes().size() == 1));
    at::Tensor q = r.new_empty(intcoords_.size());
    for (size_t i = 0; i < intcoords_.size(); i++) q[i] = intcoords_[i](r);
    return q;
}
// Return q and J given r
std::tuple<at::Tensor, at::Tensor> IntCoordSet::compute_IC_J(const at::Tensor & r) const {
    assert(("r must be a vector", r.sizes().size() == 1));
    at::Tensor q = r.new_empty(intcoords_.size());
    at::Tensor J = r.new_empty({q.size(0), r.size(0)});
    for (size_t i = 0; i < intcoords_.size(); i++) {
        at::Tensor qi, Ji;
        std::tie(qi, Ji) = intcoords_[i].compute_IC_J(r);
        q[i] = qi;
        J[i] = Ji;
    }
    return std::make_tuple(q, J);
}

} // namespace IC
} // namespace tchem