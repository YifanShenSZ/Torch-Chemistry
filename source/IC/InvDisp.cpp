#include <CppLibrary/utility.hpp>

#include <tchem/IC/InvDisp.hpp>

namespace tchem { namespace IC {

InvDisp::InvDisp() {}
InvDisp::InvDisp(const std::string & _type, const std::vector<size_t> & _atoms) : type_(_type), atoms_(_atoms) {}
InvDisp::InvDisp(const std::string & _type, const std::vector<size_t> & _atoms, const double & _min) : type_(_type), atoms_(_atoms), min_(_min) {}
InvDisp::~InvDisp() {}

const std::string & InvDisp::type() const {return type_;}
const std::vector<size_t> & InvDisp::atoms() const {return atoms_;}
const double & InvDisp::min() const {return min_;}

void InvDisp::print(std::ofstream & ofs, const std::string & format) const {
    if (format == "Columbus7") {
        if (type_ == "stretching") {
            ofs << "    STRE"
                << std::setw(9) << atoms_[0] + 1 << '.'
                << std::setw(9) << atoms_[1] + 1 << '.';
        }
        else if (type_ == "bending") {
            ofs << "    BEND"
                << std::setw(10) << atoms_[0] + 1 << '.'
                << std::setw( 9) << atoms_[2] + 1 << '.'
                << std::setw( 9) << atoms_[1] + 1 << '.';
        }
        else if (type_ == "torsion") {
            ofs << "    TORS"
            << std::setw(10) << atoms_[0] + 1 << '.'
            << std::setw( 9) << atoms_[1] + 1 << '.'
            << std::setw( 9) << atoms_[2] + 1 << '.'
            << std::setw( 9) << atoms_[3] + 1 << '.';
        }
        else if (type_ == "OutOfPlane") {
            ofs << "    OUT "
                << std::setw(10) << atoms_[0] + 1 << '.'
                << std::setw( 9) << atoms_[2] + 1 << '.'
                << std::setw( 9) << atoms_[3] + 1 << '.'
                << std::setw( 9) << atoms_[1] + 1 << '.';
        }
        else throw std::invalid_argument("Columbus does not support " + type_);
    }
    else {
        ofs << std::setw(14) << type_;
        for (const size_t & atom : atoms_) ofs << std::setw(6) << atom + 1;
    }
    ofs << '\n';
}

// Return the displacement given r
at::Tensor InvDisp::operator()(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::InvDisp::operator(): r must be a vector");
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
    else if (type_ == "cosbending") {
        at::Tensor r21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                       - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        r21 = r21 / r21.norm();
        at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                       - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        r23 = r23 / r23.norm();
        return r21.dot(r23);
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
        at::Tensor costheta = n123.dot(n234);
        at::Tensor theta = costheta.clone();
        // Fail safe for 0 and pi, but that breaks backward propagation
        if (costheta.item<double>() > 1.0) {
            theta.fill_(0.0);
        }
        else if (costheta.item<double>() < -1.0) {
            theta.fill_(M_PI);
        }
        else {
            theta = at::acos(costheta);
            if (n123.dot(n234.cross(r23)).item<double>() < 0.0) theta = -theta;
        }
        if (theta.item<double>() < min_) theta = theta + 2.0 * M_PI;
        else if(theta.item<double>() > min_ + 2.0 * M_PI) theta = theta - 2.0 * M_PI;
        return theta;
    }
    else if (type_ == "sintorsion") {
        at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                       - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
        at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                       - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor r34 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                       - r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3);
        at::Tensor n123 = r12.cross(r23); n123 = n123 / n123.norm();
        at::Tensor n234 = r23.cross(r34); n234 = n234 / n234.norm();
        at::Tensor costheta = n123.dot(n234);
        at::Tensor sintheta = costheta.clone();
        // Fail safe for 0 and pi, but that breaks backward propagation
        if (std::abs(costheta.item<double>()) > 1.0) {
            sintheta.fill_(0.0);
        }
        else {
            sintheta = at::sqrt(1.0 - costheta * costheta);
            if (n123.dot(n234.cross(r23)).item<double>() < 0.0) sintheta = -sintheta;
        }
        return sintheta;
    }
    else if (type_ == "costorsion") {
        at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                       - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
        at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                       - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor r34 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                       - r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3);
        at::Tensor n123 = r12.cross(r23); n123 = n123 / n123.norm();
        at::Tensor n234 = r23.cross(r34); n234 = n234 / n234.norm();
        return n123.dot(n234);
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
    else if (type_ == "sinoop") {
        at::Tensor r21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                       - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        r21 = r21 / r21.norm();
        at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                       - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor r24 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                       - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor n234 = r23.cross(r24); n234 = n234 / n234.norm();
        return n234.dot(r21);
    }
    else throw std::invalid_argument("Unimplemented internal coordinate type: " + type_);
}
// Return the displacement and its gradient over r given r
std::tuple<at::Tensor, at::Tensor> InvDisp::compute_IC_J(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::InvDisp::compute_IC_J: r must be a vector");
    if (type_ == "stretching") {
        // Prepare
        at::Tensor runit12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
        at::Tensor r12 = runit12.norm();
        runit12 = runit12 / r12;
        // Output
        at::Tensor J = r.new_zeros(r.sizes());
        J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = -runit12;
        J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) =  runit12;
        return std::make_tuple(r12, J);
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
        at::Tensor theta = at::acos(costheta);
        at::Tensor J = r.new_zeros(r.sizes());
        J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
        J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
        J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = - J0 - J2;
        return std::make_tuple(theta, J);
    }
    else if (type_ == "cosbending") {
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
        at::Tensor J0 = (runit23 - costheta * runit21) / r21;
        at::Tensor J2 = (runit21 - costheta * runit23) / r23;
        // Output
        at::Tensor J = r.new_zeros(r.sizes());
        J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
        J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
        J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = - J0 - J2;
        return std::make_tuple(costheta, J);
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
        at::Tensor costheta = n123.dot(n234);
        at::Tensor theta = costheta.clone();
        if (costheta.item<double>() > 1.0) {
            theta.fill_(0.0);
        }
        else if (costheta.item<double>() < -1.0) {
            theta.fill_(M_PI);
        }
        else {
            theta = at::acos(costheta);
            if (n123.dot(n234.cross(runit23)).item<double>() < 0.0) theta = -theta;
        }
        if (theta.item<double>() < min_) theta = theta + 2.0 * M_PI;
        else if(theta.item<double>() > min_ + 2.0 * M_PI) theta = theta - 2.0 * M_PI;
        // Output
        at::Tensor J = r.new_zeros(r.sizes());
        J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = -n123 / (r12 * sin123);
        J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = (r23 - r12 * cos123) / (r12 * r23 * sin123) * n123 - cos234 / (r23 * sin234) * n234;
        J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = (r34 * cos234 - r23) / (r23 * r34 * sin234) * n234 + cos123 / (r23 * sin123) * n123;
        J.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3) =  n234 / (r34 * sin234);
        return std::make_tuple(theta, J);
    }
    else if (type_ == "sintorsion") {
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
        at::Tensor costheta = n123.dot(n234);
        at::Tensor sintheta = costheta.clone();
        if (std::abs(costheta.item<double>()) > 1.0) {
            sintheta.fill_(0.0);
        }
        else {
            sintheta = at::sqrt(1.0 - costheta * costheta);
            if (n123.dot(n234.cross(runit23)).item<double>() < 0.0) sintheta = -sintheta;
        }
        // Output
        at::Tensor J = r.new_zeros(r.sizes());
        J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = -n123 / (r12 * sin123);
        J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = (r23 - r12 * cos123) / (r12 * r23 * sin123) * n123 - cos234 / (r23 * sin234) * n234;
        J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = (r34 * cos234 - r23) / (r23 * r34 * sin234) * n234 + cos123 / (r23 * sin123) * n123;
        J.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3) =  n234 / (r34 * sin234);
        return std::make_tuple(sintheta, costheta * J);
    }
    else if (type_ == "costorsion") {
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
        at::Tensor costheta = n123.dot(n234);
        at::Tensor sintheta = costheta.clone();
        if (std::abs(costheta.item<double>()) > 1.0) {
            sintheta.fill_(0.0);
        }
        else {
            sintheta = at::sqrt(1.0 - costheta * costheta);
            if (n123.dot(n234.cross(runit23)).item<double>() < 0.0) sintheta = -sintheta;
        }
        // Output
        at::Tensor J = r.new_zeros(r.sizes());
        J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = -n123 / (r12 * sin123);
        J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = (r23 - r12 * cos123) / (r12 * r23 * sin123) * n123 - cos234 / (r23 * sin234) * n234;
        J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = (r34 * cos234 - r23) / (r23 * r34 * sin234) * n234 + cos123 / (r23 * sin123) * n123;
        J.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3) =  n234 / (r34 * sin234);
        return std::make_tuple(costheta, -sintheta * J);
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
        at::Tensor sintheta = runit23.dot(runit24.cross(runit21)) / sin324;
        at::Tensor costheta = at::sqrt(1.0 - sintheta * sintheta);
        at::Tensor tantheta = sintheta / costheta;
        at::Tensor J0 = (runit23.cross(runit24) / costheta / sin324 - tantheta * runit21) / r21;
        at::Tensor J2 = (runit24.cross(runit21) / costheta / sin324 - tantheta / sin324sq * (runit23 - cos324 * runit24)) / r23;
        at::Tensor J3 = (runit21.cross(runit23) / costheta / sin324 - tantheta / sin324sq * (runit24 - cos324 * runit23)) / r24;
        // Output
        at::Tensor q = at::asin(sintheta);
        at::Tensor J = r.new_zeros(r.sizes());
        J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
        J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
        J.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3) = J3;
        J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = - J0 - J2 - J3;
        return std::make_tuple(q, J);
    }
    else if (type_ == "sinoop") {
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
        at::Tensor sintheta = runit23.dot(runit24.cross(runit21)) / sin324;
        at::Tensor J0 = (runit23.cross(runit24) / sin324 - sintheta * runit21) / r21;
        at::Tensor J2 = (runit24.cross(runit21) / sin324 - sintheta / sin324sq * (runit23 - cos324 * runit24)) / r23;
        at::Tensor J3 = (runit21.cross(runit23) / sin324 - sintheta / sin324sq * (runit24 - cos324 * runit23)) / r24;
        // Output
        at::Tensor J = r.new_zeros(r.sizes());
        J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
        J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
        J.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3) = J3;
        J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = - J0 - J2 - J3;
        return std::make_tuple(sintheta, J);
    }
    else throw std::invalid_argument("Unimplemented internal coordinate type: " + type_);
}
// Return the displacement and its 1st and 2nd order gradient over r given r
std::tuple<at::Tensor, at::Tensor, at::Tensor> InvDisp::compute_IC_J_K(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::InvDisp::compute_IC_J_K: r must be a vector");
    // Prepare
    at::Tensor q;
    std::vector<at::Tensor> rs(atoms_.size());
    for (size_t i = 0; i < atoms_.size(); i++) {
        rs[i] = r.slice(0, 3 * atoms_[i], 3 * atoms_[i] + 3);
        rs[i].set_requires_grad(true);
    }
    std::vector<at::Tensor> Js(atoms_.size());
    if (type_ == "stretching") {
        // Prepare
        at::Tensor runit12 = rs[1] - rs[0];
        at::Tensor r12 = runit12.norm();
        runit12 = runit12 / r12;
        // q
        q = r12;
        // J
        Js[0] = -runit12;
        Js[1] =  runit12;
    }
    else if (type_ == "bending") {
        // Prepare
        at::Tensor runit21 = rs[0] - rs[1];
        at::Tensor r21 = runit21.norm();
        runit21 = runit21 / r21;
        at::Tensor runit23 = rs[2] - rs[1];
        at::Tensor r23 = runit23.norm();
        runit23 = runit23 / r23;
        at::Tensor costheta = runit21.dot(runit23);
        at::Tensor sintheta = at::sqrt(1.0 - costheta * costheta);
        at::Tensor J0 = (costheta * runit21 - runit23) / (sintheta * r21);
        at::Tensor J2 = (costheta * runit23 - runit21) / (sintheta * r23);
        // q
        q = at::acos(costheta);
        // J
        Js[0] = J0;
        Js[2] = J2;
        Js[1] = - J0 - J2;
    }
    else if (type_ == "cosbending") {
        // Prepare
        at::Tensor runit21 = rs[0] - rs[1];
        at::Tensor r21 = runit21.norm();
        runit21 = runit21 / r21;
        at::Tensor runit23 = rs[2] - rs[1];
        at::Tensor r23 = runit23.norm();
        runit23 = runit23 / r23;
        at::Tensor costheta = runit21.dot(runit23);
        at::Tensor J0 = (runit23 - costheta * runit21) / r21;
        at::Tensor J2 = (runit21 - costheta * runit23) / r23;
        // q
        q = costheta;
        // J
        Js[0] = J0;
        Js[2] = J2;
        Js[1] = - J0 - J2;
    }
    else if (type_ == "torsion") {
        at::Tensor runit12 = rs[1] - rs[0];
        at::Tensor r12 = runit12.norm();
        runit12 = runit12 / r12;
        at::Tensor runit23 = rs[2] - rs[1];
        at::Tensor r23 = runit23.norm();
        runit23 = runit23 / r23;
        at::Tensor runit34 = rs[3] - rs[2];
        at::Tensor r34 = runit34.norm();
        runit34 = runit34 / r34;
        at::Tensor cos123 = -(runit12.dot(runit23));
        at::Tensor sin123 = at::sqrt(1.0 - cos123 * cos123);
        at::Tensor cos234 = -(runit23.dot(runit34));
        at::Tensor sin234 = at::sqrt(1.0 - cos234 * cos234);
        at::Tensor n123 = runit12.cross(runit23) / sin123;
        at::Tensor n234 = runit23.cross(runit34) / sin234;
        at::Tensor costheta = n123.dot(n234);
        at::Tensor theta = costheta.clone();
        theta.detach_();
        if (theta.item<double>() > 1.0) {
            theta.fill_(0.0);
        }
        else if (theta.item<double>() < -1.0) {
            theta.fill_(M_PI);
        }
        else {
            theta = at::acos(costheta);
            if (n123.dot(n234.cross(runit23)).item<double>() < 0.0) theta = -theta;
        }
        if (theta.item<double>() < min_) theta = theta + 2.0 * M_PI;
        else if(theta.item<double>() > min_ + 2.0 * M_PI) theta = theta - 2.0 * M_PI;
        // q
        q = theta;
        // J
        Js[0] = -n123 / (r12 * sin123);
        Js[1] = (r23 - r12 * cos123) / (r12 * r23 * sin123) * n123 - cos234 / (r23 * sin234) * n234;
        Js[2] = (r34 * cos234 - r23) / (r23 * r34 * sin234) * n234 + cos123 / (r23 * sin123) * n123;
        Js[3] =  n234 / (r34 * sin234);
    }
    else if (type_ == "sintorsion") {
        at::Tensor runit12 = rs[1] - rs[0];
        at::Tensor r12 = runit12.norm();
        runit12 = runit12 / r12;
        at::Tensor runit23 = rs[2] - rs[1];
        at::Tensor r23 = runit23.norm();
        runit23 = runit23 / r23;
        at::Tensor runit34 = rs[3] - rs[2];
        at::Tensor r34 = runit34.norm();
        runit34 = runit34 / r34;
        at::Tensor cos123 = -(runit12.dot(runit23));
        at::Tensor sin123 = at::sqrt(1.0 - cos123 * cos123);
        at::Tensor cos234 = -(runit23.dot(runit34));
        at::Tensor sin234 = at::sqrt(1.0 - cos234 * cos234);
        at::Tensor n123 = runit12.cross(runit23) / sin123;
        at::Tensor n234 = runit23.cross(runit34) / sin234;
        at::Tensor costheta = n123.dot(n234);
        at::Tensor sintheta = costheta.clone();
        if (std::abs(costheta.item<double>()) > 1.0) {
            sintheta.fill_(0.0);
        }
        else {
            sintheta = at::sqrt(1.0 - costheta * costheta);
            if (n123.dot(n234.cross(runit23)).item<double>() < 0.0) sintheta = -sintheta;
        }
        // q
        q = sintheta;
        // J
        Js[0] = costheta * (-n123 / (r12 * sin123));
        Js[1] = costheta * ((r23 - r12 * cos123) / (r12 * r23 * sin123) * n123 - cos234 / (r23 * sin234) * n234);
        Js[2] = costheta * ((r34 * cos234 - r23) / (r23 * r34 * sin234) * n234 + cos123 / (r23 * sin123) * n123);
        Js[3] = costheta * ( n234 / (r34 * sin234));
    }
    else if (type_ == "costorsion") {
        at::Tensor runit12 = rs[1] - rs[0];
        at::Tensor r12 = runit12.norm();
        runit12 = runit12 / r12;
        at::Tensor runit23 = rs[2] - rs[1];
        at::Tensor r23 = runit23.norm();
        runit23 = runit23 / r23;
        at::Tensor runit34 = rs[3] - rs[2];
        at::Tensor r34 = runit34.norm();
        runit34 = runit34 / r34;
        at::Tensor cos123 = -(runit12.dot(runit23));
        at::Tensor sin123 = at::sqrt(1.0 - cos123 * cos123);
        at::Tensor cos234 = -(runit23.dot(runit34));
        at::Tensor sin234 = at::sqrt(1.0 - cos234 * cos234);
        at::Tensor n123 = runit12.cross(runit23) / sin123;
        at::Tensor n234 = runit23.cross(runit34) / sin234;
        at::Tensor costheta = n123.dot(n234);
        at::Tensor sintheta = costheta.clone();
        if (std::abs(costheta.item<double>()) > 1.0) {
            sintheta.fill_(0.0);
        }
        else {
            sintheta = at::sqrt(1.0 - costheta * costheta);
            if (n123.dot(n234.cross(runit23)).item<double>() < 0.0) sintheta = -sintheta;
        }
        // q
        q = costheta;
        // J
        Js[0] = -sintheta * (-n123 / (r12 * sin123));
        Js[1] = -sintheta * ((r23 - r12 * cos123) / (r12 * r23 * sin123) * n123 - cos234 / (r23 * sin234) * n234);
        Js[2] = -sintheta * ((r34 * cos234 - r23) / (r23 * r34 * sin234) * n234 + cos123 / (r23 * sin123) * n123);
        Js[3] = -sintheta * ( n234 / (r34 * sin234));
    }
    else if (type_ == "OutOfPlane") {
        // Prepare
        at::Tensor runit21 = rs[0] - rs[1];
        at::Tensor r21 = runit21.norm();
        runit21 = runit21 / r21;
        at::Tensor runit23 = rs[2] - rs[1];
        at::Tensor r23 = runit23.norm();
        runit23 = runit23 / r23;
        at::Tensor runit24 = rs[3] - rs[1];
        at::Tensor r24 = runit24.norm();
        runit24 = runit24 / r24;
        at::Tensor cos324 = runit23.dot(runit24);
        at::Tensor sin324sq = 1.0 - cos324 * cos324;
        at::Tensor sin324 = at::sqrt(sin324sq);
        at::Tensor sintheta = runit23.dot(runit24.cross(runit21)) / sin324;
        at::Tensor costheta = at::sqrt(1.0 - sintheta * sintheta);
        at::Tensor tantheta = sintheta / costheta;
        at::Tensor J0 = (runit23.cross(runit24) / costheta / sin324 - tantheta * runit21) / r21;
        at::Tensor J2 = (runit24.cross(runit21) / costheta / sin324 - tantheta / sin324sq * (runit23 - cos324 * runit24)) / r23;
        at::Tensor J3 = (runit21.cross(runit23) / costheta / sin324 - tantheta / sin324sq * (runit24 - cos324 * runit23)) / r24;
        // q
        q = at::asin(sintheta);
        // J
        Js[0] = J0;
        Js[2] = J2;
        Js[3] = J3;
        Js[1] = - J0 - J2 - J3;
    }
    else if (type_ == "sinoop") {
        // Prepare
        at::Tensor runit21 = rs[0] - rs[1];
        at::Tensor r21 = runit21.norm();
        runit21 = runit21 / r21;
        at::Tensor runit23 = rs[2] - rs[1];
        at::Tensor r23 = runit23.norm();
        runit23 = runit23 / r23;
        at::Tensor runit24 = rs[3] - rs[1];
        at::Tensor r24 = runit24.norm();
        runit24 = runit24 / r24;
        at::Tensor cos324 = runit23.dot(runit24);
        at::Tensor sin324sq = 1.0 - cos324 * cos324;
        at::Tensor sin324 = at::sqrt(sin324sq);
        at::Tensor sintheta = runit23.dot(runit24.cross(runit21)) / sin324;
        at::Tensor J0 = (runit23.cross(runit24) / sin324 - sintheta * runit21) / r21;
        at::Tensor J2 = (runit24.cross(runit21) / sin324 - sintheta / sin324sq * (runit23 - cos324 * runit24)) / r23;
        at::Tensor J3 = (runit21.cross(runit23) / sin324 - sintheta / sin324sq * (runit24 - cos324 * runit23)) / r24;
        // q
        q = sintheta;
        // J
        Js[0] = J0;
        Js[2] = J2;
        Js[3] = J3;
        Js[1] = - J0 - J2 - J3;
    }
    else throw std::invalid_argument("Unimplemented internal coordinate type: " + type_);
    q.detach_();
    // J
    at::Tensor J = r.new_zeros(r.size(0));
    for (size_t i = 0; i < atoms_.size(); i++)
    J.slice(0, 3 * atoms_[i], 3 * atoms_[i] + 3).copy_(Js[i]);
    // K
    CL::utility::matrix<at::Tensor> Ks(atoms_.size());
    for (size_t i = 0; i < atoms_.size(); i++)
    for (size_t j = i; j < atoms_.size(); j++) {
        Ks[i][j] = r.new_empty({3, 3});
        for (size_t k = 0; k < 3; k++) {
            auto gs = torch::autograd::grad({Js[i][k]}, {rs[j]}, {}, true, false, true);
            if (gs[0].defined()) Ks[i][j][k].copy_(gs[0]);
            else                 Ks[i][j][k].fill_(0.0);
        }
    }
    at::Tensor K = r.new_zeros({(int64_t)r.size(0), (int64_t)r.size(0)});
    for (size_t i = 0; i < atoms_.size(); i++) {
        K.slice(0, 3 * atoms_[i], 3 * atoms_[i] + 3).slice(1, 3 * atoms_[i], 3 * atoms_[i] + 3).copy_(Ks[i][i]);
        for (size_t j = i + 1; j < atoms_.size(); j++) {
            K.slice(0, 3 * atoms_[i], 3 * atoms_[i] + 3).slice(1, 3 * atoms_[j], 3 * atoms_[j] + 3).copy_(Ks[i][j]);
            K.slice(0, 3 * atoms_[j], 3 * atoms_[j] + 3).slice(1, 3 * atoms_[i], 3 * atoms_[i] + 3).copy_(Ks[i][j].transpose(0, 1));
        }
    }
    return std::make_tuple(q, J, K);
}

} // namespace IC
} // namespace tchem