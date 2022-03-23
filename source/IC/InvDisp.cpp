#include <CppLibrary/utility.hpp>

#include <tchem/IC/InvDisp.hpp>

namespace tchem { namespace IC {

InvDisp::InvDisp() {}
InvDisp::InvDisp(const std::string & _type, const std::vector<size_t> & _atoms, const double & _min) : atoms_(_atoms), min_(_min) {
    size_t i = 0;
    while (i < InvDisp_typestr.size()) {
        if (_type == InvDisp_typestr[i]) {
            type_ = static_cast<InvDisp_type>(i);
            break;
        }
        i++;
    }
    if (i == InvDisp_typestr.size()) throw std::invalid_argument("Unimplemented internal coordinate type " + _type);
}
InvDisp::~InvDisp() {}

const InvDisp_type & InvDisp::type() const {return type_;}
const std::vector<size_t> & InvDisp::atoms() const {return atoms_;}
const double & InvDisp::min() const {return min_;}

void InvDisp::print(std::ofstream & ofs, const std::string & format) const {
    if (format == "Columbus7") {
        switch (type_) {
            case stretching:
                ofs << "STRE"
                    << std::setw(9) << atoms_[0] + 1 << '.'
                    << std::setw(9) << atoms_[1] + 1 << '.';
            break;
            case bending:
                ofs << "BEND"
                    << std::setw(10) << atoms_[0] + 1 << '.'
                    << std::setw( 9) << atoms_[2] + 1 << '.'
                    << std::setw( 9) << atoms_[1] + 1 << '.';
            break;
            case torsion:
                ofs << "TORS"
                    << std::setw(10) << atoms_[0] + 1 << '.'
                    << std::setw( 9) << atoms_[1] + 1 << '.'
                    << std::setw( 9) << atoms_[2] + 1 << '.'
                    << std::setw( 9) << atoms_[3] + 1 << '.';
            break;
            case OutOfPlane:
                ofs << "OUT "
                    << std::setw(10) << atoms_[0] + 1 << '.'
                    << std::setw( 9) << atoms_[2] + 1 << '.'
                    << std::setw( 9) << atoms_[3] + 1 << '.'
                    << std::setw( 9) << atoms_[1] + 1 << '.';
            break;
            default: throw std::invalid_argument("Columbus does not support " + type_);
        }
    }
    else {
        ofs << std::setw(14) << InvDisp_typestr[type_];
        for (const size_t & atom : atoms_) ofs << std::setw(6) << atom + 1;
    }
    ofs << '\n';
}

// return the displacement given r
at::Tensor InvDisp::operator()(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::InvDisp::operator(): r must be a vector");
    switch (type_) {
        case dummy: return r.new_full({}, 1.0);
        case stretching: {
            at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            return r12.norm();
        }
        case bending: {
            at::Tensor r21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            r21 /= r21.norm();
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            r23 /= r23.norm();
            at::Tensor costheta = r21.dot(r23);
            // determine theta
            at::Tensor theta = costheta.clone();
            if (costheta.item<double>() > 1.0) theta.fill_(0.0);
            else if (costheta.item<double>() < -1.0) theta.fill_(M_PI);
            else theta = at::acos(costheta);
            return theta;
        }
        case cosbending: {
            at::Tensor r21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            r21 /= r21.norm();
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            r23 /= r23.norm();
            return r21.dot(r23);
        }
        case torsion: {
            at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r34 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                           - r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3);
            at::Tensor n123 = r12.cross(r23); n123 = n123 / n123.norm();
            at::Tensor n234 = r23.cross(r34); n234 = n234 / n234.norm();
            at::Tensor cosphi = n123.dot(n234);
            // determine phi
            at::Tensor phi = cosphi.clone();
            if (cosphi.item<double>() > 1.0) phi.fill_(0.0);
            else if (cosphi.item<double>() < -1.0) phi.fill_(M_PI);
            else {
                phi = at::acos(cosphi);
                if (n123.dot(n234.cross(r23)).item<double>() < 0.0) phi.neg_();
            }
            // from [-pi, pi] to [min, min + 2pi]
            if (phi.item<double>() < min_) phi += 2.0 * M_PI;
            else if(phi.item<double>() > min_ + 2.0 * M_PI) phi -= 2.0 * M_PI;
            return phi;
        }
        case sintorsion: {
            at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r34 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                           - r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3);
            at::Tensor n123 = r12.cross(r23); n123 = n123 / n123.norm();
            at::Tensor n234 = r23.cross(r34); n234 = n234 / n234.norm();
            return n123.cross(n234).dot(r23 / r23.norm());
        }
        case costorsion: {
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
        case torsion2: {
            at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r45 = r.slice(0, 3 * atoms_[4], 3 * atoms_[4] + 3)
                           - r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3);
            at::Tensor n123  = r12.cross(r23); n123  = n123  / n123 .norm();
            at::Tensor n2345 = r23.cross(r45); n2345 = n2345 / n2345.norm();
            at::Tensor cosphi = n123.dot(n2345);
            // determine phi
            at::Tensor phi = cosphi.clone();
            if (cosphi.item<double>() > 1.0) phi.fill_(0.0);
            else if (cosphi.item<double>() < -1.0) phi.fill_(M_PI);
            else {
                phi = at::acos(cosphi);
                if (n123.dot(n2345.cross(r23)).item<double>() < 0.0) phi.neg_();
            }
            // from [-pi, pi] to [min, min + 2pi]
            if (phi.item<double>() < min_) phi += 2.0 * M_PI;
            else if(phi.item<double>() > min_ + 2.0 * M_PI) phi -= 2.0 * M_PI;
            return phi;
        }
        case sintorsion2: {
            at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r45 = r.slice(0, 3 * atoms_[4], 3 * atoms_[4] + 3)
                           - r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3);
            at::Tensor n123  = r12.cross(r23); n123  = n123  / n123 .norm();
            at::Tensor n2345 = r23.cross(r45); n2345 = n2345 / n2345.norm();
            return n123.cross(n2345).dot(r23 / r23.norm());
        }
        case costorsion2: {
            at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r45 = r.slice(0, 3 * atoms_[4], 3 * atoms_[4] + 3)
                           - r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3);
            at::Tensor n123  = r12.cross(r23); n123  = n123  / n123 .norm();
            at::Tensor n2345 = r23.cross(r45); n2345 = n2345 / n2345.norm();
            return n123.dot(n2345);
        }
        case pxtorsion2: {
            // prepare
            at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r45 = r.slice(0, 3 * atoms_[4], 3 * atoms_[4] + 3)
                           - r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3);
            // sintheta
            at::Tensor runit12 = r12 / r12.norm(),
                       runit23 = r23 / r23.norm();
            at::Tensor costheta = -(runit12.dot(runit23));
            at::Tensor sintheta = at::sqrt(1.0 - costheta * costheta);
            // cosphi
            at::Tensor n123  = r12.cross(r23); n123  = n123  / n123 .norm();
            at::Tensor n2345 = r23.cross(r45); n2345 = n2345 / n2345.norm();
            at::Tensor cosphi = n123.dot(n2345);
            return sintheta * cosphi;
        }
        case pytorsion2: {
            // prepare
            at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r45 = r.slice(0, 3 * atoms_[4], 3 * atoms_[4] + 3)
                           - r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3);
            // sintheta
            at::Tensor runit12 = r12 / r12.norm(),
                       runit23 = r23 / r23.norm();
            at::Tensor costheta = -(runit12.dot(runit23));
            at::Tensor sintheta = at::sqrt(1.0 - costheta * costheta);
            // sinphi
            at::Tensor n123  = r12.cross(r23); n123  = n123  / n123 .norm();
            at::Tensor n2345 = r23.cross(r45); n2345 = n2345 / n2345.norm();
            at::Tensor sinphi = n123.cross(n2345).dot(runit23);
            return sintheta * sinphi;
        }
        case OutOfPlane: {
            at::Tensor r21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r24 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor n234 = r23.cross(r24); n234 = n234 / n234.norm();
            at::Tensor sintheta = n234.dot(r21 / r21.norm());
            // determine theta
            at::Tensor theta = sintheta.clone();
            if (sintheta.item<double>() < -1.0) theta.fill_(-M_PI / 2.0);
            else if (sintheta.item<double>() > 1.0) theta.fill_(M_PI / 2.0);
            else theta = at::asin(sintheta);
            return theta;
        }
        case sinoop: {
            at::Tensor r21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r24 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor n234 = r23.cross(r24); n234 = n234 / n234.norm();
            return n234.dot(r21 / r21.norm());
        }
        default: throw std::domain_error("tchem::IC::InvDisp::operator(): " + InvDisp_typestr[type_] + " is not implemented");
    }
}
// return the displacement and its gradient over r given r
std::tuple<at::Tensor, at::Tensor> InvDisp::compute_IC_J(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::InvDisp::compute_IC_J: r must be a vector");
    switch (type_) {
        case dummy: return std::make_tuple(r.new_full({}, 1.0), r.new_zeros(r.sizes()));
        case stretching: {
            // prepare
            at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r12norm = r12.norm();
            at::Tensor runit12 = r12 / r12norm;
            // output
            at::Tensor J = r.new_zeros(r.sizes());
            J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = -runit12;
            J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) =  runit12;
            return std::make_tuple(r12norm, J);
        }
        case bending: {
            // prepare
            at::Tensor r21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r21norm = r21.norm();
            at::Tensor runit21 = r21 / r21norm;
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r23norm = r23.norm();
            at::Tensor runit23 = r23 / r23norm;
            at::Tensor costheta = runit21.dot(runit23);
            at::Tensor sintheta = at::sqrt(1.0 - costheta * costheta);
            at::Tensor J0 = (costheta * runit21 - runit23) / (sintheta * r21norm);
            at::Tensor J2 = (costheta * runit23 - runit21) / (sintheta * r23norm);
            // output
            at::Tensor theta = at::acos(costheta);
            at::Tensor J = r.new_zeros(r.sizes());
            J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
            J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
            J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = - J0 - J2;
            // Q: Why no fail safe at 0 or pi this time?
            // A: J diverges at 0 or pi, so it is meaningless to do fail safe there
            return std::make_tuple(theta, J);
        }
        case cosbending: {
            // prepare
            at::Tensor r21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r21norm = r21.norm();
            at::Tensor runit21 = r21 / r21norm;
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r23norm = r23.norm();
            at::Tensor runit23 = r23 / r23norm;
            at::Tensor costheta = runit21.dot(runit23);
            at::Tensor J0 = (runit23 - costheta * runit21) / r21norm;
            at::Tensor J2 = (runit21 - costheta * runit23) / r23norm;
            // output
            at::Tensor J = r.new_zeros(r.sizes());
            J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
            J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
            J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = - J0 - J2;
            return std::make_tuple(costheta, J);
        }
        case torsion: {
            // prepare
            at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r12norm = r12.norm();
            at::Tensor runit12 = r12 / r12norm;
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r23norm = r23.norm();
            at::Tensor runit23 = r23 / r23norm;
            at::Tensor r34 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                           - r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3);
            at::Tensor r34norm = r34.norm();
            at::Tensor runit34 = r34 / r34norm;
            at::Tensor cos123 = -(runit12.dot(runit23));
            at::Tensor sin123 = at::sqrt(1.0 - cos123 * cos123);
            at::Tensor cos234 = -(runit23.dot(runit34));
            at::Tensor sin234 = at::sqrt(1.0 - cos234 * cos234);
            at::Tensor n123 = runit12.cross(runit23) / sin123;
            at::Tensor n234 = runit23.cross(runit34) / sin234;
            at::Tensor cosphi = n123.dot(n234);
            // output
            at::Tensor phi = cosphi.clone();
            if (cosphi.item<double>() > 1.0) phi.fill_(0.0);
            else if (cosphi.item<double>() < -1.0) phi.fill_(M_PI);
            else {
                phi = at::acos(cosphi);
                if (n123.dot(n234.cross(runit23)).item<double>() < 0.0) phi.neg_();
            }
            if (phi.item<double>() < min_) phi += 2.0 * M_PI;
            else if(phi.item<double>() > min_ + 2.0 * M_PI) phi -= 2.0 * M_PI;
            at::Tensor J = r.new_zeros(r.sizes());
            J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = -n123 / (r12norm * sin123);
            J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = (r23norm - r12norm * cos123) / (r12norm * r23norm * sin123) * n123 - cos234 / (r23norm * sin234) * n234;
            J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = (r34norm * cos234 - r23norm) / (r23norm * r34norm * sin234) * n234 + cos123 / (r23norm * sin123) * n123;
            J.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3) =  n234 / (r34norm * sin234);
            return std::make_tuple(phi, J);
        }
        case sintorsion: {
            // prepare
            at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r12norm = r12.norm();
            at::Tensor runit12 = r12 / r12norm;
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r23norm = r23.norm();
            at::Tensor runit23 = r23 / r23norm;
            at::Tensor r34 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                           - r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3);
            at::Tensor r34norm = r34.norm();
            at::Tensor runit34 = r34 / r34norm;
            at::Tensor cos123 = -(runit12.dot(runit23));
            at::Tensor sin123 = at::sqrt(1.0 - cos123 * cos123);
            at::Tensor cos234 = -(runit23.dot(runit34));
            at::Tensor sin234 = at::sqrt(1.0 - cos234 * cos234);
            at::Tensor n123 = runit12.cross(runit23) / sin123;
            at::Tensor n234 = runit23.cross(runit34) / sin234;
            at::Tensor sinphi = n123.cross(n234).dot(runit23),
                       cosphi = n123.dot(n234);
            // output
            at::Tensor J = r.new_zeros(r.sizes());
            J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = -n123 / (r12norm * sin123);
            J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = (r23norm - r12norm * cos123) / (r12norm * r23norm * sin123) * n123 - cos234 / (r23norm * sin234) * n234;
            J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = (r34norm * cos234 - r23norm) / (r23norm * r34norm * sin234) * n234 + cos123 / (r23norm * sin123) * n123;
            J.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3) =  n234 / (r34norm * sin234);
            return std::make_tuple(sinphi, cosphi * J);
        }
        case costorsion: {
            // prepare
            at::Tensor r12 = r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r12norm = r12.norm();
            at::Tensor runit12 = r12 / r12norm;
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r23norm = r23.norm();
            at::Tensor runit23 = r23 / r23norm;
            at::Tensor r34 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                           - r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3);
            at::Tensor r34norm = r34.norm();
            at::Tensor runit34 = r34 / r34norm;
            at::Tensor cos123 = -(runit12.dot(runit23));
            at::Tensor sin123 = at::sqrt(1.0 - cos123 * cos123);
            at::Tensor cos234 = -(runit23.dot(runit34));
            at::Tensor sin234 = at::sqrt(1.0 - cos234 * cos234);
            at::Tensor n123 = runit12.cross(runit23) / sin123;
            at::Tensor n234 = runit23.cross(runit34) / sin234;
            at::Tensor sinphi = n123.cross(n234).dot(runit23),
                       cosphi = n123.dot(n234);
            // output
            at::Tensor J = r.new_zeros(r.sizes());
            J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = -n123 / (r12norm * sin123);
            J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = (r23norm - r12norm * cos123) / (r12norm * r23norm * sin123) * n123 - cos234 / (r23norm * sin234) * n234;
            J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = (r34norm * cos234 - r23norm) / (r23norm * r34norm * sin234) * n234 + cos123 / (r23norm * sin123) * n123;
            J.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3) =  n234 / (r34norm * sin234);
            return std::make_tuple(cosphi, -sinphi * J);
        }
        case torsion2: {
            at::Tensor rclone = r.clone();
            rclone.requires_grad_(true);
            // prepare
            at::Tensor r12 = rclone.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - rclone.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r23 = rclone.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - rclone.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r45 = rclone.slice(0, 3 * atoms_[4], 3 * atoms_[4] + 3)
                           - rclone.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3);
            at::Tensor n123  = r12.cross(r23); n123  = n123  / n123 .norm();
            at::Tensor n2345 = r23.cross(r45); n2345 = n2345 / n2345.norm();
            // J_phi_r from sinphi or cosphi?
            at::Tensor sinphi = n123.cross(n2345).dot(r23 / r23.norm()),
                       cosphi = n123.dot(n2345);
            at::Tensor J;
            // cosphi is closer to 0, so we use J_phi_r = J_cosphi_r / -sinphi
            if (std::abs(sinphi.item<double>()) > std::abs(cosphi.item<double>())) {
                // J_cosphi_r
                auto gs = torch::autograd::grad({cosphi}, {rclone});
                J = gs[0];
                // stop autograd tracking
                sinphi.detach_();
                cosphi.detach_();
                // J_cosphi_r -> J_phi_r
                J /= -sinphi;
            }
            // J_phi_r = J_sinphi_r / cosphi
            else {
                // J_sinphi_r
                auto gs = torch::autograd::grad({sinphi}, {rclone});
                J = gs[0];
                // stop autograd tracking
                sinphi.detach_();
                cosphi.detach_();
                // J_sinphi_r -> J_phi_r
                J /= cosphi;
            }
            // cosphi -> phi
            at::Tensor phi = cosphi.clone();
            if (cosphi.item<double>() > 1.0) {
                phi.fill_(0.0);
            }
            else if (cosphi.item<double>() < -1.0) {
                phi.fill_(M_PI);
            }
            else {
                phi = at::acos(cosphi);
                if (n123.dot(n2345.cross(r23)).item<double>() < 0.0) phi.neg_();
            }
            if (phi.item<double>() < min_) phi = phi + 2.0 * M_PI;
            else if(phi.item<double>() > min_ + 2.0 * M_PI) phi = phi - 2.0 * M_PI;
            return std::make_tuple(phi, J);
        }
        case sintorsion2: {
            at::Tensor rclone = r.clone();
            rclone.requires_grad_(true);
            // cosphi
            at::Tensor r12 = rclone.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - rclone.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r23 = rclone.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - rclone.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r45 = rclone.slice(0, 3 * atoms_[4], 3 * atoms_[4] + 3)
                           - rclone.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3);
            at::Tensor n123  = r12.cross(r23); n123  = n123  / n123 .norm();
            at::Tensor n2345 = r23.cross(r45); n2345 = n2345 / n2345.norm();
            at::Tensor sinphi = n123.cross(n2345).dot(r23 / r23.norm());
            auto gs = torch::autograd::grad({sinphi}, {rclone});
            at::Tensor J = gs[0];
            sinphi.detach_(); // stop autograd tracking
            return std::make_tuple(sinphi, J);
        }
        case costorsion2: {
            at::Tensor rclone = r.clone();
            rclone.requires_grad_(true);
            // cosphi
            at::Tensor r12 = rclone.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - rclone.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r23 = rclone.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - rclone.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r45 = rclone.slice(0, 3 * atoms_[4], 3 * atoms_[4] + 3)
                           - rclone.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3);
            at::Tensor n123  = r12.cross(r23); n123  = n123  / n123 .norm();
            at::Tensor n2345 = r23.cross(r45); n2345 = n2345 / n2345.norm();
            at::Tensor cosphi = n123.dot(n2345);
            auto gs = torch::autograd::grad({cosphi}, {rclone});
            at::Tensor J = gs[0];
            cosphi.detach_(); // stop autograd tracking
            return std::make_tuple(cosphi, J);
        }
        case pxtorsion2: {
            at::Tensor rclone = r.clone();
            rclone.requires_grad_(true);
            // cosphi
            at::Tensor r12 = rclone.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - rclone.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r23 = rclone.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - rclone.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r45 = rclone.slice(0, 3 * atoms_[4], 3 * atoms_[4] + 3)
                           - rclone.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3);
            at::Tensor n123  = r12.cross(r23); n123  = n123  / n123 .norm();
            at::Tensor n2345 = r23.cross(r45); n2345 = n2345 / n2345.norm();
            at::Tensor cosphi = n123.dot(n2345);
            auto gs = torch::autograd::grad({cosphi}, {rclone});
            at::Tensor Jcosphi = gs[0];
            cosphi.detach_();
            // sintheta
            r12.detach_();
            r23.detach_();
            at::Tensor r21norm = r12.norm(),
                       r23norm = r23.norm();
            at::Tensor runit21 = -r12 / r21norm,
                       runit23 =  r23 / r23norm;
            at::Tensor costheta = runit21.dot(runit23);
            at::Tensor sintheta = at::sqrt(1.0 - costheta * costheta);
            at::Tensor J0 = costheta * (costheta * runit21 - runit23) / (sintheta * r21norm);
            at::Tensor J2 = costheta * (costheta * runit23 - runit21) / (sintheta * r23norm);
            at::Tensor Jsintheta = r.new_zeros(r.sizes());
            Jsintheta.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
            Jsintheta.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
            Jsintheta.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = - J0 - J2;
            return std::make_tuple(sintheta * cosphi, Jsintheta * cosphi + sintheta * Jcosphi);
        }
        case pytorsion2: {
            at::Tensor rclone = r.clone();
            rclone.requires_grad_(true);
            // cosphi
            at::Tensor r12 = rclone.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                           - rclone.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
            at::Tensor r23 = rclone.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - rclone.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r45 = rclone.slice(0, 3 * atoms_[4], 3 * atoms_[4] + 3)
                           - rclone.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3);
            at::Tensor n123  = r12.cross(r23); n123  = n123  / n123 .norm();
            at::Tensor n2345 = r23.cross(r45); n2345 = n2345 / n2345.norm();
            at::Tensor sinphi = n123.cross(n2345).dot(r23 / r23.norm());
            auto gs = torch::autograd::grad({sinphi}, {rclone});
            at::Tensor Jsinphi = gs[0];
            sinphi.detach_();
            // sintheta
            r12.detach_();
            r23.detach_();
            at::Tensor r21norm = r12.norm(),
                       r23norm = r23.norm();
            at::Tensor runit21 = -r12 / r21norm,
                       runit23 =  r23 / r23norm;
            at::Tensor costheta = runit21.dot(runit23);
            at::Tensor sintheta = at::sqrt(1.0 - costheta * costheta);
            at::Tensor J0 = costheta * (costheta * runit21 - runit23) / (sintheta * r21norm);
            at::Tensor J2 = costheta * (costheta * runit23 - runit21) / (sintheta * r23norm);
            at::Tensor Jsintheta = r.new_zeros(r.sizes());
            Jsintheta.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
            Jsintheta.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
            Jsintheta.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = - J0 - J2;
            return std::make_tuple(sintheta * sinphi, Jsintheta * sinphi + sintheta * Jsinphi);
        }
        case OutOfPlane: {
            // prepare
            at::Tensor r21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r21norm = r21.norm();
            at::Tensor runit21 = r21 / r21norm;
            at::Tensor r23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r23norm = r23.norm();
            at::Tensor runit23 = r23 / r23norm;
            at::Tensor r24 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                           - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r24norm = r24.norm();
            at::Tensor runit24 = r24 / r24norm;
            at::Tensor cos324 = runit23.dot(runit24);
            at::Tensor sin324sq = 1.0 - cos324 * cos324;
            at::Tensor sin324 = at::sqrt(sin324sq);
            at::Tensor sintheta = runit23.dot(runit24.cross(runit21)) / sin324;
            at::Tensor costheta = at::sqrt(1.0 - sintheta * sintheta);
            at::Tensor tantheta = sintheta / costheta;
            at::Tensor J0 = (runit23.cross(runit24) / costheta / sin324 - tantheta * runit21) / r21norm;
            at::Tensor J2 = (runit24.cross(runit21) / costheta / sin324 - tantheta / sin324sq * (runit23 - cos324 * runit24)) / r23norm;
            at::Tensor J3 = (runit21.cross(runit23) / costheta / sin324 - tantheta / sin324sq * (runit24 - cos324 * runit23)) / r24norm;
            // output
            at::Tensor q = at::asin(sintheta);
            at::Tensor J = r.new_zeros(r.sizes());
            J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
            J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
            J.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3) = J3;
            J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = - J0 - J2 - J3;
            return std::make_tuple(q, J);
        }
        case sinoop: {
            // prepare
            at::Tensor runit21 = r.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3)
                               - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r21norm = runit21.norm();
            runit21 = runit21 / r21norm;
            at::Tensor runit23 = r.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                               - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r23norm = runit23.norm();
            runit23 = runit23 / r23norm;
            at::Tensor runit24 = r.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3)
                               - r.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
            at::Tensor r24norm = runit24.norm();
            runit24 = runit24 / r24norm;
            at::Tensor cos324 = runit23.dot(runit24);
            at::Tensor sin324sq = 1.0 - cos324 * cos324;
            at::Tensor sin324 = at::sqrt(sin324sq);
            at::Tensor sintheta = runit23.dot(runit24.cross(runit21)) / sin324;
            at::Tensor J0 = (runit23.cross(runit24) / sin324 - sintheta * runit21) / r21norm;
            at::Tensor J2 = (runit24.cross(runit21) / sin324 - sintheta / sin324sq * (runit23 - cos324 * runit24)) / r23norm;
            at::Tensor J3 = (runit21.cross(runit23) / sin324 - sintheta / sin324sq * (runit24 - cos324 * runit23)) / r24norm;
            // output
            at::Tensor J = r.new_zeros(r.sizes());
            J.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
            J.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
            J.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3) = J3;
            J.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = - J0 - J2 - J3;
            return std::make_tuple(sintheta, J);
        }
        default: throw std::domain_error("tchem::IC::InvDisp::compute_IC_J: " + InvDisp_typestr[type_] + " is not implemented");
    }
}
// return the displacement and its 1st and 2nd order gradient over r given r
std::tuple<at::Tensor, at::Tensor, at::Tensor> InvDisp::compute_IC_J_K(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::InvDisp::compute_IC_J_K: r must be a vector");
    // analytical q, J, K
    if (type_ == dummy) return std::make_tuple(r.new_full({}, 1.0), r.new_zeros(r.sizes()), r.new_zeros({r.size(0), r.size(0)}));
    // analytical q and J, backward propagation K
    else if (type_ == stretching || type_ == bending || type_ == cosbending ||
    type_ == torsion || type_ == sintorsion || type_ == costorsion ||
    type_ == OutOfPlane || type_ == sinoop) {
        // prepare
        at::Tensor q;
        std::vector<at::Tensor> rs(atoms_.size());
        for (size_t i = 0; i < atoms_.size(); i++) {
            rs[i] = r.slice(0, 3 * atoms_[i], 3 * atoms_[i] + 3);
            rs[i].requires_grad_(true);
        }
        std::vector<at::Tensor> Js(atoms_.size());
        switch (type_) {
            case stretching: {
                // prepare
                at::Tensor r12 = rs[1] - rs[0];
                at::Tensor r12norm = r12.norm();
                at::Tensor runit12 = r12 / r12norm;
                // q
                q = r12norm;
                // J
                Js[0] = -runit12;
                Js[1] =  runit12;
            }
            break;
            case bending: {
                // prepare
                at::Tensor r21 = rs[0] - rs[1];
                at::Tensor r21norm = r21.norm();
                at::Tensor runit21 = r21 / r21norm;
                at::Tensor r23 = rs[2] - rs[1];
                at::Tensor r23norm = r23.norm();
                at::Tensor runit23 = r23 / r23norm;
                at::Tensor costheta = runit21.dot(runit23);
                at::Tensor sintheta = at::sqrt(1.0 - costheta * costheta);
                at::Tensor J0 = (costheta * runit21 - runit23) / (sintheta * r21norm);
                at::Tensor J2 = (costheta * runit23 - runit21) / (sintheta * r23norm);
                // q
                q = at::acos(costheta);
                // J
                Js[0] = J0;
                Js[2] = J2;
                Js[1] = - J0 - J2;
            }
            break;
            case cosbending: {
                // prepare
                at::Tensor r21 = rs[0] - rs[1];
                at::Tensor r21norm = r21.norm();
                at::Tensor runit21 = r21 / r21norm;
                at::Tensor r23 = rs[2] - rs[1];
                at::Tensor r23norm = r23.norm();
                at::Tensor runit23 = r23 / r23norm;
                at::Tensor costheta = runit21.dot(runit23);
                at::Tensor J0 = (runit23 - costheta * runit21) / r21norm;
                at::Tensor J2 = (runit21 - costheta * runit23) / r23norm;
                // q
                q = costheta;
                // J
                Js[0] = J0;
                Js[2] = J2;
                Js[1] = - J0 - J2;
            }
            break;
            case torsion: {
                at::Tensor r12 = rs[1] - rs[0];
                at::Tensor r12norm = r12.norm();
                at::Tensor runit12 = r12 / r12norm;
                at::Tensor r23 = rs[2] - rs[1];
                at::Tensor r23norm = r23.norm();
                at::Tensor runit23 = r23 / r23norm;
                at::Tensor r34 = rs[3] - rs[2];
                at::Tensor r34norm = r34.norm();
                at::Tensor runit34 = r34 / r34norm;
                at::Tensor cos123 = -(runit12.dot(runit23));
                at::Tensor sin123 = at::sqrt(1.0 - cos123 * cos123);
                at::Tensor cos234 = -(runit23.dot(runit34));
                at::Tensor sin234 = at::sqrt(1.0 - cos234 * cos234);
                at::Tensor n123 = runit12.cross(runit23) / sin123;
                at::Tensor n234 = runit23.cross(runit34) / sin234;
                at::Tensor cosphi = n123.dot(n234);
                at::Tensor phi = cosphi.clone();
                phi.detach_();
                if (phi.item<double>() > 1.0) phi.fill_(0.0);
                else if (phi.item<double>() < -1.0) phi.fill_(M_PI);
                else {
                    phi = at::acos(cosphi);
                    if (n123.dot(n234.cross(runit23)).item<double>() < 0.0) phi = -phi;
                }
                if (phi.item<double>() < min_) phi += 2.0 * M_PI;
                else if(phi.item<double>() > min_ + 2.0 * M_PI) phi -= 2.0 * M_PI;
                // q
                q = phi;
                // J
                Js[0] = -n123 / (r12norm * sin123);
                Js[1] = (r23norm - r12norm * cos123) / (r12norm * r23norm * sin123) * n123 - cos234 / (r23norm * sin234) * n234;
                Js[2] = (r34norm * cos234 - r23norm) / (r23norm * r34norm * sin234) * n234 + cos123 / (r23norm * sin123) * n123;
                Js[3] =  n234 / (r34norm * sin234);
            }
            break;
            case sintorsion: {
                at::Tensor r12 = rs[1] - rs[0];
                at::Tensor r12norm = r12.norm();
                at::Tensor runit12 = r12 / r12norm;
                at::Tensor r23 = rs[2] - rs[1];
                at::Tensor r23norm = r23.norm();
                at::Tensor runit23 = r23 / r23norm;
                at::Tensor r34 = rs[3] - rs[2];
                at::Tensor r34norm = r34.norm();
                at::Tensor runit34 = r34 / r34norm;
                at::Tensor cos123 = -(runit12.dot(runit23));
                at::Tensor sin123 = at::sqrt(1.0 - cos123 * cos123);
                at::Tensor cos234 = -(runit23.dot(runit34));
                at::Tensor sin234 = at::sqrt(1.0 - cos234 * cos234);
                at::Tensor n123 = runit12.cross(runit23) / sin123;
                at::Tensor n234 = runit23.cross(runit34) / sin234;
                at::Tensor sinphi = n123.cross(n234).dot(runit23),
                           cosphi = n123.dot(n234);
                // q
                q = sinphi;
                // J
                Js[0] = cosphi * (-n123 / (r12norm * sin123));
                Js[1] = cosphi * ((r23norm - r12norm * cos123) / (r12norm * r23norm * sin123) * n123 - cos234 / (r23norm * sin234) * n234);
                Js[2] = cosphi * ((r34norm * cos234 - r23norm) / (r23norm * r34norm * sin234) * n234 + cos123 / (r23norm * sin123) * n123);
                Js[3] = cosphi * ( n234 / (r34norm * sin234));
            }
            break;
            case costorsion: {
                at::Tensor r12 = rs[1] - rs[0];
                at::Tensor r12norm = r12.norm();
                at::Tensor runit12 = r12 / r12norm;
                at::Tensor r23 = rs[2] - rs[1];
                at::Tensor r23norm = r23.norm();
                at::Tensor runit23 = r23 / r23norm;
                at::Tensor r34 = rs[3] - rs[2];
                at::Tensor r34norm = r34.norm();
                at::Tensor runit34 = r34 / r34norm;
                at::Tensor cos123 = -(runit12.dot(runit23));
                at::Tensor sin123 = at::sqrt(1.0 - cos123 * cos123);
                at::Tensor cos234 = -(runit23.dot(runit34));
                at::Tensor sin234 = at::sqrt(1.0 - cos234 * cos234);
                at::Tensor n123 = runit12.cross(runit23) / sin123;
                at::Tensor n234 = runit23.cross(runit34) / sin234;
                at::Tensor sinphi = n123.cross(n234).dot(runit23),
                           cosphi = n123.dot(n234);
                // q
                q = cosphi;
                // J
                Js[0] = -sinphi * (-n123 / (r12norm * sin123));
                Js[1] = -sinphi * ((r23norm - r12norm * cos123) / (r12norm * r23norm * sin123) * n123 - cos234 / (r23norm * sin234) * n234);
                Js[2] = -sinphi * ((r34norm * cos234 - r23norm) / (r23norm * r34norm * sin234) * n234 + cos123 / (r23norm * sin123) * n123);
                Js[3] = -sinphi * ( n234 / (r34norm * sin234));
            }
            break;
            case OutOfPlane: {
                // prepare
                at::Tensor r21 = rs[0] - rs[1];
                at::Tensor r21norm = r21.norm();
                at::Tensor runit21 = r21 / r21norm;
                at::Tensor r23 = rs[2] - rs[1];
                at::Tensor r23norm = r23.norm();
                at::Tensor runit23 = r23 / r23norm;
                at::Tensor r24 = rs[3] - rs[1];
                at::Tensor r24norm = r24.norm();
                at::Tensor runit24 = r24 / r24norm;
                at::Tensor cos324 = runit23.dot(runit24);
                at::Tensor sin324sq = 1.0 - cos324 * cos324;
                at::Tensor sin324 = at::sqrt(sin324sq);
                at::Tensor sintheta = runit23.dot(runit24.cross(runit21)) / sin324;
                at::Tensor costheta = at::sqrt(1.0 - sintheta * sintheta);
                at::Tensor tantheta = sintheta / costheta;
                at::Tensor J0 = (runit23.cross(runit24) / costheta / sin324 - tantheta * runit21) / r21norm;
                at::Tensor J2 = (runit24.cross(runit21) / costheta / sin324 - tantheta / sin324sq * (runit23 - cos324 * runit24)) / r23norm;
                at::Tensor J3 = (runit21.cross(runit23) / costheta / sin324 - tantheta / sin324sq * (runit24 - cos324 * runit23)) / r24norm;
                // q
                q = at::asin(sintheta);
                // J
                Js[0] = J0;
                Js[2] = J2;
                Js[3] = J3;
                Js[1] = - J0 - J2 - J3;
            }
            break;
            case sinoop: {
                // prepare
                at::Tensor r21 = rs[0] - rs[1];
                at::Tensor r21norm = r21.norm();
                at::Tensor runit21 = r21 / r21norm;
                at::Tensor r23 = rs[2] - rs[1];
                at::Tensor r23norm = r23.norm();
                at::Tensor runit23 = r23 / r23norm;
                at::Tensor r24 = rs[3] - rs[1];
                at::Tensor r24norm = r24.norm();
                at::Tensor runit24 = r24 / r24norm;
                at::Tensor cos324 = runit23.dot(runit24);
                at::Tensor sin324sq = 1.0 - cos324 * cos324;
                at::Tensor sin324 = at::sqrt(sin324sq);
                at::Tensor sintheta = runit23.dot(runit24.cross(runit21)) / sin324;
                at::Tensor J0 = (runit23.cross(runit24) / sin324 - sintheta * runit21) / r21norm;
                at::Tensor J2 = (runit24.cross(runit21) / sin324 - sintheta / sin324sq * (runit23 - cos324 * runit24)) / r23norm;
                at::Tensor J3 = (runit21.cross(runit23) / sin324 - sintheta / sin324sq * (runit24 - cos324 * runit23)) / r24norm;
                // q
                q = sintheta;
                // J
                Js[0] = J0;
                Js[2] = J2;
                Js[3] = J3;
                Js[1] = - J0 - J2 - J3;
            }
            break;
            default: throw std::domain_error("tchem::IC::InvDisp::compute_IC_J_K: " + InvDisp_typestr[type_] + " is not implemented");
        }
        q.detach_();
        // J
        at::Tensor J = r.new_zeros(r.size(0));
        for (size_t i = 0; i < atoms_.size(); i++) J.slice(0, 3 * atoms_[i], 3 * atoms_[i] + 3).copy_(Js[i]);
        J.detach_(); // Why would this .copy_ changes J.requires_grad() to Js[i].requires_grad()?
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
    // analytical q, backward propagation J and K
    else if (type_ == torsion2 || type_ == sintorsion2 || type_ == costorsion2 || type_ == pxtorsion2 || type_ == pytorsion2) {
        at::Tensor rclone = r.clone();
        rclone.requires_grad_(true);
        // prepare
        at::Tensor r12 = rclone.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3)
                       - rclone.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3);
        at::Tensor r23 = rclone.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3)
                       - rclone.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3);
        at::Tensor r45 = rclone.slice(0, 3 * atoms_[4], 3 * atoms_[4] + 3)
                       - rclone.slice(0, 3 * atoms_[3], 3 * atoms_[3] + 3);
        at::Tensor n123  = r12.cross(r23); n123  = n123  / n123 .norm();
        at::Tensor n2345 = r23.cross(r45); n2345 = n2345 / n2345.norm();
        switch (type_) {
            case torsion2: {
                at::Tensor sinphi = n123.cross(n2345).dot(r23 / r23.norm()),
                           cosphi = n123.dot(n2345);
                auto gsinphi = torch::autograd::grad({sinphi}, {rclone}, {}, true, true);
                at::Tensor Jsinphi = gsinphi[0];
                auto gcosphi = torch::autograd::grad({cosphi}, {rclone}, {}, true, true);
                at::Tensor Jcosphi = gcosphi[0];
                at::Tensor J, K;
                // cosphi is closer to 0, so we use
                // J_phi_r = J_cosphi_r / -sinphi
                // K_phi_r = (K_cosphi_r + J_sinphi_r J_phi_r) / -sinphi
                if (std::abs(sinphi.item<double>()) > std::abs(cosphi.item<double>())) {
                    // K_cosphi_r
                    K = r.new_empty({r.size(0), r.size(0)});
                    for (int64_t i = 0; i < r.size(0); i++) {
                        auto gs = torch::autograd::grad({Jcosphi[i]}, {rclone}, {}, true, false);
                        K[i].copy_(gs[0]);
                    }
                    // stop autograd tracking
                     sinphi.detach_();
                     cosphi.detach_();
                    Jsinphi.detach_();
                    Jcosphi.detach_();
                    // J_cosphi_r -> J_phi_r and K_cosphi_r -> K_phi_r
                    J = Jcosphi / -sinphi;
                    K = (K + Jsinphi.outer(J)) / -sinphi;
                }
                // J_phi_r = J_sinphi_r / cosphi
                // K_phi_r = K_sinphi_r / cosphi
                else {
                    // K_sinphi_r
                    K = r.new_empty({r.size(0), r.size(0)});
                    for (int64_t i = 0; i < r.size(0); i++) {
                        auto gs = torch::autograd::grad({Jsinphi[i]}, {rclone}, {}, true, false);
                        K[i].copy_(gs[0]);
                    }
                    // stop autograd tracking
                     sinphi.detach_();
                     cosphi.detach_();
                    Jsinphi.detach_();
                    Jcosphi.detach_();
                    // J_sinphi_r -> J_phi_r and K_sinphi_r -> K_phi_r
                    J = Jsinphi / cosphi;
                    K = (K - Jcosphi.outer(J)) / cosphi;
                }
                // cosphi -> phi
                at::Tensor phi = cosphi.clone();
                if (cosphi.item<double>() > 1.0) {
                    phi.fill_(0.0);
                }
                else if (cosphi.item<double>() < -1.0) {
                    phi.fill_(M_PI);
                }
                else {
                    phi = at::acos(cosphi);
                    if (n123.dot(n2345.cross(r23)).item<double>() < 0.0) phi.neg_();
                }
                if (phi.item<double>() < min_) phi = phi + 2.0 * M_PI;
                else if(phi.item<double>() > min_ + 2.0 * M_PI) phi = phi - 2.0 * M_PI;
                return std::make_tuple(phi, J, K);
            }
            case sintorsion2: {
                at::Tensor sinphi = n123.cross(n2345).dot(r23 / r23.norm());
                auto gs = torch::autograd::grad({sinphi}, {rclone}, {}, true, true);
                at::Tensor J = gs[0];
                at::Tensor K = r.new_empty({r.size(0), r.size(0)});
                for (int64_t i = 0; i < r.size(0); i++) {
                    auto gs = torch::autograd::grad({J[i]}, {rclone}, {}, true, false);
                    K[i].copy_(gs[0]);
                }
                // stop autograd tracking
                sinphi.detach_();
                J.detach_();
                return std::make_tuple(sinphi, J, K);
            }
            case costorsion2: {
                at::Tensor cosphi = n123.dot(n2345);
                auto gs = torch::autograd::grad({cosphi}, {rclone}, {}, true, true);
                at::Tensor J = gs[0];
                at::Tensor K = r.new_empty({r.size(0), r.size(0)});
                for (int64_t i = 0; i < r.size(0); i++) {
                    auto gs = torch::autograd::grad({J[i]}, {rclone}, {}, true, false);
                    K[i].copy_(gs[0]);
                }
                // stop autograd tracking
                cosphi.detach_();
                J.detach_();
                return std::make_tuple(cosphi, J, K);
            }
            case pxtorsion2: {
                // cosphi
                at::Tensor cosphi = n123.dot(n2345);
                auto gs = torch::autograd::grad({cosphi}, {rclone}, {}, true, true);
                at::Tensor Jcosphi = gs[0];
                at::Tensor Kcosphi = r.new_empty({r.size(0), r.size(0)});
                for (int64_t i = 0; i < r.size(0); i++) {
                    auto gs = torch::autograd::grad({Jcosphi[i]}, {rclone}, {}, true, false);
                    Kcosphi[i].copy_(gs[0]);
                }
                cosphi.detach_();
                Jcosphi.detach_();
                // sintheta
                at::Tensor r21norm = r12.norm();
                at::Tensor runit21 = -r12 / r21norm;
                at::Tensor r23norm = r23.norm();
                at::Tensor runit23 = r23 / r23norm;
                at::Tensor costheta = runit21.dot(runit23);
                at::Tensor sintheta = at::sqrt(1.0 - costheta * costheta);
                at::Tensor J0 = costheta * (costheta * runit21 - runit23) / (sintheta * r21norm);
                at::Tensor J2 = costheta * (costheta * runit23 - runit21) / (sintheta * r23norm);
                at::Tensor Jsintheta = r.new_zeros(r.sizes());
                Jsintheta.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
                Jsintheta.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
                Jsintheta.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = - J0 - J2;
                at::Tensor Ksintheta = r.new_zeros({r.size(0), r.size(0)});
                for (int64_t i = 3 * atoms_[0]; i < 3 * atoms_[0] + 3; i++) {
                    auto gs = torch::autograd::grad({Jsintheta[i]}, {rclone}, {}, true, false);
                    Ksintheta[i].copy_(gs[0]);
                }
                for (int64_t i = 3 * atoms_[1]; i < 3 * atoms_[1] + 3; i++) {
                    auto gs = torch::autograd::grad({Jsintheta[i]}, {rclone}, {}, true, false);
                    Ksintheta[i].copy_(gs[0]);
                }
                for (int64_t i = 3 * atoms_[2]; i < 3 * atoms_[2] + 3; i++) {
                    auto gs = torch::autograd::grad({Jsintheta[i]}, {rclone}, {}, true, false);
                    Ksintheta[i].copy_(gs[0]);
                }
                sintheta.detach_();
                Jsintheta.detach_();
                return std::make_tuple(sintheta * cosphi, Jsintheta * cosphi + sintheta * Jcosphi, Ksintheta * cosphi + Jsintheta.outer(Jcosphi) + Jcosphi.outer(Jsintheta) + sintheta * Kcosphi);
            }
            case pytorsion2: {
                // sinphi
                at::Tensor sinphi = n123.cross(n2345).dot(r23 / r23.norm());
                auto gs = torch::autograd::grad({sinphi}, {rclone}, {}, true, true);
                at::Tensor Jsinphi = gs[0];
                at::Tensor Ksinphi = r.new_empty({r.size(0), r.size(0)});
                for (int64_t i = 0; i < r.size(0); i++) {
                    auto gs = torch::autograd::grad({Jsinphi[i]}, {rclone}, {}, true, false);
                    Ksinphi[i].copy_(gs[0]);
                }
                sinphi.detach_();
                Jsinphi.detach_();
                // sintheta
                at::Tensor r21norm = r12.norm();
                at::Tensor runit21 = -r12 / r21norm;
                at::Tensor r23norm = r23.norm();
                at::Tensor runit23 = r23 / r23norm;
                at::Tensor costheta = runit21.dot(runit23);
                at::Tensor sintheta = at::sqrt(1.0 - costheta * costheta);
                at::Tensor J0 = costheta * (costheta * runit21 - runit23) / (sintheta * r21norm);
                at::Tensor J2 = costheta * (costheta * runit23 - runit21) / (sintheta * r23norm);
                at::Tensor Jsintheta = r.new_zeros(r.sizes());
                Jsintheta.slice(0, 3 * atoms_[0], 3 * atoms_[0] + 3) = J0;
                Jsintheta.slice(0, 3 * atoms_[2], 3 * atoms_[2] + 3) = J2;
                Jsintheta.slice(0, 3 * atoms_[1], 3 * atoms_[1] + 3) = - J0 - J2;
                at::Tensor Ksintheta = r.new_zeros({r.size(0), r.size(0)});
                for (int64_t i = 3 * atoms_[0]; i < 3 * atoms_[0] + 3; i++) {
                    auto gs = torch::autograd::grad({Jsintheta[i]}, {rclone}, {}, true, false);
                    Ksintheta[i].copy_(gs[0]);
                }
                for (int64_t i = 3 * atoms_[1]; i < 3 * atoms_[1] + 3; i++) {
                    auto gs = torch::autograd::grad({Jsintheta[i]}, {rclone}, {}, true, false);
                    Ksintheta[i].copy_(gs[0]);
                }
                for (int64_t i = 3 * atoms_[2]; i < 3 * atoms_[2] + 3; i++) {
                    auto gs = torch::autograd::grad({Jsintheta[i]}, {rclone}, {}, true, false);
                    Ksintheta[i].copy_(gs[0]);
                }
                sintheta.detach_();
                Jsintheta.detach_();
                return std::make_tuple(sintheta * sinphi, Jsintheta * sinphi + sintheta * Jsinphi, Ksintheta * sinphi + Jsintheta.outer(Jsinphi) + Jsinphi.outer(Jsintheta) + sintheta * Ksinphi);
            }
            default: throw std::domain_error("tchem::IC::InvDisp::compute_IC_J_K: " + InvDisp_typestr[type_] + " is not implemented");
        }
    }
    else throw std::domain_error("tchem::IC::InvDisp::compute_IC_J_K: " + InvDisp_typestr[type_] + " is not implemented");
}

} // namespace IC
} // namespace tchem