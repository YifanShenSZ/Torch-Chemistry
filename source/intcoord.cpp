/*
An interal coordinate is the linear combination of several translationally and rotationally invariant displacements
but only displacements under same unit can be combined, i.e. you must treat lengthes and angles separately
unless appropriate metric tensor is applied

Nomenclature:
    cartdim & intdim: Cartesian & internal space dimensionality
    r: Cartesian coordinate vector
    q: internal coordinate vector
    J: the Jacobian matrix of q over r

Warning:
    * J of bending is singular at 0 or pi,
      so please avoid using bending in those cases
    * J of out of plane is singular at +-pi/2,
      so please avoid using out of plane in those cases
    * Backward propagation through q may be problematic for torsion when q = 0 or pi,
      so please use J explicitly in those cases
*/

#include <torch/torch.h>

#include <tchem/intcoord.hpp>

namespace tchem { namespace IC {

InvolvedMotion::InvolvedMotion() {}
InvolvedMotion::InvolvedMotion(const std::string & type, const std::vector<size_t> & atom, const double & coeff, const double & min) {
    this->type  = type;
    this->atom  = atom;
    this->coeff = coeff;
    this->min   = min;
}
InvolvedMotion::~InvolvedMotion() {}

IntCoordDef::IntCoordDef() {}
IntCoordDef::~IntCoordDef() {}

// Store different internal coordinate definitions
std::vector<std::vector<IntCoordDef>> definitions;

// Input:  file format (Columbus7, default), internal coordinate definition file
// Output: intdim, internal coordinate definition ID
// This function sets the module-wide variable `definitions`
// which will be refered by all routines in this namespace
std::tuple<int64_t, size_t> define_IC(const std::string & format, const std::string & file) {
    int64_t intdim = 0;
    std::vector<IntCoordDef> def;
    if (format == "Columbus7") {
        // First line is always "TEXAS"
        // New internal coordinate line starts with 'K'
        std::ifstream ifs; ifs.open(file);
        if (! ifs.good()) {ifs.close(); ifs.open("intcfl");}
        std::string line; std::getline(ifs, line);
        while (true) {
            std::getline(ifs, line);
            if (! ifs.good()) break;
            if (line[0] == 'K') {
                intdim++;
                def.push_back(IntCoordDef());
            }
            std::string type;
            std::vector<size_t> atom;
            if (line.substr(20,4) == "STRE") {
                type = "stretching";
                atom = {std::stoul(line.substr(28, 5)) - 1, std::stoul(line.substr(34, 9)) - 1};
            } else if (line.substr(20,4) == "BEND") {
                type = "bending";
                atom = {std::stoul(line.substr(28, 6)) - 1, std::stoul(line.substr(45, 9)) - 1, std::stoul(line.substr(35, 9)) - 1};
            } else if (line.substr(20,4) == "TORS") {
                type = "torsion";
                atom = {std::stoul(line.substr(28, 6)) - 1, std::stoul(line.substr(35, 9)) - 1, std::stoul(line.substr(45, 9)) - 1, std::stoul(line.substr(55, 9)) - 1};
            } else if (line.substr(20,3) == "OUT") {
                type = "OutOfPlane";
                atom = {std::stoul(line.substr(28, 6)) - 1, std::stoul(line.substr(45, 9)) - 1, std::stoul(line.substr(55, 9)) - 1, std::stoul(line.substr(35, 9)) - 1};
            }
            double coeff = 1.0;
            if (line.substr(10, 10) != "          ") coeff = std::stod(line.substr(10, 10));
            def[intdim-1].motion.push_back(InvolvedMotion(type, atom, coeff));
        }
        ifs.close();
    }
    else {
        // First 6 spaces of a line are reserved to indicate the start of new internal coordinate
        // Example:
        //  coor |   coeff   |    type     |      atom
        // --------------------------------------------------
        //      1    1.000000    stretching     1     2          # Comment
        //           1.000000    stretching     1     3
        //      2    1.000000    stretching     1     2
        //          -1.000000    stretching     1     3
        //      3    1.000000       bending     2     1     3
        // For a line defining torsion, an additional number at the end of the line defines min
        std::ifstream ifs; ifs.open(file);
        if (! ifs.good()) {ifs.close(); ifs.open("IntCoordDef");}
        while (true) {
            std::string line;
            std::getline(ifs, line);
            if (! ifs.good()) break;
            std::istringstream iss(line);
            std::forward_list<std::string> strs(std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>());
            if (line.substr(0, 6) != "      ") {
                intdim++;
                def.push_back(IntCoordDef());
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
            double min = -M_PI;
            if (! strs.empty()) min = std::stod(strs.front());
            def[intdim-1].motion.push_back(InvolvedMotion(type, atom, coeff, min));
        }
        ifs.close();
    }
    // Normalize linear combination coefficient
    for (size_t i = 0; i < intdim; i++) {
        double norm2 = 0.0;
        for (size_t j = 0; j < def[i].motion.size(); j++) {
            norm2 += def[i].motion[j].coeff * def[i].motion[j].coeff;
        }
        norm2 = sqrt(norm2);
        for (size_t j = 0; j < def[i].motion.size(); j++) {
            def[i].motion[j].coeff /= norm2;
        }
    }
    definitions.push_back(def);
    size_t DefID = definitions.size() - 1;
    return std::make_tuple(intdim, DefID);
}

// Convert r to q according to ID-th internal coordinate definition
at::Tensor compute_IC(const at::Tensor & r, const size_t & ID) {
    auto & def = definitions[ID];
    at::Tensor q = r.new_zeros(def.size());
    for (size_t i = 0; i < def.size(); i++) {
        for (size_t j = 0; j < def[i].motion.size(); j++) {
            auto & type  = def[i].motion[j].type ;
            auto & atom  = def[i].motion[j].atom ;
            auto & coeff = def[i].motion[j].coeff;
            auto & min   = def[i].motion[j].min  ;
            if (type == "stretching") {
                at::Tensor r12 = r.slice(0, 3 * atom[1], 3 * atom[1] + 3)
                               - r.slice(0, 3 * atom[0], 3 * atom[0] + 3);
                q[i] += coeff * r12.norm();
            }
            else if (type == "bending") {
                at::Tensor r21 = r.slice(0, 3 * atom[0], 3 * atom[0] + 3)
                               - r.slice(0, 3 * atom[1], 3 * atom[1] + 3);
                r21 /= r21.norm();
                at::Tensor r23 = r.slice(0, 3 * atom[2], 3 * atom[2] + 3)
                               - r.slice(0, 3 * atom[1], 3 * atom[1] + 3);
                r23 /= r23.norm();
                q[i] += coeff * at::acos(r21.dot(r23));
            }
            else if (type == "torsion") {
                at::Tensor r12 = r.slice(0, 3 * atom[1], 3 * atom[1] + 3)
                               - r.slice(0, 3 * atom[0], 3 * atom[0] + 3);
                at::Tensor r23 = r.slice(0, 3 * atom[2], 3 * atom[2] + 3)
                               - r.slice(0, 3 * atom[1], 3 * atom[1] + 3);
                at::Tensor r34 = r.slice(0, 3 * atom[3], 3 * atom[3] + 3)
                               - r.slice(0, 3 * atom[2], 3 * atom[2] + 3);
                at::Tensor n123 = r12.cross(r23); n123 /= n123.norm();
                at::Tensor n234 = r23.cross(r34); n234 /= n234.norm();
                at::Tensor theta = n123.dot(n234);
                if (theta.item<double>() > 1.0) theta.fill_(0.0);
                else if (theta.item<double>() < -1.0) theta.fill_(M_PI);
                else theta = at::acos(theta);
                if(CL::TS::LA::triple_product(n123, n234, r23) < 0.0) theta = -theta;
                if(theta.item<double>() < min) theta += 2.0 * M_PI;
                else if(theta.item<double>() > min + 2.0 * M_PI) theta -= 2.0 * M_PI;
                q[i] += coeff * theta;
            }
            else if (type == "OutOfPlane") {
                at::Tensor r21 = r.slice(0, 3 * atom[0], 3 * atom[0] + 3)
                               - r.slice(0, 3 * atom[1], 3 * atom[1] + 3);
                r21 /= r21.norm();
                at::Tensor r23 = r.slice(0, 3 * atom[2], 3 * atom[2] + 3)
                               - r.slice(0, 3 * atom[1], 3 * atom[1] + 3);
                at::Tensor r24 = r.slice(0, 3 * atom[3], 3 * atom[3] + 3)
                               - r.slice(0, 3 * atom[1], 3 * atom[1] + 3);
                at::Tensor n234 = r23.cross(r24); n234 /= n234.norm();
                q[i] += coeff * at::asin(n234.dot(r21));
            }
        }
    }
    return q;
}

// From r, generate q & J according to ID-th internal coordinate definition
std::tuple<at::Tensor, at::Tensor> compute_IC_J(const at::Tensor & r, const size_t & ID) {
    auto & def = definitions[ID];
    at::Tensor q = r.new_zeros(def.size());
    at::Tensor J = r.new_zeros({q.size(0), r.size(0)});
    for (size_t i = 0; i < def.size(); i++) {
        for (size_t j = 0; j < def[i].motion.size(); j++) {
            auto & type  = def[i].motion[j].type ;
            auto & atom  = def[i].motion[j].atom ;
            auto & coeff = def[i].motion[j].coeff;
            auto & min   = def[i].motion[j].min  ;
            if (type == "stretching") {
                // Prepare
                at::Tensor runit12 = r.slice(0, 3 * atom[1], 3 * atom[1] + 3)
                                   - r.slice(0, 3 * atom[0], 3 * atom[0] + 3);
                at::Tensor r12 = runit12.norm();
                runit12 /= r12;
                // Output
                q[i] += coeff * r12;
                J[i].slice(0, 3 * atom[0], 3 * atom[0] + 3) += coeff * -runit12;
                J[i].slice(0, 3 * atom[1], 3 * atom[1] + 3) += coeff *  runit12;
            }
            else if (type == "bending") {
                // Prepare
                at::Tensor runit21 = r.slice(0, 3 * atom[0], 3 * atom[0] + 3)
                                   - r.slice(0, 3 * atom[1], 3 * atom[1] + 3);
                at::Tensor r21 = runit21.norm();
                runit21 /= r21;
                at::Tensor runit23 = r.slice(0, 3 * atom[2], 3 * atom[2] + 3)
                                   - r.slice(0, 3 * atom[1], 3 * atom[1] + 3);
                at::Tensor r23 = runit23.norm();
                runit23 /= r23;
                at::Tensor costheta = runit21.dot(runit23);
                at::Tensor sintheta = at::sqrt(1.0 - costheta*costheta);
                at::Tensor J0 = (costheta * runit21 - runit23) / (sintheta * r21);
                at::Tensor J2 = (costheta * runit23 - runit21) / (sintheta * r23);
                // Output
                q[i] += coeff * at::acos(costheta);
                J[i].slice(0, 3 * atom[0], 3 * atom[0] + 3) += coeff * J0;
                J[i].slice(0, 3 * atom[2], 3 * atom[2] + 3) += coeff * J2;
                J[i].slice(0, 3 * atom[1], 3 * atom[1] + 3) += coeff * (- J0 - J2);
            }
            else if (type == "torsion") {
                // Prepare
                at::Tensor runit12 = r.slice(0, 3 * atom[1], 3 * atom[1] + 3)
                                   - r.slice(0, 3 * atom[0], 3 * atom[0] + 3);
                at::Tensor r12 = runit12.norm();
                runit12 /= r12;
                at::Tensor runit23 = r.slice(0, 3 * atom[2], 3 * atom[2] + 3)
                                   - r.slice(0, 3 * atom[1], 3 * atom[1] + 3);
                at::Tensor r23 = runit23.norm();
                runit23 /= r23;
                at::Tensor runit34 = r.slice(0, 3 * atom[3], 3 * atom[3] + 3)
                                   - r.slice(0, 3 * atom[2], 3 * atom[2] + 3);
                at::Tensor r34 = runit34.norm();
                runit34 /= r34;
                at::Tensor cos123 = -(runit12.dot(runit23));
                at::Tensor sin123 = at::sqrt(1.0 - cos123*cos123);
                at::Tensor cos234 = -(runit23.dot(runit34));
                at::Tensor sin234 = at::sqrt(1.0 - cos234*cos234);
                at::Tensor n123 = runit12.cross(runit23) / sin123;
                at::Tensor n234 = runit23.cross(runit34) / sin234;
                at::Tensor theta = n123.dot(n234);
                if (theta.item<double>() > 1.0) theta.fill_(0.0);
                else if (theta.item<double>() < -1.0) theta.fill_(M_PI);
                else theta = at::acos(theta);
                if(CL::TS::LA::triple_product(n123, n234, runit23) < 0.0) theta = -theta;
                if(theta.item<double>() < min) theta += 2.0 * M_PI;
                else if(theta.item<double>() > min + 2.0 * M_PI) theta -= 2.0 * M_PI;
                // Output
                q[i] += coeff * theta;
                J[i].slice(0, 3 * atom[0], 3 * atom[0] + 3) = coeff * (-n123 / (r12 * sin123));
                J[i].slice(0, 3 * atom[1], 3 * atom[1] + 3) = coeff * ((r23 - r12 * cos123) / (r12 * r23 * sin123) * n123 - cos234 / (r23 * sin234) * n234);
                J[i].slice(0, 3 * atom[2], 3 * atom[2] + 3) = coeff * ((r34 * cos234 - r23) / (r23 * r34 * sin234) * n234 + cos123 / (r23 * sin123) * n123);
                J[i].slice(0, 3 * atom[3], 3 * atom[3] + 3) = coeff * ( n234 / (r34 * sin234));
            }
            else if (type == "OutOfPlane") {
                // Prepare
                at::Tensor runit21 = r.slice(0, 3 * atom[0], 3 * atom[0] + 3)
                                   - r.slice(0, 3 * atom[1], 3 * atom[1] + 3);
                at::Tensor r21 = runit21.norm();
                runit21 /= r21;
                at::Tensor runit23 = r.slice(0, 3 * atom[2], 3 * atom[2] + 3)
                                   - r.slice(0, 3 * atom[1], 3 * atom[1] + 3);
                at::Tensor r23 = runit23.norm();
                runit23 /= r23;
                at::Tensor runit24 = r.slice(0, 3 * atom[3], 3 * atom[3] + 3)
                                   - r.slice(0, 3 * atom[1], 3 * atom[1] + 3);
                at::Tensor r24 = runit24.norm();
                runit24 /= r24;
                at::Tensor cos324 = runit23.dot(runit24);
                at::Tensor sin324sq = 1.0 - cos324*cos324;
                at::Tensor sin324 = at::sqrt(sin324sq);
                at::Tensor sintheta = CL::TS::LA::triple_product(runit23, runit24, runit21) / sin324;
                at::Tensor costheta = at::sqrt(1.0 - sintheta*sintheta);
                at::Tensor tantheta = sintheta/costheta;
                at::Tensor J0 = (runit23.cross(runit24) / costheta / sin324 - tantheta * runit21) / r21;
                at::Tensor J2 = (runit24.cross(runit21) / costheta / sin324 - tantheta / sin324sq * (runit23 - cos324 * runit24)) / r23;
                at::Tensor J3 = (runit21.cross(runit23) / costheta / sin324 - tantheta / sin324sq * (runit24 - cos324 * runit23)) / r24;
                // Output
                q[i] += coeff * at::asin(sintheta);
                J[i].slice(0, 3 * atom[0], 3 * atom[0] + 3) += coeff * J0;
                J[i].slice(0, 3 * atom[2], 3 * atom[2] + 3) += coeff * J2;
                J[i].slice(0, 3 * atom[3], 3 * atom[3] + 3) += coeff * J3;
                J[i].slice(0, 3 * atom[1], 3 * atom[1] + 3) += coeff * (- J0 - J2 - J3);
            }
        }
    }
    return std::make_tuple(q, J);
}

} // namespace IC
} // namespace tchem