#include <regex>

#include <CppLibrary/utility.hpp>

#include <tchem/intcoord/IntCoordSet.hpp>

namespace tchem { namespace IC {

IntCoordSet::IntCoordSet() {}
// file format (Columbus7, default), internal coordinate definition file
IntCoordSet::IntCoordSet(const std::string & format, const std::string & file) {
    if (format == "Columbus7") {
        // First line is always "TEXAS"
        // New internal coordinate line starts with 'K'
        std::ifstream ifs; ifs.open(file);
        if (! ifs.good()) {ifs.close(); ifs.open("intcfl");}
        if (! ifs.good()) throw CL::utility::file_error(file + " or intcfl");
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
            else if (line.substr(20,3) == "OUT") {
                type = "OutOfPlane";
                atom = {std::stoul(line.substr(28, 6)) - 1,
                        std::stoul(line.substr(55, 9)) - 1,
                        std::stoul(line.substr(35, 9)) - 1,
                        std::stoul(line.substr(45, 9)) - 1};
            }
            else if (line.substr(20,4) == "TORS") {
                type = "torsion";
                atom = {std::stoul(line.substr(28, 6)) - 1,
                        std::stoul(line.substr(35, 9)) - 1,
                        std::stoul(line.substr(45, 9)) - 1,
                        std::stoul(line.substr(55, 9)) - 1};
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
        if (! ifs.good()) throw CL::utility::file_error(file + " or IntCoordDef");
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
            else if (type == "bending" || type == "cosbend") {
                atom.resize(3);
                atom[0] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[1] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[2] = std::stoul(strs.front()) - 1; strs.pop_front();
            }
            else if (type == "OutOfPlane" || type == "torsion" || type == "sintors" || type == "costors") {
                atom.resize(4);
                atom[0] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[1] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[2] = std::stoul(strs.front()) - 1; strs.pop_front();
                atom[3] = std::stoul(strs.front()) - 1; strs.pop_front();
            }
            else throw "Error during reading internal coordinate definition: unsupported internal coordinate type: " + type;
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

const std::vector<IntCoord> & IntCoordSet::intcoords() const {return intcoords_;}

size_t IntCoordSet::size() const {return intcoords_.size();}
// Read-only reference to an internal coordinate
const IntCoord & IntCoordSet::operator[](const size_t & index) const {return intcoords_[index];}

void IntCoordSet::print(std::ofstream & ofs, const std::string & format) const {
    if (format == "Columbus7") {
        ofs << "TEXAS\n";
        for (const IntCoord & intcoord : intcoords_) intcoord.print(ofs, format);
    }
    else
    for (const IntCoord & intcoord : intcoords_) intcoord.print(ofs, format);
}

// Return q given r
at::Tensor IntCoordSet::operator()(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::operator(): r must be a vector");
    at::Tensor q = r.new_empty(intcoords_.size());
    for (size_t i = 0; i < intcoords_.size(); i++) q[i] = intcoords_[i](r);
    return q;
}
// Return q and J given r
std::tuple<at::Tensor, at::Tensor> IntCoordSet::compute_IC_J(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::compute_IC_J: r must be a vector");
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
// Return q and J and K given r
std::tuple<at::Tensor, at::Tensor, at::Tensor> IntCoordSet::compute_IC_J_K(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::compute_IC_J_K: r must be a vector");
    at::Tensor q = r.new_empty(intcoords_.size());
    at::Tensor J = r.new_empty({q.size(0), r.size(0)});
    at::Tensor K = r.new_empty({q.size(0), r.size(0), r.size(0)});
    for (size_t i = 0; i < intcoords_.size(); i++) {
        at::Tensor qi, Ji, Ki;
        std::tie(qi, Ji, Ki) = intcoords_[i].compute_IC_J_K(r);
        q[i] = qi;
        J[i] = Ji;
        K[i] = Ki;
    }
    return std::make_tuple(q, J, K);
}

// Return internal coordinate gradient given r and Cartesian coordinate gradient
at::Tensor IntCoordSet::gradient_cart2int(const at::Tensor & r, const at::Tensor & cartgrad) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::gradient_cart2int: r must be a vector");
    if (cartgrad.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::gradient_cart2int: Cartesian coordinate gradient must be a vector");
    if (cartgrad.size(0) != r.size(0)) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::gradient_cart2int: Cartesian coordinate gradient must share a same dimension with r");
    at::Tensor q, J;
    std::tie(q, J) = this->compute_IC_J(r);
    at::Tensor JJT = J.mm(J.transpose(0, 1));
    at::Tensor cholesky = JJT.cholesky();
    at::Tensor inverse = at::cholesky_inverse(cholesky);
    at::Tensor cart2int = inverse.mm(J);
    at::Tensor intgrad = cart2int.mv(cartgrad);
    return intgrad;
}
// Return Cartesian coordinate gradient given r and internal coordinate gradient
at::Tensor IntCoordSet::gradient_int2cart(const at::Tensor & r, const at::Tensor & intgrad) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::gradient_int2cart: r must be a vector");
    if (intgrad.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::gradient_int2cart: Internal coordinate gradient must be a vector");
    at::Tensor q, J;
    std::tie(q, J) = this->compute_IC_J(r);
    if (intgrad.size(0) != q.size(0)) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::gradient_int2cart: Internal coordinate gradient must share a same dimension with q");
    at::Tensor cartgrad = J.transpose(0, 1).mv(intgrad);
    return cartgrad;
}

// Return internal coordinate Hessian given r and Cartesian coordinate gradient and Hessian
at::Tensor IntCoordSet::Hessian_cart2int(const at::Tensor & r, const at::Tensor & cartgrad, const at::Tensor & cartHess) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::Hessian_cart2int: r must be a vector");
    if (cartgrad.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::Hessian_cart2int: Cartesian coordinate gradient must be a vector");
    if (cartgrad.size(0) != r.size(0)) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::Hessian_cart2int: Cartesian coordinate gradient must share a same dimension with r");   
    if (cartHess.sizes().size() != 2) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::Hessian_cart2int: Cartesian coordinate Hessian must be a matrix");
    if (cartHess.size(0) != cartHess.size(1)) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::Hessian_cart2int: Cartesian coordinate Hessian must be a square matrix");
    if (cartgrad.size(0) != cartHess.size(0)) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::Hessian_cart2int: The dimensions of the gradient and the Hessian must match");
    at::Tensor q, J, K;
    std::tie(q, J, K) = this->compute_IC_J_K(r);
    at::Tensor JJT = J.mm(J.transpose(0, 1));
    at::Tensor cholesky = JJT.cholesky();
    at::Tensor inverse = at::cholesky_inverse(cholesky);
    at::Tensor AT = inverse.mm(J);
    at::Tensor A  = AT.transpose(0, 1);
    at::Tensor C = at::matmul(AT, at::matmul(K, A));
    at::Tensor intgrad = AT.mv(cartgrad);
    C.transpose_(0, -2); // move the contraction dimension to -2 for batched matmul
    at::Tensor intHess = AT.mm(cartHess.mm(A)) - at::matmul(intgrad, C);
    return intHess;
}
// Return Cartesian coordinate Hessian given r and internal coordinate gradient and Hessian
at::Tensor IntCoordSet::Hessian_int2cart(const at::Tensor & r, const at::Tensor & intgrad, const at::Tensor & intHess) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::Hessian_int2cart: r must be a vector");
    if (intgrad.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::Hessian_int2cart: Internal coordinate gradient must be a vector");
    if (intHess.sizes().size() != 2) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::Hessian_int2cart: Internal coordinate Hessian must be a matrix");
    if (intHess.size(0) != intHess.size(1)) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::Hessian_int2cart: Internal coordinate Hessian must be a square matrix");
    if (intgrad.size(0) != intHess.size(0)) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::Hessian_int2cart: The dimensions of the gradient and the Hessian must match");
    at::Tensor q, J, K;
    std::tie(q, J, K) = this->compute_IC_J_K(r);
    if (intgrad.size(0) != q.size(0)) throw std::invalid_argument(
    "tchem::IC::IntCoordSet::Hessian_int2cart: Internal coordinate gradient must share a same dimension with q");
    K.transpose_(0, -2); // move the contraction dimension to -2 for batched matmul
    at::Tensor cartHess = J.transpose(0, 1).mm(intHess.mm(J)) + at::matmul(intgrad, K);
    return cartHess;
}

} // namespace IC
} // namespace tchem