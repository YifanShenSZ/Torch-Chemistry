#include <regex>

#include <CppLibrary/utility.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/intcoord/SASICSet.hpp>

namespace tchem { namespace IC {

SASICSet::SASICSet() {}
// internal coordinate definition format (Columbus7, default)
// internal coordinate definition file
// symmetry adaptation and scale definition file
SASICSet::SASICSet(const std::string & format, const std::string & IC_file, const std::string & SAS_file)
: tchem::IC::IntCoordSet(format, IC_file) {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    std::ifstream ifs; ifs.open(SAS_file);
    if (! ifs.good()) throw CL::utility::file_error(SAS_file);
        std::string line;
        // internal coordinate origin
        std::string origin_file;
        std::getline(ifs, line);
        std::getline(ifs, origin_file);
        CL::utility::trim(origin_file);
        CL::chem::xyz<double> molorigin(origin_file, true);
        std::vector<double> origin_vector = molorigin.coords();
        at::Tensor origin_tensor = at::from_blob(origin_vector.data(), origin_vector.size(), top);
        origin_ = this->tchem::IC::IntCoordSet::operator()(origin_tensor);
        // internal coordinates who are scaled by others
        std::getline(ifs, line);
        while (true) {
            std::getline(ifs, line);
            std::vector<std::string> strs = CL::utility::split(line);
            if (! std::regex_match(strs[0], std::regex("\\d+"))) break;
            // self & scaler are specified, alpha is not
            if (strs.size() == 2) {
                size_t self   = std::stoul(strs[0]) - 1,
                       scaler = std::stoul(strs[1]) - 1;
                other_scaling_.push_back(OthScalRul(self, scaler));
            }
            // self & scaler & alpha are specified
            else if (strs.size() == 3) {
                size_t self   = std::stoul(strs[0]) - 1,
                       scaler = std::stoul(strs[1]) - 1;
                double alpha  = std::stod (strs[2]);
                other_scaling_.push_back(OthScalRul(self, scaler, alpha));
            }
            // wrong input format
            else throw CL::utility::file_error(SAS_file);
        }
        // internal coordinates who are scaled by themselves
        std::vector<std::pair<size_t, double>> self_vector;
        while (true) {
            std::getline(ifs, line);
            std::vector<std::string> strs = CL::utility::split(line);
            if (! std::regex_match(strs[0], std::regex("\\d+"))) break;
            // self is specified, alpha is not so default it to 1.0
            if (strs.size() == 1) {
                size_t self = std::stoul(strs[0]) - 1;
                self_vector.push_back({self, 1.0});
            }
            // self & alpha are specified
            else if (strs.size() == 2) {
                size_t self  = std::stoul(strs[0]) - 1;
                double alpha = std::stod (strs[1]);
                self_vector.push_back({self, alpha});
            }
            else throw CL::utility::file_error(SAS_file);
        }
        int64_t intdim = this->intcoords().size();
        self_alpha_    = at::zeros(intdim, top);
        self_scaling_  = at::zeros({intdim, intdim}, top);
        self_complete_ = at::eye(intdim, top);
        for (const auto & coord_alpha : self_vector) {
            const size_t & self = coord_alpha.first;
            self_alpha_[self] = coord_alpha.second;
            self_scaling_ [self][self] = 1.0;
            self_complete_[self][self] = 0.0;
        }
        // symmetry adapted linear combinations of each irreducible
        while (ifs.good()) {
            sasicss_.push_back(std::vector<SASIC>());
            std::vector<SASIC> & sasics = sasicss_.back();
            while (true) {
                std::getline(ifs, line);
                if (! ifs.good()) break;
                std::forward_list<std::string> strs;
                CL::utility::split(line, strs);
                if (! std::regex_match(strs.front(), std::regex("-?\\d+\\.?\\d*"))) break;
                if (std::regex_match(strs.front(), std::regex("\\d+"))) {
                    sasics.push_back(SASIC());
                    strs.pop_front();
                }
                double coeff = std::stod(strs.front()); strs.pop_front();
                size_t index = std::stoul(strs.front()) - 1;
                sasics.back().append(coeff, index);
            }
            // Normalize linear combination coefficients
            for (SASIC & sasic : sasics) sasic.normalize();
        }
    ifs.close();
}
SASICSet::~SASICSet() {}

const at::Tensor & SASICSet::origin() const {return origin_;}

// Return number of irreducible representations
size_t SASICSet::NIrreds() const {return sasicss_.size();}
// Return number of symmetry adapted and scaled internal coordinates per irreducible
std::vector<size_t> SASICSet::NSASICs() const {
    std::vector<size_t> N(NIrreds());
    for (size_t i = 0; i < NIrreds(); i++) N[i] = sasicss_[i].size();
    return N;
}
// Return number of internal coordinates
size_t SASICSet::intdim() const {
    size_t intdim = 0;
    for (const auto & sasics : sasicss_) intdim += sasics.size();
    return intdim;
}

std::vector<at::Tensor> SASICSet::operator()(const at::Tensor & q) {
    if (q.sizes().size() != 1) throw std::invalid_argument(
    "tchem::IC::SASICSet::operator(): q must be a vector");
    if (q.size(0) != this->size()) throw std::invalid_argument(
    "tchem::IC::SASICSet::operator(): inconsisten dimension between q and this internal coordinate system");
    // Nondimensionalize
    at::Tensor DIC = q - origin_;
    for (size_t i = 0; i < this->size(); i++)
    if ((*this)[i][0].second.type() == "stretching")
    DIC[i] = DIC[i] / origin_[i];
    // Scale
    at::Tensor SDIC = DIC.clone();
    for (const OthScalRul & osr : other_scaling_) SDIC[osr.self] = DIC[osr.self] * at::exp(-osr.alpha * DIC[osr.scaler]);
    SDIC = M_PI * at::erf(self_alpha_ * self_scaling_.mv(SDIC)) + self_complete_.mv(SDIC);
    // Symmetrize
    std::vector<at::Tensor> SASgeom(NIrreds());
    for (size_t i = 0; i < NIrreds(); i++) {
        SASgeom[i] = q.new_zeros(sasicss_[i].size());
        for (size_t j = 0; j < sasicss_[i].size(); j++) SASgeom[i][j] = sasicss_[i][j](SDIC);
    }
    return SASgeom;
}

} // namespace IC
} // namespace tchem