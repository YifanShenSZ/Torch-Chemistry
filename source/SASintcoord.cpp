#include <regex>
#include <torch/torch.h>

#include <CppLibrary/utility.hpp>
#include <CppLibrary/linalg.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/intcoord.hpp>

#include <tchem/SASintcoord.hpp>

namespace tchem { namespace IC {

OthScalRul::OthScalRul() {}
OthScalRul::OthScalRul(const std::vector<std::string> & input_strs) {
    self   = std::stoul(input_strs[0]) - 1;
    scaler = std::stoul(input_strs[1]) - 1;
    alpha  = std::stod (input_strs[2]);
}
OthScalRul::~OthScalRul() {}





SASIC::SASIC() {}
SASIC::~SASIC() {}

std::vector<double> SASIC::coeffs() const {return coeffs_;}
std::vector<size_t> SASIC::indices() const {return indices_;}

// Append a linear combination coefficient - index of scaled internal coordinate pair
void SASIC::append(const double & coeff, const size_t & index) {
    coeffs_.push_back(coeff);
    indices_.push_back(index);
}
// Normalize linear combination coefficients
void SASIC::normalize() {
    double norm2 = CL::linalg::norm2(coeffs_);
    for (double & coeff : coeffs_) coeff /= norm2;
}

// Return the symmetry adapted and scaled internal coordinate
// given the scaled internal coordinate vector
at::Tensor SASIC::operator()(const at::Tensor & SIC) const {
    at::Tensor sasic = coeffs_[0] * SIC[indices_[0]];
    for (size_t i = 1; i < coeffs_.size(); i++) sasic = sasic + coeffs_[i] * SIC[indices_[i]];
    return sasic;
}





SASICSet::SASICSet() {}
// internal coordinate definition format (Columbus7, default)
// internal coordinate definition file
// symmetry adaptation and scale definition file
SASICSet::SASICSet(const std::string & format, const std::string & IC_file, const std::string & SAS_file)
: tchem::IC::IntCoordSet(format, IC_file) {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    std::ifstream ifs; ifs.open(SAS_file);
    assert((SAS_file + " must be good", ifs));
        std::string line;
        // Internal coordinate origin
        std::string origin_file;
        std::getline(ifs, line);
        std::getline(ifs, origin_file);
        CL::utility::trim(origin_file);
        CL::chem::xyz<double> molorigin(origin_file, true);
        std::vector<double> origin_vector = molorigin.coords();
        at::Tensor origin_tensor = at::from_blob(origin_vector.data(), origin_vector.size(), top);
        origin_ = this->tchem::IC::IntCoordSet::operator()(origin_tensor);
        // Internal coordinates who are scaled by others
        std::getline(ifs, line);
        while (true) {
            std::getline(ifs, line);
            std::vector<std::string> strs = CL::utility::split(line);
            if (! std::regex_match(strs[0], std::regex("\\d+"))) break;
            other_scaling_.push_back(OthScalRul(strs));
        }
        // Internal coordinates who are scaled by themselves
        std::vector<size_t> self_vector;
        while (true) {
            std::getline(ifs, line);
            if (! std::regex_match(line, std::regex("\\ *\\d+\\ *"))) break;
            self_vector.push_back(std::stoul(line) - 1);
        }
        int64_t intdim = this->intcoords().size();
        self_scaling_ = at::zeros({intdim, intdim}, top);
        self_complete_ = at::eye(intdim, top);
        for (size_t & self : self_vector) {
            self_scaling_ [self][self] = 1.0;
            self_complete_[self][self] = 0.0;
        }
        // Symmetry adapted linear combinations of each irreducible
        while (true) {
            if (! ifs.good()) break;
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

at::Tensor SASICSet::origin() const {return origin_;}

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
    // Nondimensionalize
    at::Tensor DIC = q - origin_;
    std::vector<tchem::IC::IntCoord> IntCoordDef = intcoords();
    for (size_t i = 0; i < IntCoordDef.size(); i++)
    if (IntCoordDef[i].invdisps()[0].type() == "stretching")
    DIC[i] = DIC[i] / origin_[i];
    // Scale
    at::Tensor SDIC = DIC.clone();
    for (OthScalRul & scaling : other_scaling_)
    SDIC[scaling.self] = DIC[scaling.self] * at::exp(-scaling.alpha * DIC[scaling.scaler]);
    SDIC = M_PI * at::erf(self_scaling_.mv(SDIC)) + self_complete_.mv(SDIC);
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