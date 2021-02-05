#include <regex>

#include <CppLibrary/utility.hpp>

#include <tchem/SApolynomial.hpp>

namespace tchem { namespace polynomial {

SAP::SAP() {}
// For example, the input line of a 2nd order term made up by
// the 1st coordinate in the 2nd irreducible and
// the 3rd coordinate in the 4th irreducible is:
//     2    2,1    4,3
// The splitted input line is taken in as `strs`
SAP::SAP(const std::vector<std::string> & strs) {
    size_t order = std::stoul(strs[0]);
    irreds_.resize(order);
    coords_.resize(order);
    for (size_t i = 0; i < order; i++) {
        std::vector<std::string> irred_coord = CL::utility::split(strs[i + 1], ',');
        irreds_[i] = std::stoul(irred_coord[0]) - 1;
        coords_[i] = std::stoul(irred_coord[1]) - 1;
    }
}
SAP::~SAP() {}

std::vector<size_t> SAP::irreds() const {return irreds_;}
std::vector<size_t> SAP::coords() const {return coords_;}

// Return the symmetry adapted polynomial SAP(x) given x
at::Tensor SAP::operator()(const std::vector<at::Tensor> & xs) const {
    // assert(("Elements of xs must be vectors"));
    at::Tensor value = xs[0].new_full({}, 1.0);
    for (size_t i = 0; i < irreds_.size(); i++)
    value = value * xs[irreds_[i]][coords_[i]];
    return value;
}





SAPSet::SAPSet() {}
// Symmetry adapted polynomials definition file
SAPSet::SAPSet(const std::string sapoly_file) {
    std::ifstream ifs; ifs.open(sapoly_file);
        std::string line;
        std::getline(ifs, line);
        while (true) {
            SAPs_.push_back({});
            std::vector<SAP> & irred = SAPs_.back();
            while (true) {
                std::string line;
                std::getline(ifs, line);
                if (! ifs.good()) break;
                std::vector<std::string> strs;
                CL::utility::split(line, strs);
                if (! std::regex_match(strs[0], std::regex("\\d+"))) break;
                irred.push_back(SAP(strs));
            }
            if (! ifs.good()) break;
        }
    ifs.close();
}
SAPSet::~SAPSet() {}

std::vector<std::vector<SAP>> SAPSet::SAPs() const {return SAPs_;}

void SAPSet::pretty_print(std::ostream & stream) const {
    stream << "Number of irreducibles = " << SAPs_.size() << '\n';
    for (size_t i = 0; i < SAPs_.size(); i++) {
        stream << "Irreducible " << i << ":\n";
        stream << "Number of symmetry adapted polynomials = " << SAPs_[i].size() << '\n';
    }
}

// Return the value of each term in {P(x)} as a vector of vectors
std::vector<at::Tensor> SAPSet::operator()(const std::vector<at::Tensor> & xs) const {
    // assert(("Elements of xs must be vectors"));
    std::vector<at::Tensor> values(xs.size());
    for (size_t i = 0; i < values.size(); i++) {
        values[i] = xs[i].new_empty(SAPs_[i].size());
        for (size_t j = 0; j < SAPs_[i].size(); j++)
        values[i][j] = SAPs_[i][j](xs);
    }
    return values;
}

} // namespace polynomial
} // namespace tchem