/*
The procedure to get symmetry adapted and scaled internal coordinate (SASIC) is:
    1. Get internal coordinate (IC), which is taken care of by module `intcoord`
    2. Nondimensionalize the IC to get dimensionless internal coordinate (DIC):
       for length, DIC = (IC - origin) / origin
       for angle , DIC =  IC - origin
    3. Scale the DIC to get scaled dimensionless internal coordinate (SDIC):
       if no scaler      : SDIC = DIC
       elif scaler = self: SDIC = pi * erf(DIC)
       else              : SDIC = DIC * exp(-alpha * scaler DIC)
    4. Symmetry adapted linear combinate the SDIC to get SSAIC
*/

#ifndef tchem_SAS_hpp
#define tchem_SAS_hpp

#include <torch/torch.h>

#include <tchem/intcoord.hpp>

namespace tchem { namespace SAS {

// The rule of internal coordinates who are scaled by others
// self is scaled by scaler with alpha
struct OthScalRul {
    size_t self, scaler;
    double alpha;

    inline OthScalRul() {}
    OthScalRul(const std::vector<std::string> & input_strs);
    inline ~OthScalRul() {}
};

// A symmetry adapted and scaled internal coordinate
class SASIC {
    private:
        std::vector<double> coeffs_;
        std::vector<size_t> indices_;
    public:
        inline SASIC() {}
        inline ~SASIC() {}

        inline std::vector<double> coeffs() const {return coeffs_;}
        inline std::vector<size_t> indices() const {return indices_;}

        // Append a linear combination coefficient - index of scaled internal coordinate pair
        void append(const double & coeff, const size_t & index);
        // Normalize linear combination coefficients
        void normalize();

        // Return the symmetry adapted and scaled internal coordinate
        // given the scaled internal coordinate vector
        at::Tensor operator()(const at::Tensor & SIC) const;
};

// A set of symmetry adapted and scaled internal coordinates
class SASICSet : public tchem::IC::IntCoordSet {
    private:
        // Internal coordinate origin
        at::Tensor origin_;
        // Internal coordinates who are scaled by others
        std::vector<OthScalRul> other_scaling_;
        // Internal coordinates who are scaled by themselves are picked out by self_scaling_ matrix
        // The self scaled internal coordinate vector is q = erf(self_scaling_.mv(q)) + self_complete_.mv(q)
        at::Tensor self_scaling_, self_complete_;
        // sasicss_[i][j] contains the definition of
        // j-th symmetry adapted internal coordinate in i-th irreducible
        std::vector<std::vector<SASIC>> sasicss_;
    public:
        inline SASICSet() {}
        // internal coordinate definition format (Columbus7, default)
        // internal coordinate definition file
        // symmetry adaptation and scale definition file
        SASICSet(const std::string & format, const std::string & IC_file, const std::string & SAS_file);
        inline ~SASICSet() {}

        inline at::Tensor origin() const {return origin_;}

        // Return number of irreducible representations
        inline size_t NIrreds() const {return sasicss_.size();}
        // Return number of symmetry adapted and scaled internal coordinates per irreducible
        std::vector<size_t> NSASICs() const;

        // Return SASIC given internal coordinate q
        std::vector<at::Tensor> operator()(const at::Tensor & q);
};

} // namespace SASIC
} // namespace tchem

#endif