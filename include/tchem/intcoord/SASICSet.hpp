#ifndef tchem_IC_SASICSet_hpp
#define tchem_IC_SASICSet_hpp

#include <tchem/intcoord/IntCoordSet.hpp>
#include <tchem/intcoord/SASIC.hpp>

namespace tchem { namespace IC {

// The rule of internal coordinates who are scaled by others
// self is scaled by scaler with alpha
struct OthScalRul {
    size_t self, scaler;
    double alpha;

    inline OthScalRul() {}
    inline OthScalRul(const size_t & _self, const size_t & _scaler, const double & _alpha = 1.0) : self(_self), scaler(_scaler), alpha(_alpha) {}
    inline ~OthScalRul() {}
};

// A set of symmetry adapted and scaled internal coordinates
class SASICSet : public tchem::IC::IntCoordSet {
    private:
        // internal coordinate origin
        at::Tensor origin_;
        // internal coordinates who are scaled by others
        std::vector<OthScalRul> other_scaling_;
        // Internal coordinates who are scaled by themselves are picked out by self_scaling_ matrix
        // The self scaled internal coordinate vector is
        //     q = pi * erf(self_alpha_ * self_scaling_.mv(q)) + self_complete_.mv(q)
        at::Tensor self_alpha_, self_scaling_, self_complete_;
        // sasicss_[i][j] contains the definition of
        // j-th symmetry adapted internal coordinate in i-th irreducible
        std::vector<std::vector<SASIC>> sasicss_;
    public:
        SASICSet();
        // internal coordinate definition format (Columbus7, default)
        // internal coordinate definition file
        // symmetry adaptation and scale definition file
        SASICSet(const std::string & format, const std::string & IC_file, const std::string & SAS_file);
        ~SASICSet();

        const at::Tensor & origin() const;

        // Return number of irreducible representations
        size_t NIrreds() const;
        // Return number of symmetry adapted and scaled internal coordinates per irreducible
        std::vector<size_t> NSASICs() const;
        // Return number of internal coordinates
        size_t intdim() const;

        // Return SASIC given internal coordinate q
        std::vector<at::Tensor> operator()(const at::Tensor & q);
};

} // namespace IC
} // namespace tchem

#endif