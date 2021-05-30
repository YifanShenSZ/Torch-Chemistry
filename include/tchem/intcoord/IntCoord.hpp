#ifndef tchem_IC_IntCoord_hpp
#define tchem_IC_IntCoord_hpp

#include <tchem/intcoord/InvDisp.hpp>

namespace tchem { namespace IC {

// An internal coordinate, i.e. a linear combination of several translationally and rotationally invariant displacements
class IntCoord {
    private:
        // linear combination coefficient - invariant displacement pairs
        std::vector<std::pair<double, InvDisp>> coeff_invdisps_;
    public:
        IntCoord();
        IntCoord(const std::vector<std::pair<double, InvDisp>> & _coeff_invdisps);
        ~IntCoord();

        const std::vector<std::pair<double, InvDisp>> & coeff_invdisps() const;

        // Read-only reference to a linear combination coefficient - invariant displacement pair
        const std::pair<double, InvDisp> & operator[](const size_t & index) const;

        // Append a linear combination coefficient - invariant displacement pair
        void append(const std::pair<double, InvDisp> & coeff_invdisp);
        void append(const double & coeff, const InvDisp & invdisp);

        // Normalize linear combination coefficients
        void normalize();

        void print(std::ofstream & ofs, const std::string & format) const;

        // Return the internal coordinate given r
        at::Tensor operator()(const at::Tensor & r) const;
        // Return the internal coordinate and its gradient over r given r
        std::tuple<at::Tensor, at::Tensor> compute_IC_J(const at::Tensor & r) const;
        // Return the internal coordinate and its 1st and 2nd order gradient over r given r
        std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_IC_J_K(const at::Tensor & r) const;
};

} // namespace IC
} // namespace tchem

#endif