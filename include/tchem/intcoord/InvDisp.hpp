#ifndef tchem_IC_InvDisp_hpp
#define tchem_IC_InvDisp_hpp

#include <torch/torch.h>

namespace tchem { namespace IC {

// A translationally and rotationally invariant displacement
class InvDisp {
    private:
        // Currently only support stretching, bending, OutOfPlane, torsion
        // stretching: bond length atom1_atom2
        // bending   : bond angle atom1_atom2_atom3, range [0,pi]
        //             singular derivative at 0 and pi
        // OutOfPlane: out of plane angle atom1_atom2_atom3_atom4, range [-pi/2, pi/2]
        //             specifically, bond 12 out of plane 234
        //             singular derivative at +-pi/2
        // torsion   : dihedral angle atom1_atom2_atom3_atom4, range [min, min + 2pi]
        //             specifically, angle between plane 123 and plane 234
        //             dihedral angle has same sign to n_123 x n_234 . r_23
        //             where n_abc (the normal vector of plane abc) is the unit vector along r_ab x r_bc
        // sintors   : sin(torsion)
        // costors   : cos(torsion)
        std::string type_;
        // involved atoms
        std::vector<size_t> atoms_;
        // for torsion only, deafult = -pi
        // if (the dihedral angle < min)       angle += 2pi
        // if (the dihedral angle > min + 2pi) angle -= 2pi
        // The dihedral angle is double-valued at min and min + 2pi
        double min_ = -M_PI;
    public:
        InvDisp();
        InvDisp(const std::string & _type, const std::vector<size_t> & _atoms);
        InvDisp(const std::string & _type, const std::vector<size_t> & _atoms, const double & _min);
        ~InvDisp();

        const std::string & type() const;
        const std::vector<size_t> & atoms() const;
        const double & min() const;

        void print(std::ofstream & ofs, const std::string & format) const;

        // Return the displacement given r
        at::Tensor operator()(const at::Tensor & r) const;
        // Return the displacement and its gradient over r given r
        std::tuple<at::Tensor, at::Tensor> compute_IC_J(const at::Tensor & r) const;
        // Return the displacement and its 1st and 2nd order gradient over r given r
        std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_IC_J_K(const at::Tensor & r) const;
};

} // namespace IC
} // namespace tchem

#endif