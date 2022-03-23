#ifndef tchem_IC_InvDisp_hpp
#define tchem_IC_InvDisp_hpp

#include <torch/torch.h>

namespace tchem { namespace IC {

// available invariant displacement types
enum InvDisp_type {
    // 1, just a constant
    dummy,
    // bond length atom1_atom2
    stretching,
    // bending    : bond angle atom1_atom2_atom3, range [0,pi]
    //              singular derivative at 0 and pi
    // cosbending : cos(bending), avoiding derivative singularity
    bending, cosbending,
    // torsion    : dihedral angle atom1_atom2_atom3_atom4, range [min, min + 2pi]
    //              concretely, angle between plane 123 and plane 234
    //              sin(torsion) = n_123 x n_234 . runit_23
    //              cos(torsion) = n_123 . n_234
    //              where n_abc (the normal vector of plane abc) is the unit vector along r_ab x r_bc
    //              double-valued at min and min + 2pi
    // sintorsion : sin(torsion), avoiding double value
    // costorsion : cos(torsion), avoiding double value
    torsion, sintorsion, costorsion,
    // torsion2   : dihedral angle atom1_atom2_atom3_atom4_atom5, range [min, min + 2pi]
    //              concretely, angle between plane 123 and (pseudo) plane 2345
    //              where n_2345 is the unit vector along r_23 x r_45
    //              sin(torsion2) = n_123 x n_2345 . runit_23
    //              cos(torsion2) = n_123 . n_2345
    //              double-valued at min and min + 2pi
    // sintorsion2: sin(torsion2), avoiding double value
    // costorsion2: cos(torsion2), avoiding double value
    // When bond angle atom1_atom2_atom3 = 0 or pi, torsion2 is discontinuous
    // pxtorsion2 : sin(bond angle atom1_atom2_atom3) * cos(torsion2), avoiding discontinuity
    // pytorsion2 : sin(bond angle atom1_atom2_atom3) * sin(torsion2), avoiding discontinuity
    torsion2, sintorsion2, costorsion2, pxtorsion2, pytorsion2,
    // OutOfPlane : out of plane angle atom1_atom2_atom3_atom4, range [-pi/2, pi/2]
    //              concretely, bond 12 out of plane 234
    //              singular derivative at +-pi/2
    // sinoop     : sin(out of plane), avoiding derivative singularity
    OutOfPlane, sinoop
};
static const std::vector<std::string> InvDisp_typestr = {
    "dummy", "stretching",
    "bending", "cosbending",
    "torsion", "sintorsion", "costorsion",
    "torsion2", "sintorsion2", "costorsion2", "pxtorsion2", "pytorsion2",
    "OutOfPlane", "sinoop"
};

// A translationally and rotationally invariant displacement
class InvDisp {
    private:
        InvDisp_type type_;
        // involved atoms
        std::vector<size_t> atoms_;
        // for torsion and torsion2 only, deafult = -pi
        // if (the dihedral angle < min)       angle += 2pi
        // if (the dihedral angle > min + 2pi) angle -= 2pi
        // The dihedral angle is double-valued at min and min + 2pi
        double min_ = -M_PI;
    public:
        InvDisp();
        InvDisp(const std::string & _type, const std::vector<size_t> & _atoms, const double & _min = -M_PI);
        ~InvDisp();

        const InvDisp_type & type() const;
        const std::vector<size_t> & atoms() const;
        const double & min() const;

        void print(std::ofstream & ofs, const std::string & format) const;

        // return the displacement given r
        at::Tensor operator()(const at::Tensor & r) const;
        // return the displacement and its gradient over r given r
        std::tuple<at::Tensor, at::Tensor> compute_IC_J(const at::Tensor & r) const;
        // return the displacement and its 1st and 2nd order gradient over r given r
        std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_IC_J_K(const at::Tensor & r) const;
};

} // namespace IC
} // namespace tchem

#endif