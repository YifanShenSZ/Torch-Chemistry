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

#ifndef tchem_intcoord_hpp
#define tchem_intcoord_hpp

#include <torch/torch.h>

namespace tchem { namespace IC {
    
struct InvolvedMotion {
    // Currently only support stretching, bending, torsion, OutOfPlane
    // stretching: the motion coordinate is bond length atom1_atom2
    // bending   : the motion coordinate is bond angle atom1_atom2_atom3, range [0,pi]
    //             derivative encounters singularity at pi
    // torsion   : the motion coordinate is dihedral angle atom1_atom2_atom3_atom4, range [min, min + 2pi]
    //             specifically, angle between plane 123 and plane 234
    //             dihedral angle has same sign to n_123 x n_234 . r_23
    //             where n_abc (the normal vector of plane abc) is the unit vector along r_ab x r_bc
    // OutOfPlane: the motion coordinate is out of plane angle atom1_atom2_atom3_atom4, range [-pi/2, pi/2]
    //             specifically, bond 12 out of plane 234
    std::string type;
    // Involved atoms
    std::vector<size_t> atom;
    // Linear combination coefficient
    double coeff;
    // For torsion only, deafult = -pi
    // if (the dihedral angle < min)       angle += 2pi
    // if (the dihedral angle > min + 2pi) angle -= 2pi
    double min;

    InvolvedMotion();
    InvolvedMotion(const std::string & type, const std::vector<size_t> & atom, const double & coeff, const double & min = -M_PI);
    ~InvolvedMotion();
};
struct IntCoordDef {
    std::vector<InvolvedMotion> motion;
    
    IntCoordDef();
    ~IntCoordDef();
};

// Store different internal coordinate definitions
extern std::vector<std::vector<IntCoordDef>> definitions;

// Input:  file format (Columbus7, default), internal coordinate definition file
// Output: intdim, internal coordinate definition ID
std::tuple<int64_t, size_t> define_IC(const std::string & format, const std::string & file);

// Convert r to q according to ID-th internal coordinate definition
at::Tensor compute_IC(const at::Tensor & r, const size_t & ID = 0);

// From r, generate q & J according to ID-th internal coordinate definition
std::tuple<at::Tensor, at::Tensor> compute_IC_J(const at::Tensor & r, const size_t & ID = 0);

} // namespace IC
} // namespace tchem

#endif