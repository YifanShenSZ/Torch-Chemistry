/*
An interal coordinate is a linear combination of several translationally and rotationally invariant displacements
but only displacements under same unit can be combined, i.e. you must treat lengthes and angles separately
unless appropriate metric tensor is applied

Nomenclature:
    cartdim & intdim: Cartesian & internal space dimensionality
    r: Cartesian coordinate vector
    q: internal coordinate vector
    J: the Jacobian matrix of q over r

Warning:
    * J of bending is singular at 0 and pi,
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

// A translationally and rotationally invariant displacement
class InvDisp {
    private:
        // Currently only support stretching, bending, torsion, OutOfPlane
        // stretching: bond length atom1_atom2
        // bending   : bond angle atom1_atom2_atom3, range [0,pi]
        //             derivative encounters singularity at pi
        // torsion   : dihedral angle atom1_atom2_atom3_atom4, range [min, min + 2pi]
        //             specifically, angle between plane 123 and plane 234
        //             dihedral angle has same sign to n_123 x n_234 . r_23
        //             where n_abc (the normal vector of plane abc) is the unit vector along r_ab x r_bc
        // OutOfPlane: out of plane angle atom1_atom2_atom3_atom4, range [-pi/2, pi/2]
        //             specifically, bond 12 out of plane 234
        std::string type_;
        // involved atoms
        std::vector<size_t> atoms_;
        // for torsion only, deafult = -pi
        // if (the dihedral angle < min)       angle += 2pi
        // if (the dihedral angle > min + 2pi) angle -= 2pi
        double min_ = -M_PI;
    public:
        inline InvDisp() {}
        inline InvDisp(const std::string & _type, const std::vector<size_t> & _atoms) : type_(_type), atoms_(_atoms) {};
        inline InvDisp(const std::string & _type, const std::vector<size_t> & _atoms, const double & _min) : type_(_type), atoms_(_atoms), min_(_min) {};
        inline ~InvDisp() {}

        inline std::string type() const {return type_;}
        inline std::vector<size_t> atoms() const {return atoms_;}
        inline double min() const {return min_;}

        // Return the displacement given r
        at::Tensor operator()(const at::Tensor & r) const;
        // Return the displacement and its gradient over r given r
        std::tuple<at::Tensor, at::Tensor> compute_IC_J(const at::Tensor & r) const;
};

// An internal coordinate, i.e. a linear combination of several translationally and rotationally invariant displacements
class IntCoord {
    private:
        // linear combination coefficients
        std::vector<double> coeffs_;
        // constitutional translationally and rotationally invariant displacements
        std::vector<InvDisp> invdisps_;
    public:
        inline IntCoord() {}
        inline ~IntCoord() {}

        inline std::vector<double> coeffs() const {return coeffs_;}
        inline std::vector<InvDisp> invdisps() const {return invdisps_;}

        void append(const double & coeff, const InvDisp & invdisp);
        void normalize();

        // Return the internal coordinate given r
        at::Tensor operator()(const at::Tensor & r) const;
        // Return the internal coordinate and its gradient over r given r
        std::tuple<at::Tensor, at::Tensor> compute_IC_J(const at::Tensor & r) const;
};

// A set of internal coordinates
class IntCoordSet {
    private:
        // Internal coordinates constituting the set
        std::vector<IntCoord> intcoords_;
    public:
        inline IntCoordSet() {}
        // file format (Columbus7, default), internal coordinate definition file
        IntCoordSet(const std::string & format, const std::string & file);
        inline ~IntCoordSet() {}

        inline std::vector<IntCoord> intcoords() const {return intcoords_;}

        inline size_t size() const {return intcoords_.size();}

        // Return q given r
        at::Tensor operator()(const at::Tensor & r) const;
        // Return q and J given r
        std::tuple<at::Tensor, at::Tensor> compute_IC_J(const at::Tensor & r) const;
};

} // namespace IC
} // namespace tchem

#endif