/*
An interal coordinate is a linear combination of several translationally and rotationally invariant displacements
but only displacements under same unit can be combined, i.e. you must treat lengthes and angles separately
unless appropriate metric tensor is applied

Nomenclature:
    cartdim & intdim: Cartesian & internal space dimensionality
    r: Cartesian coordinate vector
    q: internal coordinate vector
    J: the Jacobian matrix of q over r
    K: the gradient of J over r

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
        // The dihedral angle is discontinuous at min and min + 2pi
        double min_ = -M_PI;
    public:
        InvDisp();
        InvDisp(const std::string & _type, const std::vector<size_t> & _atoms);
        InvDisp(const std::string & _type, const std::vector<size_t> & _atoms, const double & _min);
        ~InvDisp();

        std::string type() const;
        std::vector<size_t> atoms() const;
        double min() const;

        // Return the displacement given r
        at::Tensor operator()(const at::Tensor & r) const;
        // Return the displacement and its gradient over r given r
        std::tuple<at::Tensor, at::Tensor> compute_IC_J(const at::Tensor & r) const;
        // Return the displacement and its 1st and 2nd order gradient over r given r
        std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_IC_J_K(const at::Tensor & r) const;
};

// An internal coordinate, i.e. a linear combination of several translationally and rotationally invariant displacements
class IntCoord {
    private:
        // linear combination coefficients
        std::vector<double> coeffs_;
        // constitutional translationally and rotationally invariant displacements
        std::vector<InvDisp> invdisps_;
    public:
        IntCoord();
        ~IntCoord();

        std::vector<double> coeffs() const;
        std::vector<InvDisp> invdisps() const;

        // Append a linear combination coefficient - invariant displacement pair
        void append(const double & coeff, const InvDisp & invdisp);
        // Normalize linear combination coefficients
        void normalize();

        // Return the internal coordinate given r
        at::Tensor operator()(const at::Tensor & r) const;
        // Return the internal coordinate and its gradient over r given r
        std::tuple<at::Tensor, at::Tensor> compute_IC_J(const at::Tensor & r) const;
        // Return the internal coordinate and its 1st and 2nd order gradient over r given r
        std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_IC_J_K(const at::Tensor & r) const;
};

// A set of internal coordinates
class IntCoordSet {
    private:
        // Internal coordinates constituting the set
        std::vector<IntCoord> intcoords_;
    public:
        IntCoordSet();
        // file format (Columbus7, default), internal coordinate definition file
        IntCoordSet(const std::string & format, const std::string & file);
        ~IntCoordSet();

        std::vector<IntCoord> intcoords() const;

        size_t size() const;

        // Return q given r
        at::Tensor operator()(const at::Tensor & r) const;
        // Return q and J given r
        std::tuple<at::Tensor, at::Tensor> compute_IC_J(const at::Tensor & r) const;
        // Return q and J and K given r
        std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_IC_J_K(const at::Tensor & r) const;

        // Return internal coordinate gradient given r and Cartesian coordinate gradient
        at::Tensor gradient_cart2int(const at::Tensor & r, const at::Tensor & cartgrad) const;
        // Return Cartesian coordinate gradient given r and internal coordinate gradient
        at::Tensor gradient_int2cart(const at::Tensor & r, const at::Tensor & intgrad) const;

        // Return internal coordinate Hessian given r and Cartesian coordinate Hessian
        at::Tensor Hessian_cart2int(const at::Tensor & r, const at::Tensor & cartgrad, const at::Tensor & cartHess) const;
        // Return Cartesian coordinate Hessian given r and internal coordinate gradient and Hessian
        at::Tensor Hessian_int2cart(const at::Tensor & r, const at::Tensor & intgrad, const at::Tensor & intHess) const;
};

} // namespace IC
} // namespace tchem

#endif