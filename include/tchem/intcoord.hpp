#ifndef tchem_intcoord_hpp
#define tchem_intcoord_hpp

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
    * q of torsion is double-valued at min
      so please use sintors or costors for large amplitude
    * Backward propagation through q of torsion may be problematic when q = 0 or +-pi,
      so please use J explicitly in those cases
    * J of bending is singular at 0 and pi,
      so please avoid using bending in those cases
    * J of out of plane is singular at +-pi/2,
      so please avoid using out of plane in those cases
*/
#include <tchem/intcoord/InvDisp.hpp>
#include <tchem/intcoord/IntCoord.hpp>
#include <tchem/intcoord/IntCoordSet.hpp>

/*
The procedure to get symmetry adapted and scaled internal coordinate (SASIC) is:
    1. Get internal coordinate (IC), which is taken care of by module `intcoord`
    2. Nondimensionalize the IC to get dimensionless internal coordinate (DIC):
       for length, DIC = (IC - origin) / origin
       for angle , DIC =  IC - origin
    3. Scale the DIC to get scaled dimensionless internal coordinate (SDIC):
       if no scaler      : SDIC = DIC
       elif scaler = self: SDIC = pi * (1 - exp(-alpha * DIC))
       else              : SDIC = DIC * exp(-alpha * scaler DIC)
    4. Symmetry adapted linear combinate the SDIC to get SASIC
*/
#include <tchem/intcoord/SASIC.hpp>
#include <tchem/intcoord/SASICSet.hpp>

#endif