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
      so please use cosbending for large amplitude
    * J of out of plane is singular at +-pi/2,
      so please use sinoop for large amplitude
*/
#include <tchem/IC/InvDisp.hpp>
#include <tchem/IC/IntCoord.hpp>
#include <tchem/IC/IntCoordSet.hpp>

#endif