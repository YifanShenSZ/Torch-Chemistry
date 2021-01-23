# Internal coordinate
Converting Cartesian coordinate to internal coordinate

An interal coordinate is a linear combination of several translationally and rotationally invariant displacements, but only displacements under same unit can be combined, i.e. you must treat lengthes and angles separately, unless appropriate metric tensor is applied

The Jacobian of internal coordinate over Cartesian coordinate can be calculated directly instead of through backward propagation. In fact, direct Jacobian is always recommended because:
* A direct calculation is more efficient and stable than backward propagation
* In few cases (e.g. acos a number > 1, possible when an angle = 0 or pi and numerical fluctutation arises) the internal coordinate cannot be backward propagated

The supported internal coordinates are:
* bond stretching
* bond angle
* dihedral angle
* out of plane angle

## Why internal coordinate
Cartesian coordinate is a convenient but inappropriate representation for molecular properties, since they are invariant under translation and rotation. As a result, internal coordinate is adopted to remove the redundancy

## Usage
`IntCoordSet` is the engine class. An instance can be constructed by `IntCoordSet(format, file)`, where `file` is an input file defining the internal coordinates whose format is specified by `format`

The supported formats are:
* Columbus7
* default

An input file in default format obeys:
* First 6 spaces of a line are reserved to indicate the start of new internal coordinate
* For a line defining torsion, an additional number at the end of the line defines the minimum of angle (default = -pi), so the angle ranges within (min, min + 2pi). The dihedral angle is discontinuous at min and min + 2pi
* At the end of each line, anything after # is considered as comment

An example of default format input file is available in `test/intcoord/IntCoordDef`