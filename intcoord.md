# Internal coordinate
Converting Cartesian coordinate to internal coordinate

An interal coordinate is a linear combination of several translationally and rotationally invariant displacements, but only displacements under same unit can be combined, i.e. you must treat lengthes and angles separately, unless appropriate metric tensor is applied

The Jacobian of internal coordinate over Cartesian coordinate can be calculated analytically instead of through backward propagation. In fact, analytical Jacobian is always recommended because:
* A analytical calculation is more efficient and stable than backward propagation
* In few cases (e.g. acos a number > 1, possible when an angle = 0 or pi and numerical fluctutation arises) the internal coordinate cannot be backward propagated

~~Although in principle the 2nd order Jacobian also has analytical form, since a backward propagation from the Jacobian is always feasible, I'm too lazy to implement it~~ :sleeping:

Available internal coordinates are:
* bond length
* bond angle (and its cos)
* dihedral angle (and its sin and cos)
* out of plane angle (and its sin)

## Why internal coordinate?
Cartesian coordinate is a convenient but inappropriate representation for molecular properties, who are invariant under translation and rotation. As a result, internal coordinate is adopted to remove the redundancy

## Usage
`IntCoordSet` is the engine class. An instance can be constructed by `IntCoordSet(format, file)`, where `file` is an input file defining the internal coordinates whose format is specified by `format`

The supported formats are:
* Columbus7
* default

An input file in default format obeys:
* First 6 spaces of a line are reserved to indicate the start of new internal coordinate
* Internal coordinate type and the involving atoms
* For a line defining torsion, an additional number at the end of the line defines the minimum of angle (default = -pi), so the angle ranges within [min, min + 2pi]
* At the end of each line, anything after # is considered as comment

An example of default format input file is available in `test/intcoord/IntCoordDef`