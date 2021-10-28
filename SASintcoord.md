# Symmetry adaptation and scale (SAS) for internal coordinate
An internal coordinate system has no norm since some are lengthes and others are angles. To define a norm we nondimensionalize lengthes by dividing reference lengthes

The dependency of molecular properties on some internal coordinates vanishes in the dissociation limit. E.g. C6H5NH2 -> C6H5NH + H, in the dissociation limit the relative position of the H to the C6H5NH no longer matters, so the corresponding internal coordinates should be scaled: N-H -> 1 - exp(-alpha * N-H), C-N-H -> exp(-alpha * N-H) * C-N-H, C-C-N-H -> exp(-alpha * N-H) * C-C-N-H, etc.

Molecular properties usually carry some symmetry arising from the identity of the nucleus, which is called the complete nuclear permutation inversion (CNPI) group. To describe the CNPI symmetry correctly, the internal coordinate system must be adapted, letting each coordinate carries a certain irreducible

It is more appropriate to consider SAS internal coordiante as "feature" rather than "coordinate", since the exponential decay makes Jacobian degenerate and global CNPI group can be locally redundant. In another word, only when the CNPI group is isomorphic to the highest possible point group and no decay, SAS internal coordinate is real "coordinate"

## Usage
`SASICSet` is the engine class. An instance can be constructed by `SASICSet(format, IC_file, SAS_file)`, where `format` and `IC_file` are meant to construct the parent class `IntCoordSet`, `SAS_file` is an input file defining the scale and symmetry adaptation

An example of `SAS_file` is available in `test/SAS/SAS.in`
