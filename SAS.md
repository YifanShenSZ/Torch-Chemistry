# Symmetry adaptation and scale (SAS) for internal coordinate
The dependency of molecular properties on some internal coordinates vanishes in the dissociation limit. E.g. C6H5NH2 -> C6H5NH + H, in the dissociation limit the relative position of the H to the C6H5NH no longer matters, so the corresponding internal coordinates should be scaled: N-H -> pi * erf(N-H), C-N-H -> exp(- N-H) * C-N-H, C-C-N-H -> exp(- N-H) * C-C-N-H, etc.

Molecular properties usually carry some symmetry arising from the identity of the nucleus, which is called the complete nuclear permutation inversion (CNPI) group. To describe the CNPI symmetry correctly, the internal coordinate system must be adapted, letting each coordinate carries a certain irreducible

## Usage
`SASICSet` is the engine class. An instance can be constructed by `SASICSet(format, IC_file, SAS_file)`, where `format` and `IC_file` are meant to construct the parent class `IntCoordSet`, `SAS_file` is an input file defining the scale and symmetry adaptation

An example of `SAS_file` is available in `test/SAS/SAS.in`

## Issue
`SASICSet::operator()` works but if you extract part of its code and make a test, the test fails to run backward propagation. The way to fix the test is to creat more intermediate tensors

The failed test is:

    #include <torch/torch.h>
    
    int main() {
        at::Tensor q = at::tensor({1.0, 2.0, 3.0, 4.0});
        q.set_requires_grad(true);
    
        std::vector<size_t> indices = {0, 2};
    
        at::Tensor a = q.clone();
        for (size_t & index : indices) a[index] = at::erf(a[index]);
        a[0].backward();
        std::cerr << q.grad() << '\n';
    }

The fix is:

    #include <torch/torch.h>
    
    int main() {
        at::Tensor q = at::tensor({1.0, 2.0, 3.0, 4.0});
        q.set_requires_grad(true);
    
        std::vector<size_t> indices = {0, 2};
    
        at::Tensor a = q.clone(), b = q.new_empty(q.sizes());
        for (size_t & index : indices) b[index] = at::erf(a[index]);
        b[0].backward();
        std::cerr << q.grad() << '\n';
    }
