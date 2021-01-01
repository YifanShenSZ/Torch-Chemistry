/*
This library adopts gaussian functions in form of multivariate normal distribution
g(r; miu, var) = (2pi)^(-dim/2) |var|^(-1/2) exp[-1/2 (r-miu)^T.var^-1.(r-miu)]
where `r` is the variable, `dim` is the dimension of `r`
`miu` is the mean, `var` is the variance
*/

#include <torch/torch.h>

#include <CppLibrary/math.hpp>

#include <tchem/gaussian.hpp>

namespace tchem { namespace gaussian {

Gaussian::Gaussian() {}
Gaussian::Gaussian(const at::Tensor & _miu, const at::Tensor & _var) : miu_(_miu), var_(_var) {
    assert(("miu must be a vector", _miu.sizes().size() == 1));
    assert(("var must be a matrix", _var.sizes().size() == 2));
    assert(("var must be a square matrix", _var.size(0) == _var.size(1)));
    assert(("miu and var must have same dimension", _miu.size(0) == _var.size(0)));
}
Gaussian::~Gaussian() {}

// g(r; miu, var) = (2pi)^(-dim/2) |var|^(-1/2) exp[-1/2 (r-miu)^T.var^-1.(r-miu)]
at::Tensor Gaussian::operator()(const at::Tensor & r) const {
    assert(("r and miu must have same dimension", r.size(0) == miu_.size(0)));
    at::Tensor var_inv = at::inverse(var_);
    at::Tensor r_disp = r - miu_;
    at::Tensor exponent = r_disp.dot(var_inv.mv(r_disp));
    at::Tensor value = pow(6.283185307179586, -r.size(0) / 2.0)
                     / at::sqrt(at::det(var_))
                     * at::exp(-0.5 * exponent);
    return value;
}
// g1(r; miu1, var1) * g2(r; miu2, var2) = c * g3(r; miu3, var3)
// Return c and g3
std::tuple<at::Tensor, Gaussian> Gaussian::operator*(const Gaussian & g2) const {
    // Prepare
    at::Tensor miu1 =    miu(), var1 =    var(),
               miu2 = g2.miu(), var2 = g2.var();
    at::Tensor det_var1 = at::det    (var1), det_var2 = at::det    (var2),
               inv_var1 = at::inverse(var1), inv_var2 = at::inverse(var2);
    // var3
    at::Tensor inv_var3 = inv_var1 + inv_var2;
    at::Tensor var3 = at::inverse(inv_var3);
    // miu3
    at::Tensor temp = inv_var1.mv(miu1) + inv_var2.mv(miu2);
    at::Tensor miu3 = var3.mv(temp);
    // c
    at::Tensor det_var3 = at::det(var3);
    at::Tensor c = pow(6.283185307179586, -miu1.size(0)/2) * at::sqrt(det_var3 / det_var1 / det_var2)
                 * at::exp(-0.5 * (miu1.dot(inv_var1.mv(miu1)) + miu2.dot(inv_var2.mv(miu2)) - temp.dot(miu3)));
    Gaussian g3(miu3, var3);
    return std::make_tuple(c, g3);
}

// Intgerate[g(r; miu, var) * {P(r)}, {r, -Infinity, Infinity}]
// {P(r)} is specified by `set`
// The evaluation procedure for integrals is:
// 1. diagonalize `var`
// 2. transform `r` to miu-centred && `var`-diagonalized (normal) coordinate
// 3. evaluate integrals
// 4. transform integrals back to original coordinate
// The necessary integrals in normal coordinate are specified in `normal_set`
at::Tensor Gaussian::integral(const polynomial::PolynomialSet & set, const polynomial::PolynomialSet & normal_set) const {
    // Diagonalize `var`
    at::Tensor eigval, UT;
    std::tie(eigval, UT) = var_.symeig(true);
    // Evaluate the integrals in the normal coordinate
    at::Tensor normal_integrals = miu_.new_empty(normal_set.polynomials().size());
    size_t start = 0;
    auto orders = normal_set.orders();
    for (size_t order = 0; order < orders.size(); order++) {
        if (order % 2 == 1) {
            size_t stop = start + orders[order].size();
            normal_integrals.slice(0, start, stop) = 0.0;
            start = stop;
        }
        else {
            for (auto & term : orders[order]) {
                std::vector<size_t> uniques, repeats;
                std::tie(uniques, repeats) = term->uniques_orders();
                bool odd = false;
                for (size_t & repeat : repeats) if (repeat % 2 == 1) {
                    odd = true;
                    break;
                }
                if (odd) normal_integrals[start] = 0.0;
                else {
                    normal_integrals[start] = 1.0;
                    for (size_t i = 0; i < uniques.size(); i++)
                    normal_integrals[start] *= CL::math::dFactorial2(repeats[i] - 1)
                                             * at::pow(eigval[uniques[i]], (double)repeats[i]/2);
                }
                start++;
            }
        }
    }
    // Transform the integrals back to the original coordinate
    at::Tensor rotation = set.rotation(UT.transpose(0, 1));
    at::Tensor translation = set.translation(miu_);
    at::Tensor integrals = translation.mv(rotation.mv(normal_integrals));
    // Q: Why don't we multiply |U| (i.e. |dr / dq|)?
    // A: If |U| == 1, then we don't need it
    //    else, the reflected coordinates have integral range (Inifnity, -Infinity),
    //          we flip them back to (-Inifnity, Infinity), digesting the -1
    return integrals;
}
// Assuming terms are the same under transformation
at::Tensor Gaussian::integral(const polynomial::PolynomialSet & set) const {return integral(set, set);}

} // namespace gaussian
} // namespace tchem
