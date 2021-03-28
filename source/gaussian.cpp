#include <torch/torch.h>

#include <CppLibrary/utility.hpp>
#include <CppLibrary/math.hpp>

#include <tchem/polynomial.hpp>

#include <tchem/gaussian.hpp>

namespace tchem {

Gaussian::Gaussian() {}
// miu_ & var_ are deep copies of _miu & _var
Gaussian::Gaussian(const at::Tensor & _miu, const at::Tensor & _var) {
    if (_miu.sizes().size() != 1) throw std::invalid_argument(
    "tchem::Gaussian::Gaussian: miu must be a vector");
    if (_var.sizes().size() != 2) throw std::invalid_argument(
    "tchem::Gaussian::Gaussian: var must be a matrix");
    if (_var.size(0) != _var.size(1)) throw std::invalid_argument(
    "tchem::Gaussian::Gaussian: var must be a square matrix");
    if (_miu.size(0) != _var.size(0)) throw std::invalid_argument(
    "tchem::Gaussian::Gaussian: miu & var must share a same dimension");
    miu_ = _miu.clone();
    var_ = _var.clone();
}
Gaussian::~Gaussian() {}

at::Tensor Gaussian::miu() const {return miu_;}
at::Tensor Gaussian::var() const {return var_;}
bool Gaussian::random_ready() const {return random_ready_;}

// g(r; miu, var) = (2pi)^(-dim/2) |var|^(-1/2) exp[-1/2 (r-miu)^T.var^-1.(r-miu)]
at::Tensor Gaussian::operator()(const at::Tensor & r) const {
    if (r.sizes().size() != 1) throw std::invalid_argument(
    "tchem::Gaussian::operator(): r must be a vector");
    if (r.size(0) != miu_.size(0)) throw std::invalid_argument(
    "tchem::Gaussian::operator(): r & miu must share a same dimension");
    at::Tensor cholesky_var = var_.cholesky();
    at::Tensor inv_var = at::cholesky_inverse(cholesky_var);
    at::Tensor sqrtdet_var = cholesky_var.diag().prod();
    at::Tensor r_disp = r - miu_;
    at::Tensor value = pow(6.283185307179586, -r.size(0) / 2.0) / sqrtdet_var
                     * at::exp(-0.5 * r_disp.dot(inv_var.mv(r_disp)));
    return value;
}
// g1(r; miu1, var1) * g2(r; miu2, var2) = c * g3(r; miu3, var3)
// Return c and g3
std::tuple<at::Tensor, Gaussian> Gaussian::operator*(const Gaussian & g2) const {
    // Prepare
    at::Tensor miu1 =    miu(), var1 =    var(),
               miu2 = g2.miu(), var2 = g2.var();
    at::Tensor cholesky_var1 = var1.cholesky(),
               cholesky_var2 = var2.cholesky();
    at::Tensor sqrtdet_var1 = cholesky_var1.diag().prod(),
               sqrtdet_var2 = cholesky_var2.diag().prod(),
               inv_var1 = at::cholesky_inverse(cholesky_var1),
               inv_var2 = at::cholesky_inverse(cholesky_var2);
    // var3
    at::Tensor inv_var3 = inv_var1 + inv_var2;
    at::Tensor cholesky_inv_var3 = inv_var3.cholesky();
    at::Tensor sqrtdet_inv_var3 = cholesky_inv_var3.diag().prod(),
               var3 = at::cholesky_inverse(cholesky_inv_var3);
    // miu3
    at::Tensor temp = inv_var1.mv(miu1) + inv_var2.mv(miu2);
    at::Tensor miu3 = var3.mv(temp);
    // c
    at::Tensor c = pow(6.283185307179586, -miu1.size(0)/2) / sqrtdet_var1 / sqrtdet_var2 / sqrtdet_inv_var3
                 * at::exp(-0.5 * (miu1.dot(inv_var1.mv(miu1)) + miu2.dot(inv_var2.mv(miu2)) - temp.dot(miu3)));
    Gaussian g3(miu3, var3);
    return std::make_tuple(c, g3);
}

Gaussian Gaussian::clone() const {return Gaussian(miu_, var_);}

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
    at::Tensor eigvals, UT;
    std::tie(eigvals, UT) = var_.symeig(true);
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
                                             * at::pow(eigvals[uniques[i]], (double)repeats[i]/2);
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

// Initialize gaussian random tensor generation
void Gaussian::rand_init() {
    std::tie(eigvals_, eigvecs_) = var_.symeig(true);
    independent_1Dgaussians_.resize(miu_.size(0));
    for (size_t i = 0; i < miu_.size(0); i++)
    independent_1Dgaussians_[i] = std::normal_distribution<double>(0.0, sqrt(eigvals_[i].item<double>()));
    random_ready_ = true;
}
// Return a gaussian random tensor
at::Tensor Gaussian::rand(std::default_random_engine & generator) {
    if (! random_ready_) throw CL::utility::not_ready("tchem::Gaussian::rand");
    at::Tensor rand = miu_.new_empty(miu_.sizes());
    // Generate a random vector in the miu-centred && `var`-diagonalized (normal) coordinate
    for (size_t i = 0; i < rand.size(0); i++)
    rand[i] = independent_1Dgaussians_[i](generator);
    // Transform back to original coordinate
    rand = miu_ + eigvecs_.transpose(0, 1).mv(rand);
    return rand;
}

} // namespace tchem