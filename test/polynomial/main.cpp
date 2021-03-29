#include <tchem/polynomial.hpp>

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'polynomial'\n"
              << "Correct routines should print close to 0\n";

    at::Tensor answer = at::tensor({1.0,
                                    2.0, 7.0,
                                    4.0, 14.0, 49.0,
                                    8.0, 28.0, 98.0, 343.0,
                                    16.0, 56.0, 196.0, 686.0, 2401.0});

    tchem::polynomial::PolynomialSet set(2, 4);
    at::Tensor x = at::tensor({2.0, 7.0});
    std::cout << "\nValue of polynomial set: "
              << at::norm(set(x) - answer).item<double>() << '\n';

    at::Tensor U, Uinv;
    while (true) {
        U = at::rand({2, 2}, x.options());
        Uinv = U.inverse();
        if ((U.mm(Uinv) - at::eye(2)).norm().item<double>() < 1e-12) break;
    }
    at::Tensor T = set.rotation(U);
    at::Tensor q = Uinv.mv(x);
    std::cout << "\nValue of polynomial set after rotation: "
              << at::norm(T.mv(set(q)) - answer).item<double>() << '\n';

    at::Tensor a = at::rand(2, x.options());
    at::Tensor S = set.translation(a);
    at::Tensor p = x - a;
    std::cout << "\nValue of polynomial set after translation: "
              << at::norm(S.mv(set(p)) - answer).item<double>() << '\n';

    x.set_requires_grad(true);
    at::Tensor values = set(x);
    at::Tensor Jacobian = values.new_empty({values.size(0), x.size(0)});
    for (size_t i = 0; i < Jacobian.size(0); i++) {
        torch::autograd::variable_list g = torch::autograd::grad({values[i]}, {x}, {}, true);
        Jacobian[i].copy_(g[0]);
    }
    at::Tensor Jacobian_A = set.Jacobian(x);
    std::cout << "\nBackward propagation vs analytical Jacobian: "
              << (Jacobian - Jacobian_A).norm().item<double>() << '\n';
}