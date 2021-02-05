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
    at::Tensor r = at::tensor({2.0, 7.0});
    std::cout << "\nValue of polynomial set: "
              << at::norm(set(r) - answer).item<double>() << '\n';

    at::Tensor U = at::tensor({1.0,  1.0,
                               1.0, -1.0});
    U.resize_({2, 2});
    U /= sqrt(2.0);
    at::Tensor T = set.rotation(U);
    at::Tensor q = U.transpose(0, 1).mv(r);
    std::cout << "\nValue of polynomial set after rotation: "
              << at::norm(T.mv(set(q)) - answer).item<double>() << '\n';

    at::Tensor a = at::tensor({3.0, 5.0});
    at::Tensor S = set.translation(a);
    at::Tensor p = r - a;
    std::cout << "\nValue of polynomial set after translation: "
              << at::norm(S.mv(set(p)) - answer).item<double>() << '\n';

    r.set_requires_grad(true);
    at::Tensor values = set(r);
    at::Tensor Jacobian = values.new_empty({values.size(0), r.size(0)});
    for (size_t i = 0; i < Jacobian.size(0); i++) {
        torch::autograd::variable_list g = torch::autograd::grad({values[i]}, {r}, {}, true);
        Jacobian[i].copy_(g[0]);
    }
    at::Tensor Jacobian_A = values.new_zeros({values.size(0), r.size(0)});
    // 1st order
    Jacobian_A[1][0] = 1.0;
    Jacobian_A[2][1] = 1.0;
    // 2nd order
    Jacobian_A[3][0] = 2.0 * r[0];
    Jacobian_A[4][0] = r[1];
    Jacobian_A[4][1] = r[0];
    Jacobian_A[5][1] = 2.0 * r[1];
    // 3rd order
    Jacobian_A[6][0] = 3.0 * r[0] * r[0];
    Jacobian_A[7][0] = 2.0 * r[0] * r[1];
    Jacobian_A[7][1] = r[0] * r[0];
    Jacobian_A[8][0] = r[1] * r[1];
    Jacobian_A[8][1] = 2.0 * r[1] * r[0];
    Jacobian_A[9][1] = 3.0 * r[1] * r[1];
    // 4th order
    Jacobian_A[10][0] = 4.0 * r[0] * r[0] * r[0];
    Jacobian_A[11][0] = 3.0 * r[0] * r[0] * r[1];
    Jacobian_A[11][1] = r[0] * r[0] * r[0];
    Jacobian_A[12][0] = 2.0 * r[0] * r[1] * r[1];
    Jacobian_A[12][1] = 2.0 * r[1] * r[0] * r[0];
    Jacobian_A[13][0] = r[1] * r[1] * r[1];
    Jacobian_A[13][1] = 3.0 * r[1] * r[1] * r[0];
    Jacobian_A[14][1] = 4.0 * r[1] * r[1] * r[1];
    std::cout << "\nBackward propagation vs analytical Jacobian: "
              << (Jacobian - Jacobian_A).norm().item<double>() << '\n';
}