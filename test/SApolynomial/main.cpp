#include <tchem/SApolynomial.hpp>

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'SApolynomial'\n"
              << "Correct routines should print close to 0\n";

    std::vector<at::Tensor> answer = {
        at::tensor({4.0, 6.0, 9.0,
                    25.0, 35.0, 49.0,
                    2.0, 3.0}),
        at::tensor({10.0, 14.0,
                    15.0, 21.0,
                    5.0, 7.0})
    };

    tchem::polynomial::SAPSet set("sapoly.in");
    std::vector<at::Tensor> xs(2);
    xs[0] = at::tensor({2.0, 3.0});
    xs[1] = at::tensor({5.0, 7.0});
    for (at::Tensor & el : xs) el.set_requires_grad(true);

    std::vector<at::Tensor> values = set(xs);
    std::cout << "\nValue of symmetry adapted polynomial set: "
              << (values[0] - answer[0]).norm().item<double>() << ' '
              << (values[1] - answer[1]).norm().item<double>() << '\n';

    std::vector<std::vector<at::Tensor>> Jacobians(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        Jacobians[i] = std::vector<at::Tensor>(xs.size());
        for (size_t j = 0; j < xs.size(); j++)
        Jacobians[i][j] = xs[j].new_zeros({values[i].size(0), xs[j].size(0)});
    }
    for (size_t i = 0; i < values.size(); i++)
    for (size_t row = 0; row < values[i].size(0); row++) {
        torch::autograd::variable_list g = torch::autograd::grad({values[i][row]}, {xs}, {}, true);
        for (size_t j = 0; j < xs.size(); j++)
        Jacobians[i][j][row] = g[j];
    }
    std::vector<std::vector<at::Tensor>> Jacobians_A(values.size());
    for (size_t i = 0; i < values.size(); i++) {
        Jacobians_A[i] = std::vector<at::Tensor>(xs.size());
        for (size_t j = 0; j < xs.size(); j++)
        Jacobians_A[i][j] = xs[j].new_zeros({values[i].size(0), xs[j].size(0)});
    }
    // irreducible 1 terms / irreducible 1 monomials
    Jacobians_A[0][0][0][0] = 2.0 * xs[0][0];
    Jacobians_A[0][0][1][0] = xs[0][1];
    Jacobians_A[0][0][1][1] = xs[0][0];
    Jacobians_A[0][0][2][1] = 2.0 * xs[0][1];
    Jacobians_A[0][0][6][0] = 1.0;
    Jacobians_A[0][0][7][1] = 1.0;
    // irreducible 1 terms / irreducible 2 monomials
    Jacobians_A[0][1][3][0] = 2.0 * xs[1][0];
    Jacobians_A[0][1][4][0] = xs[1][1];
    Jacobians_A[0][1][4][1] = xs[1][0];
    Jacobians_A[0][1][5][1] = 2.0 * xs[1][1];
    // irreducible 2 terms / irreducible 1 monomials
    Jacobians_A[1][0][0][0] = xs[1][0];
    Jacobians_A[1][0][1][0] = xs[1][1];
    Jacobians_A[1][0][2][1] = xs[1][0];
    Jacobians_A[1][0][3][1] = xs[1][1];
    // irreducible 2 terms / irreducible 2 monomials
    Jacobians_A[1][1][0][0] = xs[0][0];
    Jacobians_A[1][1][1][1] = xs[0][0];
    Jacobians_A[1][1][2][0] = xs[0][1];
    Jacobians_A[1][1][3][1] = xs[0][1];
    Jacobians_A[1][1][4][0] = 1.0;
    Jacobians_A[1][1][5][1] = 1.0;
    std::cout << "Backward propagation vs analytical Jacobian: "
              << (Jacobians[0][0] - Jacobians_A[0][0]).norm().item<double>() << ' '
              << (Jacobians[0][1] - Jacobians_A[0][1]).norm().item<double>() << ' '
              << (Jacobians[1][0] - Jacobians_A[1][0]).norm().item<double>() << ' '
              << (Jacobians[1][1] - Jacobians_A[1][1]).norm().item<double>() << '\n';
}