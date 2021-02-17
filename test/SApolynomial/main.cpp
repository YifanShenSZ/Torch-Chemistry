#include <tchem/SApolynomial.hpp>

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'SApolynomial'\n"
              << "Correct routines should print close to 0\n";

    at::Tensor answer1 = at::tensor({4.0, 6.0, 9.0,
                                     25.0, 35.0, 49.0,
                                     2.0, 3.0}),
               answer2 = at::tensor({10.0, 14.0,
                                     15.0, 21.0,
                                     5.0, 7.0});

    tchem::polynomial::SAPSet set1("1.in"), set2("2.in");
    std::cout << "\nIrreducible 1 terms:\n";
    set1.pretty_print(std::cout);
    std::cout << "Irreducible 2 terms:\n";
    set2.pretty_print(std::cout);

    std::vector<at::Tensor> xs(2);
    xs[0] = at::tensor({2.0, 3.0});
    xs[1] = at::tensor({5.0, 7.0});
    for (at::Tensor & el : xs) el.set_requires_grad(true);

    at::Tensor value1 = set1(xs), value2 = set2(xs);
    std::cout << "\nValue of symmetry adapted polynomial set: "
              << (value1 - answer1).norm().item<double>() << ' '
              << (value2 - answer2).norm().item<double>() << '\n';

    std::vector<at::Tensor> Jacobian1(xs.size()), Jacobian2(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        Jacobian1[i] = xs[i].new_zeros({value1.size(0), xs[i].size(0)});
        Jacobian2[i] = xs[i].new_zeros({value2.size(0), xs[i].size(0)});
    }
    for (size_t row = 0; row < value1.size(0); row++) {
        torch::autograd::variable_list g = torch::autograd::grad({value1[row]}, {xs}, {}, true);
        for (size_t i = 0; i < xs.size(); i++) Jacobian1[i][row] = g[i];
    }
    for (size_t row = 0; row < value2.size(0); row++) {
        torch::autograd::variable_list g = torch::autograd::grad({value2[row]}, {xs}, {}, true);
        for (size_t i = 0; i < xs.size(); i++) Jacobian2[i][row] = g[i];
    }
    std::vector<at::Tensor> Jacobian1_A(xs.size()), Jacobian2_A(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        Jacobian1_A[i] = xs[i].new_zeros({value1.size(0), xs[i].size(0)});
        Jacobian2_A[i] = xs[i].new_zeros({value2.size(0), xs[i].size(0)});
    }
    // irreducible 1 terms / irreducible 1 monomials
    Jacobian1_A[0][0][0] = 2.0 * xs[0][0];
    Jacobian1_A[0][1][0] = xs[0][1];
    Jacobian1_A[0][1][1] = xs[0][0];
    Jacobian1_A[0][2][1] = 2.0 * xs[0][1];
    Jacobian1_A[0][6][0] = 1.0;
    Jacobian1_A[0][7][1] = 1.0;
    // irreducible 1 terms / irreducible 2 monomials
    Jacobian1_A[1][3][0] = 2.0 * xs[1][0];
    Jacobian1_A[1][4][0] = xs[1][1];
    Jacobian1_A[1][4][1] = xs[1][0];
    Jacobian1_A[1][5][1] = 2.0 * xs[1][1];
    // irreducible 2 terms / irreducible 1 monomials
    Jacobian2_A[0][0][0] = xs[1][0];
    Jacobian2_A[0][1][0] = xs[1][1];
    Jacobian2_A[0][2][1] = xs[1][0];
    Jacobian2_A[0][3][1] = xs[1][1];
    // irreducible 2 terms / irreducible 2 monomials
    Jacobian2_A[1][0][0] = xs[0][0];
    Jacobian2_A[1][1][1] = xs[0][0];
    Jacobian2_A[1][2][0] = xs[0][1];
    Jacobian2_A[1][3][1] = xs[0][1];
    Jacobian2_A[1][4][0] = 1.0;
    Jacobian2_A[1][5][1] = 1.0;
    std::cout << "Backward propagation vs analytical Jacobian: "
              << (Jacobian1[0] - Jacobian1_A[0]).norm().item<double>() << ' '
              << (Jacobian1[1] - Jacobian1_A[1]).norm().item<double>() << ' '
              << (Jacobian2[0] - Jacobian2_A[0]).norm().item<double>() << ' '
              << (Jacobian2[1] - Jacobian2_A[1]).norm().item<double>() << '\n';
}