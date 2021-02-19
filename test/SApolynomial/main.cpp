#include <tchem/SApolynomial.hpp>

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'SApolynomial'\n"
              << "Correct routines should print close to 0\n";

    at::Tensor answer1 = at::tensor({2.0, 3.0,
                                     4.0, 6.0, 9.0,
                                     25.0, 35.0, 49.0
                                    }),
               answer2 = at::tensor({5.0, 7.0,
                                     10.0, 14.0,
                                     15.0, 21.0
                                    });

    std::vector<size_t> dimensions = {2, 2};
    tchem::polynomial::SAPSet set1("1.in", dimensions), set2("2.in", dimensions);
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
    std::vector<at::Tensor> Jacobian1_A = set1.Jacobian(xs),
                            Jacobian2_A = set2.Jacobian(xs);
    std::cout << "Backward propagation vs analytical Jacobian: "
              << (Jacobian1[0] - Jacobian1_A[0]).norm().item<double>() << ' '
              << (Jacobian1[1] - Jacobian1_A[1]).norm().item<double>() << ' '
              << (Jacobian2[0] - Jacobian2_A[0]).norm().item<double>() << ' '
              << (Jacobian2[1] - Jacobian2_A[1]).norm().item<double>() << '\n';
}