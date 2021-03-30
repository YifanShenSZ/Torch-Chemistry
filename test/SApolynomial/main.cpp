#include <CppLibrary/linalg.hpp>

#include <tchem/polynomial.hpp>

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'SApolynomial'\n"
              << "Correct routines should print close to 0\n";

    at::Tensor answer1 = at::tensor({1.0,
                                     2.0, 3.0,
                                     4.0, 6.0, 9.0,
                                     25.0, 35.0, 49.0
                                    }),
               answer2 = at::tensor({5.0, 7.0,
                                     10.0, 14.0,
                                     15.0, 21.0
                                    });

    tchem::polynomial::SAPSet set1("1.in", 0, {2, 2}), set2("2.in", 1, {2, 2});
    std::vector<at::Tensor> xs(2);
    xs[0] = at::tensor({2.0, 3.0});
    xs[1] = at::tensor({5.0, 7.0});    
    std::cout << "\nValue of symmetry adapted polynomial set: "
              << (set1(xs) - answer1).norm().item<double>() << ' '
              << (set2(xs) - answer2).norm().item<double>() << '\n';

    std::vector<at::Tensor> Us(2);
    at::Tensor U1, U2, U1inv, U2inv;
    while (true) {
        U1 = at::rand({2, 2}, xs[0].options());
        U1inv = U1.inverse();
        if ((U1.mm(U1inv) - at::eye(2)).norm().item<double>() < 1e-12) break;
    }
    while (true) {
        U2 = at::rand({2, 2}, xs[0].options());
        U2inv = U2.inverse();
        if ((U2.mm(U2inv) - at::eye(2)).norm().item<double>() < 1e-12) break;
    }
    Us[0] = U1.clone();
    Us[1] = U2.clone();
    at::Tensor T1 = set1.rotation(Us), T2 = set2.rotation(Us);
    std::vector<at::Tensor> qs = {U1inv.mv(xs[0]), U2inv.mv(xs[1])};
    std::cout << "\nValue of symmetry adapted polynomial set after rotation: "
              << at::norm(T1.mv(set1(qs)) - answer1).item<double>() << ' '
              << at::norm(T2.mv(set2(qs)) - answer2).item<double>() << '\n';

    std::vector<at::Tensor> as = {at::rand(2, xs[0].options()), at::rand(2, xs[0].options())};
    at::Tensor S1 = set1.translation(as), S2 = set2.translation(as);
    std::vector<at::Tensor> ps = xs - as;
    std::cout << "\nValue of symmetry adapted polynomial set after translation: "
              << at::norm(S1.mv(set1(ps)) - answer1).item<double>() << ' '
              << at::norm(S2.mv(set2(ps)) - answer2).item<double>() << '\n';
    std::cerr << "See issue #3\n" << S1.mv(set1(ps)) << '\n';

    for (at::Tensor & el : xs) el.set_requires_grad(true);
    at::Tensor value1 = set1(xs), value2 = set2(xs);
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
    std::cout << "\nBackward propagation vs analytical Jacobian: "
              << (Jacobian1[0] - Jacobian1_A[0]).norm().item<double>() << ' '
              << (Jacobian1[1] - Jacobian1_A[1]).norm().item<double>() << ' '
              << (Jacobian2[0] - Jacobian2_A[0]).norm().item<double>() << ' '
              << (Jacobian2[1] - Jacobian2_A[1]).norm().item<double>() << '\n';
}