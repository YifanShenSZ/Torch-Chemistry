#include <tchem/tchem.hpp>

void test_polynomial() {
    at::Tensor answer = at::tensor({1.0, 2.0, 7.0, 4.0, 14.0, 49.0, 8.0, 28.0, 98.0, 343.0});

    tchem::polynomial::PolynomialSet set(2, 3);
    at::Tensor r = at::tensor({2.0, 7.0});
    std::cerr << "\nValue of polynomial set: "
              << at::norm(set(r) - answer).item<double>() << '\n';

    at::Tensor U = at::tensor({1.0,  1.0,
                               1.0, -1.0});
    U.resize_({2, 2});
    U /= sqrt(2.0);
    at::Tensor T = set.rotation(U);
    at::Tensor q = U.transpose(0, 1).mv(r);
    std::cerr << "\nValue of polynomial set after rotation: "
              << at::norm(T.mv(set(q)) - answer).item<double>() << '\n';

    at::Tensor a = at::tensor({3.0, 5.0});
    at::Tensor S = set.translation(a);
    at::Tensor p = r - a;
    std::cerr << "\nValue of polynomial set after translation: "
              << at::norm(S.mv(set(p)) - answer).item<double>() << '\n';
}

void test_gaussian() {
    at::Tensor miu1 = at::tensor({1.0, 2.0}),
               miu2 = at::tensor({3.0, 5.0});
    at::Tensor var1 = at::tensor({0.25, 0.08,
                                  0.08, 4.00}),
               var2 = at::tensor({16.00, 0.12,
                                  0.12, 0.36});
    var1.resize_({2, 2});
    var2.resize_({2, 2});
    tchem::gaussian::Gaussian g1(miu1, var1), g2(miu2, var2);
    at::Tensor c;
    tchem::gaussian::Gaussian g;
    std::tie(c, g) = g1 * g2;
    at::Tensor r = at::tensor({1.4, 1.7});
    std::cerr << "\nValue of gaussian after multiplication: "
              << (c * g(r)).item<double>() - 2.015016218884655e-9 << '\n';

    tchem::polynomial::PolynomialSet set(2, 2);
    at::Tensor integrals = g.integral(set);
    at::Tensor miu = g.miu(), var = g.var();
    integrals[0] -= 1.0;
    integrals.slice(0, 1, 3) -= miu;
    integrals[3] -= var[0][0] + miu[0] * miu[0];
    integrals[4] -= var[1][0] + miu[0] * miu[1];
    integrals[5] -= var[1][1] + miu[1] * miu[1];
    std::cerr << "\nValue of gaussian integrals: "
              << integrals.norm().item<double>() << '\n';
}

int main() {
    std::cerr << "This is a test program on Torch-Chemistry\n"
              << "Correct routines should print close to 0\n";

    std::cerr << "\n---------- Testing module polynomial... ----------\n";
    test_polynomial();
    std::cerr << "\n---------- Polynomial test passed ----------\n";

    std::cerr << "\n---------- Testing module gaussian... ----------\n";
    test_gaussian();
    std::cerr << "\n---------- Gaussian test passed ----------\n";

    return 0;
}