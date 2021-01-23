#include <tchem/polynomial.hpp>

int main() {
    std::cerr << "This is a test program on Torch-Chemistry module 'polynomial'\n"
              << "Correct routines should print close to 0\n";
    
    at::Tensor answer = at::tensor({1.0,
                                    2.0, 7.0,
                                    4.0, 14.0, 49.0,
                                    8.0, 28.0, 98.0, 343.0,
                                    16.0, 56.0, 196.0, 686.0, 2401.0});

    tchem::polynomial::PolynomialSet set(2, 4);
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