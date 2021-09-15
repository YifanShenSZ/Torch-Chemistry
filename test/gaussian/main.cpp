#include <tchem/gaussian.hpp>

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'gaussian'\n"
              << "Correct routines should print close to 0\n";

    at::Tensor miu1 = at::tensor({1.0, 2.0}),
               miu2 = at::tensor({3.0, 5.0});
    at::Tensor var1 = at::tensor({0.25, 0.08,
                                  0.08, 4.00}),
               var2 = at::tensor({16.00, 0.12,
                                  0.12, 0.36});
    var1.resize_({2, 2});
    var2.resize_({2, 2});
    tchem::Gaussian g1(miu1, var1), g2(miu2, var2);
    at::Tensor c;
    tchem::Gaussian g;
    std::tie(c, g) = g1 * g2;
    at::Tensor r = at::tensor({1.4, 1.7});
    std::cout << "\nValue of gaussian after multiplication: "
              << (c * g(r)).item<double>() - 2.015016218884655e-9 << '\n';

    tchem::polynomial::PolynomialSet set(2, 4);
    at::Tensor integrals = g.integral(set);
    at::Tensor miu = g.miu(), var = g.var();
    // 4th order terms
    integrals[10] -= 3.0 * integrals[3] * integrals[3] - 2.0 * miu[0] * miu[0] * miu[0] * miu[0];
    integrals[11] -= 3.0 * integrals[3] * integrals[4] - 2.0 * miu[0] * miu[0] * miu[0] * miu[1];
    integrals[12] -= var[0][0] * var[1][1] + 2.0 * var[0][1] * var[0][1]
                   + 2.0 * integrals[7] * miu[1] + 2.0 * integrals[8] * miu[0]
                   - integrals[3] * miu[1] * miu[1] - 4.0 * integrals[4] * miu[0] * miu[1] - integrals[5] * miu[0] * miu[0]
                   + 3.0 * miu[0] * miu[0] * miu[1] * miu[1];
    integrals[13] -= 3.0 * integrals[5] * integrals[4] - 2.0 * miu[0] * miu[1] * miu[1] * miu[1];
    integrals[14] -= 3.0 * integrals[5] * integrals[5] - 2.0 * miu[1] * miu[1] * miu[1] * miu[1];
    // 3rd order terms
    integrals[6] -= 3.0 * integrals[3] * miu[0] - 2.0 * miu[0] * miu[0] * miu[0];
    integrals[7] -= 2.0 * miu[0] * integrals[4] + miu[1] * integrals[3] - 2.0 * miu[1] * miu[0] * miu[0];
    integrals[8] -= 2.0 * miu[1] * integrals[4] + miu[0] * integrals[5] - 2.0 * miu[0] * miu[1] * miu[1];
    integrals[9] -= 3.0 * integrals[5] * miu[1] - 2.0 * miu[1] * miu[1] * miu[1];
    // 0th to 2nd order terms
    integrals[0] -= 1.0;
    integrals.slice(0, 1, 3) -= miu;
    integrals[3] -= var[0][0] + miu[0] * miu[0];
    integrals[4] -= var[1][0] + miu[0] * miu[1];
    integrals[5] -= var[1][1] + miu[1] * miu[1];
    std::cout << "\nValue of gaussian integrals: "
              << integrals.norm().item<double>() << '\n';

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator(seed);
    g.rand_init();
    at::Tensor average_r  = miu.new_zeros(miu.sizes());
    at::Tensor average_rr = var.new_zeros(var.sizes());
    const size_t NSamples = 100000;
    for (size_t i = 0; i < NSamples; i++) {
        at::Tensor rand_vec = g.rand(generator);
        average_r  += rand_vec;
        average_rr += rand_vec.outer(rand_vec);
    }
    average_r  /= (double)NSamples;
    average_rr /= (double)NSamples;
    std::cout << "\nMonte Carlo integration from gaussian random vector: "
              << (average_r - miu).norm().item<double>()
               + (average_rr - average_r.outer(average_r) - var).norm().item<double>() << '\n'
              << "100,000 averages so can only converge to around 0.00316\n";
}