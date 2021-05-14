#include <CppLibrary/linalg.hpp>

#include <tchem/polynomial.hpp>

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'SApolynomial'\n"
              << "Correct routines should print close to 0\n";

    auto answer1 = at::tensor({1.0,
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
    Us[0] = U1.clone(); Us[1] = U2.clone();
    at::Tensor T1 = set1.rotation(Us), T2 = set2.rotation(Us);
    std::vector<at::Tensor> qs = {U1inv.mv(xs[0]), U2inv.mv(xs[1])};
    std::cout << "\nValue of symmetry adapted polynomial set after rotation: "
              << at::norm(T1.mv(set1(qs)) - answer1).item<double>() << ' '
              << at::norm(T2.mv(set2(qs)) - answer2).item<double>() << '\n';

    at::Tensor a = at::rand(2, xs[0].options());
    at::Tensor S1 = set1.translation(a), S2 = set2.translation(a);
    std::vector<at::Tensor> ps = xs;
    ps[0] -= a;
    std::cout << "\nValue of symmetry adapted polynomial set after translation: "
              << at::norm(S1.mv(set1(ps)) - answer1).item<double>() << ' '
              << at::norm(S2.mv(set2(ps)) - answer2).item<double>() << '\n';

    // analytical Jacobian and 2nd-order Jacobian
    at::Tensor cat_J1, cat_J2, cat_K1, cat_K2;
    auto J1 = set1.Jacobian(xs), J2 = set2.Jacobian(xs),
         J1_ = set1.Jacobian_(xs), J2_ = set2.Jacobian_(xs),
         J1__ = set1.Jacobian_(xs, cat_J1), J2__ = set2.Jacobian_(xs, cat_J2);
    auto K1 = set1.Jacobian2nd(xs), K2 = set2.Jacobian2nd(xs),
         K1_ = set1.Jacobian2nd_(xs), K2_ = set2.Jacobian2nd_(xs),
         K1__ = set1.Jacobian2nd_(xs, cat_K1), K2__ = set2.Jacobian2nd_(xs, cat_K2);
    // backward propagation
    for (at::Tensor & el : xs) el.set_requires_grad(true);
    at::Tensor value1 = set1(xs), value2 = set2(xs);
    std::vector<at::Tensor> J1_B(xs.size()), J2_B(xs.size());
    CL::utility::matrix<at::Tensor> K1_B(xs.size()), K2_B(xs.size());
    for (size_t i = 0; i < xs.size(); i++) {
        J1_B[i] = xs[i].new_empty({value1.size(0), xs[i].size(0)});
        J2_B[i] = xs[i].new_empty({value2.size(0), xs[i].size(0)});
        for (size_t j = i; j < xs.size(); j++) {
            K1_B[i][j] = xs[i].new_empty({value1.size(0), xs[i].size(0), xs[i].size(0)});
            K2_B[i][j] = xs[i].new_empty({value2.size(0), xs[i].size(0), xs[i].size(0)});
        }
    }
    for (size_t row = 0; row < value1.size(0); row++) {
        auto g = torch::autograd::grad({value1[row]}, {xs}, {}, true, true, true);
        for (size_t i = 0; i < xs.size(); i++) {
            if (! g[i].defined()) {
                J1_B[i][row].fill_(0.0);
                for (size_t j = i; j < xs.size(); j++) K1_B[i][j][row].fill_(0.0);
            }
            else {
                J1_B[i][row].copy_(g[i]);
                for (size_t j = i; j < xs.size(); j++)
                for (size_t k = 0; k < g[i].size(0); k++) {
                    auto h = torch::autograd::grad({g[i][k]}, {xs[j]}, {}, true, false, true);
                    if (! h[0].defined()) K1_B[i][j][row][k].fill_(0.0);
                    else K1_B[i][j][row][k].copy_(h[0]);
                }
            }
        }
    }
    for (size_t row = 0; row < value2.size(0); row++) {
        auto g = torch::autograd::grad({value2[row]}, {xs}, {}, true, true, true);
        for (size_t i = 0; i < xs.size(); i++) {
            if (! g[i].defined()) {
                J2_B[i][row].fill_(0.0);
                for (size_t j = i; j < xs.size(); j++) K2_B[i][j][row].fill_(0.0);
            }
            else {
                J2_B[i][row].copy_(g[i]);
                for (size_t j = i; j < xs.size(); j++)
                for (size_t k = 0; k < g[i].size(0); k++) {
                    auto h = torch::autograd::grad({g[i][k]}, {xs[j]}, {}, true, false, true);
                    if (! h[0].defined()) K2_B[i][j][row][k].fill_(0.0);
                    else K2_B[i][j][row][k].copy_(h[0]);
                }
            }
        }
    }
    // Compare
    std::cout << "\nAnalytical Jacobian vs backward propagation: "
              << (J1[0] - J1_B[0]).norm().item<double>() << ' '
              << (J1[1] - J1_B[1]).norm().item<double>() << ' '
              << (J1_[0] - J1_B[0]).norm().item<double>() << ' '
              << (J1_[1] - J1_B[1]).norm().item<double>() << ' '
              << (J1__[0] - J1_B[0]).norm().item<double>() << ' '
              << (J1__[1] - J1_B[1]).norm().item<double>() << ' '
              << (J2[0] - J2_B[0]).norm().item<double>() << ' '
              << (J2[1] - J2_B[1]).norm().item<double>() << ' '
              << (J2_[0] - J2_B[0]).norm().item<double>() << ' '
              << (J2_[1] - J2_B[1]).norm().item<double>() << ' '
              << (J2__[0] - J2_B[0]).norm().item<double>() << ' '
              << (J2__[1] - J2_B[1]).norm().item<double>() << '\n';
    std::cout << "\nBuilt in Jacobian concatenation vs manual concatenation: "
              << (cat_J1 - at::cat(J1, 1)).norm().item<double>() << ' '
              << (cat_J2 - at::cat(J2, 1)).norm().item<double>() << '\n';
    double difference1 = 0.0;
    for (size_t i = 0; i < xs.size(); i++) {
        for (size_t el = 0; el < K1_B[i][i].size(0); el++)
        for (size_t row = 0; row < K1_B[i][i].size(1); row++)
        for (size_t col = row; col < K1_B[i][i].size(2); col++)
        difference1 += (K1[i][i][el][row][col] - K1_B[i][i][el][row][col]).norm().item<double>();
        for (size_t j = i + 1; j < xs.size(); j++)
        difference1 += (K1[i][j] - K1_B[i][j]).norm().item<double>();
    }
    double difference1_ = 0.0;
    for (size_t i = 0; i < xs.size(); i++) {
        for (size_t el = 0; el < K1_B[i][i].size(0); el++)
        for (size_t row = 0; row < K1_B[i][i].size(1); row++)
        for (size_t col = row; col < K1_B[i][i].size(2); col++)
        difference1_ += (K1_[i][i][el][row][col] - K1_B[i][i][el][row][col]).norm().item<double>();
        for (size_t j = i + 1; j < xs.size(); j++)
        difference1_ += (K1_[i][j] - K1_B[i][j]).norm().item<double>();
    }
    double difference1__ = 0.0;
    for (size_t i = 0; i < xs.size(); i++) {
        for (size_t el = 0; el < K1_B[i][i].size(0); el++)
        for (size_t row = 0; row < K1_B[i][i].size(1); row++)
        for (size_t col = row; col < K1_B[i][i].size(2); col++)
        difference1__ += (K1__[i][i][el][row][col] - K1_B[i][i][el][row][col]).norm().item<double>();
        for (size_t j = i + 1; j < xs.size(); j++)
        difference1__ += (K1__[i][j] - K1_B[i][j]).norm().item<double>();
    }
    double difference2 = 0.0;
    for (size_t i = 0; i < xs.size(); i++) {
        for (size_t el = 0; el < K2_B[i][i].size(0); el++)
        for (size_t row = 0; row < K2_B[i][i].size(1); row++)
        for (size_t col = row; col < K2_B[i][i].size(2); col++)
        difference2 += (K2[i][i][el][row][col] - K2_B[i][i][el][row][col]).norm().item<double>();
        for (size_t j = i + 1; j < xs.size(); j++)
        difference2 += (K2[i][j] - K2_B[i][j]).norm().item<double>();
    }
    double difference2_ = 0.0;
    for (size_t i = 0; i < xs.size(); i++) {
        for (size_t el = 0; el < K2_B[i][i].size(0); el++)
        for (size_t row = 0; row < K2_B[i][i].size(1); row++)
        for (size_t col = row; col < K2_B[i][i].size(2); col++)
        difference2_ += (K2_[i][i][el][row][col] - K2_B[i][i][el][row][col]).norm().item<double>();
        for (size_t j = i + 1; j < xs.size(); j++)
        difference2_ += (K2_[i][j] - K2_B[i][j]).norm().item<double>();
    }
    double difference2__ = 0.0;
    for (size_t i = 0; i < xs.size(); i++) {
        for (size_t el = 0; el < K2_B[i][i].size(0); el++)
        for (size_t row = 0; row < K2_B[i][i].size(1); row++)
        for (size_t col = row; col < K2_B[i][i].size(2); col++)
        difference2__ += (K2__[i][i][el][row][col] - K2_B[i][i][el][row][col]).norm().item<double>();
        for (size_t j = i + 1; j < xs.size(); j++)
        difference2__ += (K2__[i][j] - K2_B[i][j]).norm().item<double>();
    }
    std::cout << "\nAnalytical 2nd-order Jacobian vs backward propagation: "
              << difference1 << ' ' << difference2 << ' '
              << difference1_ << ' ' << difference2_ << ' '
              << difference1__ << ' ' << difference2__ << '\n';
    at::Tensor mcat_K1 = cat_K1.new_empty(cat_K1.sizes()),
               mcat_K2 = cat_K2.new_empty(cat_K2.sizes());
    int64_t start_row = 0, stop_row;
    for (size_t i = 0; i < xs.size(); i++) {
        stop_row = start_row + xs[i].size(0);
        int64_t start_col = start_row, stop_col;
        for (size_t j = i; j < xs.size(); j++) {
            stop_col = start_col + xs[j].size(0);
            mcat_K1.slice(1, start_row, stop_row).slice(2, start_col, stop_col).copy_(K1[i][j]);
            mcat_K2.slice(1, start_row, stop_row).slice(2, start_col, stop_col).copy_(K2[i][j]);
            start_col = stop_col;
        }
        start_row = stop_row;
    }
    for (size_t i = 0; i < mcat_K1.size(1); i++)
    for (size_t j = i + 1; j < mcat_K1.size(2); j++)
    mcat_K1.select(1, j).select(-1, i).copy_(mcat_K1.select(1, i).select(-1, j));
    for (size_t i = 0; i < mcat_K2.size(1); i++)
    for (size_t j = i + 1; j < mcat_K2.size(2); j++)
    mcat_K2.select(1, j).select(-1, i).copy_(mcat_K2.select(1, i).select(-1, j));
    std::cout << "\nBuilt in 2nd-order Jacobian concatenation vs manual concatenation: "
              << (cat_K1 - mcat_K1).norm().item<double>() << ' '
              << (cat_K2 - mcat_K2).norm().item<double>() << '\n';
}