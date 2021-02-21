#include <tchem/linalg.hpp>

void triple_product() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor a = at::rand(3, top),
               b = at::rand(3, top),
               c = at::rand(3, top);
    at::Tensor result = tchem::linalg::triple_product(a, b, c),
               answer = a.cross(b).dot(c);
    std::cout << "\nTriple product: "
              << (result - answer).norm().item<double>() << '\n';
}

void outer_product() {
    at::Tensor A = at::tensor({1.0, 2.0, 3.0, 4.0}),
               B = at::tensor({5.0, 6.0, 7.0, 8.0});
    A = A.view({2, 2});
    B = B.view({2, 2});
    at::Tensor answer = A.new_empty({2, 2, 2, 2});
    for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
    for (size_t k = 0; k < 2; k++)
    for (size_t l = 0; l < 2; l++)
    answer[i][j][k][l] = A[i][j] * B[k][l];
    at::Tensor result = tchem::linalg::outer_product(A, B);
    std::cout << "\nOuter product for general tensor: "
              << (result - answer).norm().item<double>() << '\n';
}

void vec2sytensor() {
    at::Tensor x = at::tensor({8.0, 28.0, 98.0, 343.0});
    at::Tensor answer = at::tensor({8.0, 28.0, 28.0, 98.0, 28.0, 98.0, 98.0, 343.0});
    answer = answer.view({2, 2, 2});
    at::Tensor result = tchem::linalg::vec2sytensor(x, 3, 2);
    double residue = abs((result[0][0][0] - answer[0][0][0]).item<double>())
                   + abs((result[0][0][1] - answer[0][0][1]).item<double>())
                   + abs((result[0][1][1] - answer[0][1][1]).item<double>())
                   + abs((result[1][1][1] - answer[1][1][1]).item<double>());
    x = at::tensor({16.0, 56.0, 196.0, 686.0, 2401.0});
    answer = at::tensor({16.0, 56.0, 56.0, 196.0,
                         56.0, 196.0, 196.0, 686.0,
                         56.0, 196.0, 196.0, 686.0,
                         196.0, 686.0, 686.0, 2401.0});
    answer = answer.view({2, 2, 2, 2});
    result = tchem::linalg::vec2sytensor(x, 4, 2);
    residue += abs((result[0][0][0][0] - answer[0][0][0][0]).item<double>())
             + abs((result[0][0][0][1] - answer[0][0][0][1]).item<double>())
             + abs((result[0][0][1][1] - answer[0][0][1][1]).item<double>())
             + abs((result[0][1][1][1] - answer[0][1][1][1]).item<double>())
             + abs((result[1][1][1][1] - answer[1][1][1][1]).item<double>());
    std::cout << "\nConvert a vector to a symmetric tensor: "
              << residue << '\n';
}

void matdotmul() {
    at::Tensor A = at::rand({3, 3, 5}), B = at::rand({3, 3, 5});
    at::Tensor sy = tchem::linalg::sy3matdotmul(A, B);
    for (size_t i = 0; i < 3; i++)
    for (size_t j = i + 1; j < 3; j++) {
        A[j][i].copy_(A[i][j]);
        B[j][i].copy_(B[i][j]);
    }
    at::Tensor ge = tchem::linalg::ge3matdotmul(A, B);
    std::cout << "\nMatrix dot multiplication: "
              << (sy - ge).norm().item<double>() << '\n';
}

void matmvmul() {
    at::Tensor A = at::rand({3, 3, 4, 5}), B = at::rand({3, 3, 5});
    at::Tensor sy = tchem::linalg::sy4matmvmulsy3(A, B);
    for (size_t i = 0; i < 3; i++)
    for (size_t j = i + 1; j < 3; j++) {
        A[j][i].copy_(A[i][j]);
        B[j][i].copy_(B[i][j]);
    }
    at::Tensor ge = tchem::linalg::ge4matmvmulge3(A, B);
    std::cout << "\nMatrix matrix-vector multiplication: "
              << (sy - ge).norm().item<double>() << '\n';
}

void matoutermul() {
    at::Tensor A = at::rand({3, 3}), B = at::rand({3, 3, 5});
    at::Tensor sy = tchem::linalg::symatoutermul(A, B);
    for (size_t i = 0; i < 3; i++)
    for (size_t j = i + 1; j < 3; j++) {
        A[j][i].copy_(A[i][j]);
        B[j][i].copy_(B[i][j]);
    }
    at::Tensor ge = tchem::linalg::gematoutermul(A, B);
    std::cout << "\nMatrix outer multiplication: "
              << (sy - ge).norm().item<double>() << '\n';
}

void UT_A_U() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor   H = at::rand({3, 3      }, top),
                dH = at::rand({3, 3, 5   }, top),
               ddH = at::rand({3, 3, 5, 5}, top);
    at::Tensor energies, states;
    std::tie(energies, states) = H.symeig(true);
    // sy
    at::Tensor sy2 = tchem::linalg::UT_sy_U(H, states);
    at::Tensor sy2_ = H.clone();
    tchem::linalg::UT_sy_U_(sy2_, states);
    at::Tensor sy3 = tchem::linalg::UT_sy_U(dH, states);
    at::Tensor sy3_ = dH.clone();
    tchem::linalg::UT_sy_U_(sy3_, states);
    at::Tensor sy4 = tchem::linalg::UT_sy_U(ddH, states);
    at::Tensor sy4_ = ddH.clone();
    tchem::linalg::UT_sy_U_(sy4_, states);
    // ge
    for (size_t i = 0; i < H.size(0); i++)
    for (size_t j = i + 1; j < H.size(1); j++) {
          H[j][i].copy_(  H[i][j]);
         dH[j][i].copy_( dH[i][j]);
        ddH[j][i].copy_(ddH[i][j]);
    }
    at::Tensor ge2 = tchem::linalg::UT_ge_U(H, states);
    at::Tensor ge2_ = H.clone();
    tchem::linalg::UT_ge_U_(ge2_, states);
    at::Tensor ge3 = tchem::linalg::UT_ge_U(dH, states);
    at::Tensor ge3_ = dH.clone();
    tchem::linalg::UT_ge_U_(ge3_, states);
    at::Tensor ge4 = tchem::linalg::UT_ge_U(ddH, states);
    at::Tensor ge4_ = ddH.clone();
    tchem::linalg::UT_ge_U_(ge4_, states);
    for (size_t i = 0; i < H.size(0); i++)
    for (size_t j = i + 1; j < H.size(1); j++) {
        ge2 [j][i].zero_();
        ge2_[j][i].zero_();
        ge3 [j][i].zero_();
        ge3_[j][i].zero_();
        ge4 [j][i].zero_();
        ge4_[j][i].zero_();
    }
    std::cout << "\nUnitary transformation: "
              << (sy2 - ge2 ).norm().item<double>() << ' '
              << (sy2 - sy2_).norm().item<double>() << ' '
              << (ge2 - ge2_).norm().item<double>() << ' '
              << (sy3 - ge3 ).norm().item<double>() << ' '
              << (sy3 - sy3_).norm().item<double>() << ' '
              << (ge3 - ge3_).norm().item<double>() << ' '
              << (sy4 - ge4 ).norm().item<double>() << ' '
              << (sy4 - sy4_).norm().item<double>() << ' '
              << (ge4 - ge4_).norm().item<double>() << '\n';
}

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'linalg'\n"
              << "Correct routines should print close to 0\n";
    triple_product();
    outer_product();
    vec2sytensor();
    matdotmul();
    matmvmul();
    matoutermul();
    UT_A_U();
}