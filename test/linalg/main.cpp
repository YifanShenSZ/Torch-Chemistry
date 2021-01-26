#include <tchem/linalg.hpp>

void triple_product() {
    at::Tensor a = at::rand(3), b = at::rand(3), c = at::rand(3);
    at::Tensor result = tchem::LA::triple_product(a, b, c),
               answer = a.cross(b).dot(c);
    std::cerr << "\nTriple product: "
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
    at::Tensor result = tchem::LA::outer_product(A, B);
    std::cerr << "\nOuter product for general tensor: "
              << (result - answer).norm().item<double>() << '\n';
}

void vec2sytensor() {
    at::Tensor x = at::tensor({8.0, 28.0, 98.0, 343.0});
    at::Tensor answer = at::tensor({8.0, 28.0, 28.0, 98.0, 28.0, 98.0, 98.0, 343.0});
    answer = answer.view({2, 2, 2});
    at::Tensor result = tchem::LA::vec2sytensor(x, 3, 2);
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
    result = tchem::LA::vec2sytensor(x, 4, 2);
    residue += abs((result[0][0][0][0] - answer[0][0][0][0]).item<double>())
             + abs((result[0][0][0][1] - answer[0][0][0][1]).item<double>())
             + abs((result[0][0][1][1] - answer[0][0][1][1]).item<double>())
             + abs((result[0][1][1][1] - answer[0][1][1][1]).item<double>())
             + abs((result[1][1][1][1] - answer[1][1][1][1]).item<double>());
    std::cerr << "\nConvert a vector to a symmetric tensor: "
              << residue << '\n';
}

void matdotmul() {
    at::Tensor A = at::rand({3, 3, 5}), B = at::rand({3, 3, 5});
    for (size_t i = 0; i < 3; i++)
    for (size_t j = i + 1; j < 3; j++) {
        A[j][i].copy_(A[i][j]);
        B[j][i].copy_(B[i][j]);
    }
    std::cerr << "\nMatrix dot multiplication: "
              << (tchem::LA::ge3matdotmul(A, B) - tchem::LA::sy3matdotmul(A, B)).norm().item<double>() << '\n';
}

void UT_A3_U() {
    at::Tensor H = at::rand({3, 3}), dH = at::rand({3, 3, 5});
    for (size_t i = 0; i < 3; i++)
    for (size_t j = i + 1; j < 3; j++) {
         H[j][i].copy_( H[i][j]);
        dH[j][i].copy_(dH[i][j]);
    }
    // Adiabatic representation
    at::Tensor energies, states;
    std::tie(energies, states) = H.symeig(true);
    at::Tensor dH_a = tchem::LA::UT_A3_U(dH, states);
    // Composite representation
    at::Tensor dHdH = tchem::LA::sy3matdotmul(dH, dH);
    at::Tensor eigvals, eigvecs;
    std::tie(eigvals, eigvecs) = dHdH.symeig(true);
    at::Tensor  H_c = eigvecs.transpose(0, 1).mm(H.mm(eigvecs));
    at::Tensor dH_c = tchem::LA::UT_A3_U(dH, eigvecs);
    // Composite representation -> adiabatic representation
    at::Tensor energies_c, states_c;
    std::tie(energies_c, states_c) = H_c.symeig(true);
    tchem::LA::UT_A3_U_(dH_c, states_c);
    double difference = (energies - energies_c).norm().item<double>();
    for (size_t i = 0; i < dH.size(0); i++) difference += (dH_a[i][i] - dH_c[i][i]).norm().item<double>();
    std::cerr << "\nUnitary transformation for symmetric 3rd-order tensor: "
              << difference << '\n';
}

int main() {
    std::cerr << "This is a test program on Torch-Chemistry module 'linalg'\n"
              << "Correct routines should print close to 0\n";

    triple_product();

    outer_product();

    vec2sytensor();

    matdotmul();

    UT_A3_U();
}