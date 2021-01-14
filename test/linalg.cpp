#include <tchem/linalg.hpp>

void test_linalg() {
    at::Tensor answer, result;

    at::Tensor A = at::tensor({1.0, 2.0, 3.0, 4.0}),
               B = at::tensor({5.0, 6.0, 7.0, 8.0});
    A = A.view({2, 2});
    B = B.view({2, 2});
    answer = A.new_empty({2, 2, 2, 2});
    for (size_t i = 0; i < 2; i++)
    for (size_t j = 0; j < 2; j++)
    for (size_t k = 0; k < 2; k++)
    for (size_t l = 0; l < 2; l++)
    answer[i][j][k][l] = A[i][j] * B[k][l];
    result = tchem::LA::outer_product(A, B);
    std::cerr << "\nOuter product for general tensor: "
              << (answer - result).norm().item<double>() << '\n';

    at::Tensor x = at::tensor({8.0, 28.0, 98.0, 343.0});
    answer = at::tensor({8.0, 28.0, 28.0, 98.0, 28.0, 98.0, 98.0, 343.0});
    answer = answer.view({2, 2, 2});
    result = tchem::LA::vec2sytensor(x, 3, 2);
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