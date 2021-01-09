#include <tchem/linalg.hpp>

void test_linalg() {
    at::Tensor x = at::tensor({8.0, 28.0, 98.0, 343.0});
    at::Tensor answer = at::tensor({8.0, 28.0, 28.0, 98.0, 28.0, 98.0, 98.0, 343.0});
    answer = answer.view({2, 2, 2});

    at::Tensor result = tchem::LA::vec2sytensor(x, {2, 2, 2});
    std::cerr << "\nConvert a vector to a symmetric tensor: "
              << abs((result[0][0][0] - answer[0][0][0]).item<double>())
               + abs((result[0][0][1] - answer[0][0][1]).item<double>())
               + abs((result[0][1][1] - answer[0][1][1]).item<double>())
               + abs((result[1][1][1] - answer[1][1][1]).item<double>()) << '\n';
}