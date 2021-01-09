#include <chrono>
#include <random>

#include <tchem/tchem.hpp>

void test_linalg();

void test_polynomial();
void test_gaussian();

int main() {
    std::cerr << "This is a test program on Torch-Chemistry\n"
              << "Correct routines should print close to 0\n";

    std::cerr << "\n\n---------- Testing module linear algebra... ----------\n";
    test_linalg();
    std::cerr << "\n---------- Linear algebra test passed ----------\n";

    std::cerr << "\n\n---------- Testing module polynomial... ----------\n";
    test_polynomial();
    std::cerr << "\n---------- Polynomial test passed ----------\n";

    std::cerr << "\n\n---------- Testing module gaussian... ----------\n";
    test_gaussian();
    std::cerr << "\n---------- Gaussian test passed ----------\n";

    return 0;
}