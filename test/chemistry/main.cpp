#include <tchem/linalg.hpp>

#include <tchem/chemistry.hpp>

void fix() {
    tchem::chem::Phaser phaser(3);
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
    at::Tensor H_c, dH_c;
    std::tie(H_c, dH_c) = tchem::chem::composite_representation(H, dH);
    // Composite representation -> adiabatic representation
    at::Tensor energies_c, states_c;
    std::tie(energies_c, states_c) = H_c.symeig(true);
    tchem::LA::UT_A3_U_(dH_c, states_c);
    phaser.fix_(dH_c, dH_a);
    std::cerr << "\nFixing phase of an observable: "
              << (energies - energies_c).norm().item<double>()
               + (dH_a - dH_c).norm().item<double>() << '\n';
}

void fix2() {
    tchem::chem::Phaser phaser(4);
    at::Tensor H = at::rand({4, 4}), dH = at::rand({4, 4, 5});
    for (size_t i = 0; i < 4; i++)
    for (size_t j = i + 1; j < 4; j++) {
         H[j][i].copy_( H[i][j]);
        dH[j][i].copy_(dH[i][j]);
    }
    // Adiabatic representation
    at::Tensor energies, states;
    std::tie(energies, states) = H.symeig(true);
    at::Tensor dH_a = tchem::LA::UT_A3_U(dH, states);
    // Adiabatic representation -> Composite representation
    at::Tensor H_c, dH_c;
    std::tie(H_c, dH_c) = tchem::chem::composite_representation(H, dH);
    // Composite representation
    tchem::chem::composite_representation_(H, dH);
    srand (time(NULL));
    size_t index = rand() % phaser.possible_phases().size();
    phaser.alter_( H, index);
    phaser.alter_(dH, index);
    phaser.fix_(H_c, dH_c, H, dH, 1.0);
    std::cerr << "\nFixing phase of 2 observables: "
              << ( H_c -  H).norm().item<double>()
               + (dH_c - dH).norm().item<double>() << '\n';
}

int main() {
    std::cerr << "This is a test program on Torch-Chemistry module 'chemistry'\n"
              << "Correct routines should print close to 0\n";

    fix();

    fix2();
}