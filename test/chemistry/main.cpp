#include <tchem/linalg.hpp>

#include <tchem/chemistry.hpp>

void composite_representation() {
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor  Hd = at::rand({3, 3}, top),
               dHd = at::rand({3, 3, 5}, top);
    // Adiabatic representation
    at::Tensor energies, states;
    std::tie(energies, states) = Hd.symeig();
    // Composite representation
    at::Tensor H_c, dH_c;
    std::tie(H_c, dH_c) = tchem::chem::composite_representation(Hd, dHd);
    at::Tensor  H_c_ =  Hd.clone(),
               dH_c_ = dHd.clone();
    tchem::chem::composite_representation_(H_c_, dH_c_);
    // Composite representation -> adiabatic representation
    at::Tensor energies_c, states_c;
    std::tie(energies_c, states_c) = H_c.symeig();
    at::Tensor energies_c_, states_c_;
    std::tie(energies_c_, states_c_) = H_c_.symeig();
    std::cout << "\nComposite representation: "
              << (energies_c - energies_c_).norm().item<double>() << "    "
              << (energies_c - energies  ).norm().item<double>() << '\n';
}

void fix() {
    tchem::chem::Phaser phaser(3);
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor  Hd = at::rand({3, 3}, top),
               dHd = at::rand({3, 3, 5}, top);
    // Adiabatic representation
    at::Tensor energies, states;
    std::tie(energies, states) = Hd.symeig(true);
    at::Tensor dH_a = tchem::LA::UT_sy_U(dHd, states);
    // Composite representation
    at::Tensor H_c, dH_c;
    std::tie(H_c, dH_c) = tchem::chem::composite_representation(Hd, dHd);
    // Composite representation -> adiabatic representation
    at::Tensor energies_c, states_c;
    std::tie(energies_c, states_c) = H_c.symeig(true);
    tchem::LA::UT_sy_U_(dH_c, states_c);
    at::Tensor dH_ca = phaser.fix(dH_c, dH_a);
    at::Tensor dH_ca_ = dH_c.clone();
    phaser.fix_(dH_ca_, dH_a);
    std::cout << "\nFixing phase of an observable: "
              << (dH_ca - dH_ca_).norm().item<double>() << "    "
              << (dH_ca - dH_a  ).norm().item<double>() << '\n';
}

void fix2() {
    tchem::chem::Phaser phaser(4);
    at::Tensor  Hd = at::rand({4, 4}),
               dHd = at::rand({4, 4, 5});
    // Adiabatic representation
    at::Tensor energies, states;
    std::tie(energies, states) = Hd.symeig(true);
    at::Tensor dH_a = tchem::LA::UT_sy_U(dHd, states);
    // Adiabatic representation -> Composite representation
    at::Tensor H_c, dH_c;
    std::tie(H_c, dH_c) = tchem::chem::composite_representation(Hd, dHd);
    // Composite representation
    at::Tensor  H_c_ =  Hd.clone(),
               dH_c_ = dHd.clone();
    tchem::chem::composite_representation_(H_c_, dH_c_);
    srand (time(NULL));
    size_t index = rand() % phaser.possible_phases().size();
    H_c_ = phaser.alter(H_c_, index);
    phaser.alter_(dH_c_, index);
    at::Tensor H_fixed, dH_fixed;
    std::tie(H_fixed, dH_fixed) = phaser.fix(H_c, dH_c, H_c_, dH_c_, 1.0);
    for (size_t i = 0    ; i < Hd.size(0); i++)
    for (size_t j = i + 1; j < Hd.size(1); j++) {
         H_fixed[j][i].zero_();
        dH_fixed[j][i].zero_();
    }
    at::Tensor  H_fixed_ =  H_c.clone(),
               dH_fixed_ = dH_c.clone();
    phaser.fix_(H_fixed_, dH_fixed_, H_c_, dH_c_, 1.0);
    std::cout << "\nFixing phase of 2 observables: "
              << ((H_fixed - H_fixed_).norm() + (dH_fixed - dH_fixed_).norm()).item<double>() << "    "
              << ((H_fixed - H_c_).norm() + (dH_fixed - dH_c_).norm()).item<double>() << '\n';
}

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'chemistry'\n"
              << "Correct routines should print close to 0\n";
    composite_representation();
    fix();
    fix2();
}