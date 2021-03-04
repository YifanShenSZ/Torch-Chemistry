#include <tchem/linalg.hpp>

#include <tchem/phaser.hpp>

#include <tchem/chemistry.hpp>

void alter_states() {
    tchem::Phaser phaser(3);
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor Hd = at::rand({3, 3}, top);
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig(true);
    at::Tensor states0 = phaser.alter_states(states, 0);
    double diff0 = ((states - states0.neg()).select(1, 0).norm()
                 +  (states - states0      ).select(1, 1).norm()
                 +  (states - states0      ).select(1, 2).norm()).item<double>();
    at::Tensor states1 = states.clone();
    phaser.alter_states_(states1, 1);
    double diff1 = ((states - states1      ).select(1, 0).norm()
                 +  (states - states1.neg()).select(1, 1).norm()
                 +  (states - states1      ).select(1, 2).norm()).item<double>();
    std::cout << "\nFixing phase of eigenstates: "
              << diff0 << "    "
              << diff1 << '\n';
    at::Tensor Hd4 = at::rand({4, 4}, top);
    at::Tensor energy4, states4;
    std::tie(energy4, states4) = Hd4.symeig(true);
    at::Tensor states40 = states4.clone();
    at::Tensor states40_view = states40.slice(1, 0, 3);
    phaser.alter_states_(states40_view, 0);
    double diff4 = ((states4 - states40.neg()).select(1, 0).norm()
                 +  (states4 - states40      ).select(1, 1).norm()
                 +  (states4 - states40      ).select(1, 2).norm()
                 +  (states4 - states40      ).select(1, 3).norm()).item<double>();
    std::cout << "\nFixing more eigenstates than phaser's definition: "
              << diff4 << '\n';
}

void fix_ob() {
    tchem::Phaser phaser(3);
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor  Hd = at::rand({3, 3}, top),
               dHd = at::rand({3, 3, 5}, top);
    // Adiabatic representation
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig(true);
    at::Tensor dH_a = tchem::linalg::UT_sy_U(dHd, states);
    // Composite representation
    at::Tensor H_c, dH_c;
    std::tie(H_c, dH_c) = tchem::chem::composite_representation(Hd, dHd);
    // Composite representation -> adiabatic representation
    at::Tensor energy_c, states_c;
    std::tie(energy_c, states_c) = H_c.symeig(true);
    tchem::linalg::UT_sy_U_(dH_c, states_c);
    at::Tensor dH_ca = phaser.fix_ob(dH_c, dH_a);
    at::Tensor dH_ca_ = dH_c.clone();
    phaser.fix_ob_(dH_ca_, dH_a);
    std::cout << "\nFixing phase of an observable: "
              << (dH_ca - dH_ca_).norm().item<double>() << "    "
              << (dH_ca - dH_a  ).norm().item<double>() << '\n';
}

void fix_ob2() {
    tchem::Phaser phaser(4);
    at::Tensor  Hd = at::rand({4, 4}),
               dHd = at::rand({4, 4, 5});
    // Adiabatic representation
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig(true);
    at::Tensor dH_a = tchem::linalg::UT_sy_U(dHd, states);
    // Adiabatic representation -> Composite representation
    at::Tensor H_c, dH_c;
    std::tie(H_c, dH_c) = tchem::chem::composite_representation(energy, dH_a);
    // Composite representation
    at::Tensor  H_c_ =  Hd.clone(),
               dH_c_ = dHd.clone();
    tchem::chem::composite_representation_(H_c_, dH_c_);
    srand(time(NULL));
    size_t index = rand() % phaser.possible_phases().size();
    H_c_ = phaser.alter_ob(H_c_, index);
    phaser.alter_ob_(dH_c_, index);
    at::Tensor H_fixed, dH_fixed;
    std::tie(H_fixed, dH_fixed) = phaser.fix_ob(H_c, dH_c, H_c_, dH_c_, 1.0);
    for (size_t i = 0    ; i < Hd.size(0); i++)
    for (size_t j = i + 1; j < Hd.size(1); j++) {
         H_fixed[j][i].zero_();
        dH_fixed[j][i].zero_();
    }
    at::Tensor  H_fixed_ =  H_c.clone(),
               dH_fixed_ = dH_c.clone();
    phaser.fix_ob_(H_fixed_, dH_fixed_, H_c_, dH_c_, 1.0);
    for (size_t i = 0    ; i < Hd.size(0); i++)
    for (size_t j = i + 1; j < Hd.size(1); j++) {
         H_fixed_[j][i].zero_();
        dH_fixed_[j][i].zero_();
    }
    std::cout << "\nFixing phase of 2 observables: "
              << ((H_fixed - H_fixed_).norm() + (dH_fixed - dH_fixed_).norm()).item<double>() << "    "
              << ((H_fixed - H_c_).norm() + (dH_fixed - dH_c_).norm()).item<double>() << '\n';
}

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'phaser'\n"
              << "Correct routines should print close to 0\n";
    alter_states();
    fix_ob();
    fix_ob2();
}