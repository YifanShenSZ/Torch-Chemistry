#include <tchem/linalg.hpp>

#include <tchem/chemistry.hpp>

void alter_states() {
    tchem::chem::Orderer orderer(3);
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor Hd = at::rand({3, 3}, top);
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig(true);
    at::Tensor states0 = orderer.alter_states(states, 0);
    double diff0 = ((states.select(1, 0) - states0.select(1, 0)).norm()
                 +  (states.select(1, 1) - states0.select(1, 2)).norm()
                 +  (states.select(1, 2) - states0.select(1, 1)).norm()).item<double>();
    at::Tensor states1 = states.clone();
    orderer.alter_states_(states1, 1);
    double diff1 = ((states.select(1, 0) - states1.select(1, 1)).norm()
                 +  (states.select(1, 1) - states1.select(1, 0)).norm()
                 +  (states.select(1, 2) - states1.select(1, 2)).norm()).item<double>();
    std::cout << "\nFixing ordering of eigenstates: "
              << diff0 << ' ' << diff1 << '\n';
}

void fix_ob() {
    tchem::chem::Orderer orderer(3);
    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);
    at::Tensor  Hd = at::rand({3, 3}, top),
               dHd = at::rand({3, 3, 5}, top);
    // adiabatic representation
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig(true);
    at::Tensor dH_a = tchem::linalg::UT_sy_U(dHd, states);
    // composite representation
    at::Tensor H_c, dH_c;
    std::tie(H_c, dH_c) = tchem::chem::composite_representation(Hd, dHd);
    // composite representation -> adiabatic representation
    at::Tensor energy_c, states_c;
    std::tie(energy_c, states_c) = H_c.symeig(true);
    orderer.alter_states_(states_c, rand() % 5);
    tchem::linalg::UT_sy_U_(dH_c, states_c);
    at::Tensor dH_ca = orderer.fix_ob(dH_c, dH_a);
    at::Tensor dH_ca_ = dH_c.clone();
    orderer.fix_ob_(dH_ca_, dH_a);
    std::cout << "\nFixing ordering of an observable: "
              << (dH_ca - dH_ca_).norm().item<double>() << ' '
              << (dH_ca - dH_a  ).norm().item<double>() << '\n';
}

void fix_ob2() {
    tchem::chem::Orderer orderer(4);
    at::Tensor  Hd = at::rand({4, 4}),
               dHd = at::rand({4, 4, 5});
    // adiabatic representation
    at::Tensor energy, states;
    std::tie(energy, states) = Hd.symeig(true);
    at::Tensor dH_a = tchem::linalg::UT_sy_U(dHd, states);
    // adiabatic representation -> composite representation
    at::Tensor H_c, dH_c;
    std::tie(H_c, dH_c) = tchem::chem::composite_representation(energy, dH_a);
    // composite representation
    at::Tensor  H_c_ =  Hd.clone(),
               dH_c_ = dHd.clone();
    tchem::chem::composite_representation_(H_c_, dH_c_);
    size_t index = rand() % 23;
    H_c_ = orderer.alter_ob(H_c_, index);
    orderer.alter_ob_(dH_c_, index);
    at::Tensor H_fixed, dH_fixed;
    std::tie(H_fixed, dH_fixed) = orderer.fix_ob(H_c, dH_c, H_c_, dH_c_, 1.0);
    at::Tensor  H_fixed_ =  H_c.clone(),
               dH_fixed_ = dH_c.clone();
    orderer.fix_ob_(H_fixed_, dH_fixed_, H_c_, dH_c_, 1.0);
    for (size_t i = 0    ; i < Hd.size(0); i++)
    for (size_t j = i + 1; j < Hd.size(1); j++) {
         H_fixed [j][i].zero_();
        dH_fixed [j][i].zero_();
         H_fixed_[j][i].zero_();
        dH_fixed_[j][i].zero_();
         H_c_    [j][i].zero_();
        dH_c_    [j][i].zero_();
    }
    std::cout << "\nFixing ordering of 2 observables: "
              << ((H_fixed - H_fixed_).norm() + (dH_fixed - dH_fixed_).norm()).item<double>() << ' '
              << ((H_fixed - H_c_    ).norm() + (dH_fixed - dH_c_    ).norm()).item<double>() << '\n';
}

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'orderer'\n"
              << "Correct routines should print close to 0\n";
    srand(time(NULL));

    alter_states();
    fix_ob();
    fix_ob2();
}