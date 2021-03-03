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

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'chemistry'\n"
              << "Correct routines should print close to 0\n";
    composite_representation();
}