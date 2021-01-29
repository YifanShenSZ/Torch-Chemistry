#include <CppLibrary/chemistry.hpp>

#include <tchem/utility.hpp>
#include <tchem/intcoord.hpp>

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'intcoord'\n"
              << "Correct routines should print close to 0\n";

    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);

    at::Tensor intgeom = tchem::utility::read_vector("intgeom");
    intgeom.slice(0, 0 , 7 ) *= 1.8897261339212517;
    intgeom.slice(0, 11, 16) *= 1.8897261339212517;
    intgeom.slice(0, 21, 23) *= 1.8897261339212517;

    CL::chem::xyz_mass<double> geom_col("geom", true);
    std::vector<double> coords_col = geom_col.coords();
    at::Tensor r_col = at::from_blob(coords_col.data(), coords_col.size(), top);

    tchem::IC::IntCoordSet set_col("Columbus7", "whatever");
    at::Tensor q0_col = set_col(r_col);
    std::cout << "\nColumbus7 format internal coordinate: "
              << ((q0_col - intgeom) / intgeom).norm().item<double>() << '\n';

    r_col.set_requires_grad(true);
    at::Tensor q_col, J_col;
    std::tie(q_col, J_col) = set_col.compute_IC_J(r_col);
    std::cout << "\nInternal coordinate calculated with Jacobian: "
              << (q0_col - q_col).norm().item<double>() << '\n';

    at::Tensor J_col_back = J_col.new_empty(J_col.sizes());
    for (size_t i = 0; i < q_col.size(0); i++) {
        if (r_col.grad().defined()) {
            r_col.grad().detach_();
            r_col.grad().zero_();
        }
        q_col[i].backward({}, true);
        J_col_back[i].copy_(r_col.grad());
    }
    std::cout << "\nDirect Jacobian vs backward propagation: "
              << (J_col - J_col_back).norm().item<double>() << '\n';

    CL::chem::xyz<double> geom_def("slow-1.5.xyz", true);
    std::vector<double> coords_def = geom_def.coords();
    at::Tensor r_def = at::from_blob(coords_def.data(), coords_def.size(), top);

    tchem::IC::IntCoordSet set_def("whatever", "whatever");
    at::Tensor q_def, J_def;
    std::tie(q_def, J_def) = set_def.compute_IC_J(r_def);
    std::cout << "\nDefault internal coordinate and Jacobian: "
              << (q_col - q_def).norm().item<double>()
               + (J_col - J_def).norm().item<double>() << '\n';
}