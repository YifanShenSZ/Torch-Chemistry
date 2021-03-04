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
    at::Tensor q1_col, J1_col;
    std::tie(q1_col, J1_col) = set_col.compute_IC_J(r_col);
    std::cout << "\nInternal coordinate calculated with Jacobian: "
              << (q0_col - q1_col).norm().item<double>() << '\n';

    at::Tensor J_col_back = J1_col.new_empty(J1_col.sizes());
    for (size_t i = 0; i < q1_col.size(0); i++) {
        if (r_col.grad().defined()) {
            r_col.grad().detach_();
            r_col.grad().zero_();
        }
        q1_col[i].backward({}, true);
        J_col_back[i].copy_(r_col.grad());
    }
    r_col.set_requires_grad(false);
    std::cout << "\nBackward propagation vs analytical Jacobian: "
              << (J1_col - J_col_back).norm().item<double>() << '\n';

    at::Tensor q2_col, J2_col, K2_col;
    std::tie(q2_col, J2_col, K2_col) = set_col.compute_IC_J_K(r_col);
    std::cout << "\nInternal coordinate calculated with 1st and 2nd order Jacobian: "
              << (q0_col - q2_col).norm().item<double>() << '\n';
    std::cout << "\nJacobian calculated with 2nd order Jacobian: "
              << (J1_col - J2_col).norm().item<double>() << '\n';

    const double dr = 1e-5;
    std::vector<at::Tensor> plus(r_col.size(0)), minus(r_col.size(0));
    for (size_t i = 0; i < r_col.size(0); i++) {
        at::Tensor q;
        plus[i] = r_col.clone();
        plus[i][i] += dr;
        std::tie(q, plus[i]) = set_col.compute_IC_J(plus[i]);
        minus[i] = r_col.clone();
        minus[i][i] -= dr;
        std::tie(q, minus[i]) = set_col.compute_IC_J(minus[i]);
    }
    at::Tensor K_numerical = K2_col.new_empty(K2_col.sizes());
    for (size_t i = 0; i < r_col.size(0); i++) K_numerical.select(1, i).copy_((plus[i] - minus[i]) / 2.0 / dr);
    std::cout << "\nFinite difference vs analytical 2nd order Jacobian: "
              << (K2_col - K_numerical).norm().item<double>() << '\n';

    CL::chem::xyz<double> geom_def("slow-1.5.xyz", true);
    std::vector<double> coords_def = geom_def.coords();
    at::Tensor r_def = at::from_blob(coords_def.data(), coords_def.size(), top);

    tchem::IC::IntCoordSet set_def("whatever", "whatever");
    at::Tensor q_def, J_def;
    std::tie(q_def, J_def) = set_def.compute_IC_J(r_def);
    std::cout << "\nDefault internal coordinate and Jacobian: "
              << (q1_col - q_def).norm().item<double>()
               + (J1_col - J_def).norm().item<double>() << '\n';
}