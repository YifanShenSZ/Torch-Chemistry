#include <CppLibrary/chemistry.hpp>

#include <tchem/utility.hpp>
#include <tchem/intcoord.hpp>

c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);

void q_J_K() {
    at::Tensor intgeom = tchem::utility::read_vector("intgeom");
    intgeom.slice(0, 0 , 7 ) *= 1.8897261339212517;
    intgeom.slice(0, 11, 16) *= 1.8897261339212517;
    intgeom.slice(0, 21, 23) *= 1.8897261339212517;

    CL::chem::xyz_mass<double> geom_col("geom", true);
    std::vector<double> coords_col = geom_col.coords();
    at::Tensor r_col = at::from_blob(coords_col.data(), coords_col.size(), top);

    tchem::IC::IntCoordSet set_col("Columbus7", "intcfl");
    at::Tensor q0_col = set_col(r_col);
    std::cout << "\nColumbus7 format internal coordinate: "
              << ((q0_col - intgeom) / intgeom).norm().item<double>() << '\n';

    at::Tensor q1_col, J1_col;
    std::tie(q1_col, J1_col) = set_col.compute_IC_J(r_col);

    at::Tensor q2_col, J2_col, K2_col;
    std::tie(q2_col, J2_col, K2_col) = set_col.compute_IC_J_K(r_col);
    std::cout << "\nInternal coordinates calculated with Jacobians: "
              << (q0_col - q1_col).norm().item<double>() << ' '
              << (q0_col - q2_col).norm().item<double>() << '\n';
    std::cout << "\nJacobian calculated with 2nd order Jacobian: "
              << (J1_col - J2_col).norm().item<double>() << '\n';

    at::Tensor J_f = J2_col.new_empty(J2_col.sizes()),
               K_f = K2_col.new_empty(K2_col.sizes());
    for (size_t i = 0; i < r_col.size(0); i++) {
        // +1e-5
        at::Tensor rp = r_col.clone();
        rp[i] += 1e-5;
        at::Tensor qp, Jp;
        std::tie(qp, Jp) = set_col.compute_IC_J(rp);
        // -1e-5
        at::Tensor rm = r_col.clone();
        rm[i] -= 1e-5;
        at::Tensor qm, Jm;
        std::tie(qm, Jm) = set_col.compute_IC_J(rm);
        // finite difference
        J_f.select(1, i) = (qp - qm) / 2e-5;
        K_f.select(2, i) = (Jp - Jm) / 2e-5;
    }
    std::cout << "\nFinite difference vs analytical Jacobian: "
              << (J2_col - J_f).abs_().max().item<double>() << ' '
              << (K2_col - K_f).abs_().max().item<double>() << '\n';

    CL::chem::xyz<double> geom_def("slow-1.5.xyz", true);
    std::vector<double> coords_def = geom_def.coords();
    at::Tensor r_def = at::from_blob(coords_def.data(), coords_def.size(), top);

    tchem::IC::IntCoordSet set_def("default", "IntCoordDef");
    at::Tensor q_def, J_def, K_def;
    std::tie(q_def, J_def, K_def) = set_def.compute_IC_J_K(r_def);
    std::cout << "\nDefault internal coordinate and Jacobian: "
              << (q2_col - q_def).norm().item<double>() << ' '
              << (J2_col - J_def).norm().item<double>() << ' '
              << (K2_col - K_def).norm().item<double>() << '\n';
}

std::tuple<double, double, double, double> advanced_q_J_K(const std::string & geom_file) {
    CL::chem::xyz<double> geom(geom_file, true);
    std::vector<double> coords = geom.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), top);
    tchem::IC::IntCoordSet set("whatever", "advanced_IntCoordDef");

    at::Tensor q = set(r);

    at::Tensor qJ, J;
    std::tie(qJ, J) = set.compute_IC_J(r);

    at::Tensor qK, JK, K;
    std::tie(qK, JK, K) = set.compute_IC_J_K(r);

    at::Tensor J_f = J.new_empty(J.sizes()),
               K_f = K.new_empty(K.sizes());
    for (size_t i = 0; i < r.size(0); i++) {
        // +1e-3
        at::Tensor rp = r.clone();
        rp[i] += 1e-3;
        at::Tensor qp, Jp;
        std::tie(qp, Jp) = set.compute_IC_J(rp);
        // -1e-3
        at::Tensor rm = r.clone();
        rm[i] -= 1e-3;
        at::Tensor qm, Jm;
        std::tie(qm, Jm) = set.compute_IC_J(rm);
        // finite difference
        J_f.select(1, i) = (qp - qm) / 2e-3;
        K_f.select(2, i) = (Jp - Jm) / 2e-3;
    }

    double qerror = q[0].item<double>() - 1.0
                  + (at::cos(q[1]) - q[2]).item<double>()
                  + (at::cos(q[3]) - q[4]).item<double>()
                  + (at::cos(q[5]) - q[6]).item<double>()
                  + (at::sin(q[7 ]) - q[8 ]).item<double>()
                  + (at::sin(q[10]) - q[11]).item<double>()
                  + (at::cos(q[7 ]) - q[9 ]).item<double>()
                  + (at::cos(q[10]) - q[12]).item<double>()
                  + (at::sin(q[13]) - q[14]).item<double>()
                  + (at::sin(q[15]) - q[16]).item<double>()
                  + (at::sin(q[17]) - q[18]).item<double>()
                  + (at::cos(q[17]) - q[19]).item<double>()
                  + (at::sin(q[1]) * q[19] - q[20]).item<double>()
                  + (at::sin(q[1]) * q[18] - q[21]).item<double>(),
           Jerror = (J - J_f).norm().item<double>() + (K - K_f).norm().item<double>(),
           qdiff = (q - qJ).norm().item<double>() + (q - qK).norm().item<double>(),
           Jdiff = (J - JK).norm().item<double>();
    return std::make_tuple(qerror, Jerror, qdiff, Jdiff);
}

void grad_hess() {
    tchem::IC::IntCoordSet set("default", "IntCoordDef");

    CL::chem::xyz<double> origin("min-B1.xyz", true);
    std::vector<double> origin_coords = origin.coords();
    at::Tensor r_origin = at::from_blob(origin_coords.data(), origin_coords.size(), top);

    CL::chem::xyz<double> geom("slow-1.5.xyz", true);
    std::vector<double> coords = geom.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), top);

    at::Tensor q = set(r) - set(r_origin);
    at::Tensor  intgrad = q.clone(),
                intHess = at::eye(36, top);
    at::Tensor cartgrad = set.gradient_int2cart(r, intgrad);
    at::Tensor cartHess = set.Hessian_int2cart(r, intgrad, intHess);

    at::Tensor intgrad0 = set.gradient_cart2int_matrix(r).mv(cartgrad);

    at::Tensor  intgrad1 = set.gradient_cart2int(r, cartgrad),
                intHess1 = set.Hessian_cart2int(r, cartgrad, cartHess);
    at::Tensor cartgrad1 = set.gradient_int2cart(r, intgrad1),
               cartHess1 = set.Hessian_int2cart(r, intgrad1, intHess1);
    std::cout << "\nGradient and Hessian transformations: "
              << ( intgrad -  intgrad0).norm().item<double>()
               + ( intgrad -  intgrad1).norm().item<double>() << ' '
              << ( intHess -  intHess1).norm().item<double>() << ' '
              << (cartgrad - cartgrad1).norm().item<double>() << ' '
              << (cartHess - cartHess1).norm().item<double>() << '\n';
}

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'intcoord'\n"
              << "Correct routines should print close to 0\n";
    q_J_K();

    double qerror1, Jerror1, qdiff1, Jdiff1;
    std::tie(qerror1, Jerror1, qdiff1, Jdiff1) = advanced_q_J_K("slow-1.5.xyz");
    double qerror2, Jerror2, qdiff2, Jdiff2;
    std::tie(qerror2, Jerror2, qdiff2, Jdiff2) = advanced_q_J_K("min-B1.xyz");
    double qerror3, Jerror3, qdiff3, Jdiff3;
    std::tie(qerror3, Jerror3, qdiff3, Jdiff3) = advanced_q_J_K("B1rot-2.2.xyz");
    std::cout << "\nAdvanced internal coordinates: "
              << qerror1 << ' ' << qerror2 << ' ' << qerror3 << '\n';
    std::cout << "\nAnalytical Jacobian vs finite difference: "
              << Jerror1 << ' ' << Jerror2 << ' ' << Jerror3 << '\n'; 
    std::cout << "\nAdvanced internal coordinates calculated with Jacobians: "
              << qdiff1 << ' ' << qdiff2 << ' ' << qdiff3 << '\n';
    std::cout << "\nJacobian calculated with 2nd order Jacobian: "
              << Jdiff1 << ' ' << Jdiff2 << ' ' << Jdiff3 << '\n';

    grad_hess();
}
