#include <CppLibrary/chemistry.hpp>

#include <tchem/SASintcoord.hpp>

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'SASintcoord'\n"
              << "Correct routines should print close to 0\n";

    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);

    tchem::IC::SASICSet set("whatever", "whatever", "SAS.in");

    CL::chem::xyz<double> geom_c2v("sad-A1.xyz", true);
    std::vector<double> coords_c2v = geom_c2v.coords();
    at::Tensor r_c2v = at::from_blob(coords_c2v.data(), coords_c2v.size(), top);
    at::Tensor q_c2v = set.IntCoordSet::operator()(r_c2v);
    std::vector<at::Tensor> SASgeom_c2v = set(q_c2v);
    std::cout << "\nC2v: "
              << SASgeom_c2v[1].norm().item<double>() << ' '
              << SASgeom_c2v[2].norm().item<double>() << ' '
              << SASgeom_c2v[3].norm().item<double>() << '\n';

    CL::chem::xyz<double> geom_csoop("min-A1.xyz", true);
    std::vector<double> coords_csoop = geom_csoop.coords();
    at::Tensor r_csoop = at::from_blob(coords_csoop.data(), coords_csoop.size(), top);
    at::Tensor q_csoop = set.IntCoordSet::operator()(r_csoop);
    std::vector<at::Tensor> SASgeom_csoop = set(q_csoop);
    std::cout << "\nCs out of plane: "
              << SASgeom_csoop[2].norm().item<double>() << ' '
              << SASgeom_csoop[3].norm().item<double>() << '\n';

    CL::chem::xyz<double> geom_csplanar("sad-B1.xyz", true);
    std::vector<double> coords_csplanar = geom_csplanar.coords();
    at::Tensor r_csplanar = at::from_blob(coords_csplanar.data(), coords_csplanar.size(), top);
    at::Tensor q_csplanar = set.IntCoordSet::operator()(r_csplanar);
    std::vector<at::Tensor> SASgeom_csplanar = set(q_csplanar);
    std::cout << "\nCs planar: "
              << SASgeom_csplanar[1].norm().item<double>() << ' '
              << SASgeom_csplanar[3].norm().item<double>() << '\n';

    CL::chem::xyz<double> geom("slow-1.5.xyz", true);
    std::vector<double> coords = geom.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), top);
    at::Tensor q = set.IntCoordSet::operator()(r);
    // Backward propagation
    q.set_requires_grad(true);
    std::vector<at::Tensor> SASq = set(q);
    std::vector<at::Tensor> SASJ(SASq.size());
    for (size_t irred = 0; irred < SASq.size(); irred++) {
        SASJ[irred] = SASq[irred].new_empty({SASq[irred].size(0), q.size(0)});
        for (size_t row = 0; row < SASJ[irred].size(0); row++) {
            if (q.grad().defined()) {
                q.grad().detach_();
                q.grad().zero_();
            }
            SASq[irred][row].backward({}, true);
            SASJ[irred][row].copy_(q.grad());
        }
    }
    // Numerical Jacobian
    q.set_requires_grad(false);
    std::vector<at::Tensor> SASJ_N(SASq.size());
    for (size_t irred = 0; irred < SASq.size(); irred++) SASJ_N[irred] = SASq[irred].new_empty({SASq[irred].size(0), q.size(0)});
    for (size_t column = 0; column < q.size(0); column++) {
        at::Tensor q_plus = q.clone();
        q_plus[column] += 1e-5;
        std::vector<at::Tensor> SASq_plus = set(q_plus);
        at::Tensor q_minus = q.clone();
        q_minus[column] -= 1e-5;
        std::vector<at::Tensor> SASq_minus = set(q_minus);
        for (size_t irred = 0; irred < SASq.size(); irred++) {
            at::Tensor subvec = (SASq_plus[irred] - SASq_minus[irred]) / 2e-5;
            for (size_t row = 0; row < SASJ_N[irred].size(0); row++) SASJ_N[irred][row][column] = subvec[row];
        }
    }
    double difference = 0.0;
    for (size_t irred = 0; irred < SASq.size(); irred++) difference += (SASJ[irred] - SASJ_N[irred]).norm().item<double>();
    std::cout << "\nBackward propagation vs numerical Jacobian: " << difference << '\n';
}