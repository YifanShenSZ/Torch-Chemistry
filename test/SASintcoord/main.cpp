#include <unordered_map>

#include <CppLibrary/chemistry.hpp>

#include <tchem/intcoord.hpp>

c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);

void test_sasic() {
    tchem::IC::SASICSet set("whatever", "whatever", "SAS.in");

    std::vector<std::string> prefixes({"E", "N", "B", "I", "NB", "NI", "BI", "NBI"});
    std::vector<std::vector<at::Tensor>> SASgeoms(8);
    for (size_t i = 0; i < 8; i++) {
        std::string file = prefixes[i] + ".xyz";
        CL::chem::xyz<double> geom(file, true);
        std::vector<double> coords = geom.coords();
        at::Tensor r = at::from_blob(coords.data(), coords.size(), top);
        at::Tensor q = set.IntCoordSet::operator()(r);
        SASgeoms[i] = set(q);
    }

    double difference = 0.0;
    std::unordered_map<std::string, size_t> prefix2index{
        {"E", 0}, {"N", 1}, {"B", 2}, {"I", 3},
        {"NB", 4}, {"NI", 5}, {"BI", 6}, {"NBI", 7}
    };
    std::cout << "\nSymmetry adapted and scaled internal coordinate:\n";
    // irreducible 1: totally symmetric
    for (const std::string & prefix : {"N", "B", "I", "NB", "NI", "BI", "NBI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][0] - SASgeoms[0][0]).norm().item<double>();
    }
    std::cout << "irreducible 1: " << difference << '\n';
    // irreducible 2: asymmetric N
    difference = 0.0;
    difference += (SASgeoms[0][1] + SASgeoms[1][1]).norm().item<double>();
    for (const std::string & prefix : {"B", "I", "BI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][1] - SASgeoms[0][1]).norm().item<double>();
    }
    for (const std::string & prefix : {"NB", "NI", "NBI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][1] - SASgeoms[1][1]).norm().item<double>();
    }
    std::cout << "irreducible 2: " << difference << '\n';
    // irreducible 3: asymmetric B
    difference = 0.0;
    difference += (SASgeoms[0][2] + SASgeoms[2][2]).norm().item<double>();
    for (const std::string & prefix : {"N", "I", "NI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][2] - SASgeoms[0][2]).norm().item<double>();
    }
    for (const std::string & prefix : {"NB", "BI", "NBI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][2] - SASgeoms[2][2]).norm().item<double>();
    }
    std::cout << "irreducible 3: " << difference << '\n';
    // irreducible 4: asymmetric I
    difference = 0.0;
    difference += (SASgeoms[0][3] + SASgeoms[3][3]).norm().item<double>();
    for (const std::string & prefix : {"N", "B", "NB"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][3] - SASgeoms[0][3]).norm().item<double>();
    }
    for (const std::string & prefix : {"NI", "BI", "NBI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][3] - SASgeoms[3][3]).norm().item<double>();
    }
    std::cout << "irreducible 4: " << difference << '\n';
    // irreducible 5: asymmetric N & B
    difference = 0.0;
    difference += (SASgeoms[0][4] + SASgeoms[1][4]).norm().item<double>();
    for (const std::string & prefix : {"I", "NB", "NBI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][4] - SASgeoms[0][4]).norm().item<double>();
    }
    for (const std::string & prefix : {"B", "NI", "BI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][4] - SASgeoms[1][4]).norm().item<double>();
    }
    std::cout << "irreducible 5: " << difference << '\n';
    // irreducible 6: asymmetric N & I
    difference = 0.0;
    difference += (SASgeoms[0][5] + SASgeoms[1][5]).norm().item<double>();
    for (const std::string & prefix : {"B", "NI", "NBI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][5] - SASgeoms[0][5]).norm().item<double>();
    }
    for (const std::string & prefix : {"I", "NB", "BI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][5] - SASgeoms[1][5]).norm().item<double>();
    }
    std::cout << "irreducible 6: " << difference << '\n';
    // irreducible 7: asymmetric B & I
    difference = 0.0;
    difference += (SASgeoms[0][6] + SASgeoms[2][6]).norm().item<double>();
    for (const std::string & prefix : {"N", "BI", "NBI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][6] - SASgeoms[0][6]).norm().item<double>();
    }
    for (const std::string & prefix : {"I", "NB", "NI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][6] - SASgeoms[2][6]).norm().item<double>();
    }
    std::cout << "irreducible 7: " << difference << '\n';
    // irreducible 8: asymmetric N & B & I
    difference = 0.0;
    difference += (SASgeoms[0][7] + SASgeoms[1][7]).norm().item<double>();
    for (const std::string & prefix : {"NB", "NI", "BI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][7] - SASgeoms[0][7]).norm().item<double>();
    }
    for (const std::string & prefix : {"B", "I", "NBI"}) {
        size_t index = prefix2index[prefix];
        difference += (SASgeoms[index][7] - SASgeoms[1][7]).norm().item<double>();
    }
    std::cout << "irreducible 8: " << difference << '\n';
}

void test_Jacobian() {
    tchem::IC::SASICSet set("whatever", "whatever", "SAS.in");

    CL::chem::xyz<double> geom("E.xyz", true);
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

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'SASintcoord'\n"
              << "Correct routines should print close to 0\n";

    test_sasic();
    test_Jacobian();
}