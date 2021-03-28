#include <CppLibrary/chemistry.hpp>

#include <tchem/intcoord.hpp>

#include <tchem/chem/normal_mode.hpp>

int main() {
    std::cout << "This is a test program on Torch-Chemistry module 'normal mode'\n"
              << "Correct routines should print close to 0\n";

    c10::TensorOptions top = at::TensorOptions().dtype(torch::kFloat64);

    tchem::IC::IntCoordSet set("whatever", "whatever");
    CL::chem::xyz_mass<double> geom("geom");
    std::vector<double> coords = geom.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), top);
    at::Tensor q, J;
    std::tie(q, J) = set.compute_IC_J(r);

    at::Tensor inthess = q.new_empty({q.size(0), q.size(0)});
    std::ifstream ifs; ifs.open("hessian");
    for (size_t i = 0; i < q.size(0); i++)
    for (size_t j = 0; j < 6; j++) {
        double dbletemp;
        ifs >> dbletemp; inthess[i][8 * j + 0] = dbletemp;
        ifs >> dbletemp; inthess[i][8 * j + 1] = dbletemp;
        ifs >> dbletemp; inthess[i][8 * j + 2] = dbletemp;
        ifs >> dbletemp; inthess[i][8 * j + 3] = dbletemp;
        ifs >> dbletemp; inthess[i][8 * j + 4] = dbletemp;
        ifs >> dbletemp; inthess[i][8 * j + 5] = dbletemp;
        ifs >> dbletemp; inthess[i][8 * j + 6] = dbletemp;
        ifs >> dbletemp; inthess[i][8 * j + 7] = dbletemp;
    }
    ifs.close();
    // The internal coordinate and vibration routines of Columbus use weird unit:
    //     energy in 10^-18 J, length in A (to be continued)
    inthess /= 4.35974417; // 1 Hatree = 4.35974417 * 10^-18 J
    inthess.slice(0, 0, 8)   /= 1.8897261339212517;
    inthess.slice(1, 0, 8)   /= 1.8897261339212517;
    inthess.slice(0, 16, 19) /= 1.8897261339212517;
    inthess.slice(1, 16, 19) /= 1.8897261339212517;
    inthess.slice(0, 27, 32) /= 1.8897261339212517;
    inthess.slice(1, 27, 32) /= 1.8897261339212517;
    inthess.slice(0, 39, 41) /= 1.8897261339212517;
    inthess.slice(1, 39, 41) /= 1.8897261339212517;
    at::Tensor intgrad = q.new_zeros(q.size(0));
    at::Tensor carthess = set.Hessian_int2cart(r, intgrad, inthess);

    tchem::chem::CartNormalMode cartvib(geom.masses(), carthess);
    cartvib.kernel();
    
    tchem::chem::IntNormalMode intvib(geom.masses(), J, inthess);
    intvib.kernel();

    at::Tensor sections = at::tensor({27, 21});
    std::vector<at::Tensor> Js(2), inthesses(2);
    Js[0] = J.slice(0, 0, 27);
    Js[1] = J.slice(0, 27);
    inthesses[0] = inthess.slice(0, 0, 27).slice(1, 0, 27);
    inthesses[1] = inthess.slice(0, 27).slice(1, 27);
    tchem::chem::SANormalMode SAvib(geom.masses(), Js, inthesses);
    SAvib.kernel();
    at::Tensor SAfreq, indices;
    std::tie(SAfreq, indices) = at::cat(SAvib.frequencies()).sort();
    at::Tensor unsorted_SAcartmode = at::cat(SAvib.cartmodes());
    at::Tensor SAcartmode = unsorted_SAcartmode.new_empty(unsorted_SAcartmode.sizes());
    for (size_t i = 0; i < indices.size(0); i++) {
        int64_t index = indices[i].item<int64_t>();
        SAcartmode[i].copy_(unsorted_SAcartmode[index]);
    }

    std::cout << "\nVibrational frequency: "
              << (cartvib.frequency() - intvib.frequency()).norm().item<double>() << ' '
              << (cartvib.frequency() - SAfreq).norm().item<double>() << '\n';

    std::cout << "\nNormal modes: "
              << (at::abs(intvib.cartmode()) - at::abs(SAcartmode)).norm().item<double>() << '\n';

    // Although not far, normal modes produced by Cartesian analysis differ from internal ones
    // However, other kinds of tests (e.g. normalization to mass metric) are passed
    // So maybe they are fine to differ?
}