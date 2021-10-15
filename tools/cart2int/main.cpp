#include <CppLibrary/argparse.hpp>
#include <CppLibrary/utility.hpp>
#include <CppLibrary/chemistry.hpp>

#include <tchem/intcoord.hpp>

argparse::ArgumentParser parse_args(const size_t & argc, const char ** & argv) {
    CL::utility::echo_command(argc, argv, std::cout);
    std::cout << '\n';
    argparse::ArgumentParser parser("cart2int: Convert geometry from Cartesian coordinate to internal coordinate");

    // required arguments
    parser.add_argument("-f","--format",   1, false, "internal coordinate definition format (Columbus7, default)");
    parser.add_argument("-i","--IC",       1, false, "internal coordinate definition file");
    parser.add_argument("-g","--geometry", 1, false, "input Cartesian coordinate geometry");

    // optional argument
    parser.add_argument("-o","--output", 1, true, "output internal coordinate (default = `geom`.int)");

    parser.parse_args(argc, argv);
    return parser;
}

int main(size_t argc, const char ** argv) {
    std::cout << "Convert geometry from Cartesian coordinate to symmetry adapted and scaled internal coordinate\n";
    argparse::ArgumentParser args = parse_args(argc, argv);
    CL::utility::show_time(std::cout);
    std::cout << '\n';

    std::string format = args.retrieve<std::string>("format");
    std::string IC     = args.retrieve<std::string>("IC");
    tchem::IC::IntCoordSet icset(format, IC);

    std::string geom_input = args.retrieve<std::string>("geometry");
    CL::chem::xyz<double> geom(geom_input, true);
    std::vector<double> coords = geom.coords();
    at::Tensor r = at::from_blob(coords.data(), coords.size(), at::TensorOptions().dtype(torch::kFloat64));
    at::Tensor q = icset(r);

    std::ofstream ofs;
    if (args.gotArgument("output")) ofs.open(args.retrieve<std::string>("output"));
    else ofs.open(geom_input + ".int");
    const double * p = q.data_ptr<double>();
    for (size_t j = 0; j < q.numel(); j++) ofs << std::fixed << std::setw(18) << std::setprecision(15) << p[j] << '\n';
    ofs.close();

    CL::utility::show_time(std::cout);
    std::cout << "Mission success\n";
}