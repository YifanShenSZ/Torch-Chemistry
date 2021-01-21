// Additional general basic routine

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace tchem { namespace utility {

// Read a vector from file
at::Tensor read_vector(const std::string & file) {
    std::vector<double> data = CL::utility::read_vector(file);
    at::Tensor vector = at::from_blob(data.data(), data.size(), at::TensorOptions().dtype(torch::kFloat64));
    // Q: Why clone?
    // A: Because of from_blob, `vector` shares memory with `data`,
    //    who goes out of scope after this function call
    return vector.clone();
}

// Number of trainable network parameters
size_t NParameters(const std::vector<at::Tensor> & parameters) {
    size_t N = 0;
    for (auto & p : parameters) if (p.requires_grad()) N += p.numel();
    return N;
}

// Norm of the network parameter gradient
double NetGradNorm(const std::vector<at::Tensor> & parameters) {
    double norm = 0.0;
    for (auto & p : parameters) norm += p.grad().norm().item<double>() * p.grad().norm().item<double>();
    return std::sqrt(norm);
}

} // namespace utility
} // namespace tchem