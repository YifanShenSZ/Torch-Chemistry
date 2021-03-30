// Additional general basic routine

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace tchem { namespace utility {

// Convert an at::Tensor to a std::vector<double>
std::vector<double> tensor2vector(const at::Tensor & tensor) {
    at::Tensor vec_view = tensor.view(tensor.numel());
    std::vector<double> vector(vec_view.size(0));
    for (size_t i = 0; i < vector.size(); i++) vector[i] = vec_view[i].item<double>();
    return vector;
}

// Convert an at::Tensor to CL::utility::matrix<double>
CL::utility::matrix<double> tensor2matrix(const at::Tensor & tensor) {
    if (tensor.sizes().size() != 2) throw std::invalid_argument(
    "tchem::utility::tensor2matrix: tensor must be a matrix");
    CL::utility::matrix<double> matrix(tensor.size(0), tensor.size(1));
    for (size_t i = 0; i < matrix.size(0); i++)
    for (size_t j = 0; j < matrix.size(1); j++)
    matrix[i][j] = tensor[i][j].item<double>();
    return matrix;
}

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