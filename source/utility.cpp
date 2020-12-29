// Additional general basic routine

#include <torch/torch.h>

namespace tchem {

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

} // namespace tchem