// Additional general basic routine

#ifndef tchem_utility_hpp
#define tchem_utility_hpp

#include <torch/torch.h>

namespace tchem { namespace utility {

// Read a vector from file
at::Tensor read_vector(const std::string & file);

// Number of trainable network parameters
size_t NParameters(const std::vector<at::Tensor> & parameters);

// 1-norm of the network parameter gradient
double NetGradNorm(const std::vector<at::Tensor> & parameters);

} // namespace utility
} // namespace tchem

#endif