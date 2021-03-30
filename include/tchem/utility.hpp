// Additional general basic routine

#ifndef tchem_utility_hpp
#define tchem_utility_hpp

#include <torch/torch.h>

#include <CppLibrary/utility.hpp>

namespace tchem { namespace utility {

// Convert an at::Tensor to a std::vector<double>
std::vector<double> tensor2vector(const at::Tensor & tensor);

// Convert an at::Tensor to CL::utility::matrix<double>
CL::utility::matrix<double> tensor2matrix(const at::Tensor & tensor);

// Read a vector from file
at::Tensor read_vector(const std::string & file);

// Number of trainable network parameters
size_t NParameters(const std::vector<at::Tensor> & parameters);

// 1-norm of the network parameter gradient
double NetGradNorm(const std::vector<at::Tensor> & parameters);

} // namespace utility
} // namespace tchem

#endif