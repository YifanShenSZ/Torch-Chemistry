#ifndef tchem_IC_IntCoordSet_hpp
#define tchem_IC_IntCoordSet_hpp

#include <tchem/intcoord/IntCoord.hpp>

namespace tchem { namespace IC {

// A set of internal coordinates
class IntCoordSet {
    private:
        // Internal coordinates constituting the set
        std::vector<IntCoord> intcoords_;
    public:
        IntCoordSet();
        // file format (Columbus7, default), internal coordinate definition file
        IntCoordSet(const std::string & format, const std::string & file);
        ~IntCoordSet();

        const std::vector<IntCoord> & intcoords() const;

        size_t size() const;
        // Read-only reference to an internal coordinate
        const IntCoord & operator[](const size_t & index) const;

        void print(std::ofstream & ofs, const std::string & format) const;

        // Return q given r
        at::Tensor operator()(const at::Tensor & r) const;
        // Return q and J given r
        std::tuple<at::Tensor, at::Tensor> compute_IC_J(const at::Tensor & r) const;
        // Return q and J and K given r
        std::tuple<at::Tensor, at::Tensor, at::Tensor> compute_IC_J_K(const at::Tensor & r) const;

        // Return matrix M who satisfies M . ▽r = ▽q
        at::Tensor gradient_cart2int_matrix(const at::Tensor & r) const;
        // Return internal coordinate gradient given r and Cartesian coordinate gradient
        at::Tensor gradient_cart2int(const at::Tensor & r, const at::Tensor & cartgrad) const;
        // Return Cartesian coordinate gradient given r and internal coordinate gradient
        at::Tensor gradient_int2cart(const at::Tensor & r, const at::Tensor & intgrad) const;

        // Return internal coordinate Hessian given r and Cartesian coordinate Hessian
        at::Tensor Hessian_cart2int(const at::Tensor & r, const at::Tensor & cartgrad, const at::Tensor & cartHess) const;
        // Return Cartesian coordinate Hessian given r and internal coordinate gradient and Hessian
        at::Tensor Hessian_int2cart(const at::Tensor & r, const at::Tensor & intgrad, const at::Tensor & intHess) const;
};

} // namespace IC
} // namespace tchem

#endif