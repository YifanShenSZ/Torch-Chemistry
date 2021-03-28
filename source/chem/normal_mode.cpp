#include <CppLibrary/utility.hpp>

#include <tchem/FORTRAN.hpp>

#include <tchem/chem/normal_mode.hpp>

namespace tchem { namespace chem {

CartNormalMode::CartNormalMode() {}
CartNormalMode::CartNormalMode(const std::vector<double> & _masses, const at::Tensor & _Hessian) {
    if (_Hessian.sizes().size() != 2) throw std::invalid_argument(
    "tchem::chem::CartNormalMode: Hessian must be a matrix");
    if (_Hessian.size(0) != _Hessian.size(1)) throw std::invalid_argument(
    "tchem::chem::CartNormalMode: Hessian must be a square matrix");
    if (3 * _masses.size() != _Hessian.size(0)) throw std::invalid_argument(
    "tchem::chem::CartNormalMode: inconsistent dimension between masses and Hessian");
    masses_  = _masses;
    Hessian_ = _Hessian.clone();
}
CartNormalMode::~CartNormalMode() {}

void CartNormalMode::kernel() {
    // Cartesian coordinate -> mass weighed Cartesian coordinate
    std::vector<double> sqrt_masses = masses_;
    for (double & sqrt_mass : sqrt_masses) sqrt_mass = sqrt(sqrt_mass);
    for (size_t i = 0; i < masses_.size(); i++) {
        Hessian_.slice(0, 3 * i, 3 * i + 3) /= sqrt_masses[i];
        Hessian_.slice(1, 3 * i, 3 * i + 3) /= sqrt_masses[i];
    }
    // Obtain frequency^2 and mass weighed normal modes
    at::Tensor raw_freq, raw_mode;
    std::tie(raw_freq, raw_mode) = Hessian_.symeig(true);
    raw_mode.transpose_(0, 1);
    // Rule out 6 translations and rotations
    std::vector<std::pair<double, size_t>> abs_indices(raw_freq.size(0));
    for (size_t i = 0; i < abs_indices.size(); i++) {
        abs_indices[i].first  = abs(raw_freq[i].item<double>());
        abs_indices[i].second = i;
    }
    std::sort(abs_indices.begin(), abs_indices.end());
    at::Tensor unsorted_freq = raw_freq.new_empty( raw_freq.size(0) - 6),
               unsorted_mode = raw_mode.new_empty({raw_mode.size(0) - 6, raw_mode.size(1)});
    for (size_t i = 0; i < unsorted_freq.size(0); i++) {
        const size_t & index = abs_indices[i + 6].second;
        unsorted_freq[i].copy_(raw_freq[index]);
        unsorted_mode[i].copy_(raw_mode[index]);
    }
    // frequency^2 -> frequency
    for (size_t i = 0; i < unsorted_freq.size(0); i++) {
        double frequency = unsorted_freq[i].item<double>();
        if (frequency > 0.0) unsorted_freq[i].fill_( sqrt( frequency));
        else                 unsorted_freq[i].fill_(-sqrt(-frequency));
    }
    // mass weighed normal modes -> normal modes
    for (size_t i = 0; i < masses_.size(); i++)
    unsorted_mode.slice(1, 3 * i, 3 * i + 3) /= sqrt_masses[i];
    // Sort frequency ascendingly and normal modes accordingly
    std::vector<std::pair<double, size_t>> freq_indices(unsorted_freq.size(0));
    for (size_t i = 0; i < unsorted_freq.size(0); i++) {
        freq_indices[i].first  = unsorted_freq[i].item<double>();
        freq_indices[i].second = i;
    }
    std::sort(freq_indices.begin(), freq_indices.end());
    frequency_ = unsorted_freq.new_empty(unsorted_freq.sizes());
    cartmode_  = unsorted_mode.new_empty(unsorted_mode.sizes());
    for (size_t i = 0; i < unsorted_freq.size(0); i++) {
        const size_t & index = freq_indices[i].second;
        frequency_[i].copy_(unsorted_freq[index]);
        cartmode_ [i].copy_(unsorted_mode[index]);
    }
    // done
    ready_ = true;
}

// Harmonic frequencies (negative if imaginary)
const at::Tensor & CartNormalMode::frequency() const {
    if (ready_) return frequency_;
    else throw CL::utility::not_ready("tchem::chem::CartNormalMode::frequency");
}
// Cartesian coordinate normal modes (normalized by mass metric)
const at::Tensor & CartNormalMode::cartmode() const {
    if (ready_) return cartmode_;
    else throw CL::utility::not_ready("tchem::chem::CartNormalMode::cartmode");
}





IntNormalMode::IntNormalMode() {}
IntNormalMode::IntNormalMode(const std::vector<double> & _masses, const at::Tensor & _Jacobian, const at::Tensor & _Hessian) {
    if (_Jacobian.sizes().size() != 2) throw std::invalid_argument(
    "tchem::chem::IntNormalMode: Jacobian must be a matrix");
    if (3 * _masses.size() != _Jacobian.size(1)) throw std::invalid_argument(
    "tchem::chem::IntNormalMode: inconsistent dimension between masses and Jacobian");
    if (_Hessian.sizes().size() != 2) throw std::invalid_argument(
    "tchem::chem::IntNormalMode: Hessian must be a matrix");
    if (_Hessian.size(0) != _Hessian.size(1)) throw std::invalid_argument(
    "tchem::chem::IntNormalMode: Hessian must be a square matrix");
    if (_Jacobian.size(0) != _Hessian.size(0)) throw std::invalid_argument(
    "tchem::chem::IntNormalMode: inconsistent dimension between Jacobian and Hessian");
    masses_   = _masses;
    Jacobian_ = _Jacobian.clone();
    Hessian_  = _Hessian .clone();
}
IntNormalMode::~IntNormalMode() {}

// Perform normal mode analysis, then observables are ready for fetching
void IntNormalMode::kernel() {
    // Obtain frequency^2 and normal modes
    at::Tensor inv_mass = Jacobian_.new_empty(3 * masses_.size());
    for (size_t i = 0; i < masses_.size(); i++) inv_mass.slice(0, 3 * i, 3 * i + 3).fill_(1.0 / masses_[i]);
    inv_mass = inv_mass.diag();
    at::Tensor G = Jacobian_.mm(inv_mass.mm(Jacobian_.transpose(0, 1)));
    std::tie(frequency_, intmode_) = at::dsygv(Hessian_, G, at::gv_type::type3, true);
    intmode_.transpose_(0, 1);
    // frequency^2 -> frequency
    for (size_t i = 0; i < frequency_.size(0); i++) {
        double frequency = frequency_[i].item<double>();
        if (frequency > 0.0) frequency_[i].fill_( sqrt( frequency));
        else                 frequency_[i].fill_(-sqrt(-frequency));
    }
    // Linv
    at::Tensor cholesky_G = G.cholesky();
    at::Tensor inv_G = at::cholesky_inverse(cholesky_G);
    Linv_ = intmode_.mm(inv_G);
    // intmode -> cartmode
    // L . dQ = dq = J . dr, where Q denotes internal coordinate normal mode
    // So L contains internal coordinate normal mode in each column
    //    J^g . L contains Cartesian coordinate normal mode in each column
    at::Tensor JJT = Jacobian_.mm(Jacobian_.transpose(0, 1));
    at::Tensor cholesky_JJT = JJT.cholesky();
    at::Tensor inv_JJT = at::cholesky_inverse(cholesky_JJT);
    at::Tensor JgT = inv_JJT.mm(Jacobian_);
    cartmode_ = intmode_.mm(JgT);
    // done
    ready_ = true;
    // Reference: https://en.wikipedia.org/wiki/GF_method
}

// Internal coordinate normal modes (normalized by G^-1 metric)
const at::Tensor & IntNormalMode::intmode() const {
    if (ready_) return intmode_;
    else throw CL::utility::not_ready("tchem::chem::IntNormalMode::intmode");
}
// intmode_^-1
const at::Tensor & IntNormalMode::Linv() const {
    if (ready_) return Linv_;
    else throw CL::utility::not_ready("tchem::chem::IntNormalMode::Linv");
}





SANormalMode::SANormalMode() {}
SANormalMode::SANormalMode(const std::vector<double> & _masses, const std::vector<at::Tensor> & _Jacobians, const std::vector<at::Tensor> & _Hessians) {
    if (_Jacobians.size() != _Hessians.size()) throw std::invalid_argument(
    "tchem::chem::SANormalMode: inconsistent number of irreducibles between Jacobian and Hessian");
    for (size_t i = 0; i < _Jacobians.size(); i++) {
        if (_Jacobians[i].sizes().size() != 2) throw std::invalid_argument(
        "tchem::chem::SANormalMode: Jacobian must be a matrix");
        if (3 * _masses.size() != _Jacobians[i].size(1)) throw std::invalid_argument(
        "tchem::chem::SANormalMode: inconsistent dimension between masses and Jacobian");
        if (_Hessians[i].sizes().size() != 2) throw std::invalid_argument(
        "tchem::chem::SANormalMode: Hessian must be a matrix");
        if (_Hessians[i].size(0) != _Hessians[i].size(1)) throw std::invalid_argument(
        "tchem::chem::SANormalMode: Hessian must be a square matrix");
        if (_Jacobians[i].size(0) != _Hessians[i].size(0)) throw std::invalid_argument(
        "tchem::chem::SANormalMode: inconsistent dimension between Jacobian and Hessian");
    }
    masses_ = _masses;
    Jacobians_.resize(_Jacobians.size());
     Hessians_.resize( _Hessians.size());
    for (size_t i = 0; i < _Jacobians.size(); i++) {
        Jacobians_[i] = _Jacobians[i].clone();
         Hessians_[i] =  _Hessians[i].clone();
    }
}
SANormalMode::~SANormalMode() {}

// Perform normal mode analysis, then observables are ready for fetching
void SANormalMode::kernel() {
    // Obtain M^-1
    at::Tensor inv_mass = Jacobians_[0].new_empty(3 * masses_.size());
    for (size_t i = 0; i < masses_.size(); i++) inv_mass.slice(0, 3 * i, 3 * i + 3).fill_(1.0 / masses_[i]);
    inv_mass = inv_mass.diag();
    // Loop over irreducibles
    size_t NIrreds = Jacobians_.size();
    frequencies_.resize(NIrreds);
    intmodes_   .resize(NIrreds);
    Linvs_      .resize(NIrreds);
    cartmodes_  .resize(NIrreds);
    for (size_t i = 0; i < NIrreds; i++) {
        // Obtain frequency^2 and normal modes
        at::Tensor G = Jacobians_[i].mm(inv_mass.mm(Jacobians_[i].transpose(0, 1)));
        std::tie(frequencies_[i], intmodes_[i]) = at::dsygv(Hessians_[i], G, at::gv_type::type3, true);
        intmodes_[i].transpose_(0, 1);
        // frequency^2 -> frequency
        for (size_t j = 0; j < frequencies_[i].size(0); j++) {
            double frequency = frequencies_[i][j].item<double>();
            if (frequency > 0.0) frequencies_[i][j].fill_( sqrt( frequency));
            else                 frequencies_[i][j].fill_(-sqrt(-frequency));
        }
        // Linv
        at::Tensor cholesky_G = G.cholesky();
        at::Tensor inv_G = at::cholesky_inverse(cholesky_G);
        Linvs_[i] = intmodes_[i].mm(inv_G);
        // intmode -> cartmode
        // L . dQ = dq = J . dr, where Q denotes internal coordinate normal mode
        // So L contains internal coordinate normal mode in each column
        //    J^g . L contains Cartesian coordinate normal mode in each column
        at::Tensor JJT = Jacobians_[i].mm(Jacobians_[i].transpose(0, 1));
        at::Tensor cholesky_JJT = JJT.cholesky();
        at::Tensor inv_JJT = at::cholesky_inverse(cholesky_JJT);
        at::Tensor JgT = inv_JJT.mm(Jacobians_[i]);
        cartmodes_[i] = intmodes_[i].mm(JgT);
    }
    // done
    ready_ = true;
    // Reference: https://en.wikipedia.org/wiki/GF_method
}

// Harmonic frequencies (negative if imaginary)
const std::vector<at::Tensor> & SANormalMode::frequencies() const {
    if (ready_) return frequencies_;
    else throw CL::utility::not_ready("tchem::chem::SANormalMode::frequencies");
}
// Internal coordinate normal modes (normalized by (J . M^-1 . J^T)^-1 metric)
const std::vector<at::Tensor> & SANormalMode::intmodes() const {
    if (ready_) return intmodes_;
    else throw CL::utility::not_ready("tchem::chem::SANormalMode::intmodes");
}
// intmode_^-1
const std::vector<at::Tensor> & SANormalMode::Linvs() const {
    if (ready_) return Linvs_;
    else throw CL::utility::not_ready("tchem::chem::SANormalMode::Linvs");
}
// Cartesian coordinate normal modes
const std::vector<at::Tensor> & SANormalMode::cartmodes() const {
    if (ready_) return cartmodes_;
    else throw CL::utility::not_ready("tchem::chem::SANormalMode::cartmodes");
}

} // namespace chem
} // namespace tchem