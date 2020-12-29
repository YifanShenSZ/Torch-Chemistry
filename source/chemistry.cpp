#include <torch/torch.h>

#include <tchem/linalg.hpp>

namespace tchem { namespace chemistry {

bool check_degeneracy(const double & threshold, const at::Tensor & energy) {
    bool deg = false;
    for (size_t i = 0; i < energy.numel() - 1; i++) {
        if (energy[i+1].item<double>() - energy[i].item<double>() < threshold) {
            deg = true;
            break;
        }
    }
    return deg;
}

// Transform adiabatic energy (H) and gradient (dH) to composite representation
void composite_representation(at::Tensor & H, at::Tensor & dH) {
    at::Tensor dHdH = LA::sy3matdotmul(dH, dH);
    at::Tensor eigval, eigvec;
    std::tie(eigval, eigvec) = dHdH.symeig(true, true);
    dHdH = eigvec.transpose(0, 1);
    H = eigvec.transpose(0, 1).mm(H.diag().mm(eigvec));
    LA::UT_A3_U_(dH, eigvec);
}    

/*
Matrix off-diagonal elements do not have determinate phase, because
the eigenvectors defining a representation have indeterminate phase difference
*/

// For a certain number of electronic states (NStates),
// there are 2^(NStates-1) possibilities in total
// The user input matrix indicates the base case and is excluded from trying,
// so possible_phases[NStates-2].size() = 2^(NStates-1) - 1
// possible_phases[NStates-2][i] contains one of the phases of NStates electronic states
// where true means -, false means +,
// the phase of the last state is always arbitrarily assigned to +,
// so possible_phases[NStates-2][i].size() == NStates-1
std::vector<std::vector<std::vector<bool>>> possible_phases;

// Prepare possible_phases for up to NStates electronic states
void initialize_phase_fixing(const size_t & NStates) {
    possible_phases.resize(NStates-1);
    for (size_t N = 0; N < NStates-1; N++) {
        // Unchanged case is exculded
        possible_phases[N].resize(1 << (N+1) - 1);
        for (auto & phase : possible_phases[N]) phase.resize(N+1);
        possible_phases[N][0][0] = true;
        for (size_t j = 1; j < N+1; j++) possible_phases[N][0][j] = false;
        for (size_t i = 1; i < possible_phases[N].size(); i++) {
            for (size_t j = 0; j < N+1; j++) possible_phases[N][i][j] = possible_phases[N][i-1][j];
            size_t count = 0;
            while(possible_phases[N][i][count]) {
                possible_phases[N][i][count] = false;
                count++;
            }
            possible_phases[N][i][count] = true;
        }
    }
}

// Fix M by minimizing || M - ref ||_F^2
void fix(at::Tensor & M, const at::Tensor & ref) {
    size_t NStates = M.size(0);
    const auto & possibilities = possible_phases[NStates - 2];
    double change_min = 0.0;
    int     phase_min = -1;
    if (M.sizes().size() > 2) {
        std::vector<int64_t> dim_vec(M.sizes().size()-2);
        for (size_t i = 0; i < dim_vec.size(); i++) dim_vec[i] = i+2;
        c10::IntArrayRef sum_dim(dim_vec.data(), dim_vec.size());
        at::Tensor diff = (M - ref).pow_(2).sum(sum_dim);
        if (sum_dim.size() < 3) sum_dim = {};
        else {
            dim_vec.resize(M.sizes().size()-4);
            for (size_t i = 0; i < dim_vec.size(); i++) dim_vec[i] = i;
            sum_dim = c10::IntArrayRef(dim_vec.data(), dim_vec.size());
        }
        // Try out phase possibilities
        for (int phase = 0; phase < possibilities.size(); phase++) {
            at::Tensor change = M.new_zeros({});
            for (size_t i = 0; i < NStates; i++) {
                for (size_t j = i+1; j < NStates-1; j++)
                if (possibilities[phase][i] != possibilities[phase][j])
                change += (-M[i][j] - ref[i][j]).pow_(2).sum(sum_dim) - diff[i][j];
                if (possibilities[phase][i])
                change += (-M[i][NStates-1] - ref[i][NStates-1]).pow_(2).sum(sum_dim) - diff[i][NStates-1];
            }
            if (change.item<double>() < change_min) {
                change_min = change.item<double>();
                phase_min  = phase;
            }
        }
    }
    else {
        at::Tensor diff = (M - ref).pow_(2);
        // Try out phase possibilities
        for (int phase = 0; phase < possibilities.size(); phase++) {
            at::Tensor change = M.new_zeros({});
            for (size_t i = 0; i < NStates; i++) {
                for (size_t j = i+1; j < NStates-1; j++)
                if (possibilities[phase][i] != possibilities[phase][j])
                change += (-M[i][j] - ref[i][j]).pow_(2) - diff[i][j];
                if (possibilities[phase][i])
                change += (-M[i][NStates-1] - ref[i][NStates-1]).pow_(2) - diff[i][NStates-1];
            }
            if (change.item<double>() < change_min) {
                change_min = change.item<double>();
                phase_min  = phase;
            }
        }
    }
    // Modify M if the best phase is different from the input
    if (phase_min > -1) {
        for (size_t i = 0; i < NStates; i++) {
            for (size_t j = i+1; j < NStates-1; j++)
            if (possibilities[phase_min][i] != possibilities[phase_min][j])
            M[i][j] = -M[i][j];
            if (possibilities[phase_min][i])
            M[i][NStates-1] = -M[i][NStates-1];
        }
    }
}
// Fix M1 and M2 by minimizing weight * || M1 - ref1 ||_F^2 + || M2 - ref2 ||_F^2
void fix(at::Tensor & M1, at::Tensor & M2, const at::Tensor & ref1, const at::Tensor & ref2, const double & weight) {
    size_t NStates = M1.size(0);
    const auto & possibilities = possible_phases[NStates - 2];
    double change_min = 0.0;
    int     phase_min = -1;
    if (M1.sizes().size() > 2 && M2.sizes().size() > 2) {
        std::vector<int64_t> dim_vec1(M1.sizes().size()-2);
        for (size_t i = 0; i < dim_vec1.size(); i++) dim_vec1[i] = i+2;
        c10::IntArrayRef sum_dim1(dim_vec1.data(), dim_vec1.size());
        std::vector<int64_t> dim_vec2(M2.sizes().size()-2);
        for (size_t i = 0; i < dim_vec2.size(); i++) dim_vec2[i] = i+2;
        c10::IntArrayRef sum_dim2(dim_vec2.data(), dim_vec2.size());
        at::Tensor diff = weight * (M1 - ref1).pow_(2).sum(sum_dim1) + (M2 - ref2).pow_(2).sum(sum_dim2);
        if (sum_dim1.size() < 3) sum_dim1 = {};
        else {
            dim_vec1.resize(M1.sizes().size()-4);
            for (size_t i = 0; i < dim_vec1.size(); i++) dim_vec1[i] = i;
            sum_dim1 = c10::IntArrayRef(dim_vec1.data(), dim_vec1.size());
        }
        if (sum_dim2.size() < 3) sum_dim2 = {};
        else {
            dim_vec2.resize(M2.sizes().size()-4);
            for (size_t i = 0; i < dim_vec2.size(); i++) dim_vec2[i] = i;
            sum_dim2 = c10::IntArrayRef(dim_vec2.data(), dim_vec2.size());
        }
        // Try out phase possibilities
        for (int phase = 0; phase < possibilities.size(); phase++) {
            at::Tensor change = M1.new_zeros({});
            for (size_t i = 0; i < NStates; i++) {
                for (size_t j = i+1; j < NStates-1; j++)
                if (possibilities[phase][i] != possibilities[phase][j])
                change += weight * (-M1[i][j] - ref1[i][j]).pow_(2).sum(sum_dim1)
                          + (-M2[i][j] - ref2[i][j]).pow_(2).sum(sum_dim2)
                          - diff[i][j];
                if (possibilities[phase][i])
                change += weight * (-M1[i][NStates-1] - ref1[i][NStates-1]).pow_(2).sum(sum_dim1)
                          + (-M2[i][NStates-1] - ref2[i][NStates-1]).pow_(2).sum(sum_dim2)
                          - diff[i][NStates-1];
            }
            if (change.item<double>() < change_min) {
                change_min = change.item<double>();
                phase_min  = phase;
            }
        }
    }
    else if (M1.sizes().size() > 2) {
        std::vector<int64_t> dim_vec1(M1.sizes().size()-2);
        for (size_t i = 0; i < dim_vec1.size(); i++) dim_vec1[i] = i+2;
        c10::IntArrayRef sum_dim1(dim_vec1.data(), dim_vec1.size());
        at::Tensor diff = weight * (M1 - ref1).pow_(2).sum(sum_dim1) + (M2 - ref2).pow_(2);
        if (sum_dim1.size() < 3) sum_dim1 = {};
        else {
            dim_vec1.resize(M1.sizes().size()-4);
            for (size_t i = 0; i < dim_vec1.size(); i++) dim_vec1[i] = i;
            sum_dim1 = c10::IntArrayRef(dim_vec1.data(), dim_vec1.size());
        }
        // Try out phase possibilities
        for (int phase = 0; phase < possibilities.size(); phase++) {
            at::Tensor change = M1.new_zeros({});
            for (size_t i = 0; i < NStates; i++) {
                for (size_t j = i+1; j < NStates-1; j++)
                if (possibilities[phase][i] != possibilities[phase][j])
                change += weight * (-M1[i][j] - ref1[i][j]).pow_(2).sum(sum_dim1)
                          + (-M2[i][j] - ref2[i][j]).pow_(2)
                          - diff[i][j];
                if (possibilities[phase][i])
                change += weight * (-M1[i][NStates-1] - ref1[i][NStates-1]).pow_(2).sum(sum_dim1)
                          + (-M2[i][NStates-1] - ref2[i][NStates-1]).pow_(2)
                          - diff[i][NStates-1];
            }
            if (change.item<double>() < change_min) {
                change_min = change.item<double>();
                phase_min  = phase;
            }
        }
    }
    else if (M2.sizes().size() > 2) {
        std::vector<int64_t> dim_vec2(M2.sizes().size()-2);
        for (size_t i = 0; i < dim_vec2.size(); i++) dim_vec2[i] = i+2;
        c10::IntArrayRef sum_dim2(dim_vec2.data(), dim_vec2.size());
        at::Tensor diff = weight * (M1 - ref1).pow_(2) + (M2 - ref2).pow_(2).sum(sum_dim2);
        if (sum_dim2.size() < 3) sum_dim2 = {};
        else {
            dim_vec2.resize(M2.sizes().size()-4);
            for (size_t i = 0; i < dim_vec2.size(); i++) dim_vec2[i] = i;
            sum_dim2 = c10::IntArrayRef(dim_vec2.data(), dim_vec2.size());
        }
        // Try out phase possibilities
        for (int phase = 0; phase < possibilities.size(); phase++) {
            at::Tensor change = M1.new_zeros({});
            for (size_t i = 0; i < NStates; i++) {
                for (size_t j = i+1; j < NStates-1; j++)
                if (possibilities[phase][i] != possibilities[phase][j])
                change += weight * (-M1[i][j] - ref1[i][j]).pow_(2)
                          + (-M2[i][j] - ref2[i][j]).pow_(2).sum(sum_dim2)
                          - diff[i][j];
                if (possibilities[phase][i])
                change += weight * (-M1[i][NStates-1] - ref1[i][NStates-1]).pow_(2)
                          + (-M2[i][NStates-1] - ref2[i][NStates-1]).pow_(2).sum(sum_dim2)
                          - diff[i][NStates-1];
            }
            if (change.item<double>() < change_min) {
                change_min = change.item<double>();
                phase_min  = phase;
            }
        }
    }
    else {
        at::Tensor diff = weight * (M1 - ref1).pow_(2) + (M2 - ref2).pow_(2);
        // Try out phase possibilities
        for (int phase = 0; phase < possibilities.size(); phase++) {
            at::Tensor change = M1.new_zeros({});
            for (size_t i = 0; i < NStates; i++) {
                for (size_t j = i+1; j < NStates-1; j++)
                if (possibilities[phase][i] != possibilities[phase][j])
                change += weight * (-M1[i][j] - ref1[i][j]).pow_(2)
                          + (-M2[i][j] - ref2[i][j]).pow_(2)
                          - diff[i][j];
                if (possibilities[phase][i])
                change += weight * (-M1[i][NStates-1] - ref1[i][NStates-1]).pow_(2)
                          + (-M2[i][NStates-1] - ref2[i][NStates-1]).pow_(2)
                          - diff[i][NStates-1];
            }
            if (change.item<double>() < change_min) {
                change_min = change.item<double>();
                phase_min  = phase;
            }
        }
    }
    // Modify M1 and M2 if the best phase is different from the input
    if (phase_min > -1) {
        for (size_t i = 0; i < NStates; i++) {
            for (size_t j = i+1; j < NStates-1; j++)
            if (possibilities[phase_min][i] != possibilities[phase_min][j]) {
                M1[i][j] = -M1[i][j];
                M2[i][j] = -M2[i][j];
            }
            if (possibilities[phase_min][i]) {
                M1[i][NStates-1] = -M1[i][NStates-1];
                M2[i][NStates-1] = -M2[i][NStates-1];
            }
        }
    }
}

} // namespace chemistry
} // namespace TS