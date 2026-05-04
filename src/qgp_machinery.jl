module QuantumGPs

using AbstractGPs
using KernelFunctions
using LinearAlgebra
using Statistics
using Distributions
using Random
using DoubleFloats

# ==============================================================================
# 1. Custom Kernel for Precomputed Matrices
# ==============================================================================
struct PrecomputedIndexKernel{T<:AbstractMatrix} <: KernelFunctions.Kernel
    K::T
end

# Fast path: build the whole kernel matrix for integer index inputs
KernelFunctions.kernelmatrix(k::PrecomputedIndexKernel,
                             x::AbstractVector{<:Integer},
                             y::AbstractVector{<:Integer}; obsdim=1) = k.K[x, y]

KernelFunctions.kernelmatrix_diag(k::PrecomputedIndexKernel,
                                  x::AbstractVector{<:Integer}) = view(diag(k.K), x)

# Scalar fallback
(k::PrecomputedIndexKernel)(i::Integer, j::Integer) = k.K[i, j]


# ==============================================================================
# 2. Helper Functions (PSD & Noise)
# ==============================================================================

"""
Enforce Positive Semi-Definiteness on a matrix.
"""
function psdify(K::AbstractMatrix; mode::Symbol=:shift, eps::Real=0.0, σ²_kernel_entries=NaN, return_as_diag_shift::Bool=false)
    A = (K + K')/2
    ntp = size(A, 1)
    F = eigen(A)                        # A = Q * Diagonal(λ) * Q'
    λshift = 0.0
    if mode == :shift
        λneg = minimum(F.values) - eps  
        if λneg < 0 
            λshift = -λneg  # only shift if there's a negative eigenvalue, and shift just enough to make it zero (plus a small eps for numerical stability)
        end
    elseif mode == :trim
        λtrim = max.(F.values, eps)             # clamp
        A = F.vectors * diagm(λtrim) * F.vectors'
        if return_as_diag_shift
            error("psdify :trim mode cannot be realized as a simple diagonal shift.")
        end
    elseif mode == :semicircle
        λtrim = F.values

        if isnan(σ²_kernel_entries)
            error("σ²_kernel_entries must be provided for semicircle mode")
        end

        λshift = 2*sqrt(ntp)*sqrt(σ²_kernel_entries)  # Wigner semi-circle shift

        # check if this guarantees PSD, otherwise shift more:
        λneg = minimum(F.values) 
        if λneg < -λshift
            println("Wigner semicircle shift $λshift insufficient ntp=$(ntp) σ²_k=$(σ²_kernel_entries), shift by most negative eigenvalue instead $(-λneg) + eps $(eps)")
            λshift = -λneg + abs(λshift + λneg + eps)
            println("Total shift applied: $λshift")
        end
    else
        error("Not Implemented yet")
    end

    if return_as_diag_shift
        return λshift
    else
        Kpsd = A + diagm(fill(λshift, ntp))
    end

    return Kpsd
end

"""
Simulate shot noise on the kernel matrix.
Note: Requires nq and m (qubit/measurement params) for the 'upper_bound' model logic.
"""
function add_noise_2_kernel(K, nshots; nq=nothing, m=nothing, model::Symbol=:upper_bound, rng=Random.default_rng())
    nshots <= 0 && error("nshots must be positive")
    d = size(K, 1)
    @assert size(K, 2) == d

    ker = Matrix{Float64}(undef, d, d)
    ker_vars = Matrix{Float64}(undef, d, d)

    if model == :upper_bound  # normal approximation of the kernel Bell measurement protocol
        size_of_module = binomial(2nq, m)
        σ²_est = size_of_module / nshots
        σ_est = sqrt(σ²_est)

        max_overlap_magnitude = binomial(nq, m ÷ 2)  # automatically floors (for odd m)
        #println("max overlap magnitude: $(max_overlap_magnitude)")
        @assert maximum(abs.(K)) <= max_overlap_magnitude + 1e-10 "Kernel entries exceed theoretical maximum overlap magnitude $(max_overlap_magnitude) with $(maximum(abs.(K)))"

        for i in 1:d
            ker[i, i] = rand(rng, Normal(K[i, i], σ_est))
            for j in (i + 1):d
                begin
                    ovp = rand(rng, Normal(K[i, j], σ_est))
                    ker[i, j] = ovp
                    ker[j, i] = ovp
                end
            end
        end

        ker = clamp.(ker, -max_overlap_magnitude, max_overlap_magnitude)  # guarantee estimates within theoretical bounds
        ker_vars = fill(σ²_est, d, d)
    elseif model == :pauli_binomial
        for i in 1:d
            ker[i, i] = sample_pauli_mean(K[i, i], nshots)
            for j in (i + 1):d
                begin
                    ovp = sample_pauli_mean(K[i, j], nshots)
                    ker[i, j] = ovp
                    ker[j, i] = ovp
                    ovp_var = pauli_mean_var(ovp, nshots)
                    ker_vars[i, j] = ovp_var
                    ker_vars[j, i] = ovp_var
                end
            end
        end
    else 
        error("Shot noise model $(model) not available")
    end
    return ker, ker_vars
end

# ----------------------------------------------------------------------
# Shot-noise samplers (projective Pauli measurements)
# ----------------------------------------------------------------------

"""
    sample_pauli_mean(μ, shots)

Simulate `shots` projective measurements of a Pauli observable with true
expectation value `μ ∈ [-1,1]` (outcomes ±1), and return the sample mean.
"""
@inline function sample_pauli_mean(μ::Real, shots)
    if shots == Inf
        return μ
    end
    shots <= 0 && error("shots must be positive, not $shots")
    μc = clamp(Float64(μ), -1.0, 1.0)
    p = 0.5 * (1 + μc)
    k = rand(Binomial(shots, p))
    return (2k - shots) / shots
end

"""
    pauli_mean_var(μ, shots)

Variance of the sample mean produced by `sample_pauli_mean`.
"""
@inline function pauli_mean_var(μ::Real, shots)
    shots <= 0 && error("shots must be positive")
    μc = clamp(Float64(μ), -1.0, 1.0)
    return (1 - μc^2) / shots
end


# ==============================================================================
# 3. Main Routine
# ==============================================================================

"""
# Returns
A NamedTuple containing:
- `mean`: Posterior mean vector
- `var`: Posterior variance vector (corrected for shot noise)
- `gp_posterior_obj`: The AbstractGPs posterior object
- `kernel_reg_shift`: The total diagonal shift applied to the training sub-block of the kernel (useful for diagnostics)
"""
function qgp_fit_predict(
    kernel::AbstractMatrix,
    y_train::AbstractVector,
    train_idxs::AbstractVector{<:Integer};
    obs_σ²::Union{Real, AbstractVector{<:Real}} = 0.0,
    ker_σ²::Union{Real, AbstractMatrix{<:Real}} = 0.0,
    ker_ε::Real = 1e-10,  # small diagonal shift to ensure numerical stability
    psd_method::Symbol = :semicircle
)
    ntp = length(y_train)
    @assert ntp == length(train_idxs) "Length of y_train must match number of training indices"

    ### Pre-processing Kernel ###
    # Accumulator for (training) diagonal shift:
    kernel_reg_shift = fill(ker_ε, ntp)
    kernel_reg_shift .+= obs_σ²

    # Ensure symmetry
    kernel = (kernel .+ kernel') ./ 2
    
    if isempty(train_idxs)
        @warn "No training indices provided. The GP will return the prior predictive distribution."
        k_obj = PrecomputedIndexKernel(kernel)
        post = GP(k_obj)  # no data, post = prior
    else
        ker_σ² = mean(ker_σ²)  # Simplified since current noise model yields uniform kernel noise variance. For more complex models (or real noise), this should be handled entrywise.

        # PSD Correction (On Training Sub-block only)
        psd_shift = psdify(kernel[train_idxs, train_idxs]; 
                            mode=psd_method, return_as_diag_shift=true, σ²_kernel_entries=ker_σ²)
        kernel_reg_shift .+= psd_shift  # accumulate total diagonal shift for training points

        ### Construct GP  ###
        k_obj = PrecomputedIndexKernel(kernel)
        f_prior = GP(k_obj)
        fx = f_prior(train_idxs, kernel_reg_shift) # handles the (K + R) inversion correctly while keeping 'k_obj' clean.
        post = posterior(fx, y_train)  # train GP
    end
    pred_dist = post(1:size(kernel, 1))
    μ_pred = mean(pred_dist)
    σ²_pred = abs.(var(pred_dist))  # ensure non-negative variance from GP prediction
    
    # Add Shot Noise Uncertainty
    # The GP assumes the kernel is ground truth. We know the kernel itself is noisy.
    # We add the estimated kernel variance to the predictive variance.
    # σ²_pred = abs.(σ²_pred) .+ ker_σ²
    
    # We predict on all indices (which can include training points)
    return (
        mean = μ_pred,
        var = σ²_pred,
        std = sqrt.(σ²_pred),
        gp_posterior_obj = post,
        kernel_reg_shift = kernel_reg_shift,
    )
end

export psdify, add_noise_2_kernel, sample_pauli_mean, pauli_mean_var, qgp_fit_predict

end # module
