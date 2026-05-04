module QuantumGPs

using AbstractGPs
using Distributions
using KernelFunctions
using LinearAlgebra
using Random
using Statistics

export PrecomputedIndexKernel,
       psdify,
       sample_pauli_mean,
       pauli_mean_var,
       noisy_sum_Z,
       shotvar_sumZ,
       noisy_sum_from_S,
       add_noise_2_kernel,
       add_noise_2_kernel_sum,
       noisy_xxyyzz_from_true,
       xxyyzz_shot_se_from_true,
       kernel_scaling_factor,
       linear_diagonal_indices,
       linear_antidiagonal_indices,
       qgp_fit_predict,
       gp_predict_xxz

# ==============================================================================
# Precomputed-kernel interface
# ==============================================================================

"""
    PrecomputedIndexKernel(K)

KernelFunctions-compatible wrapper for a precomputed kernel matrix `K`.
Inputs to the kernel are integer indices selecting rows and columns of `K`.
"""
struct PrecomputedIndexKernel{T<:AbstractMatrix} <: KernelFunctions.Kernel
    K::T
end

"""
    KernelFunctions.kernelmatrix(k, x, y; obsdim=1)

Return the submatrix `k.K[x, y]` for integer index vectors `x` and `y`.
"""
function KernelFunctions.kernelmatrix(
    k::PrecomputedIndexKernel,
    x::AbstractVector{<:Integer},
    y::AbstractVector{<:Integer};
    obsdim::Integer = 1,
)
    return k.K[x, y]
end

"""
    KernelFunctions.kernelmatrix_diag(k, x)

Return the diagonal entries of the precomputed kernel selected by `x`.
"""
function KernelFunctions.kernelmatrix_diag(
    k::PrecomputedIndexKernel,
    x::AbstractVector{<:Integer},
)
    return diag(k.K)[x]
end

"""
    k(i, j)

Scalar kernel evaluation for integer indices.
"""
(k::PrecomputedIndexKernel)(i::Integer, j::Integer) = k.K[i, j]

# ==============================================================================
# Positive-semidefinite regularization
# ==============================================================================

"""
    psdify(K; mode=:shift, eps=0.0, σ²_kernel_entries=NaN, return_as_diag_shift=false)

Symmetrize a kernel matrix and regularize it to be positive semidefinite.

Supported modes are:

- `:shift`: add the minimal diagonal shift needed to lift the smallest
  eigenvalue to `eps`.
- `:trim`: clamp eigenvalues below `eps` and reconstruct the matrix.
- `:semicircle`: add the Wigner-semircle-inspired shift
  `2sqrt(size(K,1)) * sqrt(σ²_kernel_entries)`, falling back to the exact
  minimal shift if the estimate is insufficient.

When `return_as_diag_shift=true`, return only the diagonal shift. This is
useful when a GP package should receive the regularizer as observation noise
rather than through a modified kernel matrix.
"""
function psdify(
    K::AbstractMatrix;
    mode::Symbol = :shift,
    eps::Real = 0.0,
    σ²_kernel_entries::Real = NaN,
    return_as_diag_shift::Bool = false,
)
    A = Matrix((K + K') / 2)
    n = size(A, 1)
    size(A, 2) == n || throw(ArgumentError("K must be square."))

    F = eigen(Hermitian(A))
    λmin = minimum(real.(F.values))
    λshift = 0.0

    if mode == :shift
        λshift = max(eps - λmin, 0.0)

    elseif mode == :trim
        return_as_diag_shift && throw(ArgumentError(":trim mode cannot be represented as a scalar diagonal shift."))
        λtrim = max.(real.(F.values), eps)
        return Matrix(F.vectors * Diagonal(λtrim) * F.vectors')

    elseif mode == :semicircle
        isnan(σ²_kernel_entries) && throw(ArgumentError("σ²_kernel_entries must be provided in :semicircle mode."))
        σ²_kernel_entries >= 0 || throw(ArgumentError("σ²_kernel_entries must be non-negative."))

        λshift = 2 * sqrt(n) * sqrt(Float64(σ²_kernel_entries))
        if λmin + λshift < eps
            fallback_shift = eps - λmin
            @warn "Semicircle shift was insufficient; using the exact minimal PSD shift instead." λshift fallback_shift λmin eps
            λshift = fallback_shift
        end

    else
        throw(ArgumentError("Unsupported PSD regularization mode: $mode."))
    end

    return return_as_diag_shift ? λshift : Matrix(A + λshift * I)
end

# ==============================================================================
# Shot-noise models
# ==============================================================================

"""
    sample_pauli_mean(μ, shots; rng=Random.default_rng())

Simulate `shots` projective measurements of a Pauli observable with true
expectation value `μ ∈ [-1, 1]`, and return the sample mean of ±1 outcomes.

Passing `shots=Inf` returns the noiseless value.
"""
@inline function sample_pauli_mean(μ::Real, shots; rng::AbstractRNG = Random.default_rng())
    shots == Inf && return Float64(μ)
    shots > 0 || throw(ArgumentError("shots must be positive."))

    nshots = Int(shots)
    μc = clamp(Float64(μ), -1.0, 1.0)
    p = 0.5 * (1 + μc)
    nplus = rand(rng, Binomial(nshots, p))

    return (2 * nplus - nshots) / nshots
end

"""
    pauli_mean_var(μ, shots)

Variance of the sample mean produced by [`sample_pauli_mean`](@ref).
"""
@inline function pauli_mean_var(μ::Real, shots)
    shots == Inf && return 0.0
    shots > 0 || throw(ArgumentError("shots must be positive."))

    μc = clamp(Float64(μ), -1.0, 1.0)
    return (1 - μc^2) / Int(shots)
end

"""
    noisy_sum_Z(S_true, n, shots; model=:upper_bound, rng=Random.default_rng())

Add shot noise to the expectation value
`S_true = ⟨∑_{q=1}^n Z_q⟩` estimated from `shots` measurements.

Available models:

- `:upper_bound`: Gaussian approximation using the support bound
  `Var(Ŝ) ≤ n² / shots`.
- `:iid_equal_mean`: assume independent qubits with identical marginal mean
  `S_true / n`, giving `Var(Ŝ) = n(1 - (S_true/n)^2) / shots`.
- `:binomial_exact`: exact binomial sampler under the same iid equal-mean
  assumption.
"""
function noisy_sum_Z(
    S_true::Real,
    n::Integer,
    shots;
    model::Symbol = :upper_bound,
    rng::AbstractRNG = Random.default_rng(),
)
    n > 0 || throw(ArgumentError("n must be positive."))
    shots == Inf && return clamp(Float64(S_true), -Float64(n), Float64(n))
    shots > 0 || throw(ArgumentError("shots must be positive."))

    nshots = Int(shots)
    S = clamp(Float64(S_true), -Float64(n), Float64(n))

    if model == :upper_bound
        return rand(rng, Normal(S, n / sqrt(nshots)))

    elseif model == :iid_equal_mean
        μ = S / n
        σ² = (n / nshots) * max(1 - μ^2, 0.0)
        return rand(rng, Normal(S, sqrt(σ²)))

    elseif model == :binomial_exact
        μ = S / n
        p = clamp(0.5 * (1 + μ), 0.0, 1.0)
        nplus = rand(rng, Binomial(nshots * n, p))
        return (2 * nplus - nshots * n) / nshots

    else
        throw(ArgumentError("Unsupported sum-Z noise model: $model."))
    end
end

"""
    shotvar_sumZ(S_true, n, shots; model=:upper_bound)

Return the variance of the estimator modeled by [`noisy_sum_Z`](@ref).
"""
function shotvar_sumZ(
    S_true::Real,
    n::Integer,
    shots;
    model::Symbol = :upper_bound,
)
    n > 0 || throw(ArgumentError("n must be positive."))
    shots == Inf && return 0.0
    shots > 0 || throw(ArgumentError("shots must be positive."))

    nshots = Int(shots)
    S = clamp(Float64(S_true), -Float64(n), Float64(n))

    if model == :upper_bound
        return n^2 / nshots
    elseif model == :iid_equal_mean
        μ = S / n
        return (n / nshots) * max(1 - μ^2, 0.0)
    else
        throw(ArgumentError("Unsupported sum-Z variance model: $model."))
    end
end

"""
    noisy_sum_from_S(S_true, support_size, shots; model=:upper_bound, rng=Random.default_rng())

Gaussian shot-noise model for a sum estimator with `support_size` terms. The
`:upper_bound` model uses variance `support_size / shots`. The `:equal_terms`
model assumes all terms have the same mean `S_true / support_size`.
"""
function noisy_sum_from_S(
    S_true::Real,
    support_size::Integer,
    shots;
    model::Symbol = :upper_bound,
    rng::AbstractRNG = Random.default_rng(),
)
    support_size > 0 || throw(ArgumentError("support_size must be positive."))
    shots == Inf && return Float64(S_true)
    shots > 0 || throw(ArgumentError("shots must be positive."))

    nshots = Int(shots)
    σ² = if model == :upper_bound
        support_size / nshots
    elseif model == :equal_terms
        μ = clamp(Float64(S_true) / support_size, -1.0, 1.0)
        (support_size / nshots) * (1 - μ^2)
    else
        throw(ArgumentError("Unsupported sum noise model: $model."))
    end

    return rand(rng, Normal(Float64(S_true), sqrt(max(σ², 0.0))))
end

"""
    add_noise_2_kernel(K, nshots; nq, m, model=:upper_bound, rng=Random.default_rng())

Add symmetric shot noise to a precomputed kernel matrix.

The default `:upper_bound` model uses the Bell-measurement normal
approximation with variance `binomial(2nq, m) / nshots` per entry and clamps
the noisy estimates to the theoretical overlap bound
`binomial(nq, m ÷ 2)`.

Returns `(K_noisy, K_entry_variances)`.
"""
function add_noise_2_kernel(
    K::AbstractMatrix,
    nshots;
    nq = nothing,
    m = nothing,
    model::Symbol = :upper_bound,
    rng::AbstractRNG = Random.default_rng(),
)
    nshots == Inf || nshots > 0 || throw(ArgumentError("nshots must be positive."))

    d = size(K, 1)
    size(K, 2) == d || throw(ArgumentError("K must be square."))

    K_noisy = Matrix{Float64}(undef, d, d)
    K_vars = Matrix{Float64}(undef, d, d)

    if model == :upper_bound
        (nq === nothing || m === nothing) && throw(ArgumentError("nq and m are required for model=:upper_bound."))

        support_size = binomial(2 * nq, m)
        σ²_est = nshots == Inf ? 0.0 : support_size / Int(nshots)
        σ_est = sqrt(σ²_est)
        max_overlap = binomial(nq, m ÷ 2)

        maximum(abs.(K)) <= max_overlap + 1e-10 ||
            throw(ArgumentError("Kernel entries exceed the theoretical maximum overlap magnitude $max_overlap."))

        @inbounds for i in 1:d
            diag_estimate = nshots == Inf ? real(K[i, i]) : rand(rng, Normal(real(K[i, i]), σ_est))
            K_noisy[i, i] = clamp(diag_estimate, -max_overlap, max_overlap)
            K_vars[i, i] = σ²_est

            for j in (i + 1):d
                estimate = nshots == Inf ? real(K[i, j]) : rand(rng, Normal(real(K[i, j]), σ_est))
                estimate = clamp(estimate, -max_overlap, max_overlap)
                K_noisy[i, j] = estimate
                K_noisy[j, i] = estimate
                K_vars[i, j] = σ²_est
                K_vars[j, i] = σ²_est
            end
        end

    elseif model == :pauli_binomial
        @inbounds for i in 1:d
            K_noisy[i, i] = sample_pauli_mean(real(K[i, i]), nshots; rng=rng)
            K_vars[i, i] = pauli_mean_var(K_noisy[i, i], nshots)

            for j in (i + 1):d
                estimate = sample_pauli_mean(real(K[i, j]), nshots; rng=rng)
                variance = pauli_mean_var(estimate, nshots)
                K_noisy[i, j] = estimate
                K_noisy[j, i] = estimate
                K_vars[i, j] = variance
                K_vars[j, i] = variance
            end
        end

    else
        throw(ArgumentError("Unsupported kernel shot-noise model: $model."))
    end

    return K_noisy, K_vars
end

"""
    add_noise_2_kernel_sum(K, support_size, nshots; model=:upper_bound, rng=Random.default_rng())

Add symmetric Gaussian shot noise to a kernel whose entries are estimated as
sums over `support_size` Pauli-like terms. Returns the noisy kernel and a
matrix of entrywise variances.
"""
function add_noise_2_kernel_sum(
    K::AbstractMatrix,
    support_size::Integer,
    nshots;
    model::Symbol = :upper_bound,
    rng::AbstractRNG = Random.default_rng(),
)
    support_size > 0 || throw(ArgumentError("support_size must be positive."))
    nshots == Inf || nshots > 0 || throw(ArgumentError("nshots must be positive."))

    d = size(K, 1)
    size(K, 2) == d || throw(ArgumentError("K must be square."))

    K_noisy = Matrix{Float64}(undef, d, d)
    σ² = nshots == Inf ? 0.0 : support_size / Int(nshots)
    K_vars = fill(σ², d, d)

    @inbounds for i in 1:d
        K_noisy[i, i] = noisy_sum_from_S(real(K[i, i]), support_size, nshots; model=model, rng=rng)
        for j in (i + 1):d
            estimate = noisy_sum_from_S(real(K[i, j]), support_size, nshots; model=model, rng=rng)
            K_noisy[i, j] = estimate
            K_noisy[j, i] = estimate
        end
    end

    return K_noisy, K_vars
end

"""
    noisy_xxyyzz_from_true(true_xxyyzz, nshots; rng=Random.default_rng())

Shot-noise a theoretical value of `⟨XX + YY + ZZ⟩` assuming it is obtained
from a SWAP-test estimate through `XX + YY + ZZ = 2SWAP - 1`.
"""
function noisy_xxyyzz_from_true(
    true_xxyyzz::Real,
    nshots;
    rng::AbstractRNG = Random.default_rng(),
)
    nshots == Inf && return Float64(true_xxyyzz)
    nshots > 0 || throw(ArgumentError("nshots must be positive."))

    shots = Int(nshots)
    swap_true = clamp((Float64(true_xxyyzz) + 1) / 2, -1.0, 1.0)
    p0 = clamp((1 + swap_true) / 2, 0.0, 1.0)
    k0 = rand(rng, Binomial(shots, p0))
    swap_hat = 2 * (k0 / shots) - 1

    return 2 * swap_hat - 1
end

"""
    xxyyzz_shot_se_from_true(true_xxyyzz, nshots)

Standard error of the estimator modeled by [`noisy_xxyyzz_from_true`](@ref).
"""
function xxyyzz_shot_se_from_true(true_xxyyzz::Real, nshots)
    nshots == Inf && return 0.0
    nshots > 0 || throw(ArgumentError("nshots must be positive."))

    swap_true = clamp((Float64(true_xxyyzz) + 1) / 2, -1.0, 1.0)
    return 2 * sqrt((1 - swap_true^2) / Int(nshots))
end

# ==============================================================================
# Small experiment helpers
# ==============================================================================

"""
    kernel_scaling_factor(n, m) -> Float64

Scaling factor used to convert the raw order-`m` overlap-sum estimator into
the normalized QGP kernel used in the XXZ experiments.
"""
kernel_scaling_factor(n::Integer, m::Integer) = (n * factorial(m)) / (2n)^m

"""
    linear_diagonal_indices(n)

Column-major linear indices of the diagonal of an `n × n` array.
"""
linear_diagonal_indices(n::Integer) = [i + (i - 1) * n for i in 1:n]

"""
    linear_antidiagonal_indices(n)

Column-major linear indices of the anti-diagonal of an `n × n` array.
"""
linear_antidiagonal_indices(n::Integer) = [i + (n - i) * n for i in 1:n]

# ==============================================================================
# Main QGP routine
# ==============================================================================

"""
    qgp_fit_predict(kernel, y_train, train_idxs; obs_σ²=0.0, ker_σ²=0.0,
                    ker_ε=1e-10, psd_method=:semicircle)

Fit a Gaussian process using a precomputed kernel and return predictions on
all kernel indices.

Arguments:

- `kernel`: full precomputed kernel matrix.
- `y_train`: noisy training observations.
- `train_idxs`: integer indices of the training points.
- `obs_σ²`: scalar or vector of training-label noise variances.
- `ker_σ²`: scalar or matrix of kernel-entry noise variances. The current
  implementation reduces this to its mean when computing the PSD shift.
- `ker_ε`: small baseline diagonal regularizer.
- `psd_method`: PSD regularization strategy passed to [`psdify`](@ref).

Returns a named tuple with fields `mean`, `var`, `std`, `gp_posterior_obj`,
and `kernel_reg_shift`.
"""
function qgp_fit_predict(
    kernel::AbstractMatrix,
    y_train::AbstractVector,
    train_idxs::AbstractVector{<:Integer};
    obs_σ²::Union{Real, AbstractVector{<:Real}} = 0.0,
    ker_σ²::Union{Real, AbstractMatrix{<:Real}} = 0.0,
    ker_ε::Real = 1e-10,
    psd_method::Symbol = :semicircle,
)
    ntp = length(y_train)
    ntp == length(train_idxs) || throw(ArgumentError("y_train and train_idxs must have the same length."))

    kernel_sym = Matrix((kernel + kernel') / 2)
    kernel_noise_variance = ker_σ² isa Real ? Float64(ker_σ²) : Float64(mean(ker_σ²))

    kernel_reg_shift = fill(Float64(ker_ε), ntp)
    kernel_reg_shift .+= obs_σ²

    psd_shift = psdify(
        kernel_sym[train_idxs, train_idxs];
        mode = psd_method,
        return_as_diag_shift = true,
        σ²_kernel_entries = kernel_noise_variance,
    )
    kernel_reg_shift .+= psd_shift

    kernel_object = PrecomputedIndexKernel(kernel_sym)
    prior = GP(kernel_object)
    finite_train = prior(train_idxs, kernel_reg_shift)
    post = posterior(finite_train, y_train)

    pred_dist = post(1:size(kernel_sym, 1))
    μ_pred = mean(pred_dist)
    σ²_pred = max.(var(pred_dist), 0.0)

    return (
        mean = μ_pred,
        var = σ²_pred,
        std = sqrt.(σ²_pred),
        gp_posterior_obj = post,
        kernel_reg_shift = kernel_reg_shift,
    )
end

"""
    gp_predict_xxz(n, true_outputs, true_overlaps2, ntp, nshots_obs, nshots_ker;
                   tps=nothing, obs_noise_model=:upper_bound,
                   psd_method=:shift, rng=Random.default_rng(), verbose=false)

Convenience wrapper for the XXZ QGP experiments. It constructs a noisy kernel,
selects training points, samples noisy observations, fits the GP, and returns
predictions together with the sampled training data.

Returns `(μ_all, σ_all, y_obs, σ_obs, train_idxs, test_idxs, kernel_used)`.
"""
function gp_predict_xxz(
    n::Integer,
    true_outputs::AbstractVector,
    true_overlaps2::AbstractMatrix,
    ntp::Integer,
    nshots_obs,
    nshots_ker;
    tps = nothing,
    obs_noise_model::Symbol = :upper_bound,
    psd_method::Symbol = :shift,
    rng::AbstractRNG = Random.default_rng(),
    verbose::Bool = false,
)
    dsize = length(true_outputs)
    size(true_overlaps2, 1) >= dsize || throw(ArgumentError("Kernel matrix is smaller than the output vector."))
    size(true_overlaps2, 2) >= dsize || throw(ArgumentError("Kernel matrix is smaller than the output vector."))

    K2 = @view true_overlaps2[1:dsize, 1:dsize]
    support_size = binomial(2n, 2)
    kernel_raw, kernel_vars_raw = add_noise_2_kernel_sum(K2, support_size, nshots_ker; rng=rng)

    scale = kernel_scaling_factor(n, 2)
    kernel = kernel_raw .* scale
    kernel_vars = kernel_vars_raw .* scale^2

    train_idxs = if isnothing(tps)
        idxs = Int[]
        for k in 1:ntp
            push!(idxs, isodd(k) ? (k + 1) ÷ 2 : dsize + 1 - k ÷ 2)
        end
        unique(sort(idxs))
    else
        Int.(tps)
    end
    test_idxs = setdiff(collect(1:dsize), train_idxs)

    σ²_obs = [shotvar_sumZ(true_outputs[i], n, nshots_obs; model=obs_noise_model) for i in train_idxs]
    y_obs = [noisy_sum_Z(true_outputs[i], n, nshots_obs; model=obs_noise_model, rng=rng) for i in train_idxs]

    verbose && @info "Running qgp_fit_predict" dsize ntp nshots_obs nshots_ker psd_method

    fit = qgp_fit_predict(
        kernel,
        y_obs,
        train_idxs;
        obs_σ² = σ²_obs,
        ker_σ² = kernel_vars,
        ker_ε = 1e-10,
        psd_method = psd_method,
    )

    return fit.mean, fit.std, y_obs, sqrt.(σ²_obs), train_idxs, test_idxs, kernel
end

end # module QuantumGPs
