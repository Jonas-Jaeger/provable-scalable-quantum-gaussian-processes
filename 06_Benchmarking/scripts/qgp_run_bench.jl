#!/usr/bin/env julia
# GP vs TN comparison driver (with multi-run averaging & prediction saving)
#
# This script is an optimized, non-notebook version of `qgp_run_bench.ipynb`.
#
# It:
#   * loads precomputed data for a given (n, m)
#   * runs GP, Hamiltonian learning, and parametric baselines
#   * sweeps over numbers of observation / prediction shots and training points
#   * averages metrics over `nruns` independent noisy runs
#   * saves:
#       - a summary JLD2 file with mean/std metrics
#       - per-run prediction files (GP / HL / param) in a folder hierarchy
#
# Usage (inside the scripts directory, so that ../data/ exists):
#   julia --threads auto qgp_run_bench.jl 25 2 [nruns]
#
# If `nruns` is omitted, it defaults to the environment variable `NRUNS` (or 1).
#
# The script expects files:
#   ../data/majo_n<n>_m<m>/{times,true_module_projs,true_outputs,true_overlaps,true_obs_coeffs}.npy

using LinearAlgebra
using Statistics
using Random
using Distributions
using Combinatorics
using NPZ
using JLD2
using AbstractGPs
using KernelFunctions
using Base.Threads
using Dierckx

#import KernelFunctions: kernelmatrix
# New script with the machinery to perform QGPs with noisy kernels
include("../../src/qgp_machinery.jl");
using .QuantumGPs


# TODO add active learning to benchmark!

# Toggle threaded GP over test points (default: false, for stability)
const USE_GP_THREADS = get(ENV, "USE_GP_THREADS", "false") == "true"


# ----------------------------------------------------------------------
# Simple Hamiltonian learning toy model and parametric baseline
# ----------------------------------------------------------------------

"""
    ham_learn(feat_vecs, true_coeffs, nshots_obs, nshots_pred)

Toy Hamiltonian learning:
  * Noisy estimate each coefficient with `nshots_obs`
  * Predict observables as a noisy dot product with `nshots_pred` per feature
Returns a vector of predictions for all time points.
"""
function ham_learn(feat_vecs, true_coeffs, nshots_obs, nshots_pred)
    # Each coefficient / feature entry is treated as a Pauli expectation value,
    # estimated from projective measurements (±1 outcomes).
    learnt_coeffs = [sample_pauli_mean(c, nshots_obs) for c in true_coeffs]
    preds = Vector{Float64}(undef, size(feat_vecs, 1))

    for (row_idx, row) in enumerate(eachrow(feat_vecs))
        noisy_feats = [sample_pauli_mean(comp, nshots_pred) for comp in row]
        preds[row_idx] = dot(learnt_coeffs, noisy_feats)
    end
    return preds
end


"""
    fit_least_squares(labels, X)

Return the least-squares solution `params` minimizing `‖X*params - labels‖₂²`.
"""
function fit_least_squares(labels, X, σ²_labels=0.0)
    if σ²_labels == 0.0
        params = X \ labels  # ordinary least squares
    elseif σ²_labels isa Number  # least squares with scalar ridge regularization
        params = (X'*X + σ²_labels * I) \ (X' * labels)
    else  # least squares with per-point regularization
        W = Diagonal(1.0 ./ σ²_labels)  # weight matrix is inverse of covariance
        params = (X' * W * X + I) \ (X' * W * labels)
    end
    return params
end

"""
    param_predict(true_outputs, feature_vecs,
                  nshots_obs_par, nshots_feats_par,
                  train_idxs_par, test_idxs_par,
                  use_single_precision)

Parametric linear regression baseline:
  * Noisy labels y_obs_par for training indices
  * Noisy features X with per-entry std = 1/sqrt(nshots_feats_par)
  * Fit least-squares, then return predictions on all points.
  Note that single precision mode maintains numerical stability fairness with GP: condition number of dual problem (GP) is squared of the condition numer of primal problem (param)
Returns:
  preds_par :: Vector
  y_obs_par :: Vector (noisy train labels)
  σ_obs_par :: Float64 (their per-point noise std)
"""
function param_predict(true_outputs::AbstractVector,
                       feature_vecs::AbstractMatrix,
                       nshots_obs_par::Int,
                       nshots_feats_par::Int,
                       train_idxs_par::AbstractVector{<:Integer},
                       test_idxs_par::AbstractVector{<:Integer},
                       use_single_precision::Bool = false)  # single 

    # Noisy labels via Pauli (±1) measurement shot noise
    y_obs_par = [sample_pauli_mean(true_outputs[i], nshots_obs_par) for i in train_idxs_par]

    # Keep a representative scalar noise std for compatibility with existing callers.
    # (True variance depends on the expectation value.)
    σ_obs_par = 1 / sqrt(nshots_obs_par)

    # Noisy features: each entry is a Pauli expectation value estimated from shots
    X = Matrix{Float64}(undef, size(feature_vecs, 1), size(feature_vecs, 2))
    for (k, row) in enumerate(eachrow(feature_vecs))
        @inbounds X[k, :] .= [sample_pauli_mean(comp, nshots_feats_par) for comp in row]
    end

    Xt = @view X[train_idxs_par, :]
    σ²_obs_par = [pauli_mean_var(true_outputs[i], nshots_obs_par) for i in train_idxs_par]  # noise variance per training point for ridge regularization (matches GP mean)
    
    if use_single_precision
        X = Float32.(X)
        Xt = Float32.(Xt)
        y_obs_par = Float32.(y_obs_par)
        σ²_obs_par = Float32.(σ²_obs_par)
    end
    
    params_hat = fit_least_squares(y_obs_par, Xt, σ²_obs_par)
    preds_par = X * params_hat

    return preds_par, y_obs_par, σ_obs_par
end

function basic_baselines_predict(ts, y_obs, train_idxs)
    # Caution: y_obs should already contain the noise
    t_train = ts[train_idxs]

    # 1. Constant Mean Prediction
    mean_val = mean(y_obs)
    mean_preds = fill(mean_val, length(ts))
    
    # 2. Linear Spline Interpolation (k=1)
    # bc="extrapolate" prevents bounds errors on test points outside the train range
    linear_itp = Spline1D(t_train, y_obs, k=1, bc="extrapolate")
    lin_preds = linear_itp(ts)  # Dierckx allows direct array evaluation, no dot needed
    
    # 3. Cubic Spline Interpolation (k=3)
    cubic_itp = Spline1D(t_train, y_obs, k=3, bc="extrapolate")
    cub_preds = cubic_itp(ts)
    
    return mean_preds, lin_preds, cub_preds
end

# ----------------------------------------------------------------------
# GP machinery
# ----------------------------------------------------------------------


"""
    gp_predict(true_outputs, true_overlaps, ntp,
               nshots_obs, nshots_ker, n, m; verbose=false)

Gaussian process regression using a *precomputed* kernel matrix:

  * Select `ntp` training points, evenly spaced in index space.
  * Build a noisy kernel `ker` by sampling each entry.
  * For each test index, build a (ntp+1)×(ntp+1) kernel including that point,
    project to PSD, and run a small GP to get mean and variance.

Returns
  μ         :: Vector{Float64}  (means at test indices)
  σ         :: Vector{Float64}  (stds at test indices, rescaled by binomial(2n,m))
  y_obs     :: Vector{Float64}  (noisy training labels)
  σ_obs     :: Float64          (training noise std)
  train_idxs:: Vector{Int}
  test_idxs :: Vector{Int}
"""
function gp_predict(true_outputs::AbstractVector,
                    true_overlaps::AbstractMatrix,
                    ntp::Int,
                    nshots_obs::Int,
                    nshots_ker::Int,
                    n::Int,
                    m::Int;
                    verbose::Bool = false)

    dsize = length(true_outputs)
    @assert size(true_overlaps, 1) >= dsize
    @assert size(true_overlaps, 2) >= dsize

    # Training / test split (unchanged)
    train_idxs = floor.(Int, collect(range(1, dsize; length = ntp)))
    train_idxs = unique(sort(train_idxs))
    test_idxs  = setdiff(collect(1:dsize), train_idxs)

    # Base kernel (exact overlaps restricted to dataset size)
    K = copy(true_overlaps[1:dsize, 1:dsize])

    ### Simulate noise
    # Add noise to observations on train indices
    Random.seed!(123)
    y_obs  = [sample_pauli_mean(true_outputs[i], nshots_obs) for i in train_idxs]
    σ²_obs = [pauli_mean_var(true_outputs[idx], nshots_obs) for idx in train_idxs]
    σ_obs = sqrt.(σ²_obs)

    # Add noise to kernel
    Random.seed!(42)
    ker, ker_vars = add_noise_2_kernel(copy(K), nshots_ker; nq=n, m=m)

    # Apply kernel scaling factor (after noise simulation!)
    kernel_factor = factorial(m) / (2n)^m
    ker .*= kernel_factor
    ker_vars .*= kernel_factor^2
    K .*= kernel_factor
    
    ### Learn GP and predict on test points
        qgp_res = qgp_fit_predict(ker, y_obs, train_idxs; obs_σ²=σ²_obs, ker_σ²=ker_vars, 
    ker_ε=1e-10, psd_method=:semicircle)
        μ_pred = qgp_res.mean[test_idxs]
        σ_pred = qgp_res.std[test_idxs]

        return μ_pred, σ_pred, y_obs, σ_obs, train_idxs, test_idxs
end


function gp_active_predict(true_outputs::AbstractVector,
                    true_overlaps::AbstractMatrix,
                    ntp::Int,
                    nshots_obs::Int,
                    nshots_ker::Int,
                    n::Int,
                    m::Int;
                    verbose::Bool = false)

    dsize = length(true_outputs)
    @assert size(true_overlaps, 1) >= dsize
    @assert size(true_overlaps, 2) >= dsize

    # Base kernel (exact overlaps restricted to dataset size)
    K = true_overlaps[1:dsize, 1:dsize]

    # Add noise to kernel
    Random.seed!(42)
    ker, ker_vars = add_noise_2_kernel(copy(K), nshots_ker; nq=n, m=m)

    # Apply kernel scaling factor (after noise simulation!)
    kernel_factor = factorial(m) / (2n)^m
    ker .*= kernel_factor
    ker_vars .*= kernel_factor^2
    K .*= kernel_factor

    train_idxs = Int[]
    test_idxs = collect(1:dsize)
    y_obs = Float64[]
    σ²_obs = Float64[]
    
    Random.seed!(123)
    for iter in 1:ntp
        ### Learn GP and predict on test points
        qgp_res = qgp_fit_predict(ker, y_obs, train_idxs; obs_σ²=σ²_obs, ker_σ²=ker_vars, 
        ker_ε=1e-10, psd_method=:semicircle)
        σ_pred = qgp_res.std[test_idxs]

        ### Next to query: max σ_pred
        if isempty(σ_pred)
            @warn "All points have been selected for training, but ntp=$ntp is larger than the number of available points. Ending active learning loop early."
            break
        end
        next_idx_rel = argmax(σ_pred)
        next_idx = test_idxs[next_idx_rel]  # map back to original index
        push!(train_idxs, next_idx)
        test_idxs = setdiff(collect(1:dsize), train_idxs)

        ### Simulate noise
        # Add noise to observations on train indices
        new_obs = sample_pauli_mean(true_outputs[next_idx], nshots_obs)
        new_σ²_obs = pauli_mean_var(true_outputs[next_idx], nshots_obs)
        push!(y_obs, new_obs)
        push!(σ²_obs, new_σ²_obs)
    end

    # Final refit with all points
    qgp_res = qgp_fit_predict(ker, y_obs, train_idxs; obs_σ²=σ²_obs, ker_σ²=ker_vars, 
                ker_ε=1e-10, psd_method=:semicircle)
    μ_pred = qgp_res.mean[test_idxs]
    σ_pred = qgp_res.std[test_idxs]

    return μ_pred, σ_pred, y_obs, sqrt.(σ²_obs), train_idxs, test_idxs

end



# ----------------------------------------------------------------------
# Comparison wrapper
# ----------------------------------------------------------------------

"""
    compare_gp2hl2par(ts, true_outputs, true_obs_coeffs, feature_vecs, true_overlaps,
                      ntp, nshots_obs, nshots_preds, n, m)

Run GP, Hamiltonian learning, and parametric baseline for a single choice
of (ntp, nshots_obs, nshots_preds).

Returns:
  gp_mu, gp_sigma, hl_preds, par_preds, train_idxs, test_idxs,
  mse_gp, relerr_gp, mse_hl, relerr_hl, mse_par, relerr_par
"""
function compare_gp2hl2par(ts::AbstractVector,
                           true_outputs::AbstractVector,
                           true_obs_coeffs::AbstractVector,
                           feature_vecs::AbstractMatrix,
                           true_overlaps::AbstractMatrix,
                           ntp::Int,
                           nshots_obs::Int,
                           nshots_preds::Int,
                           n::Int,
                           m::Int)

    dsize = length(true_outputs)

    # --- GP ---
    gp_mu, gp_sigma, y_obs, σ_obs, train_idxs, test_idxs =
        gp_predict(true_outputs, true_overlaps, ntp, nshots_obs, nshots_preds, n, m; verbose = false)

    mse_gp = mean((gp_mu .- true_outputs[test_idxs]) .^ 2)
    relerr_gp = mean(abs.(gp_mu .- true_outputs[test_idxs]) ./ abs.(true_outputs[test_idxs]))

    # --- Budget accounting (match notebook formulas) ---
    blkbox_gp_budget = ntp * nshots_obs
    prediction_gp_budget = (ntp * (ntp + 1) ÷ 2 + (ntp) * (dsize - ntp)) * nshots_preds  # another dsize - ntp for certainty predictions of test points but ignore here as uncertainty not compared/benchmarked
    println("GP Budget: obs=$(nshots_obs), prediction=$(nshots_preds)")
    println("GP metrics: mse=$(mse_gp), relerr=$(relerr_gp)")

    prediction_gp_active_budget = prediction_gp_budget + (dsize - ntp) * nshots_preds  # additional predictions for uncertainties required for active learning selection
    nshots_preds_active = round(Int, ((prediction_gp_budget ÷ nshots_preds) / (prediction_gp_active_budget ÷ nshots_preds)) * nshots_preds)  # rescaled prediction shots to fit active learning budget (numerical stability rewrite)

    blkbox_hl_ops = length(true_obs_coeffs)
    prediction_hl_ops = length(true_obs_coeffs) * dsize

    blkbox_hl_shots = ceil(Int, blkbox_gp_budget / blkbox_hl_ops)
    prediction_hl_shots = ceil(Int, prediction_gp_budget / prediction_hl_ops)

    # --- GP active learning ---
    println("GP active learning Budget: obs=$(nshots_obs), prediction=$(nshots_preds_active)")
    gp_active_mu, gp_active_sigma, y_active_obs, σ_active_obs, train_active_idxs, test_active_idxs =
        gp_active_predict(true_outputs, true_overlaps, ntp, nshots_obs, nshots_preds_active, n, m; verbose = false)

    mse_active_gp = mean((gp_active_mu .- true_outputs[test_active_idxs]) .^ 2)
    relerr_active_gp = mean(abs.(gp_active_mu .- true_outputs[test_active_idxs]) ./ abs.(true_outputs[test_active_idxs]))
    println("GP active learning metrics: mse=$(mse_active_gp), relerr=$(relerr_active_gp)")

    # --- Hamiltonian learning baseline ---
    hl_preds = ham_learn(feature_vecs, true_obs_coeffs, blkbox_hl_shots, prediction_hl_shots)
    mse_hl = mean((hl_preds[test_idxs] .- true_outputs[test_idxs]) .^ 2)
    relerr_hl = mean(abs.(hl_preds[test_idxs] .- true_outputs[test_idxs]) ./ abs.(true_outputs[test_idxs]))

    # --- Parametric baseline ---
    nshots_feats_par = ceil(Int, prediction_gp_budget / prod(size(feature_vecs)))
    println("Param Budget: obs=$(nshots_obs), feat=$(nshots_feats_par)")
    preds_par, y_obs_par, σ_obs_par =
        param_predict(true_outputs, feature_vecs, nshots_obs, nshots_feats_par, train_idxs, test_idxs)

    mse_par = mean((true_outputs[test_idxs] .- preds_par[test_idxs]) .^ 2)
    relerr_par = mean(abs.(preds_par[test_idxs] .- true_outputs[test_idxs]) ./ abs.(true_outputs[test_idxs]))
    println("Param metrics: mse=$(mse_par), relerr=$(relerr_par)")

    # --- Basic Baselines ---
    # We pass `y_obs` from the GP to ensure we are interpolating the exact same noisy data
    mean_preds, lin_preds, cub_preds = basic_baselines_predict(ts, y_obs, train_idxs)

    mse_mean = mean((mean_preds[test_idxs] .- true_outputs[test_idxs]) .^ 2)
    relerr_mean = mean(abs.(mean_preds[test_idxs] .- true_outputs[test_idxs]) ./ abs.(true_outputs[test_idxs]))

    mse_lin = mean((lin_preds[test_idxs] .- true_outputs[test_idxs]) .^ 2)
    relerr_lin = mean(abs.(lin_preds[test_idxs] .- true_outputs[test_idxs]) ./ abs.(true_outputs[test_idxs]))

    mse_cub = mean((cub_preds[test_idxs] .- true_outputs[test_idxs]) .^ 2)
    relerr_cub = mean(abs.(cub_preds[test_idxs] .- true_outputs[test_idxs]) ./ abs.(true_outputs[test_idxs]))

    mse_gp < mse_par && mse_gp < mse_hl && mse_gp < mse_active_gp && println("GP wins") 
    mse_active_gp < mse_par && mse_active_gp < mse_hl && mse_active_gp < mse_gp && println("GP (active learning) wins") 

    return y_obs, σ_obs, gp_mu, gp_sigma, hl_preds, preds_par, train_idxs, test_idxs,
           mse_gp, relerr_gp, mse_hl, relerr_hl, mse_par, relerr_par,
              gp_active_mu, gp_active_sigma, train_active_idxs, test_active_idxs, mse_active_gp, relerr_active_gp, y_active_obs, σ_active_obs,
              mean_preds, lin_preds, cub_preds, mse_mean, relerr_mean, mse_lin, relerr_lin, mse_cub, relerr_cub
end

# ----------------------------------------------------------------------
# Prediction saving
# ----------------------------------------------------------------------

"""
    save_run_predictions(pred_base_dir,
                         n, m, ntp, nshots_obs, nshots_preds, run,
                         ts, true_outputs,
                         train_idxs, test_idxs,
                         gp_mu, gp_sigma,
                         hl_preds, param_preds)

Save per-run predictions for all three protocols in a JLD2 file, in a
hierarchy:

  pred_base_dir/n{n}_m{m}/ntp_{ntp}/obs_{nshots_obs}_pred_{nshots_preds}/run_{run}.jld2
"""
function save_run_predictions(pred_base_dir::AbstractString,
                              n::Int, m::Int,
                              ntp::Int, nshots_obs::Int, nshots_preds::Int,
                              run::Int,
                              ts,
                              true_outputs,
                              train_idxs,
                              test_idxs,
                              y_obs,
                              σ_obs,
                              gp_mu,
                              gp_sigma,
                              hl_preds,
                              param_preds,
                              gp_active_mu, gp_active_sigma, train_active_idxs, test_active_idxs, y_active_obs, σ_active_obs,
                              mean_preds, lin_preds, cub_preds
                              )

    dir = joinpath(pred_base_dir,
                   "n$(n)_m$(m)",
                   "ntp_$(ntp)",
                   "obs_$(nshots_obs)_pred_$(nshots_preds)")
    isdir(dir) || mkpath(dir)

    filename = joinpath(dir, "run_$(run).jld2")

    pred_data = Dict(
        "n" => n,
        "m" => m,
        "ntp" => ntp,
        "nshots_obs" => nshots_obs,
        "nshots_preds" => nshots_preds,
        "run" => run,
        "ts" => ts,
        "true_outputs" => true_outputs,
        "train_idxs" => train_idxs,
        "y_obs" => y_obs,
        "σ_obs" => σ_obs,
        "test_idxs" => test_idxs,
        "gp_mu" => gp_mu,
        "gp_sigma" => gp_sigma,
        "hl_preds" => hl_preds,
        "param_preds" => param_preds,
        "gp_active_mu" => gp_active_mu,
        "gp_active_sigma" => gp_active_sigma,
        "train_active_idxs" => train_active_idxs,
        "test_active_idxs" => test_active_idxs,
        "y_active_obs" => y_active_obs,
        "σ_active_obs" => σ_active_obs,
        "mean_preds" => mean_preds, 
        "lin_preds" => lin_preds,   
        "cub_preds" => cub_preds    
    )

    JLD2.save_object(filename, pred_data)
end

# ----------------------------------------------------------------------
# run_comparison
# ----------------------------------------------------------------------

"""
    run_comparison(ts, true_outputs, true_obs_coeffs, feature_vecs, true_overlaps,
                   nshots_obs_list, nshots_preds_list, ntps, n, m, nruns;
                   pred_base_dir = "data/preds")

Sweep over:
  * observation shots `nshots_obs_list`
  * prediction shots `nshots_preds_list`
  * number of training points `ntps`
and average metrics over `nruns` runs.

Returns a Dict with:
  * ntps, nshots_obs_list, nshots_preds_list, nruns
  * metrics[ntp][key] where key ∈
      "gp_mse", "gp_mse_std",
      "gp_relerr", "gp_relerr_std",
      "hl_mse", "hl_mse_std",
      "hl_relerr", "hl_relerr_std",
      "param_mse", "param_mse_std",
      "param_relerr", "param_relerr_std"
"""
function run_comparison(ts::AbstractVector,
                        true_outputs::AbstractVector,
                        true_obs_coeffs::AbstractVector,
                        feature_vecs::AbstractMatrix,
                        true_overlaps::AbstractMatrix,
                        nshots_obs_list::AbstractVector{<:Integer},
                        nshots_preds_list::AbstractVector{<:Integer},
                        ntps::AbstractVector{<:Integer},
                        n::Int,
                        m::Int,
                        nruns::Int;
                        pred_base_dir::AbstractString = "data/preds")

    data = Dict{String, Any}()
    data["ntps"] = collect(ntps)
    data["nshots_obs_list"] = collect(nshots_obs_list)
    data["nshots_preds_list"] = collect(nshots_preds_list)
    data["nruns"] = nruns
    data["predictions_base_dir"] = pred_base_dir
    data["metrics"] = Dict{Int, Dict{String, Array{Float64, 2}}}()

    nobs = length(nshots_obs_list)
    npred = length(nshots_preds_list)

    for ntp in ntps
        println("=== Running ntp = $ntp over $nruns runs ===")
        flush(stdout)

        # Sums for mean / std
        gp_mse_sum        = zeros(Float64, nobs, npred)
        gp_mse_sqsum      = zeros(Float64, nobs, npred)
        gp_relerr_sum     = zeros(Float64, nobs, npred)
        gp_relerr_sqsum   = zeros(Float64, nobs, npred)

        gp_active_mse_sum        = zeros(Float64, nobs, npred)
        gp_active_mse_sqsum      = zeros(Float64, nobs, npred)
        gp_active_relerr_sum     = zeros(Float64, nobs, npred)
        gp_active_relerr_sqsum   = zeros(Float64, nobs, npred)

        hl_mse_sum        = zeros(Float64, nobs, npred)
        hl_mse_sqsum      = zeros(Float64, nobs, npred)
        hl_relerr_sum     = zeros(Float64, nobs, npred)
        hl_relerr_sqsum   = zeros(Float64, nobs, npred)

        par_mse_sum       = zeros(Float64, nobs, npred)
        par_mse_sqsum     = zeros(Float64, nobs, npred)
        par_relerr_sum    = zeros(Float64, nobs, npred)
        par_relerr_sqsum  = zeros(Float64, nobs, npred)

        mean_mse_sum      = zeros(Float64, nobs, npred)
        mean_mse_sqsum    = zeros(Float64, nobs, npred)
        mean_relerr_sum   = zeros(Float64, nobs, npred)
        mean_relerr_sqsum = zeros(Float64, nobs, npred)

        lin_mse_sum       = zeros(Float64, nobs, npred)
        lin_mse_sqsum     = zeros(Float64, nobs, npred)
        lin_relerr_sum    = zeros(Float64, nobs, npred)
        lin_relerr_sqsum  = zeros(Float64, nobs, npred)

        cub_mse_sum       = zeros(Float64, nobs, npred)
        cub_mse_sqsum     = zeros(Float64, nobs, npred)
        cub_relerr_sum    = zeros(Float64, nobs, npred)
        cub_relerr_sqsum  = zeros(Float64, nobs, npred)

        for run in 1:nruns
            println("  Run $run / $nruns for ntp = $ntp")
            flush(stdout)

            for (i, nshots_obs) in enumerate(nshots_obs_list)
                for (j, nshots_preds) in enumerate(nshots_preds_list)
                    y_obs, σ_obs, gp_mu, gp_sigma, hl_preds, par_preds, train_idxs, test_idxs,
                    mse_gp, relerr_gp, mse_hl, relerr_hl, mse_par, relerr_par,
                    gp_active_mu, gp_active_sigma, train_active_idxs, test_active_idxs, mse_active_gp, relerr_active_gp, y_active_obs, σ_active_obs,
                    mean_preds, lin_preds, cub_preds, mse_mean, relerr_mean, mse_lin, relerr_lin, mse_cub, relerr_cub =
                        compare_gp2hl2par(ts, true_outputs, true_obs_coeffs, feature_vecs,
                                          true_overlaps, ntp, nshots_obs, nshots_preds, n, m)

                    # Accumulate metrics
                    gp_mse_sum[i, j]       += mse_gp
                    gp_mse_sqsum[i, j]     += mse_gp^2
                    gp_relerr_sum[i, j]    += relerr_gp
                    gp_relerr_sqsum[i, j]  += relerr_gp^2

                    hl_mse_sum[i, j]       += mse_hl
                    hl_mse_sqsum[i, j]     += mse_hl^2
                    hl_relerr_sum[i, j]    += relerr_hl
                    hl_relerr_sqsum[i, j]  += relerr_hl^2

                    par_mse_sum[i, j]      += mse_par
                    par_mse_sqsum[i, j]    += mse_par^2
                    par_relerr_sum[i, j]   += relerr_par
                    par_relerr_sqsum[i, j] += relerr_par^2

                    gp_active_mse_sum[i, j]       += mse_active_gp
                    gp_active_mse_sqsum[i, j]     += mse_active_gp^2
                    gp_active_relerr_sum[i, j]    += relerr_active_gp
                    gp_active_relerr_sqsum[i, j]  += relerr_active_gp^2

                    mean_mse_sum[i, j]      += mse_mean
                    mean_mse_sqsum[i, j]    += mse_mean^2
                    mean_relerr_sum[i, j]   += relerr_mean
                    mean_relerr_sqsum[i, j] += relerr_mean^2

                    lin_mse_sum[i, j]       += mse_lin
                    lin_mse_sqsum[i, j]     += mse_lin^2
                    lin_relerr_sum[i, j]    += relerr_lin
                    lin_relerr_sqsum[i, j]  += relerr_lin^2

                    cub_mse_sum[i, j]       += mse_cub
                    cub_mse_sqsum[i, j]     += mse_cub^2
                    cub_relerr_sum[i, j]    += relerr_cub
                    cub_relerr_sqsum[i, j]  += relerr_cub^2


                    # Save predictions for this (ntp, obs, pred, run)
                    save_run_predictions(pred_base_dir,
                                         n, m, ntp, nshots_obs, nshots_preds, run,
                                         ts, true_outputs,
                                         train_idxs, test_idxs,
                                         y_obs, σ_obs,
                                         gp_mu, gp_sigma,
                                         hl_preds, par_preds,
                                         gp_active_mu, gp_active_sigma, train_active_idxs, test_active_idxs, y_active_obs, σ_active_obs,
                                         mean_preds, lin_preds, cub_preds
                                         )
                end
            end
        end

        # Compute means and stds
        gp_mse_mean      = gp_mse_sum ./ nruns
        gp_mse_var       = gp_mse_sqsum ./ nruns .- gp_mse_mean.^2
        gp_mse_std       = sqrt.(max.(gp_mse_var, 0.0))

        gp_relerr_mean   = gp_relerr_sum ./ nruns
        gp_relerr_var    = gp_relerr_sqsum ./ nruns .- gp_relerr_mean.^2
        gp_relerr_std    = sqrt.(max.(gp_relerr_var, 0.0))

        hl_mse_mean      = hl_mse_sum ./ nruns
        hl_mse_var       = hl_mse_sqsum ./ nruns .- hl_mse_mean.^2
        hl_mse_std       = sqrt.(max.(hl_mse_var, 0.0))

        hl_relerr_mean   = hl_relerr_sum ./ nruns
        hl_relerr_var    = hl_relerr_sqsum ./ nruns .- hl_relerr_mean.^2
        hl_relerr_std    = sqrt.(max.(hl_relerr_var, 0.0))

        par_mse_mean     = par_mse_sum ./ nruns
        par_mse_var      = par_mse_sqsum ./ nruns .- par_mse_mean.^2
        par_mse_std      = sqrt.(max.(par_mse_var, 0.0))

        par_relerr_mean  = par_relerr_sum ./ nruns
        par_relerr_var   = par_relerr_sqsum ./ nruns .- par_relerr_mean.^2
        par_relerr_std   = sqrt.(max.(par_relerr_var, 0.0))

        gp_active_mse_mean     = gp_active_mse_sum ./ nruns
        gp_active_mse_var      = gp_active_mse_sqsum ./ nruns .- gp_active_mse_mean.^2
        gp_active_mse_std      = sqrt.(max.(gp_active_mse_var, 0.0))

        gp_active_relerr_mean  = gp_active_relerr_sum ./ nruns
        gp_active_relerr_var   = gp_active_relerr_sqsum ./ nruns .- gp_active_relerr_mean.^2
        gp_active_relerr_std   = sqrt.(max.(gp_active_relerr_var, 0.0))

        mean_mse_mean     = mean_mse_sum ./ nruns
        mean_mse_var      = mean_mse_sqsum ./ nruns .- mean_mse_mean.^2
        mean_mse_std      = sqrt.(max.(mean_mse_var, 0.0))

        mean_relerr_mean  = mean_relerr_sum ./ nruns
        mean_relerr_var   = mean_relerr_sqsum ./ nruns .- mean_relerr_mean.^2
        mean_relerr_std   = sqrt.(max.(mean_relerr_var, 0.0))

        lin_mse_mean      = lin_mse_sum ./ nruns
        lin_mse_var       = lin_mse_sqsum ./ nruns .- lin_mse_mean.^2
        lin_mse_std       = sqrt.(max.(lin_mse_var, 0.0))

        lin_relerr_mean   = lin_relerr_sum ./ nruns
        lin_relerr_var    = lin_relerr_sqsum ./ nruns .- lin_relerr_mean.^2
        lin_relerr_std    = sqrt.(max.(lin_relerr_var, 0.0))

        cub_mse_mean      = cub_mse_sum ./ nruns
        cub_mse_var       = cub_mse_sqsum ./ nruns .- cub_mse_mean.^2
        cub_mse_std       = sqrt.(max.(cub_mse_var, 0.0))

        cub_relerr_mean   = cub_relerr_sum ./ nruns
        cub_relerr_var    = cub_relerr_sqsum ./ nruns .- cub_relerr_mean.^2
        cub_relerr_std    = sqrt.(max.(cub_relerr_var, 0.0))

        data["metrics"][ntp] = Dict(
            "gp_mse"           => gp_mse_mean,
            "gp_mse_std"       => gp_mse_std,
            "gp_relerr"        => gp_relerr_mean,
            "gp_relerr_std"    => gp_relerr_std,
            "hl_mse"           => hl_mse_mean,
            "hl_mse_std"       => hl_mse_std,
            "hl_relerr"        => hl_relerr_mean,
            "hl_relerr_std"    => hl_relerr_std,
            "param_mse"        => par_mse_mean,
            "param_mse_std"    => par_mse_std,
            "param_relerr"     => par_relerr_mean,
            "param_relerr_std" => par_relerr_std,
            "gp_active_mse"           => gp_active_mse_mean,
            "gp_active_mse_std"       => gp_active_mse_std,
            "gp_active_relerr"        => gp_active_relerr_mean,
            "gp_active_relerr_std"    => gp_active_relerr_std,
            "mean_mse"              => mean_mse_mean,
            "mean_mse_std"          => mean_mse_std,
            "mean_relerr"           => mean_relerr_mean,
            "mean_relerr_std"       => mean_relerr_std,
            "lin_mse"               => lin_mse_mean,
            "lin_mse_std"           => lin_mse_std,
            "lin_relerr"            => lin_relerr_mean,
            "lin_relerr_std"        => lin_relerr_std,
            "cub_mse"               => cub_mse_mean,
            "cub_mse_std"           => cub_mse_std,
            "cub_relerr"            => cub_relerr_mean,
            "cub_relerr_std"        => cub_relerr_std,
        )
    end

    return data
end

# ----------------------------------------------------------------------
# I/O and CLI
# ----------------------------------------------------------------------

"""
    load_data(n, m)

Load all required arrays from `../data/majo_n<n>_m<m>/`.
Returns:
  ts, feature_vecs, true_outputs, true_overlaps, true_obs_coeffs
"""
function load_data(n::Int, m::Int)
    base = "../data/majo_n$(n)_m$(m)"
    ts               = npzread(joinpath(base, "times.npy"))
    feature_vecs     = npzread(joinpath(base, "true_module_projs.npy"))
    true_outputs     = npzread(joinpath(base, "true_outputs.npy"))
    true_overlaps    = npzread(joinpath(base, "true_overlaps.npy"))
    true_obs_coeffs  = npzread(joinpath(base, "true_obs_coeffs.npy"))

    tmax = maximum(ts)
    ts = ts[ts .<= tmax]
    dsize = length(ts)

    feature_vecs = feature_vecs[1:dsize, :]
    true_outputs = true_outputs[1:dsize]
    true_overlaps = true_overlaps[1:dsize, 1:dsize]

    return ts, feature_vecs, true_outputs, true_overlaps, true_obs_coeffs
end

"""
    main(ARGS)

Entry point. Expects:
  ARGS[1] = n
  ARGS[2] = m
  ARGS[3] = nruns (optional; otherwise env `NRUNS` or 1)

Optional environment variables:
  NOBS_MIN, NOBS_MAX   # exponents for 10^k in nshots_obs_list  (default 1:10)
  NPRED_MIN, NPRED_MAX # exponents for 10^k in nshots_preds_list (default 1:10)
  NRUNS                # default number of runs if ARGS[3] not provided
  SEED                 # if set to an Int, seeds the RNG
  USE_GP_THREADS       # if "true", parallelize GP over test points
"""
function main(ARGS)
    if length(ARGS) < 2
        println("Usage: julia --threads auto qgp_run_bench.jl n m [nruns]")
        println("Example: julia --threads 8 qgp_run_bench.jl 18 2 5")
        return
    end

    n = parse(Int, ARGS[1])
    m = parse(Int, ARGS[2])

    nruns =
        if length(ARGS) >= 3
            parse(Int, ARGS[3])
        else
            parse(Int, get(ENV, "NRUNS", "1"))
        end

    # Optional seeding for reproducibility
    seed_str = get(ENV, "SEED", "")
    if seed_str != ""
        seed = parse(Int, seed_str)
        @info "Seeding RNG" seed
        Random.seed!(seed)
    end

    println(">>> Running GP/TN comparison for n = $n, m = $m")
    println("    Threads: $(nthreads())")
    println("    nruns   : $nruns")
    println("    USE_GP_THREADS = $(USE_GP_THREADS)")

    ts, feature_vecs, true_outputs, true_overlaps, true_obs_coeffs = load_data(n, m)
    println("    Loaded dataset of size d = $(length(ts))")

    # Shot grids (10^k)
    nobs_min  = parse(Int, get(ENV, "NOBS_MIN", "1"))
    nobs_max  = parse(Int, get(ENV, "NOBS_MAX", "10"))
    npred_min = parse(Int, get(ENV, "NPRED_MIN", "1"))
    npred_max = parse(Int, get(ENV, "NPRED_MAX", "10"))

    nshots_obs_list   = [10^k for k in nobs_min:nobs_max]
    nshots_preds_list = [10^k for k in npred_min:npred_max]

    # ntp sweep: 25, 50, ..., floor(binomial(2n,2)/4)
    max_ntp = floor(Int, binomial(2n, 2) ÷ 4)
    ntps = collect(25:25:max_ntp)
    ntps = [5, 10, 25, 50, 100, 250] #, 500, 1000]

    println("    ntps = $(ntps)")
    println("    nshots_obs_list   = $(nshots_obs_list)")
    println("    nshots_preds_list = $(nshots_preds_list)")


    outdir = "../results"
    isdir(outdir) || mkpath(outdir)

    pred_base_dir = joinpath(outdir, "preds")

    data = run_comparison(ts, true_outputs, true_obs_coeffs, feature_vecs,
                          true_overlaps, nshots_obs_list, nshots_preds_list, ntps,
                          n, m, nruns; pred_base_dir = pred_base_dir)

    # Save summary metrics
    outfile = joinpath(outdir, "n$(n)_m$(m)_comparison.jld2")
    println("    Saving summary results to $outfile")
    save_object(outfile, data)

    println(">>> Done.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
