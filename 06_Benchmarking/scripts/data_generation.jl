#!/usr/bin/env julia

using Combinatorics
using ITensors
using JLD2
using ITensorMPS
using NPZ
using LinearAlgebra
using Random


include("../../src/apply_gate_as_mpo.jl")
include("../../src/gates_utils.jl")
# Script with operations w/ Majoranas
include("../../src/majo_machinery.jl")
# Script with possible initial configurations
include("../src/initial_circuits.jl")  # use custom initial states from benchmarking directory ../src
# Script with potential quantum gates
include("../../src/gates_circuit.jl")

# Build single-site operators from a Pauli symbol
function make_operator(sites::Vector{Index{Int}}, op::Symbol, index::Int)
    ITensor(paulis[op], sites[index]', sites[index])
end

function get_state!(gates, state::MPS, sites::Vector{Index{Int}}, indices::Vector{Vector{Int}})
    final_state = copy(state)
    
    for (j, gate) in enumerate(gates)
        inds = indices[j]
        it = length(size(gate)) == 2 ?
             ITensor(gate, sites[inds[1]]', sites[inds[1]]) :
             ITensor(gate, sites[inds[1]]', sites[inds[2]]', sites[inds[1]], sites[inds[2]])

        final_state = noprime(ITensors.apply([it], final_state));
    end

    return final_state
end

# Function to group MPOs (kept for future use, but not used in main)
function group_mpos(mpos::Vector{MPO}, size_g::Int; maxbond::Int)
    ng = ceil(Int, length(mpos) / size_g)
    grouped = Vector{MPO}(undef, ng)

    for i in 1:ng
        range = (i - 1) * size_g + 1 : min(i * size_g, length(mpos))
        grouped[i] = truncate(sum(mpos[range]), maxdim = maxbond)
    end

    return grouped
end

# Haar-random special orthogonal matrix in SO(dim)
function random_SO(dim::Int; rng = Random.default_rng())
    A = randn(rng, dim, dim)
    F = qr(A)
    Q = Matrix(F.Q)
    R = F.R

    # Absorb the signs of diag(R) into Q (standard Haar trick)
    s = sign.(diag(R))
    # avoid zeros -> treat as +1
    s[s .== 0] .= 1
    Q *= Diagonal(s)

    # Fix determinant to +1 (SO, not O)
    if det(Q) < 0
        Q[:, 1] .*= -1
    end

    return Q
end

# Coefficients in B_m obtained by conjugating γ_1...γ_m
# with a random Gaussian unitary (represented by R ∈ SO(2nq))
function random_Bm_coeffs(nq::Int, m::Int; rng = Random.default_rng())
    nmaj = 2 * nq               # total number of Majoranas
    len_maj = binomial(nmaj, m) # dim B_m

    R = random_SO(nmaj; rng)    # SO(2nq) in standard rep
    R_sub = @view R[1:m, :]     # first m rows

    coeffs = Vector{Float64}(undef, len_maj)
    idx = 1
    for cols_tuple in combinations(1:nmaj, m)  # lexicographic order of subsets
        cols = collect(cols_tuple)
        # m x m minor corresponding to subset I
        submat = Matrix(R_sub[:, cols])
        coeffs[idx] = det(submat)
        idx += 1
    end
    
    return coeffs
end


function main(nq::Int, m::Int)
    @info "Starting dataset generation (features only)" nq=nq m=m Threads=Threads.nthreads()

    # === Majorana operator construction ===
    list_majoranas = majorana_products(nq, m)
    len_maj = length(list_majoranas)

    # Operators out of list_majoranas
    list_maj_op = [ops for (_, ops) in list_majoranas]
    # Coefficients out of list_majoranas (phases/signs already in here)
    coeff_maj_op = [real(coef) for (coef, _) in list_majoranas]

    # Random coefficients for FLO (in the Majorana basis)
    random_coeffs = random_Bm_coeffs(nq, m)

    # === Output directory ===
    output_dir = "../data/majo_n$(nq)_m$(m)"
    if !isdir(output_dir)
        mkpath(output_dir)
        @info "Created directory $output_dir"
    else
        @info "Directory already exists: $output_dir"
    end

    # === Build Pauli / Majorana MPOs ===
    sites_maj = siteinds("Qubit", nq)

    paulis_maj = [MPO([make_operator(sites_maj, list_maj_op[i][j], j)
                       for j in 1:nq])
                  for i in 1:len_maj]

    # These are the actual Majorana products Γ_k as MPOs
    op_maj = paulis_maj .* coeff_maj_op

    # === FLO operator: we only need the coefficients, not the MPOs ===

    # Save the subset indices and coefficients (for use in Python or later)
    npzwrite("$(output_dir)/true_obs_coeffs.npy", random_coeffs)

    # === Initial states ===
    zero_state = fill("0", nq)
    MPS_zero = productMPS(sites_maj, zero_state)

    # Number of points (different angles t)
    num_points = 2 * binomial(2nq, m)
    tmax = 2π
    t_list = collect(range(0, tmax, num_points))
    npzwrite("$(output_dir)/times.npy", t_list)

    # Circuit parameters
    nlayers = 1

    # Maximum bond dimension for states (used in truncations inside the circuits)
    maxbond = 64

    # Types of circuits for initialization
    state_generators = [
        random_rotations_with_entanglement_state_circuit,
        simple_extent_state_circuit,
        first_qubit_rotation_state_circuit,
        random_rotations_state_circuit,
        random_fermionic_gaussian_circuit,
    ]

    # Initial state after applying generator circuit
    # (as in the notebook: use state_generators[4])
    @info "Building initial states"
    t0 = time()
    initial_state = [state_generators[4](MPS_zero, sites_maj, t; seed = 42)
                     for t in t_list]
    t1 = time()
    @info " - Generating the dataset took $(round(Int, t1-t0)) s."

    # === Observables (Majorana basis) ===
    observable_maj = op_maj

    # === Compute feature vectors: f[i, k] = ⟨ψ_i | Γ_k | ψ_i⟩ ===
    @info "Computing feature matrix of size $(num_points) × $(len_maj)"
    feat_matrix = Matrix{Float64}(undef, num_points, len_maj)

    t0 = time()
    # IMPORTANT: use inner(psi', A, psi) to avoid deprecated index-matching path
    Threads.@threads for i in 1:num_points
        ψ = initial_state[i]
        ψbra = ψ'  # primes the site indices on the bra, matches MPO convention
        for j in 1:len_maj
            feat_matrix[i, j] = real(inner(ψbra, observable_maj[j], ψ))
        end
    end
    t1 = time()
    @info " - Computing feature matrix took $(round(Int, t1-t0)) s."
    
    # Save as a 2D array directly
    npzwrite("$(output_dir)/true_module_projs.npy", feat_matrix)

    # === Compute Gram matrix of features: overlaps via matrix multiplication ===
    @info "Computing Gram matrix of feature vectors"
    true_ovps = (feat_matrix * feat_matrix')
    npzwrite("$(output_dir)/true_overlaps.npy", true_ovps)

    @info "Computing true outputs"
    true_outputs = [dot(feat_matrix[i,:], random_coeffs) for i in 1:num_points]
    npzwrite("$(output_dir)/true_outputs.npy", true_outputs)

    @info "Done generating feature dataset" output_dir=output_dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) < 2
        println("Usage: julia data_generation.jl n m")
        exit(1)
    end
    nq = parse(Int, ARGS[1])
    m = parse(Int, ARGS[2])
    main(nq, m)
end
