module FFGPUtils

# Conservative module extraction from `ff_gp.ipynb`.
#
# Main structural edits relative to the sandbox notebook:
# - all reusable function definitions are centralized here
# - duplicate definitions were consolidated
# - repeated notebook-wide imports/constants were hoisted to module scope
# - ambiguous `levicivita(...)` calls were replaced by a local `_permutation_sign(...)` helper
# - backward-compatible helper names were kept where the notebook used more than one name
#
# The goal here is to reduce notebook statefulness without doing risky algorithmic surgery.

using Arpack
using Combinatorics
using DelimitedFiles
using Distributions
using ITensors
using ITensorMPS
using LinearAlgebra
using Plots
using Random
using SparseArrays
using SpecialFunctions
using Statistics
using Yao

export σx, σy, σz, I2, PAULI_MATS, PAULI_MULTIPLICATION_RULES, PAULI_MULTIPLICATION_RULES_BARE, PAULI_CHARS, majorana_operator, two_majoranas_matrix, get_contr, plot_data, get_y_magicstate, check_2majo_magic_randmeas_gp, check_2majo_magic_singlepauli_gp, get_ratios_magic, create_pairs, four_majoranas_ops, fill_antisymm_tensor, fill_antisymm_tensor_as_matrix, unique_combs_fast, unique_combs_indices_fast, lex_index, pair_combinations_fast, pair_combination_indices_fast, haar_rand_o, sample_4_majos, sample_4_majos_new, sample_4_majos_gaussian, get_4majos_zzs, init_zerostate_y_4majos, init_singlepauli_o_4majos, sample_4_majos_gaussian_singlepauli, sample_4_majos_singlepauli, matchings_linear_system, reduce_matching_coeff_mat_to_clusters, generate_brauer, transpose_brauer, check_non_vanishing_brauer_elem, what_we_know, draw_brauer, draw_moment_contrib, get_valid_choices, count_pairings, flo_unitary, moment_operator, algebra_dimension, random_flo, random_flo_unitaries, zero_state_MPS, get_input_states, mps_mpo_expval, get_y_comps, get_y_tensor, pauli_product, pauli_product_bare, majorana_symbol, majorana_products, majorana_products_bare, read_pauli_symb_string, pauli_mpo, m_majoranas_ops, majoranas_ops, maintest

# Shared single-qubit Pauli operators used throughout the notebook.
const σx = ComplexF64[0 1; 1 0]
const σy = ComplexF64[0 -1im; 1im 0]
const σz = ComplexF64[1 0; 0 -1]
const I2 = Matrix{ComplexF64}(I, 2, 2)

# Pauli-string helpers used in the later symbolic/MPO section.
const PAULI_MATS = Dict(:x => σx, :y => σy, :z => σz, :id => I2)

const PAULI_MULTIPLICATION_RULES = Dict(
    (:x, :x) => (:id, 1),
    (:y, :y) => (:id, 1),
    (:z, :z) => (:id, 1),
    (:x, :y) => (:z, im),
    (:y, :z) => (:x, im),
    (:z, :x) => (:y, im),
    (:y, :x) => (:z, -im),
    (:z, :y) => (:x, -im),
    (:x, :z) => (:y, -im),
    (:x, :id) => (:x, 1),
    (:y, :id) => (:y, 1),
    (:z, :id) => (:z, 1),
    (:id, :x) => (:x, 1),
    (:id, :y) => (:y, 1),
    (:id, :z) => (:z, 1),
    (:id, :id) => (:id, 1),
)

const PAULI_MULTIPLICATION_RULES_BARE = Dict(
    (:x, :x) => :id,
    (:y, :y) => :id,
    (:z, :z) => :id,
    (:x, :y) => :z,
    (:y, :z) => :x,
    (:z, :x) => :y,
    (:y, :x) => :z,
    (:z, :y) => :x,
    (:x, :z) => :y,
    (:x, :id) => :x,
    (:y, :id) => :y,
    (:z, :id) => :z,
    (:id, :x) => :x,
    (:id, :y) => :y,
    (:id, :z) => :z,
    (:id, :id) => :id,
)

const PAULI_CHARS = Dict(:x => "X", :y => "Y", :z => "Z", :id => "I")

# Two-mode passive-FLO generators used by the later moment-operator scratch work.
const Z1 = kron(σz, I2)
const Z2 = kron(I2, σz)
const X1 = kron(σx, I2)
const Y1 = kron(σy, I2)
const X2 = kron(I2, σx)
const Y2 = kron(I2, σy)
const Jz = (Z1 - Z2) / 2
const Jy = (X1 * Y2 - Y1 * X2) / 2

# Small internal helper: sign of a permutation, used to fill antisymmetric tensors.
function _permutation_sign(perm)
    ninv = 0
    @inbounds for i in 1:length(perm)-1
        for j in i+1:length(perm)
            ninv += perm[i] > perm[j]
        end
    end
    return iseven(ninv) ? 1 : -1
end


# ==================================================================================================
# Core Majorana / two-point helpers
# ==================================================================================================

# ------------------------------------------------------------------------------------------------
# majorana_operator
# Build the sparse Jordan–Wigner matrix for the j-th Majorana operator on n qubits.
# Origin: notebook cells 9 and 81 (duplicate definitions).
# Edit: kept one shared definition in the module instead of redefining it twice in the notebook.
# Edit: left the Jordan–Wigner construction logic essentially unchanged.
# ------------------------------------------------------------------------------------------------
function majorana_operator(j::Int, n::Int)
    I = I2
    σx = [0 1; 1 0]
    σy = [0 -im; im 0]
    σz = [1 0; 0 -1]
    op = 1.0
    for k in 1:n
        if k < j÷2 + mod(j,2)
            op = sparse(kron(op, σz))
        elseif k == j÷2 + mod(j,2)
            mod(j,2) == 1 ? op = sparse(kron(op, σx)) : op = sparse(kron(op, σy))
        else
            op = sparse(kron(op, I))
        end
    end
    return op
end;

# ------------------------------------------------------------------------------------------------
# two_majoranas_matrix
# Precompute the bilinears iγ_iγ_j used in the two-Majorana experiments.
# Origin: notebook cell 10.
# Edit: moved out of the notebook so the expensive helper is defined once and reused.
# ------------------------------------------------------------------------------------------------
function two_majoranas_matrix(n::Int)
    cab = [spzeros(ComplexF64, 2n, 2n) for i in 1:2n, j in 1:2n]
    for i in 1:2n
        for j in i+1:2n
            cab[i,j] = 1im * (majorana_operator(i, n) * majorana_operator(j, n))
        end
    end
    return cab
end

# ------------------------------------------------------------------------------------------------
# get_contr
# Sample contraction statistics from random parity-even states up to a chosen moment order.
# Origin: notebook cell 11.
# Edit: logic kept intentionally close to the sandbox version; only documentation/packaging changed.
# ------------------------------------------------------------------------------------------------
function get_contr(n::Int; nsamples::Int=100, maxmoment::Int=10)
    data = Dict(k => zeros(nsamples, k÷2) for k in 2:2:maxmoment)
    majos = two_majoranas_matrix(n)
    for i in 1:nsamples
        # Y = [-1 + 2rand() for i in 1:2n, j in 1:2n]
        state = rand_state(n)
        # impose parity
        state = state + state |> repeat(n, Z)
        state = normalize(state)
        y = zeros(2n, 2n)
        for j in 1:2n, k in j+1:2n
            y[j, k] = Yao.expect(GeneralMatrixBlock(majos[j, k]), state)
        end
        Y = y - transpose(y)
        A = Y*transpose(Y)
        eigvals = eigen(A).values
        innerp = sum(eigvals)
        for k in 2:2:maxmoment
            maxcontr = innerp ^ (k÷2)
            othercontr = [innerp ^ ((k-2j)÷2) * sum(eigvals.^j) for j in 2:k÷2]
            data[k][i, :] = [maxcontr; othercontr...]
        end
    end;
    print("\r - n=$(n) done ...")
    flush(stdout)
    data
end

# ------------------------------------------------------------------------------------------------
# plot_data
# Plot the contraction-ratio data produced by `get_contr`.
# Origin: notebook cell 13.
# Edit: moved plotting helper into the module so figures can be regenerated from a clean notebook.
# ------------------------------------------------------------------------------------------------
function plot_data(data)
    p = Plots.plot(legend=false, yscale=:log10, fontfamily="Computer Modern", xlabel="\$n\$", ylabel="ratio")
    ns = sort(collect(keys(data)))
    for n in ns
        vals = data[n]
        for k in sort(collect(keys(vals)))
            valss = vals[k]
            ratios = [valss[:, j] ./ valss[:, 1] for j in 2:(k÷2)]
            avg_ratios = [mean(x) for x in ratios]
            err_ratios = [std(x) for x in ratios]
            for m in 1:(k÷2)-1
                p = scatter!([n], [avg_ratios[m]], c=m)
            end
            p = Plots.plot!(ns, 1 ./ ((1 .* ns) .^(k÷2)), c=k÷2)
        end
    end
    p
end

# ------------------------------------------------------------------------------------------------
# get_y_magicstate
# Construct the antisymmetric two-Majorana correlation matrix for the repeated 4-qubit magic-state ansatz.
# Origin: notebook cell 22.
# Edit: logic kept close to the original exploratory version.
# ------------------------------------------------------------------------------------------------
function get_y_magicstate(n::Int, theta::Float64)
    n4 = n÷4
    psi = zeros(ComplexF64, 2^4)
    # psi[1] = 1/2
    # psi[4] = 1/2
    # psi[13] = 1/2
    # psi[end] = exp(1im*theta)/2
    # psi[1] = 1/sqrt(2)
    # psi[end] = exp(1im*theta)/sqrt(2)
    psi[1] = cos(theta/4)/sqrt(2)
    psi[end] = sin(theta/4)/sqrt(2)
    # extent state
    state = ArrayReg(LinearAlgebra.kron([psi for _ in 1:n4]...))
    # state = normalize(state)
    y = zeros(2n, 2n)
    majos = two_majoranas_matrix(n)
    for j in 1:2n, k in j+1:2n
        y[j, k] = Yao.expect(GeneralMatrixBlock(majos[j, k]), state)
    end
    y = y - transpose(y)
    y
end

# ------------------------------------------------------------------------------------------------
# check_2majo_magic_randmeas_gp
# Estimate the GP-relevant statistic for the magic-state ansatz using a random antisymmetric probe observable.
# Origin: notebook cell 23.
# Edit: no algorithmic rewrite; this is mainly a module lift with comments.
# ------------------------------------------------------------------------------------------------
function check_2majo_magic_randmeas_gp(n::Int, theta::Float64, nsamples::Int)
    # get y of magic
    y = get_y_magicstate(n, theta)
    # prepare random measurement
    o = zeros(2n,2n)
    for i in 1:2n, j in i+1:2n
        o[i, j] = randn()
        o[j, i] = - o[i, j]
    end
    # get samples
    results = zeros(nsamples)
    @inbounds for i in 1:nsamples
        #sample a random orthogoal
        Q = haar_rand_o(2n)
        #apply
        results[i] = dot(transpose(o)*Q, Q*y)
        print("\r - sampled $(i)/$(nsamples)")
        # flush(stdout)
    end
end

# ------------------------------------------------------------------------------------------------
# check_2majo_magic_singlepauli_gp
# Estimate the same statistic as `check_2majo_magic_randmeas_gp`, but for one fixed Pauli probe.
# Origin: notebook cell 24.
# Edit: no algorithmic rewrite; this is mainly a module lift with comments.
# ------------------------------------------------------------------------------------------------
function check_2majo_magic_singlepauli_gp(n::Int, theta::Float64, pauliind::Tuple{Int,Int}, nsamples::Int)
    # get y of magic
    y = get_y_magicstate(n, theta)
    # prepare pauli measurement
    o = spzeros(2n,2n)
    c1,c2 = pauliind
    o[c1,c2] = +1
    o[c2,c1] = -1
    # get samples
    results = zeros(nsamples)
    @inbounds for i in 1:nsamples
        #sample a random orthogoal
        Q = haar_rand_o(2n)
        #apply
        results[i] = dot(transpose(o)*Q, Q*y)
        print("\r - sampled $(i)/$(nsamples)")
        # flush(stdout)
    end
end

# ------------------------------------------------------------------------------------------------
# get_ratios_magic
# Sweep system sizes and summarize the two-Majorana magic-state ratios.
# Origin: notebook cell 27.
# Edit: moved into the module so size sweeps are callable from a fresh notebook.
# ------------------------------------------------------------------------------------------------
function get_ratios_magic(nmax::Int, theta::Float64)
    vals = []
    # eigvs_y = []
    # eigvs = []
    # corr_mats = []
    for m in 8:4:nmax
        y = get_y_magicstate(m, theta)
        _eis = eigen(y).values
        # push!(eigvs_y, _eis)
        a = transpose(y) * y
        _eis = eigen(a).values
        # push!(eigvs, _eis)
        # push!(corr_mats, outerprod(_eis, _eis))
        push!(vals, tr(a)^2 / tr(a*a))
    end
    vals
end


# ==================================================================================================
# Combinatorial tensor helpers
# ==================================================================================================

# ------------------------------------------------------------------------------------------------
# create_pairs
# Enumerate all unordered pairs drawn from 1:2n.
# Origin: notebook cell 46.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function create_pairs(n::Int)
    range = 1:2n
    pairs = [(i, j) for i in range, j in range if j > i]
    return pairs
end

# ------------------------------------------------------------------------------------------------
# four_majoranas_ops
# Construct all products of four Majorana operators as sparse matrices.
# Origin: notebook cell 82.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function four_majoranas_ops(n::Int)
    combs = collect(combinations(1:2n, 4))
    c_abcd = fill(spzeros(ComplexF64, 2^n, 2^n), length(combs))
    for (idx, (i,j,k,l)) in enumerate(combs)
        c_abcd[idx] = majorana_operator(i, n) * majorana_operator(j, n) * majorana_operator(k, n) * majorana_operator(l, n)
    end
    return c_abcd
end

# ------------------------------------------------------------------------------------------------
# fill_antisymm_tensor
# Fill an antisymmetric tensor from values indexed by increasing combinations.
# Origin: notebook cells 83 and 220 (two arities).
# Edit: both arities were kept; the generalized MPS-facing version now uses a local `_permutation_sign` helper instead of relying on an external `levicivita` symbol.
# ------------------------------------------------------------------------------------------------
function fill_antisymm_tensor(n, values)
    # Initialize the 2n x 2n x 2n x 2n tensor with zeros
    tensor = zeros(Float64, 2n, 2n, 2n, 2n)
    combs = collect(combinations(1:2n, 4))
    # Fill the tensor with provided values
    for ((i, j, k, l), value) in zip(combs, values)
        tensor[i, j, k, l] = value
        tensor[i, j, l, k] = -value
        tensor[i, k, j, l] = -value
        tensor[i, k, l, j] = value
        tensor[i, l, j, k] = value
        tensor[i, l, k, j] = -value
        tensor[j, i, k, l] = -value
        tensor[j, i, l, k] = value
        tensor[j, k, i, l] = value
        tensor[j, k, l, i] = -value
        tensor[j, l, i, k] = -value
        tensor[j, l, k, i] = value
        tensor[k, i, j, l] = value
        tensor[k, i, l, j] = -value
        tensor[k, j, i, l] = -value
        tensor[k, j, l, i] = value
        tensor[k, l, i, j] = value
        tensor[k, l, j, i] = -value
        tensor[l, i, j, k] = -value
        tensor[l, i, k, j] = value
        tensor[l, j, i, k] = value
        tensor[l, j, k, i] = -value
        tensor[l, k, i, j] = -value
        tensor[l, k, j, i] = value
    end
    
    return tensor
end

# ------------------------------------------------------------------------------------------------
# fill_antisymm_tensor_as_matrix
# Store an antisymmetric tensor in matrix form by flattening half of the indices into rows and half into columns.
# Origin: notebook cells 94, 97, and 98.
# Edit: all overloads were kept.
# Edit: replaced `levicivita(...)` with `_permutation_sign(...)` to make the module self-contained.
# ------------------------------------------------------------------------------------------------
function fill_antisymm_tensor_as_matrix(n, values)
    # Initialize the 2n x 2n x 2n x 2n tensor with zeros
    t_mat = spzeros(Float64, (2n)^2, (2n)^2)
    # t inds to mat
    mapInd(p,q) = p + (q-1)*2n
    # Fill the matrix with provided values
    for ((i, j, k, l), value) in zip(combinations(1:2n, 4), values)
        t_mat[mapInd(i, j), mapInd(k, l)] = value
        t_mat[mapInd(i, j), mapInd(l, k)] = -value
        t_mat[mapInd(i, k), mapInd(j, l)] = -value
        t_mat[mapInd(i, k), mapInd(l, j)] = value
        t_mat[mapInd(i, l), mapInd(j, k)] = value
        t_mat[mapInd(i, l), mapInd(k, j)] = -value
        t_mat[mapInd(j, i), mapInd(k, l)] = -value
        t_mat[mapInd(j, i), mapInd(l, k)] = value
        t_mat[mapInd(j, k), mapInd(i, l)] = value
        t_mat[mapInd(j, k), mapInd(l, i)] = -value
        t_mat[mapInd(j, l), mapInd(i, k)] = -value
        t_mat[mapInd(j, l), mapInd(k, i)] = value
        t_mat[mapInd(k, i), mapInd(j, l)] = value
        t_mat[mapInd(k, i), mapInd(l, j)] = -value
        t_mat[mapInd(k, j), mapInd(i, l)] = -value
        t_mat[mapInd(k, j), mapInd(l, i)] = value
        t_mat[mapInd(k, l), mapInd(i, j)] = value
        t_mat[mapInd(k, l), mapInd(j, i)] = -value
        t_mat[mapInd(l, i), mapInd(j, k)] = -value
        t_mat[mapInd(l, i), mapInd(k, j)] = value
        t_mat[mapInd(l, j), mapInd(i, k)] = value
        t_mat[mapInd(l, j), mapInd(k, i)] = -value
        t_mat[mapInd(l, k), mapInd(i, j)] = -value
        t_mat[mapInd(l, k), mapInd(j, i)] = value
    end
    
    return t_mat
end

# ------------------------------------------------------------------------------------------------
# fill_antisymm_tensor_as_matrix
# Store an antisymmetric tensor in matrix form by flattening half of the indices into rows and half into columns.
# Origin: notebook cells 94, 97, and 98.
# Edit: all overloads were kept.
# Edit: replaced `levicivita(...)` with `_permutation_sign(...)` to make the module self-contained.
# ------------------------------------------------------------------------------------------------
function fill_antisymm_tensor_as_matrix(n, m, values)
    # Initialize the 2n .. 2n tensor with zeros
    t_mat = spzeros(Float64, (2n)^(m÷2), (2n)^(m÷2))
    # t inds to mat
    mapInd(q) = sum([(2n)^(j) for j in 0:length(q)-1] .* (q .- 1)) + 1
    # Fill the matrix with provided values
    for (c, value) in zip(combinations(1:2n, m), values)
        for (pc, parperm) in zip(permutations(c), permutations([1:m]...))
            t_mat[mapInd(pc[1:m÷2]), mapInd(pc[m÷2+1:end])] = _permutation_sign(parperm) * value
        end
    end
    
    return t_mat
end

# ------------------------------------------------------------------------------------------------
# fill_antisymm_tensor_as_matrix
# Store an antisymmetric tensor in matrix form by flattening half of the indices into rows and half into columns.
# Origin: notebook cells 94, 97, and 98.
# Edit: all overloads were kept.
# Edit: replaced `levicivita(...)` with `_permutation_sign(...)` to make the module self-contained.
# ------------------------------------------------------------------------------------------------
function fill_antisymm_tensor_as_matrix(n, m, combs, values)
    # Initialize the 2n .. 2n tensor with zeros
    t_mat = spzeros(Float64, (2n)^(m÷2), (2n)^(m÷2))
    # t inds to mat
    mapInd(q) = sum([(2n)^(j) for j in 0:length(q)-1] .* (q .- 1)) + 1
    # Fill the matrix with provided values
    for (c, value) in zip(combs, values)
        for (pc, parperm) in zip(permutations(c), permutations([1:m]...))
            t_mat[mapInd(pc[1:m÷2]), mapInd(pc[m÷2+1:end])] = _permutation_sign(parperm) * value
        end
    end
    
    return t_mat
end

# ------------------------------------------------------------------------------------------------
# unique_combs_fast
# Return a simple block-structured subset of combinations used as a fast special case.
# Origin: notebook cell 99.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function unique_combs_fast(n::Int, m::Int)
    N = 2n
    # @assert N % m == 0 "2n must be divisible by m"
    t = div(N, m)
    return [ collect((k-1)*m+1 : k*m) for k in 1:t ]
end

# ------------------------------------------------------------------------------------------------
# unique_combs_indices_fast
# Return the lexicographic indices of the block-structured combinations from `unique_combs_fast`.
# Origin: notebook cell 99.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function unique_combs_indices_fast(n, m)
    N = 2n
    t = div(N, m)
    # S[j+1] will hold sum_{i=1}^j binomial(N-i, m-1),
    # and S[1] == 0 by construction.
    S = zeros(Int, N+1)      # indices 1:(N+1)
    for j in 1:N
        S[j+1] = S[j] + binomial(N-j, m-1)
    end

    # for block k, (k-1)*m elements come before it
    # so its 1-based idx is S[(k-1)*m + 1] + 1
    idx = [ S[(k-1)*m + 1] + 1 for k in 1:t ]
    return idx
end

# ------------------------------------------------------------------------------------------------
# lex_index
# Compute the 1-based lexicographic index of a sorted combination.
# Origin: notebook cell 100.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function lex_index(c::Vector{Int}, N::Int)
    m = length(c)
    idx = 1
    prev = 0
    for i in 1:m
        for j in prev+1:(c[i]-1)
            idx += binomial(N - j, m - i)
        end
        prev = c[i]
    end
    return idx
end

# ------------------------------------------------------------------------------------------------
# pair_combinations_fast
# Generate combinations built from adjacent pairs [2k-1,2k].
# Origin: notebook cell 100.
# Edit: extracted carefully because the notebook cell also contained the next helper right after it.
# ------------------------------------------------------------------------------------------------
function pair_combinations_fast(n::Int, m::Int)
    @assert m % 2 == 0 "m must be even"
    r = m ÷ 2
    # loop over which r blocks (of 1:n) you pick,
    # then flatten each block k → [2k-1,2k]
    return [ vcat(( [2*k-1, 2*k] for k in blocks )...) 
             for blocks in combinations(1:n, r) ]
end

# ------------------------------------------------------------------------------------------------
# pair_combination_indices_fast
# Return the lexicographic indices of the adjacent-pair combinations.
# Origin: notebook cell 100.
# Edit: restored explicitly in the module so the helper is available without relying on notebook cell state.
# ------------------------------------------------------------------------------------------------
function pair_combination_indices_fast(n::Int, m::Int)
    @assert m % 2 == 0 "m must be even"
    N = 2n
    r = div(m, 2)
    inds = Int[]
    for blocks in combinations(1:n, r)
        # build [2k1-1,2k1, 2k2-1,2k2, …]
        comb = vcat([2*k .+ [-1, 0] for k in blocks]...)
        push!(inds, lex_index(comb, N))
    end
    sort!(inds)
    return inds
end


# ==================================================================================================
# Four-Majorana sampling
# ==================================================================================================

# ------------------------------------------------------------------------------------------------
# haar_rand_o
# Sample a random orthogonal matrix from the Haar measure.
# Origin: notebook cell 110.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function haar_rand_o(n::Int)
    """
    from: http://www.ams.org/notices/200705/fea-mezzadri-web.pdf
    """
    # Create a complex matrix with random Gaussian entries
    z = (randn(n, n)) / sqrt(2.0)

    # Perform QR decomposition
    q, r = qr(z)

    # Normalize to make the diagonal of r have modulus 1
    d = diag(r)
    ph = d ./ abs.(d)
    q = q * Diagonal(ph)

    return (1. +0im) * q
end;

# ------------------------------------------------------------------------------------------------
# sample_4_majos
# Sample a four-Majorana statistic for a concrete input state.
# Origin: notebook cell 113.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function sample_4_majos(state::ArrayReg, coeffs::Array{Float64, 4}, nsamples::Int)
    # get n
    n = nqubits(state)
    # init majos
    majos = four_majoranas_ops(n)
    # construct Y tensor
    yvals = [Yao.expect(GeneralMatrixBlock(majo), state) for majo in majos]
    y = fill_antisymm_tensor(n, yvals)
    
    println("Y tensor ready!")
    flush(stdout)
    
    # create inds
    ss = siteinds(2n, 4)
    # make y and coeffs into tensors
    o = ITensor(coeffs, ss)
    y = ITensor(y, ss)
    
    # get samples
    results = zeros(nsamples)
    @inbounds for i in 1:nsamples
        #sample a random orthogoal
        Q = haar_rand_o(2n)
        #apply
        tmp = y * ITensor(Q, ss[1]', ss[1])
        tmp = tmp * ITensor(Q, ss[2]', ss[2])
        tmp = tmp * ITensor(Q, ss[3]', ss[3])
        tmp = tmp * ITensor(Q, ss[4]', ss[4])
        tmp = noprime(tmp)
        results[i] = (tmp*o)[]
        i%100 == 0 ? print("\r - sampled $(i)/$(nsamples)") : nothing
        flush(stdout)
    end
end

# ------------------------------------------------------------------------------------------------
# sample_4_majos_new
# Variant of `sample_4_majos` with a flattened coefficient container.
# Origin: notebook cell 114.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function sample_4_majos_new(state::ArrayReg, coeffs::Array{Float64}, nsamples::Int)
    # get n
    n = nqubits(state)
    # init majos
    majos = four_majoranas_ops(n)
    # get indpendent vals
    combs = collect(combinations(1:2n, 4))
    # construct Y tensor
    yvals = [Yao.expect(GeneralMatrixBlock(majo), state) for majo in majos]
    y = fill_antisymm_tensor(n, yvals)
    
    println("Y tensor ready!")
    flush(stdout)
    
    # create inds
    ss = siteinds(2n, 4)
    # make y and coeffs into tensors
    # o = ITensor(coeffs, ss)
    y = ITensor(y, ss)
    
    # get samples
    results = zeros(nsamples)
    @inbounds for i in 1:nsamples
        #sample a random orthogoal
        Q = haar_rand_o(2n)
        #apply
        tmp = y * ITensor(Q, ss[1]', ss[1])
        tmp = tmp * ITensor(Q, ss[2]', ss[2])
        tmp = tmp * ITensor(Q, ss[3]', ss[3])
        tmp = tmp * ITensor(Q, ss[4]', ss[4])
        tmp = noprime(tmp)
        
        # get independent coeffs
        tmpvals = [array(tmp)[c...] for c in combs]
        
        results[i] = factorial(4) * dot(tmpvals, coeffs)
        print("\r - sampled $(i)/$(nsamples)")
        flush(stdout)
    end
end

# ------------------------------------------------------------------------------------------------
# sample_4_majos_gaussian
# Sample the four-Majorana statistic for a Gaussian-state input.
# Origin: notebook cell 115.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function sample_4_majos_gaussian(n::Int, coeffs::Array{Float64}, nsamples::Int)
    
    # get indpendent vals
    # combs = collect(combinations(1:2n, 4))
    # construct Y tensor for gaussian states (i.e. the zero state) we only get contribs from the ZZs -> c2i-1c2i c2j-1c2j 
    yvals = zeros(binomial(2n, 4))
    for (ind, (i, j, k, l)) in enumerate(combinations(1:2n, 4))
        if i%2==1 && k%2==1 && j==i+1 && l==k+1
            yvals[ind] = 1
        end
    end
    y = fill_antisymm_tensor(n, yvals)
    
    println("Y tensor ready!")
    flush(stdout)
    
    # create inds
    ss = siteinds(2n, 4)
    # make y and coeffs into tensors
    # o = ITensor(coeffs, ss)
    y = ITensor(y, ss)
    
    # get samples
    results = zeros(nsamples)
    @inbounds for i in 1:nsamples
        #sample a random orthogoal
        Q = haar_rand_o(2n)
        #apply
        tmp = y * ITensor(Q, ss[1]', ss[1])
        tmp = tmp * ITensor(Q, ss[2]', ss[2])
        tmp = tmp * ITensor(Q, ss[3]', ss[3])
        tmp = tmp * ITensor(Q, ss[4]', ss[4])
        tmp = noprime(tmp)
        
        # get independent coeffs
        tmpvals = [array(tmp)[c...] for c in combinations(1:2n, 4)]
        
        results[i] = factorial(4) * dot(tmpvals, coeffs)
        print("\r - sampled $(i)/$(nsamples)")
        flush(stdout)
    end
end

# ------------------------------------------------------------------------------------------------
# get_4majos_zzs
# Return the four-Majorana tuples corresponding to ZZ-type locations on the zero state.
# Origin: notebook cell 116.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function get_4majos_zzs(n::Int)
    zz_inds = []
    for i in 1:2:2n
        for k in i+2:2:2n
            push!(zz_inds, (i,i+1,k,k+1))
        end
    end
    zz_inds
end

# ------------------------------------------------------------------------------------------------
# init_zerostate_y_4majos
# Build the matrixized four-index tensor for the zero-state four-Majorana signal.
# Origin: notebook cell 116.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function init_zerostate_y_4majos(n::Int)
    # Initialize the 2n x 2n x 2n x 2n tensor with zeros
    t_mat = spzeros(Float64, (2n)^2, (2n)^2)
    # t inds to mat
    mapInd(p,q) = p + (q-1)*2n
    # Fill the matrix with provided values
    locs = get_4majos_zzs(n)
    value = 1
    for (i, j, k, l) in locs
        t_mat[mapInd(i, j), mapInd(k, l)] = value
        t_mat[mapInd(i, j), mapInd(l, k)] = -value
        t_mat[mapInd(i, k), mapInd(j, l)] = -value
        t_mat[mapInd(i, k), mapInd(l, j)] = value
        t_mat[mapInd(i, l), mapInd(j, k)] = value
        t_mat[mapInd(i, l), mapInd(k, j)] = -value
        t_mat[mapInd(j, i), mapInd(k, l)] = -value
        t_mat[mapInd(j, i), mapInd(l, k)] = value
        t_mat[mapInd(j, k), mapInd(i, l)] = value
        t_mat[mapInd(j, k), mapInd(l, i)] = -value
        t_mat[mapInd(j, l), mapInd(i, k)] = -value
        t_mat[mapInd(j, l), mapInd(k, i)] = value
        t_mat[mapInd(k, i), mapInd(j, l)] = value
        t_mat[mapInd(k, i), mapInd(l, j)] = -value
        t_mat[mapInd(k, j), mapInd(i, l)] = -value
        t_mat[mapInd(k, j), mapInd(l, i)] = value
        t_mat[mapInd(k, l), mapInd(i, j)] = value
        t_mat[mapInd(k, l), mapInd(j, i)] = -value
        t_mat[mapInd(l, i), mapInd(j, k)] = -value
        t_mat[mapInd(l, i), mapInd(k, j)] = value
        t_mat[mapInd(l, j), mapInd(i, k)] = value
        t_mat[mapInd(l, j), mapInd(k, i)] = -value
        t_mat[mapInd(l, k), mapInd(i, j)] = -value
        t_mat[mapInd(l, k), mapInd(j, i)] = value
    end
    
    return t_mat
end

# ------------------------------------------------------------------------------------------------
# init_singlepauli_o_4majos
# Build the matrixized four-index probe tensor for a single selected Pauli combination.
# Origin: notebook cell 116.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function init_singlepauli_o_4majos(n::Int, comb)
    # Initialize the 2n x 2n x 2n x 2n tensor with zeros
    t_mat = spzeros(Float64, (2n)^2, (2n)^2)
    # t inds to mat
    mapInd(p,q) = p + (q-1)*2n
    # Fill the matrix with provided values
    locs = get_4majos_zzs(n)
    value = 1/factorial(4)
    (i, j, k, l) = comb
    
    t_mat[mapInd(i, j), mapInd(k, l)] = value
    t_mat[mapInd(i, j), mapInd(l, k)] = -value
    t_mat[mapInd(i, k), mapInd(j, l)] = -value
    t_mat[mapInd(i, k), mapInd(l, j)] = value
    t_mat[mapInd(i, l), mapInd(j, k)] = value
    t_mat[mapInd(i, l), mapInd(k, j)] = -value
    t_mat[mapInd(j, i), mapInd(k, l)] = -value
    t_mat[mapInd(j, i), mapInd(l, k)] = value
    t_mat[mapInd(j, k), mapInd(i, l)] = value
    t_mat[mapInd(j, k), mapInd(l, i)] = -value
    t_mat[mapInd(j, l), mapInd(i, k)] = -value
    t_mat[mapInd(j, l), mapInd(k, i)] = value
    t_mat[mapInd(k, i), mapInd(j, l)] = value
    t_mat[mapInd(k, i), mapInd(l, j)] = -value
    t_mat[mapInd(k, j), mapInd(i, l)] = -value
    t_mat[mapInd(k, j), mapInd(l, i)] = value
    t_mat[mapInd(k, l), mapInd(i, j)] = value
    t_mat[mapInd(k, l), mapInd(j, i)] = -value
    t_mat[mapInd(l, i), mapInd(j, k)] = -value
    t_mat[mapInd(l, i), mapInd(k, j)] = value
    t_mat[mapInd(l, j), mapInd(i, k)] = value
    t_mat[mapInd(l, j), mapInd(k, i)] = -value
    t_mat[mapInd(l, k), mapInd(i, j)] = -value
    t_mat[mapInd(l, k), mapInd(j, i)] = value
    
    return t_mat
end

# ------------------------------------------------------------------------------------------------
# sample_4_majos_gaussian_singlepauli
# Gaussian-state sampling routine for one fixed Pauli probe.
# Origin: notebook cell 137.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function sample_4_majos_gaussian_singlepauli(n::Int, pauliComb, nsamples::Int)
    
    y = init_zerostate_y_4majos(n)
    println("Y matrix ready!")
    flush(stdout)
    
    #same for o 
    o = init_singlepauli_o_4majos(n, pauliComb)
    println("O matrix ready!")
    flush(stdout)
    
    # get samples
    results = zeros(nsamples)
    @inbounds for i in 1:nsamples
        #sample a random orthogoal
        Q = haar_rand_o(2n)
        Q2 = kron(Q, Q)
        #apply
        results[i] = dot(transpose(o)*Q2, Q2*y)
        print("\r - sampled $(i)/$(nsamples)")
        flush(stdout)
    end
end

# ------------------------------------------------------------------------------------------------
# sample_4_majos_singlepauli
# Sampling routine for one fixed Pauli probe with a supplied Y tensor.
# Origin: notebook cell 137.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function sample_4_majos_singlepauli(n::Int, y, pauliComb, nsamples::Int)
    
    #same for o 
    o = init_singlepauli_o_4majos(n, pauliComb)
    println("O matrix ready!")
    flush(stdout)
    
    # get samples
    results = zeros(nsamples)
    @inbounds for i in 1:nsamples
        #sample a random orthogoal
        Q = haar_rand_o(2n)
        Q2 = kron(Q, Q)
        #apply
        results[i] = dot(transpose(o)*Q2, Q2*y)
        print("\r - sampled $(i)/$(nsamples)")
        flush(stdout)
    end
    results
end


# ==================================================================================================
# Matching and Brauer scratch helpers
# ==================================================================================================

# ------------------------------------------------------------------------------------------------
# matchings_linear_system
# Assemble the linear constraints that encode the allowed matching problem.
# Origin: notebook cell 166.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function matchings_linear_system(m::Int, k::Int)::Tuple{Matrix{Int},Vector{Int}}
    # there are km*(km-1) vars x_ij encoding presence (1) or absence (0) of the edge between i and j
    # we order them as x_12, x_13 ... x_1km, x_21, x_23 ... etc
    
    nverts = k*m
    nvars = nverts*(nverts - 1)
    
    # we initialize the matrix as an empty row that we will discard at the end
    coeff_mat = zeros(Int, 1, nvars)
    # the constant term is initialized as an empty vector
    b = Int[]
    
    # the first condition is given by having only one edge coming in/out of any vertex
    for i in 1:nverts
        cond = zeros(Int, nvars)
        cond[1+(i-1)*(nverts-1):i*(nverts-1)] .= 1 # this takes care of the batch of variables x_ij
        # now we need the variables x_ji
        for j in 1:nverts
            if j != i
                j < i ? (id = i-1) : (id = i)
                cond[id+(j-1)*(nverts-1)] = 1
            end
        end
        coeff_mat = vcat(coeff_mat, cond')
        push!(b, 2)
    end

    # now we enforce x_ij = x_ji
    for i in 1:nverts
        for j in i+1:nverts
            if j != i
                cond = zeros(Int, nvars)
                cond[j-1+(i-1)*(nverts-1)] = 1
                cond[i+(j-1)*(nverts-1)] = -1
                coeff_mat = vcat(coeff_mat, cond')
                push!(b, 0)
            end
        end
    end

    # now we enforce clusters, meaning that there should be no edges inside any of the clusters, only between them
    cluster(i::Int) = ceil(Int, i/m)
    iscluster(i::Int, j::Int) = cluster(i) == cluster(j)
    for i in 1:nverts
        for j in 1:nverts
            cond = zeros(Int, nvars)
            if j != i
                j < i ? (id = j) : (id = j-1)
                if iscluster(i, j)
                    cond[id+(i-1)*(nverts-1)] = 1
                end
            end
            coeff_mat = vcat(coeff_mat, cond')
            push!(b, 0)
        end
    end

    # finally we enforce transpose clusters, meaning that there should be no edges inside any of the clusters, only between them
    transpose_vertex(i::Int) = mod1(i+nverts÷2, nverts)
    for i in 1:nverts
        for j in 1:nverts
            cond = zeros(Int, nvars)
            if j != i
                j < i ? (id = j) : (id = j-1)
                if iscluster(transpose_vertex(i), transpose_vertex(j))
                    cond[id+(i-1)*(nverts-1)] = 1
                end
            end
            coeff_mat = vcat(coeff_mat, cond')
            push!(b, 0)
        end
    end
    coeff_mat[2:end, :], b
end

# ------------------------------------------------------------------------------------------------
# reduce_matching_coeff_mat_to_clusters
# Coarse-grain the matching-constraint matrix by clustering equivalent variables.
# Origin: notebook cell 172.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function reduce_matching_coeff_mat_to_clusters(cmat::Matrix{Int}, m::Int, k::Int)::Matrix{Int}
    # we cluster the variables corresponding to edges between tensors
    # the variables n_IJ are now the sum of all edges connecting I to J
    # there are K clusters, hence k choose 2 variables
    nverts = k*m
    nvars = nverts*(nverts - 1)
    red_coeff_mat = zeros(Int, size(cmat, 1), binomial(k, 2))

    varnames = ["x_$(j)$(i)" for i in 1:m*k, j in 1:m*k if i != j]
    
    for (k, (ci, cj)) in enumerate(combinations(1:k, 2))
        # we need to sum all the columns of cmat corresponding to edges between clusters ci and cj
        cols = Int[]
        println("Merging cluster connection $(ci) <=> $(cj)")
        for vi in 1:m
            for vj in 1:m
                i = m*(ci-1)+vi
                j = m*(cj-1)+vj
                println("i=$(i), j=$(j)")
                if j != i
                    j < i ? (id = j) : (id = j-1)
                    push!(cols, id+(i-1)*(nverts-1))
                    println(" - collected variable $(varnames[id+(i-1)*(nverts-1)])")
                    j < i ? (id = i-1) : (id = i)
                    push!(cols, id+(j-1)*(nverts-1))
                    println(" - collected variable $(varnames[id+(j-1)*(nverts-1)])")
                    flush(stdout)
                end
            end
        end
        red_coeff_mat[:,k] = sum([cmat[:, col] for col in cols])
    end

    # now we aggregate rows
    

    red_coeff_mat
end

# ------------------------------------------------------------------------------------------------
# generate_brauer
# Generate Brauer-diagram data for the chosen order.
# Origin: notebook cell 182.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function generate_brauer(k::Int)#::Int
    n = 2k

    function generate_partitions(n::Int)
        pairs = collect(combinations(1:n, 2))

        function recursive_partitions(remaining::Vector{Int}, current_partition::Vector{Any})
            if isempty(remaining)
                return [current_partition]
            end
            partitions = []
            for i in 2:length(remaining)
                new_pair = [remaining[1], remaining[i]]
                new_remaining = remaining[2:i-1] ∪ remaining[i+1:end]
                new_partition = copy(current_partition)
                push!(new_partition, new_pair)
                append!(partitions, recursive_partitions(new_remaining, new_partition))
            end
            return partitions
        end

        return recursive_partitions(collect(1:n), [])
    end

    pairings = generate_partitions(n)
    
    return pairings
end;

# ------------------------------------------------------------------------------------------------
# transpose_brauer
# Transpose a Brauer element by swapping its pair endpoints.
# Origin: notebook cell 184.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function transpose_brauer(b_elem)
    n = maximum(maximum.(b_elem))
    mid = n÷2
    rules = [[k => k+mid for k in 1:mid]...; [k => k-mid for k in mid+1:n]...]
    t_b_elem = [sort(replace(pair, rules...)) for pair in b_elem]
    return t_b_elem
end;

# ------------------------------------------------------------------------------------------------
# check_non_vanishing_brauer_elem
# Filter Brauer elements against the non-vanishing rules used in the notebook.
# Origin: notebook cell 185.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function check_non_vanishing_brauer_elem(m::Int, k::Int)#::Int
    n = m * k
    pairings = generate_brauer(n ÷ 2)
    
    # Function to check if a partition contains at least one pair in each group
    function group_pairs_q(partition)
        for i in 1:k
            group_range = ((i - 1) * m + 1):(i * m)
            group_pairs = collect(combinations(group_range, 2))
            # return group_pairs
            if any(x -> x in group_pairs, partition)
                return true
            end
        end
        return false
    end

    # Filter the partitions that satisfy the condition
    # valid_partitions = [!(group_pairs_q(p)||group_pairs_q(transpose_brauer(p))) ? p : nothing for p in pairings]
    valid_partitions = [p for p in pairings if !(group_pairs_q(p)||group_pairs_q(transpose_brauer(p)))]

    # return valid partitions
    return valid_partitions
end

# ------------------------------------------------------------------------------------------------
# what_we_know
# Return the hand-curated set of Brauer/moment patterns that were known from earlier reasoning.
# Origin: notebook cell 187.
# Edit: kept as a scratch/helper routine, but moved out of the notebook to reduce statefulness.
# ------------------------------------------------------------------------------------------------
function what_we_know()
    # identity
    fookers = [[[1,7], [2,8], [3,9], [4,10], [5,11], [6,12]]]
    # ~equivalent contractions
    nrules = [(i=>i+1, i+1=>i) for i in 1:2:5]
    for j in 1:3
        this_rules = collect(combinations(nrules, j))
        for rule in this_rules
            push!(fookers, replace.(fookers[1], [x for t in rule for x in t]...))
        end
    end
    
    # swaps inside thicc legs
    dtmp = length(fookers)
    for j in 1:dtmp
        push!(fookers, replace.(fookers[j], 1=>3, 2=>4, 3=>1, 4=>2))
        push!(fookers, replace.(fookers[j], 5=>7, 6=>8, 7=>5, 8=>6))
        push!(fookers, replace.(fookers[j], 9=>11, 10=>12, 11=>9, 12=>10))
    end
    
    fookers = [sort.(f) for f in fookers]
    
#     # thicc legs splitting
#     dtmp = length(fookers)
#     for j in 1:dtmp
#         # find thicc legs that skip 2
#         this_elem = fookers[j]
#         good_legs = []
#         for (a, b) in this_elem
#             if 5 <= b-a <= 7
#                 push!(good_legs, (ceil(Int, a/2), ceil(Int, b/2)))
#             end
#         end
#         good_legs = unique(good_legs)
#         # return good_legs
#         for (l1, l2) in good_legs
#             s1, s2 = 2(l1 + 1), 2(l2 - 1) - 1
#             push!(fookers, replace.(this_elem, s1=>s2, s2=>s1))
#         end
        
#     end
    
    return [sort.(f) for f in fookers]
end

# ------------------------------------------------------------------------------------------------
# draw_brauer
# Visualize a Brauer diagram using Plots.jl.
# Origin: notebook cell 190.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function draw_brauer(b_elem, m, k)
    if m * k  != maximum(maximum.(b_elem))
        throw(ArgumentError("Given m and k do not match size of brauer elem!"))
    end
    
    mypalette = palette(:roma, (m*k)÷2)
    
    # Calculate the spacing and position of boxes
    box_height = 8
    box_width = 4
    spacing = 1
    leg_length = 2
    canvas_height = k * (box_height + spacing)
    canvas_width = box_width + canvas_height/2 + 8
    
    boxes_xs = [0, 0, box_width, box_width]
    
    p = plot(ratio=:equal, framestyle=:none, xlims=(0, canvas_width), ylims=(0, canvas_height))
    
    leg_pos = zeros(m*k)
    
    for i in 0:(k-1)
        y_top = canvas_height - i * (box_height + spacing)
        y_bottom = y_top - box_height
        
        # Draw the box
        p = plot!(boxes_xs, [y_bottom, y_top, y_top, y_bottom], seriestype=:shape, color=:lightgray, label=false)
        p = annotate!(box_width/2, y_top - box_height/2, text("\$T_{$(i+1)}\$", :black))
        
        # Draw legs
        for j in 1:m
            p = plot!([box_width, box_width+leg_length], [y_top - j*box_height/(m+1), y_top - j*box_height/(m+1)], color=:black, label=false,linewidth=2)
            p = annotate!(box_width+leg_length/2, y_top - (j-0.5)*box_height/(m+1), text("\$l_{$(m*i + j)}\$", :black))
            leg_pos[i*m+j] = y_top - j*box_height/(m+1)
        end
    end
    
    # add connections
    sc(y, r, cx, cy) = cx + sqrt(abs(r^2 - (y-cy)^2)) 
    for (k, (a, b)) in enumerate(b_elem)
        y1, y2 = leg_pos[a], leg_pos[b]
        r = abs(y1 - y2)/2
        cy = y2 + r
        cx = box_width+leg_length
        ys = range(y2, stop=y1, length=1000)
        
        p = plot!(sc.(ys, r, cx, cy), ys, label="\$$(a)\\leftrightarrow $(b)\$", style=:dash, c=mypalette[k], linewidth=2)
    end
    # display(p)
    return p
end

# ------------------------------------------------------------------------------------------------
# draw_moment_contrib
# Visualize the contribution of a Brauer element to the moment diagrammatics.
# Origin: notebook cell 199.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function draw_moment_contrib(b_elem, m, k)
    if m * k  != maximum(maximum.(b_elem))
        throw(ArgumentError("Given m and k do not match size of brauer elem!"))
    end
    
    mypalette = palette(:roma, (m*k)÷2)
    
    # Calculate the spacing and position of boxes
    box_height = 8
    box_width = 4
    spacing = 1
    leg_length = 2
    canvas_height = k * (box_height + spacing)
    canvas_width = 2box_width + canvas_height + 8
    
    o_boxes_xs = [0, 0, box_width, box_width]
    y_boxes_xs = [canvas_width-box_width, canvas_width-box_width, canvas_width, canvas_width]
    
    p = plot(ratio=:equal, framestyle=:none, xlims=(0, canvas_width), ylims=(0, canvas_height))
    
    leg_pos = zeros(m*k)
    
    for i in 0:(k-1)
        y_top = canvas_height - i * (box_height + spacing)
        y_bottom = y_top - box_height
        
        # Draw the O box
        p = plot!(o_boxes_xs, [y_bottom, y_top, y_top, y_bottom], seriestype=:shape, color=:lightgray, label=false)
        p = annotate!(box_width/2, y_top - box_height/2, text("\$O_{$(i+1)}\$", :black))
        # Draw the Y box
        p = plot!(y_boxes_xs, [y_bottom, y_top, y_top, y_bottom], seriestype=:shape, color=:lightgray, label=false)
        p = annotate!(canvas_width-box_width/2, y_top - box_height/2, text("\$Y_{$(i+1)}\$", :black))
        
        # Draw legs
        for j in 1:m
            # O
            p = plot!([box_width, box_width+leg_length], [y_top - j*box_height/5, y_top - j*box_height/5], color=:black, label=false,linewidth=2)
            p = annotate!(box_width+leg_length/2, y_top - (j-0.5)*box_height/5, text("\$\\mu_{$(m*i + j)}\$", :black))
            
            # Y
            p = plot!([canvas_width-box_width-leg_length, canvas_width-box_width], [y_top - j*box_height/5, y_top - j*box_height/5], color=:black, label=false,linewidth=2)
            p = annotate!(canvas_width-box_width-leg_length/2, y_top - (j-0.5)*box_height/5, text("\$\\alpha_{$(m*i + j)}\$", :black))
            
            leg_pos[i*m+j] = y_top - j*box_height/5
        end
    end
    
    # add connections
    o_sc(y, r, cx, cy) = cx + sqrt(abs(r^2 - (y-cy)^2)) 
    y_sc(y, r, cx, cy) = cx - sqrt(abs(r^2 - (y-cy)^2))
    for (k, (a, b)) in enumerate(b_elem)
        y1, y2 = leg_pos[a], leg_pos[b]
        r = abs(y1 - y2)/2
        cy = y2 + r
        cx = box_width+leg_length
        ys = range(y2, stop=y1, length=1000)
        
        p = plot!(o_sc.(ys, r, cx, cy), ys, label="\$$(a)\\leftrightarrow $(b)\$", style=:dash, c=mypalette[k], linewidth=2)
    end
    t_b_elem = transpose_brauer(b_elem)
    for (k, (a, b)) in enumerate(t_b_elem)
        y1, y2 = leg_pos[a], leg_pos[b]
        r = abs(y1 - y2)/2
        cy = y2 + r
        cx = canvas_width-box_width-leg_length
        ys = range(y2, stop=y1, length=1000)
        
        p = plot!(y_sc.(ys, r, cx, cy), ys, label=false, style=:dash, c=mypalette[k], linewidth=2)
    end
    # display(p)
    p = plot!(legend=:inside)
    return p
end

# ------------------------------------------------------------------------------------------------
# get_valid_choices
# Standalone helper for the local pairing constraints.
# Origin: notebook cell 206.
# Edit: kept as the standalone debug/helper version; `count_pairings` still carries its own local recursion.
# ------------------------------------------------------------------------------------------------
function get_valid_choices(i, remaining, m, k)
    g_i = ceil(Int, i / (m÷2)) # (i + 1) ÷ (m÷2)
    l_i = g_i - 1 == 0 ? 2k : g_i - 1
    r_i = g_i + 1 == 2k+1 ? 1 : g_i + 1
    
    println(l_i, " ", g_i, " ", r_i)
    
    excluded = [(m÷2)*(l_i-1)+1:(m÷2)*l_i...; (m÷2)*(g_i-1)+1:(m÷2)*g_i...; (m÷2)*(r_i-1)+1:(m÷2)*r_i...]
    return setdiff(remaining, excluded)
    
end

# ------------------------------------------------------------------------------------------------
# count_pairings
# Count the valid pairings subject to the notebook's local exclusion rules.
# Origin: notebook cell 207.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function count_pairings(m::Int, k::Int)
    n = m * k

    # Check if n is even, otherwise return 0 since pairs are not possible
    if n % 2 != 0
        return 0
    end
    
    # construcct dict of choices
    excluded = Dict()
    for g_i in 1:n
        l_i = g_i - 1 == 0 ? 2k : g_i - 1
        r_i = g_i + 1 == 2k+1 ? 1 : g_i + 1
        excluded[g_i] = [(m÷2)*(l_i-1)+1:(m÷2)*l_i...; (m÷2)*(g_i-1)+1:(m÷2)*g_i...; (m÷2)*(r_i-1)+1:(m÷2)*r_i...]
    end
    
    function get_valid_choices(i, remaining, m, k)
        g_i = ceil(Int, i / (m÷2))
        return setdiff(remaining, excluded[g_i])
    end

    # Recursive function to count valid pairings
    function valid_pairings(remaining, m, k)
        if length(remaining) == 0
            return 1
        end

        count = 0
        for i in get_valid_choices(remaining[1], remaining, m, k)
            count += valid_pairings(setdiff(remaining[2:end], [i]), m, k)
        end
        return count
    end

    return valid_pairings([1:n...], m, k)
end


# ==================================================================================================
# Passive-FLO helpers
# ==================================================================================================

# ------------------------------------------------------------------------------------------------
# flo_unitary
# Construct the passive-FLO two-mode unitary from Euler angles.
# Origin: notebook cells 211 and 213 (duplicate definitions).
# Edit: kept the later shared definition and removed the earlier duplicate from the notebook flow.
# ------------------------------------------------------------------------------------------------
function flo_unitary(φ, θ, ψ)
    exp(-im*φ*Jz) * exp(-im*θ*Jy) * exp(-im*ψ*Jz)
end

# ------------------------------------------------------------------------------------------------
# moment_operator
# Build the empirical moment operator from a list of sample unitaries.
# Origin: notebook cells 211 and 213.
# Edit: kept the more general later version with the `conj_second` keyword.
# Edit: this still supports the old no-keyword call pattern.
# ------------------------------------------------------------------------------------------------
function moment_operator(us::Vector{<:AbstractMatrix}; conj_second::Bool=true)
    d = size(us[1],1)
    M = zeros(ComplexF64, d^2, d^2)
    for U in us
        M .+= conj_second ? kron(U, conj(U)) : kron(U, U)
    end
    return M / length(us)
end

# ------------------------------------------------------------------------------------------------
# algebra_dimension
# Estimate the linear dimension of the span generated by the sampled unitaries.
# Origin: notebook cell 213.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function algebra_dimension(us::Vector{<:AbstractMatrix}; atol=1e-6)
    d = size(us[1],1)
    V = hcat([vec(U) for U in us]...)
    # numeric rank
    return rank(V; atol=atol)
end

# ------------------------------------------------------------------------------------------------
# random_flo
# Sample passive-FLO unitaries by drawing Euler angles with Haar-compatible measure.
# Origin: notebook cell 213.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function random_flo(N::Int)
    us = Vector{Matrix{ComplexF64}}(undef, N)
    for i in 1:N
        φ = 2π * rand()
        ψ = 2π * rand()
        u = 2 * rand() - 1
        θ = acos(clamp(u, -1, 1))
        us[i] = flo_unitary(φ, θ, ψ)
    end
    return us
end

# ------------------------------------------------------------------------------------------------
# random_flo_unitaries
# Backward-compatible name for sampling passive-FLO unitaries.
# Origin: notebook cell 211.
# Edit: kept for backward compatibility even though `random_flo` is the newer name used later in the notebook.
# ------------------------------------------------------------------------------------------------
function random_flo_unitaries(N::Int)
    return random_flo(N)
end


# ==================================================================================================
# MPS / MPO Majorana-GP helpers
# ==================================================================================================

# ------------------------------------------------------------------------------------------------
# zero_state_MPS
# Build the all-zero computational-basis MPS on the supplied sites.
# Origin: notebook cell 216.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function zero_state_MPS(sites)::MPS
    return truncate(MPS([ITensor([1., 0.], s) for s in sites]))
end;

# ------------------------------------------------------------------------------------------------
# get_input_states
# Generate random or structured MPS input states used in the later GP experiments.
# Origin: notebook cell 219.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function get_input_states(nq::Int, nstates::Int; nl::Int=1, p0=nothing)::Vector{MPS}
    _model = hea(nq, nl, Dict(q => "X" for q in 1:nq))
    if p0==nothing
        p0 = π*randn(_model.nparams)
    end
    net(t::Number) = get_forward_net(_model, p0 .* t)

    psi0 = zero_state_MPS(_model.d2_sites)
    states = [contract_net(copy(psi0), net(t), _model.targets_list) for t in 0:(1/(nstates-1)):1]
    return states
end

# ------------------------------------------------------------------------------------------------
# fill_antisymm_tensor
# Fill an antisymmetric tensor from values indexed by increasing combinations.
# Origin: notebook cells 83 and 220 (two arities).
# Edit: both arities were kept; the generalized MPS-facing version now uses a local `_permutation_sign` helper instead of relying on an external `levicivita` symbol.
# ------------------------------------------------------------------------------------------------
function fill_antisymm_tensor(n, m, values)
    # Initialize the 2n x 2n x 2n x 2n tensor with zeros
    tensor = zeros(Float64, [2n for _ in 1:m]...)
    # Fill the tensor with provided values
    for (c, value) in zip(combinations(1:2n, m), values)
        for (pc, parperm) in zip(permutations(c), permutations([1:m]...))
            tensor[pc...] = _permutation_sign(parperm) * value
        end
    end
    return tensor
end

# ------------------------------------------------------------------------------------------------
# mps_mpo_expval
# Evaluate an MPO expectation value between two MPS objects by explicit sequential contraction.
# Origin: notebook cell 221.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function mps_mpo_expval(
    ydag::MPS, A::MPO, x::MPS, loginner::Bool=false; make_inds_match::Bool=true, kwargs...
    )::Number
    N = length(A)
    sim!(linkinds, ydag)
    O = ydag[1] * A[1] * x[1]
    for j in 2:N
        O = O * ydag[j] * A[j] * x[j]
    end
    return O[]
end

# ------------------------------------------------------------------------------------------------
# get_y_comps
# Compute all expectation values of a list of MPOs on one MPS.
# Origin: notebook cell 222.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function get_y_comps(psi::MPS, ops::Vector{MPO})
    expvals = zeros(length(ops))
    Base.Threads.@threads for k in 1:length(ops)
        op = @views ops[k]
        expvals[k] = real(mps_mpo_expval(dag(psi)', op, psi))
    end
end

# ------------------------------------------------------------------------------------------------
# get_y_tensor
# Lift `get_y_comps` into an antisymmetric tensor representation.
# Origin: notebook cell 222.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function get_y_tensor(psi::MPS, ops::Vector{MPO}, m::Int)
    n = length(psi)
    expvals = get_y_comps(psi, ops)
    return fill_antisymm_tensor(n, m, expvals)
end

# ------------------------------------------------------------------------------------------------
# pauli_product
# Multiply two Pauli strings while keeping track of the phase.
# Origin: notebook cell 223.
# Edit: unchanged aside from documentation; moved next to the Pauli-symbol helpers it feeds.
# ------------------------------------------------------------------------------------------------
function pauli_product(ops1::Vector{Symbol}, ops2::Vector{Symbol})::Tuple{Vector{Symbol}, Number}
    length(ops1) == length(ops2) ? nothing : error("Incompatible Pauli string lengths.")
    result = fill(:id, length(ops1))
    coeff = 1

    for (k, (op1, op2)) in enumerate(zip(ops1, ops2))
        res, c = PAULI_MULTIPLICATION_RULES[(op1, op2)]
        coeff *= c
        result[k] = res
    end

    return (result, coeff)
end

# ------------------------------------------------------------------------------------------------
# pauli_product_bare
# Multiply two Pauli strings while discarding the phase.
# Origin: notebook cell 223.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function pauli_product_bare(ops1::Vector{Symbol}, ops2::Vector{Symbol})::Vector{Symbol}
    length(ops1) == length(ops2) ? nothing : error("Incompatible Pauli string lengths.")
    result = fill(:id, length(ops1))
    
    for (k, (op1, op2)) in enumerate(zip(ops1, ops2))
        result[k] = PAULI_MULTIPLICATION_RULES_BARE[(op1, op2)]
    end

    return result
end

# ------------------------------------------------------------------------------------------------
# majorana_symbol
# Return the Pauli-symbol representation of one Jordan–Wigner Majorana operator.
# Origin: notebook cell 223.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function majorana_symbol(n::Int, j::Int)::Vector{Symbol}
    majo = fill(:id, n)
    for k in 1:n
        if k < j÷2 + mod(j,2)
            majo[k] = :z
        elseif k == j÷2 + mod(j,2)
            mod(j,2) == 1 ? majo[k] = :x : majo[k] = :y
        else
            majo[k] = :id
        end
    end
    return majo
end

# ------------------------------------------------------------------------------------------------
# majorana_products
# Enumerate m-Majorana products together with their Pauli-string image and coefficient.
# Origin: notebook cell 223.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function majorana_products(n::Int, m::Int)
    
    bm_elems = Vector{Tuple{Vector{Symbol}, Number, Vector{Int}}}(undef, binomial(2n, m))

    majos = [majorana_symbol(n, j) for j in 1:2n]

    for (k, combo) in enumerate(Combinatorics.combinations(1:2n, m))
        res = fill(:id, n)
        # dummy = 1.
        for c in combo
            # res, dummy .= pauli_product(res, majos[c])
            res .= pauli_product_bare(res, majos[c])
        end
        coeff = (-1im)^floor(m/2)
        bm_elems[k] = (res, coeff, combo)
    end

    return bm_elems
end

# ------------------------------------------------------------------------------------------------
# majorana_products_bare
# Enumerate only the Pauli-string image of the m-Majorana products.
# Origin: notebook cell 223.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function majorana_products_bare(n::Int, m::Int)::Vector{Vector{Symbol}}
    
    bm_elems = fill(fill(:id, n), binomial(2n, m))
    majos = [majorana_symbol(n, j) for j in 1:2n]

    for (k, combo) in enumerate(Combinatorics.combinations(1:2n, m))
        res = fill(:id, n)
        for c in combo
            res .= pauli_product_bare(res, majos[c])
        end
        bm_elems[k] = res
    end

    return bm_elems
end

# ------------------------------------------------------------------------------------------------
# read_pauli_symb_string
# Convert a Pauli-symbol vector into a readable string like `IXYZ`.
# Origin: notebook cell 223.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function read_pauli_symb_string(pauli::Vector{Symbol})::String
    pstring = ""
    for psymb in pauli
        pstring *= PAULI_CHARS[psymb]
    end
    pstring
end

# ------------------------------------------------------------------------------------------------
# pauli_mpo
# Build an MPO from a symbolic Pauli string.
# Origin: notebook cell 224.
# Edit: unchanged aside from documentation; now lives next to the module-level `PAULI_MATS` constant.
# ------------------------------------------------------------------------------------------------
function pauli_mpo(sites::Vector{<:Index}, pauli::Vector{Symbol})::MPO
    n = length(sites)
    # op = MPO(ComplexF64, sites)
    # for k in 1:n
    #     op[k][fill(:, length(dims(op[k])))...] = PAULI_MATS[pauli[k]]
    # end
    
    op = MPO([ITensor(PAULI_MATS[pauli[k]], sites[k]', sites[k]) for k in 1:n])
    
    return op
end;

# ------------------------------------------------------------------------------------------------
# m_majoranas_ops
# Construct all MPOs associated with m-Majorana products.
# Origin: notebook cell 224.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function m_majoranas_ops(sites::Vector{<:Index}, m::Int)::Vector{MPO}
    n = length(sites)
    majo_symbs = majorana_products_bare(n, m)
    ops_list = [pauli_mpo(sites, pauli) for pauli in majo_symbs]
    return ops_list
end

# ------------------------------------------------------------------------------------------------
# majoranas_ops
# Construct MPOs from an explicitly supplied list of Majorana/Pauli symbols.
# Origin: notebook cell 224.
# Edit: unchanged aside from documentation.
# ------------------------------------------------------------------------------------------------
function majoranas_ops(sites::Vector{<:Index}, majo_symbs::Vector{Vector{Symbol}})::Vector{MPO}
    n = length(sites)
    ops_list = [pauli_mpo(sites, pauli) for pauli in majo_symbs]
    return ops_list
end

# ------------------------------------------------------------------------------------------------
# maintest
# Small smoke test for the MPS/MPO Majorana pipeline.
# Origin: notebook cell 231.
# Edit: kept as a smoke-test helper for the MPS/MPO pipeline.
# ------------------------------------------------------------------------------------------------
function maintest(tnq::Int, tm::Int)::Vector{Float64}
    # @time tstates = get_input_states(tnq, 100)
    tstates = [zero_state_MPS(siteinds(2,tnq))]
    ss = siteinds(tstates[1])

    @time tmajos = majorana_products_bare(tnq, tm);
    @time tops = majoranas_ops(ss, tmajos);
    @time aaa = get_y_comps(tstates[1], tops)
    aaa
end

end # module FFGPUtils
