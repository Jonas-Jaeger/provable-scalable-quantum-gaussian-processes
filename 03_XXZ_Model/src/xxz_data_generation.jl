module XXZDataGeneration

using Dates
using JSON
using LinearAlgebra
using NPZ
using JLD2
using ITensors
using ITensorMPS
using ProgressMeter

include(joinpath(@__DIR__, "../../src/majo_machinery.jl"))

export build_majorana_mpos,
       compute_dataset,
       generate_ground_state,
       gram_matrix,
       grid_points,
       circle_points,
       read_required,
       save_array,
       sum_z_mpo,
       write_metadata_json,
       xxz_complex_hamiltonian

const PAULI_MATRICES = Dict{Symbol, Matrix{ComplexF64}}(
    :id => ComplexF64[1 0; 0 1],
    :x  => ComplexF64[0 1; 1 0],
    :y  => ComplexF64[0 -1im; 1im 0],
    :z  => ComplexF64[1 0; 0 -1],
)

"""
    read_required(metadata, key, T)

Read `key` from a JSON metadata dictionary and convert it to type `T`.
An informative error is thrown if the key is absent.
"""
function read_required(metadata::AbstractDict, key::AbstractString, ::Type{T}) where {T}
    haskey(metadata, key) || throw(ArgumentError("Missing required metadata key: $key"))
    return T(metadata[key])
end

"""
    write_metadata_json(path, metadata)

Write `metadata` as a pretty-printed JSON file. Parent directories are created
if needed.
"""
function write_metadata_json(path::AbstractString, metadata::AbstractDict)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, metadata, 4)
        println(io)
    end
    return path
end

"""
    pauli_itensor(sites, op, site)

Construct the one-site ITensor representation of a Pauli operator. The symbols
accepted are `:id`, `:x`, `:y`, and `:z`, matching the output convention of
`majorana_products` from `majo_machinery.jl`.
"""
function pauli_itensor(sites::AbstractVector, op::Symbol, site::Integer)
    1 <= site <= length(sites) || throw(BoundsError(sites, site))
    haskey(PAULI_MATRICES, op) || throw(ArgumentError("Unknown Pauli symbol: $op"))
    return ITensor(PAULI_MATRICES[op], sites[site]', sites[site])
end

"""
    xxz_complex_hamiltonian(N, h1, h2; J=1.0, Jz=1.0, include_closing_term=true)

Build the OpSum used in the XXZ/QGP dataset generation. The local field is
`h1` on odd sites and `h2` on even sites. The two-body interaction follows the
original script: all pairs are included, and an additional `(N, 1)` closing term
is added when `include_closing_term=true`.

The closing term is kept enabled by default to preserve the convention used to
generate the original data. Set `include_closing_term=false` only when you
intend to regenerate a modified dataset.
"""
function xxz_complex_hamiltonian(
    N::Integer,
    h1::Real,
    h2::Real;
    J::Real = 1.0,
    Jz::Real = 1.0,
    include_closing_term::Bool = true,
)
    N > 0 || throw(ArgumentError("N must be positive."))

    H = OpSum()

    for j in 1:N
        H += 2.0 * (isodd(j) ? h1 : h2), "Sz", j
    end

    for i in 1:N
        for j in 1:(i - 1)
            H += -4.0 * J,  "Sx", i, "Sx", j
            H += -4.0 * J,  "Sy", i, "Sy", j
            H += -4.0 * Jz, "Sz", i, "Sz", j
        end
    end

    if include_closing_term && N > 1
        H += -4.0 * J,  "Sx", N, "Sx", 1
        H += -4.0 * J,  "Sy", N, "Sy", 1
        H += -4.0 * Jz, "Sz", N, "Sz", 1
    end

    return H
end

"""
    generate_ground_state(sites, h1, h2; kwargs...) -> MPS

Compute the DMRG ground state of `xxz_complex_hamiltonian(length(sites), h1, h2)`.
The defaults reproduce the settings of the original data-generation script.

Keyword arguments:

- `nsweeps=30`
- `maxdim=32`
- `cutoff=1e-8`
- `init_linkdim=4`
- `outputlevel=0`
- `hamiltonian=xxz_complex_hamiltonian`
"""
function generate_ground_state(
    sites::AbstractVector,
    h1::Real,
    h2::Real;
    nsweeps::Integer = 30,
    maxdim::Integer = 32,
    cutoff::Real = 1e-8,
    init_linkdim::Integer = 4,
    outputlevel::Integer = 0,
    hamiltonian::Function = xxz_complex_hamiltonian,
)
    H = MPO(hamiltonian(length(sites), Float64(h1), Float64(h2)), sites)
    psi0 = randomMPS(sites, init_linkdim)
    _, psi = dmrg(
        H,
        psi0;
        nsweeps = nsweeps,
        maxdim = [maxdim],
        cutoff = [cutoff],
        outputlevel = outputlevel,
    )
    return psi
end

"""
    build_majorana_mpos(sites, m) -> Vector{MPO}

Build the Majorana-product MPOs `Γ_k` of degree `m`, including the scalar phase
coefficient returned by `majorana_products`. The Pauli-string convention is
inherited from `majo_machinery.jl`.
"""
function build_majorana_mpos(sites::AbstractVector, m::Integer)
    products = majorana_products(length(sites), m)
    operators = Vector{MPO}(undef, length(products))

    @inbounds for k in eachindex(products)
        coeff, ops = products[k]
        mpo = MPO([pauli_itensor(sites, ops[j], j) for j in eachindex(ops)])
        operators[k] = mpo * real(coeff)
    end

    return operators
end

"""
    sum_z_mpo(sites) -> MPO

Construct the observable `2 * sum_j Sz_j` as an MPO.
"""
function sum_z_mpo(sites::AbstractVector)
    obs = OpSum()
    for j in eachindex(sites)
        obs += 2.0, "Sz", j
    end
    return MPO(obs, sites)
end

"""
    grid_points(t1_values, t2_values) -> Vector{Tuple{Float64, Float64}}

Return the grid points `(t1, t2)` in Julia column-major order, i.e. `t1` is the
fast index. Consequently, an observable vector returned from `compute_dataset`
can be reshaped as `reshape(y, length(t1_values), length(t2_values))`.
"""
function grid_points(t1_values, t2_values)
    points = Vector{Tuple{Float64, Float64}}(undef, length(t1_values) * length(t2_values))
    i = 1
    for t2 in t2_values
        for t1 in t1_values
            points[i] = (Float64(t1), Float64(t2))
            i += 1
        end
    end
    return points
end

"""
    circle_points(theta_values, radius) -> Vector{Tuple{Float64, Float64}}

Return the circular-cut parameters `(radius*cos(theta), radius*sin(theta))`.
"""
function circle_points(theta_values, radius::Real)
    return [(Float64(radius) * cos(θ), Float64(radius) * sin(θ)) for θ in theta_values]
end

"""
    compute_dataset(points, sites, majorana_mpos, observable_mpo; kwargs...)

For every parameter pair `(h1, h2)` in `points`, compute the DMRG ground state,
the Majorana feature vector

```math
f_k(h_1,h_2) = \\langle \\psi(h_1,h_2) | \\Gamma_k | \\psi(h_1,h_2) \\rangle,
```

and the observable value `⟨2 * sum_j Sz_j⟩`.

The ground states are streamed and discarded immediately, so the script does not
store a vector of MPS objects. This is important for the `n=100` data.

Keyword arguments:

- `storage_eltype=Float32`: type used for the feature and observable arrays.
- `threaded=Threads.nthreads() > 1`: evaluate parameter points in parallel.
- `show_progress=true`: display a progress meter.
- DMRG options accepted by `generate_ground_state`, such as `maxdim` and `cutoff`.
"""
function compute_dataset(
    points::AbstractVector{<:Tuple{<:Real, <:Real}},
    sites::AbstractVector,
    majorana_mpos::AbstractVector{<:MPO},
    observable_mpo::MPO;
    storage_eltype::Type{<:AbstractFloat} = Float32,
    threaded::Bool = Threads.nthreads() > 1,
    show_progress::Bool = true,
    ground_state_kwargs...,
)
    npoints = length(points)
    nfeatures = length(majorana_mpos)

    features = Matrix{storage_eltype}(undef, npoints, nfeatures)
    observables = Vector{storage_eltype}(undef, npoints)

    progress = show_progress ? Progress(npoints; desc = "ground states", showspeed = true) : nothing

    function do_point!(i::Int)
        h1, h2 = points[i]
        psi = generate_ground_state(sites, h1, h2; ground_state_kwargs...)
        psibra = psi'

        @inbounds begin
            for k in eachindex(majorana_mpos)
                features[i, k] = storage_eltype(real(inner(psibra, majorana_mpos[k], psi)))
            end
            observables[i] = storage_eltype(real(inner(psibra, observable_mpo, psi)))
        end

        progress === nothing || next!(progress)
        return nothing
    end

    if threaded
        BLAS.set_num_threads(1)
        Threads.@threads for i in 1:npoints
            do_point!(i)
        end
    else
        for i in 1:npoints
            do_point!(i)
        end
    end

    return features, observables
end

"""
    gram_matrix(features; storage_eltype=eltype(features)) -> Matrix

Compute the precomputed QGP overlap matrix `features * features'` using BLAS and
without forming extra large temporaries.
"""
function gram_matrix(features::AbstractMatrix; storage_eltype::Type{<:AbstractFloat} = eltype(features))
    K = Matrix{storage_eltype}(undef, size(features, 1), size(features, 1))
    mul!(K, features, transpose(features))
    return K
end

"""
    save_array(stem, array, key; kwargs...)

Save `array` to one or both supported output formats. The extension is appended
automatically to `stem`.

Keyword arguments:

- `save_npy=true`: write `stem * ".npy"` using NPZ.jl.
- `save_jld2=true`: write `stem * ".jld2"` using JLD2.
- `storage_eltype=Float32`: type stored in the JLD2 file.
- `compression_level=9`: Zstd compression level for the JLD2 output.
- `metadata=Dict()`: metadata stored together with the JLD2 dataset.
"""
function save_array(
    stem::AbstractString,
    array::AbstractArray,
    key::Symbol;
    save_npy::Bool = true,
    save_jld2::Bool = true,
    storage_eltype::Type{<:AbstractFloat} = Float32,
    compression_level::Integer = 9,
    metadata::AbstractDict = Dict{String, Any}(),
)
    mkpath(dirname(stem))

    if save_npy
        npy_path = stem * ".npy"
        npzwrite(npy_path, array)
        @info "Saved NumPy array" path = npy_path size_gb = round(filesize(npy_path) / 1024^3; digits = 3)
    end

    if save_jld2
        jld2_path = stem * ".jld2"
        stored_array = storage_eltype.(array)
        payload = NamedTuple{(key, :metadata)}((stored_array, metadata))
        jldsave(jld2_path; compress = ZstdFilter(level = compression_level), payload...)
        @info "Saved compressed JLD2 array" path = jld2_path size_gb = round(filesize(jld2_path) / 1024^3; digits = 3)
    end

    return stem
end

end # module
