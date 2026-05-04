#!/usr/bin/env julia

# Regenerate the circular-cut XXZ/QGP dataset.
#
# This script is intentionally safe to `include`: all work is inside `main`, and
# `main()` is called only when the file is executed directly from the Julia CLI.
#
# Default output files:
#   data/circle_true_outputs_m2_<n>q.npy
#   data/circle_true_outputs_m2_<n>q.jld2
#   data/circle_true_overlaps_m2_<n>q.npy
#   data/circle_true_overlaps_m2_<n>q.jld2

using Dates
using JSON
using ITensors
using ITensorMPS

const SCRIPT_DIR = @__DIR__
const PROJECT_ROOT = dirname(SCRIPT_DIR)
const SRC_DIR = joinpath(PROJECT_ROOT, "src")
const INPUT_DIR = joinpath(PROJECT_ROOT, "config")
const OUTPUT_DIR = joinpath(PROJECT_ROOT, "data")

include(joinpath(SRC_DIR, "xxz_data_generation.jl"))
using .XXZDataGeneration

"""
    main(; kwargs...)

Regenerate the circular-cut dataset used by the XXZ/QGP regression notebook.

Keyword arguments:

- `input_file`: JSON metadata file with keys `n`, `thetas_len`, and `r`.
  Optional keys `theta_start` and `theta_stop` override the default interval
  `[4π/6, 5π/6]`.
- `output_dir`: directory where arrays and a copy of the metadata are saved.
- `m=2`: Majorana degree used for the QGP feature map.
- `threaded=Threads.nthreads() > 1`: parallelize over angular points.
- `storage_eltype=Float32`: precision used for features and compressed output.
- `save_npy=true`: save notebook-compatible `.npy` arrays.
- `save_jld2=true`: save compressed `.jld2` arrays.
- `compression_level=9`: Zstd compression level for JLD2 output.
- `dmrg_maxdim=32`, `dmrg_cutoff=1e-8`, `dmrg_nsweeps=30`: DMRG settings.
"""
function main(;
    input_file::AbstractString = joinpath(INPUT_DIR, "circle_initial_variables.json"),
    output_dir::AbstractString = OUTPUT_DIR,
    m::Integer = 2,
    threaded::Bool = Threads.nthreads() > 1,
    storage_eltype::Type{<:AbstractFloat} = Float32,
    save_npy::Bool = true,
    save_jld2::Bool = true,
    compression_level::Integer = 9,
    dmrg_maxdim::Integer = 32,
    dmrg_cutoff::Real = 1e-8,
    dmrg_nsweeps::Integer = 30,
)
    metadata = JSON.parsefile(input_file)

    nq = read_required(metadata, "n", Int)
    thetas_len = read_required(metadata, "thetas_len", Int)
    radius = read_required(metadata, "r", Float64)

    theta_start = Float64(get(metadata, "theta_start", 4π / 6))
    theta_stop = Float64(get(metadata, "theta_stop", 5π / 6))

    theta_values = collect(range(theta_start; stop = theta_stop, length = thetas_len))
    points = circle_points(theta_values, radius)

    @info "Starting circular-cut dataset generation" n = nq m = m npoints = length(points) threaded = threaded storage_eltype = storage_eltype

    sites = siteinds("Qubit", nq)
    majorana_mpos = build_majorana_mpos(sites, m)
    observable_mpo = sum_z_mpo(sites)

    t0 = time()
    features, observable_vec = compute_dataset(
        points,
        sites,
        majorana_mpos,
        observable_mpo;
        storage_eltype = storage_eltype,
        threaded = threaded,
        nsweeps = dmrg_nsweeps,
        maxdim = dmrg_maxdim,
        cutoff = dmrg_cutoff,
    )
    @info "Computed features and observables" elapsed_minutes = round((time() - t0) / 60; digits = 2)

    @info "Computing overlap matrix" size = (length(points), length(points))
    overlaps = gram_matrix(features; storage_eltype = storage_eltype)

    mkpath(output_dir)

    metadata_to_write = copy(metadata)
    metadata_to_write["theta_start"] = theta_start
    metadata_to_write["theta_stop"] = theta_stop
    write_metadata_json(joinpath(output_dir, "circle_initial_variables.json"), metadata_to_write)

    metadata_with_run_info = copy(metadata_to_write)
    metadata_with_run_info["majorana_degree"] = m
    metadata_with_run_info["storage_eltype"] = string(storage_eltype)
    metadata_with_run_info["generated_at_utc"] = string(Dates.now(Dates.UTC))
    metadata_with_run_info["dmrg_maxdim"] = dmrg_maxdim
    metadata_with_run_info["dmrg_cutoff"] = dmrg_cutoff
    metadata_with_run_info["dmrg_nsweeps"] = dmrg_nsweeps

    prefix = joinpath(output_dir, "circle_true")
    suffix = "m$(m)_$(nq)q"

    save_array(
        "$(prefix)_outputs_$(suffix)",
        observable_vec,
        :true_outputs;
        save_npy = save_npy,
        save_jld2 = save_jld2,
        storage_eltype = storage_eltype,
        compression_level = compression_level,
        metadata = metadata_with_run_info,
    )

    save_array(
        "$(prefix)_overlaps_$(suffix)",
        overlaps,
        :true_overlaps;
        save_npy = save_npy,
        save_jld2 = save_jld2,
        storage_eltype = storage_eltype,
        compression_level = compression_level,
        metadata = metadata_with_run_info,
    )

    @info "Finished circular-cut dataset generation" output_dir = output_dir
    return nothing
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    main()
end
