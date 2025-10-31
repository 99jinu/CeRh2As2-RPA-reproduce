using JLD2
using Distributed

# cpu_number = 64
# addprocs(cpu_number)

# @everywhere include(joinpath(@__DIR__, "../src/core.jl"))
include("../src/test.jl")

println("Core functions loaded on all workers.")
N_urange = 3
N = 64
mu = 0.5
U_range = LinRange(0.0, 3.0, N_urange)

file_name = pwd()*"/analysis/raw_data/static_suscep_BZmap_$mu"*"_64x64.jld2"
data = JLD2.load(file_name)
BZ = BrillouinZone(:square, N)
chi_grid = JLD2.load(file_name)["chi_BZ_map"]


println("Projection method has started.")
t1 = time()
lambda_gap_U = array_compute(lambda_proj(), U_range, chi_grid, BZ, mu, run, weight_array = compute_module.spect_weight_array)
t2 = time()
println("Projection method ended", t2-t1, "seconds")

title_string = string("lambda_U_projection_$mu.jld2")
@save title_string U_range lambda_gap_U

println("Diagonalization method has started.")
t1 = time()
lambda_diag_U = array_compute(lambda_diag(), U_range, chi_grid, BZ, mu, run, weight_array = compute_module.spect_weight_array)
t2 = time()
println("Diagonalization method ended", t2-t1, "seconds")
title_string = string("lambda_U_diagonalization_$mu.jld2")
@save title_string U_range lambda_diag_U