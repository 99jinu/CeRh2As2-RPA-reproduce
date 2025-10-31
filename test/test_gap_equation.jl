using Test, Random, JLD2, LinearAlgebra, Plots, StaticArrays

include("../src/physics.jl"); using .physics_module
include("../src/types.jl"); using .types_module
include("../src/gap_equation.jl")
using .LGE_solver: spin_diagonalized_interaction, test_interaction, tanh_weight_array, spect_weight_array, weight_array2, gap_equation_solver

@testset "test_interaction_Creating Procedure" begin
    BZ = BrillouinZone(:square, 12)
    k_grid = BZ.k_grid
    run = (soft_eta = 1e-6, T = 0.01)
    w_array = tanh_weight_array(BZ, 0.5, run)
    test_V = test_interaction(BZ, w_array)
    i1, i2 = rand(1:144), rand(1:144)
    k1, k2 = k_grid[i1], k_grid[i2]
    dx2y2w2 = (cos(k1[1])-cos(k1[2])) *(cos(k2[1])-cos(k2[2])) * w_array[i2] 
    error = test_V(i1, i2) - dx2y2w2
    println("error: $error")
end


N = 64
BZ = BrillouinZone(:square, N)
k_range = BZ.k_range
k_grid = BZ.k_grid
mu = 0.5 
run = (soft_eta = 1e-6, T = 0.01)
file_name = pwd()*"/analysis/raw_data/static_suscep_BZmap_$mu"*"_64x64.jld2"
BZ = BrillouinZone(:square, N)
chi_grid = JLD2.load(file_name)["chi_BZ_map"]

w_array = spect_weight_array(BZ, mu, run)

idx_max = argmax(w_array)
U = 0.5
gap_type = :dx2y2
gap_basis = GapBasis(gap_type)
eff_V = spin_diagonalized_interaction(gap_basis.parity, chi_grid, U, w_array)
eff_V_matrix = reshape([eff_V(idx_max, idx) for idx in 1:N^2], N, N)
p1 = heatmap(k_range, k_range, real(eff_V_matrix))

half = div(N,2)
idx_zero = half*N + half + 1
eff_V_matrix = reshape([eff_V(idx_zero, idx) for idx in 1:N^2], N, N)
p2 = heatmap(k_range, k_range, real(eff_V_matrix))
p = plot(p1, p2)
savefig(p, "eff_V(k_FS, k_vec).png")

@testset "spin_diagonalized_interaction check" begin
    N = 64
    mu, U = 0.5, 0.5
    run = (soft_eta = 1e-6, T = 0.01)
    gap_type = :dx2y2
    gap_basis = GapBasis(gap_type)
    file_name = pwd()*"/analysis/raw_data/static_suscep_BZmap_$mu"*"_64x64.jld2"
    BZ = BrillouinZone(:square, N)
    chi_grid = JLD2.load(file_name)["chi_BZ_map"]
    w_array = tanh_weight_array(BZ, mu, run)
      
    eff_V = spin_diagonalized_interaction(gap_basis.parity, chi_grid, U, w_array)
    half = div(N,2)
    zero_point_flatten = half*N + half + 1
    k_range = BZ.k_range
    k_grid = BZ.k_grid 

    @test k_range[half+1] == 0.0
    eff_vector = [eff_V(idx, zero_point_flatten) for idx in 1:N^2]
    @test k_grid[zero_point_flatten] == [0.0, 0.0]
    e0 = eigen(LGE_solver.H[](@SVector[0.0, 0.0], mu)).values[1]
    w0 = tanh(e0/(2*run.T)) / (2*e0)
    @test w_array[zero_point_flatten] == w0

    idx = rand(1:N^2)
    i1, i2 = Tuple(CartesianIndices((N,N))[idx])
    eff_matrix = reshape(eff_vector, (N,N))
    @test eff_vector[idx] == eff_matrix[idx]
    @test eff_vector[idx] == eff_matrix[i1, i2]

    chi_0 = chi_grid[i1, i2]
    chi_0_flatten = chi_grid[idx]
    @test chi_0 == chi_0_flatten
    chi_p = prompt_matrix_mod(chi_grid, idx, :plus, zero_point_flatten)
    chi_m = prompt_matrix_mod(chi_grid, idx, :minus, zero_point_flatten)
    @test chi_p == chi_0
    @test chi_m == chi_0
    @test eff_V(idx, zero_point_flatten) == eff_matrix[i1, i2]
    @test -eff_matrix[i1, i2] == (-U^2*chi_0/(1.0+U*chi_0) + U^3*chi_0^2/(1-U^2*chi_0^2)) * w_array[zero_point_flatten]
    p = heatmap(k_range, k_range, real(eff_matrix))
    savefig(p, "eff_V(k_vec, 0_vec).png")
end

@testset "Proj method with eff_V = 1" begin
    N = 32
    BZ = BrillouinZone(:square, N)
    k_grid = BZ.k_grid
    run = (soft_eta = 1e-6, T = 0.01)
    w_array = [1.0 for i in 1:N^2]
    test_V = (i1,i2) -> 
    if i1==i2
        1
    else
        0
    end

    gapbasis = GapBasis(:s_const)
    lge_projmethod = LGE_ProjMethod(test_V, gapbasis, BZ)
    lge_diagmethod = LGE_DiagMethod(test_V, BZ)

    proj_result = gap_equation_solver(lge_projmethod)
    diag_result, diag_vector = gap_equation_solver(lge_diagmethod)
    
    println("proj_result = $proj_result")
    println("diag_result = $diag_result")
    @test proj_result == diag_result 
end

@testset "GapBasis check" begin
    N = 32
    BZ = BrillouinZone(:square, N)
    k_range = BZ.k_range
    println(typeof(k_range))
    k_grid = BZ.k_grid
    type = :dx2y2
    gapbasis = GapBasis(type)
    gapftn = gapbasis.ftn
    idx1 = rand(1:N^2)
    i1, i2 = Tuple(CartesianIndices((N,N))[idx1])
    @test gapftn(k_grid[idx1]) == cos(k_range[i1])-cos(k_range[i2])
end

@testset "Proj and Diag consistency check for test interaction" begin
    N = 32
    BZ = BrillouinZone(:square, N)
    k_grid = BZ.k_grid
    k_range = BZ.k_range
    
    gap_idx = 5
    gap_type_list = [:s_const, :s_ext, :dx2y2, :dxy, :px, :py]
    gap_weight_array=zeros(Float64, length(gap_type_list))
    gap_weight_array[gap_idx] = 1.
    gap_type = gap_type_list[gap_idx]
    w_array = [1.0 for i in 1:N^2]
    
    test_V = test_interaction(BZ, w_array, gap_weight_array)
    gapbasis = GapBasis(gap_type)
    gapftn = gapbasis.ftn
    lge_projmethod = LGE_ProjMethod(test_V, gapbasis, BZ)
    lge_diagmethod = LGE_DiagMethod(test_V, BZ)

    proj_result = gap_equation_solver(lge_projmethod)
    diag_result, diag_vector = gap_equation_solver(lge_diagmethod)
    println("gap type: ", gap_type)
    println("proj_result = $proj_result")
    println("diag_result = $diag_result")
    @test abs(proj_result-diag_result)<1e-6
    p = heatmap(k_range, k_range, real(diag_vector)', title = "η = $gap_type at $N x $N grid")
    savefig(p, "diag_at_testV_$gap_type")

    idx1, idx2 = rand(1:N^2), rand(1:N^2)
    k1, k2 = k_grid[idx1], k_grid[idx2]
    dx2y2w2 = gapftn(k1)*gapftn(k2)*w_array[idx2]
    @test test_V(idx1,idx2) ==dx2y2w2
end

@testset "Proj consistency check for rand matrix" begin
    N = 32
    BZ = BrillouinZone(:square, N)
    k_grid = BZ.k_grid
    norm = N^2

    rand_matrix = Hermitian(rand(N^2, N^2))
    eigvals, eigvecs = eigen(rand_matrix)
    idx = rand(1:N^2)
    test_V = (idx1, idx2) -> rand_matrix[idx1, idx2]

    eig_ftn = k -> begin 
    i1, i2 = Tuple(findfirst(x -> x == k, k_grid)) 
    flatten_idx = i1 +N*(i2-1)
    return eigvecs[flatten_idx, idx]
    end

    vector = [eig_ftn(k_grid[idx]) for idx in 1:N^2]

    rand_gap_basis = GapBasis(:rand, eig_ftn, :rand)
    rand_lge_method = LGE_ProjMethod(test_V, rand_gap_basis, BZ)

    proj_result = gap_equation_solver(rand_lge_method)
    println(eigvals[idx]/norm)
    println(proj_result)
    @test abs(eigvals[idx]/norm-proj_result)<1e-6
end

@testset "Proj method with spin_diag_V check" begin
    run = (soft_eta = 1e-6, T = 0.01)
    mu, U = 3.9, 10.0
    N = 64
    gap_type = :dxy
    gap_basis = GapBasis(gap_type)
    file_name = pwd()*"/analysis/raw_data/static_suscep_BZmap_$mu"*"_64x64.jld2"
    BZ = BrillouinZone(:square, N)
    chi_grid = JLD2.load(file_name)["chi_BZ_map"]
    w_array = tanh_weight_array(BZ, mu, run)
    
    eff_V = spin_diagonalized_interaction(gap_basis.parity, chi_grid, U, w_array)
    lge_projmethod = LGE_ProjMethod(eff_V, gap_basis, BZ)
    lge_diagmethod = LGE_DiagMethod(eff_V, BZ)

    proj_result = gap_equation_solver(lge_projmethod)
    diag_result, _ = gap_equation_solver(lge_diagmethod)

    println(gap_type, "proj_result: $proj_result")
    println(gap_type, "diag_result: $diag_result")
end

@testset "prompt_matrix_mod test" begin
    N = 4
    BZ = BrillouinZone(:square, N)
    k_range = BZ.k_range
    k_grid = BZ.k_grid
    half = div(N,2)
    zero_point_flatten = half*N + half + 1

    idx1 = rand(1:N^2) 
    result_plus = prompt_matrix_mod(k_grid, idx1, :plus, zero_point_flatten)
    result_minus = prompt_matrix_mod(k_grid, idx1, :minus, zero_point_flatten)
    @test result_plus == result_minus
    @test result_plus == k_grid[idx1]

    result2_minus = prompt_matrix_mod(k_grid, idx1, :minus, idx1)
    @test result2_minus == k_grid[zero_point_flatten]
end

@testset "spect_weight_array test" begin
    N = 64
    mu = 0.5
    run = (soft_eta = 1e-6, T = 0.01)
    BZ = BrillouinZone(:square, N)
    k_range = BZ.k_range
    k_grid = BZ.k_grid
    half = div(N,2)
    zero_point_flatten = half*N + half + 1
    @test k_grid[zero_point_flatten] == [0.0, 0.0]
    w_array = log.(spect_weight_array(BZ, mu, run, epsilon = 1e-3))
    w_matrix = reshape(w_array, N, N)
    p = heatmap(k_range, k_range, w_matrix')
    savefig(p, pwd() * "/analysis/figure/spect_w_array_μ=$mu.png")
end