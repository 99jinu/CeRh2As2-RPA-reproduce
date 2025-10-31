include("../src/test.jl")

using JLD2, Plots
begin
    mu, N, U = 0.5, 64, 1.5
    run = (soft_eta = 1e-6, T = 0.01)
    gap_type = :dx2y2

    file_name = pwd()*"/analysis/raw_data/static_suscep_BZmap_$mu"*"_64x64.jld2"
    chi_grid = JLD2.load(file_name)["chi_BZ_map"]

    gap_basis = GapBasis(gap_type)
    gap_ftn = gap_basis.ftn
    parity = gap_basis.parity
    BZ = BrillouinZone(:square, N)
    k_range, k_grid = BZ.k_range, BZ.k_grid
    w_array = compute_module.spect_weight_array(BZ, mu, run)
    FS = reshape(w_array, N, N)
    eff_V_ftn_singlet = compute_module.spin_diagonalized_interaction(:singlet, chi_grid, U, w_array)
    eff_V_ftn_triplet = compute_module.spin_diagonalized_interaction(:triplet, chi_grid, U, w_array)

end

# for idx in eachindex(w_array)
#     w_max = maximum(w_array)
#     if w_array[idx] == w_max
#         kx, ky = round.(k_grid[idx]/pi, digits=2)
#         println("idx: $idx, kx: $kx, ky: $ky")
#     end
# end

# idx: 1540, kx: -0.91, ky: -0.25 # idx: 217, kx: -0.25, ky: -0.91 
# idx: 3945, kx: 0.25, ky: 0.91 # idx: 2622, kx: 0.91, ky: 0.25 

# idx: 1598, kx: 0.91, ky: -0.25 # idx: 233, kx: 0.25, ky: -0.91 
# idx: 3929, kx: -0.25, ky: 0.91 # idx: 2564, kx: -0.91, ky: 0.25 
begin 
    idx1, idx2 = 2564, 3929
    χ_p = prompt_matrix_mod(chi_grid, idx1, :plus, idx2)
    χ_m = prompt_matrix_mod(chi_grid, idx1, :minus, idx2)
    singlet_int= compute_module.LGE_solver.singlet_interaction(U, chi_grid, idx1, idx2)
    triplet_int= compute_module.LGE_solver.triplet_interaction(U, chi_grid, idx1, idx2)
end

singlet_int == -U^2*χ_p/(1.0+U*χ_p) + U^3*χ_m^2/(1-U^2*χ_m^2)
triplet_int == U^2*χ_m/(1-U^2*χ_m^2)

function tmp_BZ_plot(matrix; xlabel = "kx", ylabel="ky",title="")
    return heatmap(k_range, k_range, real(matrix)',xlabel = xlabel, ylabel=ylabel, title = title, size = (500, 400))
end
eff_V_ftn_triplet(idx1, idx2)
eff_interaction_matrix_singlet = reshape([eff_V_ftn_singlet(idx, idx2) for idx in eachindex(k_grid)], N, N)    
eff_interaction_matrix_triplet = reshape([eff_V_ftn_triplet(idx, idx2) for idx in eachindex(k_grid)], N, N)
tmp_BZ_plot(eff_interaction_matrix_triplet, title="V_same(k, k2)")
tmp_BZ_plot(eff_interaction_matrix_singlet, title="V_opp(k, k2)")

lge_diagmethod = LGE_DiagMethod(eff_V_ftn, BZ)

diag_eigval, diag_eigvec = compute_module.gap_equation_solver(lge_diagmethod)
@save "tmp_LGE_diag_mu=$mu, U=$U.jld2" diag_eigval diag_eigvec

c_eigval, c_eigvec = compute(lambda_diag(), U, chi_grid, BZ, w_array)
@save "tmp_LGE_compute_mu=$mu, U=$U.jld2" c_eigval c_eigvec

tmp_BZ_plot(c_eigvec)

eff_V_ftn_triplet = compute_module.spin_diagonalized_interaction(:triplet, chi_grid, U, w_array)
lge_diagmethod_triplet = LGE_DiagMethod(eff_V_ftn_triplet, BZ)

diag_eigval, diag_eigvec = compute_module.gap_equation_solver(lge_diagmethod_triplet)
@save "tmp_LGE_diag_triplet_mu=$mu, U=$U.jld2" diag_eigval diag_eigvec
tmp_BZ_plot(diag_eigvec)