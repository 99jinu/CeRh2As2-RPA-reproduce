using JLD2, Plots

include("../src/test.jl")

N=64
mu = 0.5
file_name = pwd()*"/analysis/raw_data/static_suscep_BZmap_$mu"*"_64x64.jld2"
BZ = BrillouinZone(:square, N)
chi_grid = JLD2.load(file_name)["chi_BZ_map"]
k_grid = BZ.k_grid

method = :plus
idx2 = 1 + 1024 + 16 
kx_pi, ky_pi = k_grid[idx2]/pi
begin shifted_chi_grid = zeros(ComplexF64, N, N)
    for i in 1:N, j in 1:N
        idx1 = i + N * (j-1)
        shifted_chi_grid[i,j] = prompt_matrix_mod(chi_grid, idx1, method, idx2)
    end
    p = heatmap(BZ.k_range, BZ.k_range, -real(chi_grid)', title = "original")
    p_shifted = heatmap(BZ.k_range, BZ.k_range, -real(shifted_chi_grid)', title = "$method: $kx_pi, $ky_pi")
    p_tot = plot(p, p_shifted, size = (1000, 400))
end

#transponse is essential
begin test_grid = zeros(Float64, N, N)
    for i in 1:N, j in 1:N
        if i%2 == 0
            test_grid[i, j] = 1
        end
    end
    heatmap(1:N, 1:N, test_grid, title="Transponse necessary! grid[even, :] = 1")
end