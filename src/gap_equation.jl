module LGE_solver

using ..physics_module
using ..types_module
using LinearAlgebra, StaticArrays


const H = Ref{Function}((k, mu) 
-> H = ((-2.0)*(cos(k[1]) + cos(k[2])) - mu) * LinearAlgebra.I(2))  


export gap_equation_solver, tanh_weight_array, spin_diagonalized_interaction, spin_diagonalized_interaction_sym, spect_weight_array

function gap_equation_solver(method::LGE_ProjMethod)
    eff_interaction = method.eff_interaction
    w_array = method.weight_array
    gap_function = method.gap_basis.ftn
    k_grid = method.BZ.k_grid
    dim = length(k_grid) # k_grid_number
    vector = [gap_function(k_grid[idx]) for idx in 1:dim]
    
    lambda = 0.0
    norm = 0.0
    for i1 in 1:dim
        for i2 in 1:dim
            lambda += vector[i1]' * eff_interaction(i1, i2) * w_array[i2] * vector[i2]
        end
        norm += vector[i1]' * vector[i1]
    end
    return lambda/norm/dim
end

function gap_equation_solver(method::LGE_ProjMethod, sym::Symbol)
    eff_interaction = method.eff_interaction
    w_array = method.weight_array
    gap_function = method.gap_basis.ftn
    k_grid = method.BZ.k_grid
    dim = length(k_grid) # k_grid_number
    vector = [gap_function(k_grid[idx]) for idx in 1:dim]
    
    lambda = 0.0
    norm = 0.0
    for i1 in 1:dim
        for i2 in 1:dim
            lambda += vector[i1]' * w_array[i1] *eff_interaction(i1, i2) * w_array[i2] * vector[i2]
        end
        norm += vector[i1]' * w_array[i1] * vector[i1]
    end
    return lambda/norm/dim
end


function gap_equation_solver(method::LGE_DiagMethod)
    eff_interaction = method.eff_interaction
    w_array = method.weight_array
    k_grid = method.BZ.k_grid
    k_range = method.BZ.k_range
    dim = length(k_grid)
    N = length(k_range)

    linear_operator_mat = [eff_interaction(i1, i2) * w_array[i2] for i1 in 1:dim, i2 in 1:dim]
    #proj_mat = 
    eigvals, eigvecs = eigen(linear_operator_mat)

    real_eigvals = real.(eigvals)
    max_eigval = eigvals[argmax(real_eigvals)]
    max_eigvec = eigvecs[:, argmax(real_eigvals)]

    Δ_final = reshape(max_eigvec, N, N)
    Δ_final ./= sqrt(sum(abs2, Δ_final))
    
    return max_eigval/dim, Δ_final
end


function gap_equation_solver(method::LGE_DiagMethod, sym::Symbol)
    eff_interaction = method.eff_interaction
    w_array = method.weight_array
    k_grid = method.BZ.k_grid
    k_range = method.BZ.k_range
    dim = length(k_grid)
    N = length(k_range)

    linear_operator_mat = [w_array[i1] * eff_interaction(i1, i2) * w_array[i2] for i1 in 1:dim, i2 in 1:dim]
    #proj_mat = 
    eigvals, eigvecs = eigen(linear_operator_mat)

    real_eigvals = real.(eigvals)
    max_eigval = eigvals[argmax(real_eigvals)]
    max_eigvec = eigvecs[:, argmax(real_eigvals)]

    Δ_final = reshape(max_eigvec, N, N)
    Δ_final ./= sqrt(sum(abs2, Δ_final))
    
    return max_eigval/dim, Δ_final
end


function singlet_int(U, χ_m, χ_p)
    return  -U^2*χ_p/(1.0+U*χ_p) + U^3*χ_m^2/(1-U^2*χ_m^2)
end
function singlet_interaction(U, chi_grid, idx1, idx2)
    chi_p = prompt_matrix_mod(chi_grid, idx1, :plus, idx2)
    chi_m = prompt_matrix_mod(chi_grid, idx1, :minus, idx2)
    return 1/2*(singlet_int(U, chi_m, chi_p) + singlet_int(U, chi_p, chi_m))
end
function triplet_int(U, χ_m)
    return U^2*χ_m/(1-U^2*χ_m^2)
end
function triplet_interaction(U, chi_grid, idx1, idx2)
    chi_p = prompt_matrix_mod(chi_grid, idx1, :plus, idx2)
    chi_m = prompt_matrix_mod(chi_grid, idx1, :minus, idx2)
    return 1/2*(triplet_int(U, chi_m) - triplet_int(U, chi_p))
end

function spin_diagonalized_interaction(
    parity::Symbol, chi_grid::Matrix, U::Float64
)::Function

    sgn = -1.0
    if parity == :singlet
        return (idx1, idx2) -> sgn * singlet_interaction(U, chi_grid, idx1, idx2)
    elseif parity == :triplet
        return (idx1, idx2) -> sgn * triplet_interaction(U, chi_grid, idx1, idx2)
    end
end

function test_interaction(BZ::BrillouinZone, gap_weight_array=[0., 0., 1., 0., 0., 0.];
    gap_type_list = [:s_const, :s_ext, :dx2y2, :dxy, :px, :py]
)
    @assert length(gap_weight_array) == length(gap_type_list)
    k_grid = BZ.k_grid
    test_interaction = (idx1, idx2) -> sum(
        gap_weight_array[i] * GapBasis(gap_type).ftn(k_grid[idx1])*GapBasis(gap_type).ftn(k_grid[idx2])
        for (i, gap_type) in enumerate(gap_type_list)
    ) # d_i f_i(ka)f_i(kb) * tanh(Ea/2T)/2Eb

    return test_interaction
end

function tanh_weight_array(BZ::BrillouinZone, mu::Float64, run::NamedTuple; epsilon::Float64 = 1e-12)
    ϵ = epsilon 
    T = run.T
    k_grid = BZ.k_grid
    N = length(k_grid)
    w_array = zeros(Float64, N)
    for idx in eachindex(w_array)
        eigvals, _ = eigen(H[](SVector(k_grid[idx]...), mu))
        E = eigvals[1]
        if abs(E) < ϵ
            w_array[idx] = 1 / (4*T)
        else
            w_array[idx] = tanh(E / (2*T)) / (2 * E)
        end
    end
    return w_array
end

function spect_weight_array(BZ::BrillouinZone, mu::Float64, run::NamedTuple; epsilon::Float64 = 1e-1)
    ϵ = epsilon 
    k_grid = BZ.k_grid
    N = length(k_grid)
    w_array = zeros(Float64, N)
    for idx in eachindex(w_array)
        eigvals, _ = eigen(H[](SVector(k_grid[idx]...), mu))
        E = eigvals[1]
        w_array[idx] = ϵ / (ϵ^2 + E^2) / pi  # Nascent Dirac Delta function, δ(Ek=0)  
    end
    return w_array
end

end