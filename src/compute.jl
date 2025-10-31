module compute_module

using ..physics_module
using ..types_module

include("./setting.jl"); using.setting_module
include("./parallelize_distributed.jl"); using.parallelize_distributed

export init_H, H
const H = Ref{Function}()
# Hamiltonian 초기화 함수
function init_H(hparams_json::String)
    H[] = hamiltonian_from_json2(hparams_json)
end
function link_submodules()
    suscep_module.H[] = H[]
    LGE_solver.H[] = H[]
end

include("./suscep.jl"); using .suscep_module
include("./gap_equation.jl"); using .LGE_solver

export compute, array_compute
"""
compute(Susceptype, mu, BZ, run); args = [BZ, @NamedTuple(soft_eta, rtol, atol, maxevals, T)]
"""
compute(obs::SuscepType, mu::Float64, BZ, run) = q_vec -> static_chi_hcubature(obs, q_vec, mu, BZ, run)
array_compute(obs::SuscepType, array::AbstractArray, mu, BZ, run) = parallelize_array_distributed(array, compute(obs, mu, BZ, run))

begin
    function compute(obs::lambda_proj, U, chi_grid, BZ, w_array)
        lambda_gap = zeros(Float64, length(gap_type_list))
        for (i,gap_type) in enumerate(gap_type_list)
            gap_basis = GapBasis(gap_type)
            eff_V = spin_diagonalized_interaction(gap_basis.parity, chi_grid, U)
            lge_projmethod = LGE_ProjMethod(eff_V, w_array, gap_basis, BZ)
            lambda_gap[i] = real(gap_equation_solver(lge_projmethod))
        end

        return lambda_gap
    end
    function compute(obs::lambda_diag, U, chi_grid, BZ, w_array)
        eff_V_singlet = spin_diagonalized_interaction(:singlet, chi_grid, U)
        eff_V_triplet = spin_diagonalized_interaction(:triplet, chi_grid, U)
        lge_diagmethod_singlet = LGE_DiagMethod(eff_V_singlet, w_array, BZ)
        lge_diagmethod_triplet = LGE_DiagMethod(eff_V_triplet, w_array, BZ)

        s_result, s_vec = gap_equation_solver(lge_diagmethod_singlet)
        t_result, t_vec = gap_equation_solver(lge_diagmethod_triplet)

        if real(s_result) > real(t_result)
            who_win = :singlet
            return s_result, s_vec, who_win
        else
            who_win = :triplet
            return t_result, t_vec, who_win
        end
    end
    compute(obs::lambda, chi_grid, BZ, w_array) = U -> compute(obs, U, chi_grid, BZ, w_array)

    function array_compute(obs::lambda, U_array, chi_grid, BZ, mu, run; weight_array::Function=tanh_weight_array) 
        w_array = weight_array(BZ, mu, run)
        return parallelize_array_distributed(U_array, compute(obs, chi_grid, BZ, w_array))
    end


    export compute_sym, array_compute_sym
    function compute_sym(obs::lambda_proj, U, chi_grid, BZ, w_array)
        lambda_gap = zeros(Float64, length(gap_type_list))
        for (i,gap_type) in enumerate(gap_type_list)
            gap_basis = GapBasis(gap_type)
            eff_V = spin_diagonalized_interaction(gap_basis.parity, chi_grid, U)
            lge_projmethod = LGE_ProjMethod(eff_V, w_array, gap_basis, BZ)
            lambda_gap[i] = real(gap_equation_solver(lge_projmethod, :sym))
        end

        return lambda_gap
    end
    function compute_sym(obs::lambda_diag, U, chi_grid, BZ, w_array)
        eff_V_singlet = spin_diagonalized_interaction(:singlet, chi_grid, U)
        eff_V_triplet = spin_diagonalized_interaction(:triplet, chi_grid, U)
        lge_diagmethod_singlet = LGE_DiagMethod(eff_V_singlet, w_array, BZ)
        lge_diagmethod_triplet = LGE_DiagMethod(eff_V_triplet, w_array, BZ)

        s_result, s_vec = gap_equation_solver(lge_diagmethod_singlet, :sym)
        t_result, t_vec = gap_equation_solver(lge_diagmethod_triplet, :sym)

        if real(s_result) > real(t_result)
            who_win = :singlet
            return s_result, s_vec, who_win
        else
            who_win = :triplet
            return t_result, t_vec, who_win
        end
    end
    compute_sym(obs::lambda, chi_grid, BZ, w_array) = U -> compute_sym(obs, U, chi_grid, BZ, w_array)
    function array_compute_sym(obs::lambda, U_array, chi_grid, BZ, mu, run; weight_array::Function=tanh_weight_array) 
        w_array = weight_array(BZ, mu, run)
        return parallelize_array_distributed(U_array, compute_sym(obs, chi_grid, BZ, w_array))
    end
end

end


