module suscep_module

using ..physics_module
using ..types_module: BrillouinZone, SuscepType, MasterSuscep, QuickSuscep, DensitySuscep
using HCubature, LinearAlgebra, JLD2

const H = Ref{Function}((k, μ) -> μ * 2)  

export static_chi_hcubature

function static_chi_hcubature(stype::SuscepType, q_vec::AbstractVector{<:Real}, mu::Float64, BZ::BrillouinZone, run::NamedTuple)
    return chi_hcubature(stype, q_vec, 0.0, mu, BZ, run)
end

function chi_hcubature(stype::SuscepType, q_vec::AbstractVector{<:Real}, omega::Float64, mu::Float64, BZ::BrillouinZone, run::NamedTuple)
    kdim = length(BZ.k_grid[1,1])
    lower_bound = fill(-π, kdim)
    upper_bound = fill( π, kdim)
    rtol, atol, maxevals = run.rtol, run.atol, Int(run.maxevals)

    f = k_vec -> chi_integrand(stype, k_vec, q_vec, omega, mu, run) + zero_integral_function(k_vec)
    result, _ = hcubature(f,  lower_bound, upper_bound, rtol=rtol, atol=atol, maxevals=maxevals)

    return result /(2.0*pi)^kdim 
end

function chi_integrand(S::MasterSuscep, k_vec::AbstractVector{<:Real}, q_vec::AbstractVector{<:Real}, omega::Float64, mu::Float64, run::NamedTuple)
    Hdim = size(H(k_vec, mu), 1)
    T, eta = run.T, run.soft_eta
    eig_k, U_k, eig_kq, U_kq = eig_setting(k_vec, q_vec, mu)

    integrand = zeros(ComplexF64, Hdim^4)
    
    for idx_d in 1:Hdim, idx_c in 1:Hdim, idx_b in 1:Hdim, idx_a in 1:Hdim
        idx = idx_a + Hdim * (idx_b-1) + Hdim^2 * (idx_c-1) + Hdim^3 * (idx_d-1)
        for idx_k in 1:Hdim, idx_kq in 1:Hdim
            e_k = eig_k[idx_k]
            a_uk = U_k[idx_a, idx_k]
            uk_b = conj(U_k[idx_b, idx_k])
            e_kq = eig_kq[idx_kq]
            c_ukq = U_kq[idx_c, idx_kq]
            ukq_d = conj(U_kq[idx_d, idx_kq])
    
            overlap_factor = a_uk * uk_b * c_ukq * ukq_d
            integrand[idx] += Lindhard_numerator_denominator(e_k, e_kq, omega, T, eta) * overlap_factor
        end
    end

    return integrand
end

function chi_integrand(S::QuickSuscep, k_vec::AbstractVector{<:Real}, q_vec::AbstractVector{<:Real}, mu::Float64, omega::Float64, run::NamedTuple)
    Hdim = size(H[](k_vec, mu), 1)
    T, eta = run.T, run.soft_eta
    eig_k, U_k, eig_kq, U_kq = eig_setting(k_vec, q_vec, mu)

    e_k = eig_k[1]
    e_kq = eig_kq[1]
    
    result = Lindhard_numerator_denominator(e_k, e_kq, omega, T, eta)
    return result
end


function chi_integrand(S::DensitySuscep, k_vec::AbstractVector{<:Real}, q_vec::AbstractVector{<:Real}, mu::Float64, omega::Float64, run::NamedTuple)
    Hdim = size(H[](k_vec, mu), 1)
    T, eta = run.T, run.soft_eta
    eig_k, U_k, eig_kq, U_kq = eig_setting(k_vec, q_vec, mu)

    integrand = 0.0
    for idx_k in 1:Hdim, idx_kq in 1:Hdim
        e_k = eig_k[idx_k]
        uk = U_k[:, idx_k]
        e_kq = eig_kq[idx_kq]
        ukq = U_kq[:, idx_kq]
        
        overlap_factor = dot(uk, ukq) * dot(ukq, uk)
        integrand += Lindhard_numerator_denominator(e_k, e_kq, omega, T, eta) * overlap_factor
    end

    return integrand
end

function eig_setting(k_vec, q_vec, mu)
    eig_k, U_k = eigen(H[](k_vec, mu))
    kq_vec = mod.(k_vec + q_vec, 2π)
    eig_kq, U_kq = eigen(H[](kq_vec, mu))
    return eig_k, U_k, eig_kq, U_kq
end

"""
Lindhard numerator and denominator that comes as 

numerator: f(e_k) - f(e_kq)

denominator : e_k - e_kq + omega + im * eta

This function handles the case where omega is zero and the energies are close to each other,
by returning the derivative of the Fermi-Dirac function.
"""
function Lindhard_numerator_denominator(e_k, e_kq, omega, T, eta)
    if omega == 0.0 && abs(e_k - e_kq) < eta
        return ComplexF64(derivative_Fermi_Dirac(e_k, T))
    else
        numerator = Fermi_Dirac(e_k, T) - Fermi_Dirac(e_kq, T)
        denominator = e_k - e_kq + omega + im * eta
        return numerator/denominator
    end
end

using SpecialFunctions
"""
zero_integral_function is a function that returns a value that integrates to zero over the range [-π, π] in each dimension.
"""
function zero_integral_function(k_vec; width=0.1)
    # 1. 넓은 변동: Fourier-like 기본 파트
    freq_list = [1.0, 2.0, 4.0, 8.0, 16.0]
    base_part(k) = sum(cos(f * k[1]) + sin(f * k[2]) for f in freq_list)

    # 2. 봉우리 위치 & 높이
    centers = [-π, -π/2, 0.0, π/2, π]
    peaks = length(centers)
    heights = fill(1.0, peaks)
    σ = width

    # 3. 가우시안 봉우리 정의
    function raw_peak(x)
        val = 0.0
        for i in 1:peaks
            w = (centers[i] == -π || centers[i] == π) ? 0.5 : 1.0
            val += w * heights[i] * exp(-((x - centers[i])^2) / (2σ^2))
        end
        return val
    end

    # 4. 해석적 평균 제거 (경계 봉우리 가중치 포함)
    function analytic_avg_peak()
        s = 0.0
        for i in 1:peaks
            w = (centers[i] == -π || centers[i] == π) ? 0.5 : 1.0
            s += w * heights[i] * σ * sqrt(π/2) *
                 (erf((π - centers[i]) / (√2 * σ)) -
                  erf((-π - centers[i]) / (√2 * σ)))
        end
        return s / (2π) # 2π for average over the full range
    end

    avg_peak = analytic_avg_peak()
    peak_part(x) = raw_peak(x) - avg_peak

    # 5. 전체 합
    combined(k) = base_part(k) + peak_part(k[1]) + peak_part(k[2])

    return combined(k_vec)
end

end