module types_module

export Observable, SuscepType, MasterSuscep, QuickSuscep, DensitySuscep

abstract type Observable end

abstract type SuscepType <: Observable end
struct MasterSuscep <: SuscepType end
struct QuickSuscep  <: SuscepType end
struct DensitySuscep <: SuscepType end

export BrillouinZone
struct BrillouinZone
    type::Symbol
    k_range::AbstractVector{Float64}
    k_grid::Matrix{Vector{Float64}}
end
BrillouinZone(type, N) = begin
    if type == :square
        k_range = LinRange(-pi, pi, N+1)[1:end-1]
        k_grid = [[kx,ky] for kx in k_range, ky in k_range]
    else
        @error "BZ not implemented"
    end
    BrillouinZone(type, k_range, k_grid) 
end


export GapBasis
struct GapBasis
    type::Symbol
    ftn::Function
    parity::Symbol
end
GapBasis(gap_type) = begin
    if gap_type == :s_const
        return GapBasis(gap_type, p -> 1.0, :singlet)
    elseif gap_type == :s_ext
        return GapBasis(gap_type, p -> cos(p[1]) + cos(p[2]), :singlet)  # 2D extended s-wave
    elseif gap_type == :dx2y2
        return GapBasis(gap_type, p -> cos(p[1]) - cos(p[2]), :singlet)  # d_{x^2-y^2}
    elseif gap_type == :dxy
        return GapBasis(gap_type, p -> sin(p[1]) * sin(p[2]), :singlet)  # d_{xy}
    elseif gap_type == :px
        return GapBasis(gap_type, p -> sin(p[1]), :triplet)              # p_x
    elseif gap_type == :py
        return GapBasis(gap_type, p -> sin(p[2]), :triplet)              # p_y
    else
        error("Unsupported gap_type: $gap_type")
    end
end

abstract type LGE_Method end
export LGE_Method, LGE_ProjMethod, LGE_DiagMethod
struct LGE_ProjMethod <:LGE_Method
    eff_interaction::Function
    weight_array::Vector{Float64}
    gap_basis::GapBasis
    BZ::BrillouinZone
end
struct LGE_DiagMethod <:LGE_Method
    eff_interaction::Function
    weight_array::Vector{Float64}
    BZ::BrillouinZone
end


export lambda, lambda_proj, lambda_diag
abstract type lambda <: Observable end
struct lambda_proj <: lambda end
struct lambda_diag <: lambda end


end