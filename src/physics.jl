module physics_module

using LinearAlgebra
using StaticArrays  # 성능을 위해 SMatrix로 정의

export σx, σy, σz, I2, Fermi_Dirac, derivative_Fermi_Dirac

const σx = @SMatrix [0.0 1.0;
                    1.0 0.0]
const σy = @SMatrix [0.0 -im;
                    im  0.0]
const σz = @SMatrix [1.0  0.0;
                    0.0 -1.0]
const I2 = @SMatrix [1.0 0.0;
                        0.0 1.0]
function Fermi_Dirac(e::Float64, T::Float64)
    if T == 0.0
        if e==0
            return 0.5
        end

        return e > 0 ? 0.0 : 1.0
    end
    return 1 / (exp(e/T)+1)
end

function derivative_Fermi_Dirac(e::Float64, T::Float64)
    if T == 0.0
        error("Derivative of Fermi-Dirac function is not implemented at T=0.")
    else 
        return - Fermi_Dirac(e, T) * (1 - Fermi_Dirac(e,T)) / T
    end
end


export prompt_matrix_mod
"""
From flatten idx, return elements of matrix in modular.
"""
function prompt_matrix_mod(m::AbstractMatrix, idx_k1::Tuple{Int, Int}, method::Symbol, idx_k2::Tuple{Int,Int})
    i1, j1 = idx_k1
    i2, j2 = idx_k2
    N = size(m, 1)
    @assert N%2 == 0
    half = div(N,2)

    # 이해 가능한 형태로 식을 적기
    if method == :plus
        i = mod(i1+i2+half-2, N) + 1
        j = mod(j1+j2+half-2, N) + 1
        return m[i,j]
    elseif method == :minus
        i = mod(i1-i2+half, N) +1
        j = mod(j1-j2+half, N) +1
        return m[i,j]
    end
end

function prompt_matrix_mod(m::AbstractMatrix, idx1::Int, method::Symbol, idx2::Int)
    N = size(m, 1)
    i1, j1 = Tuple(CartesianIndices((N, N))[idx1])
    i2, j2 = Tuple(CartesianIndices((N, N))[idx2])
    return prompt_matrix_mod(m, (i1, j1), method, (i2, j2))
end

export gap_type_list
gap_type_list = [:s_const, :s_ext, :dx2y2, :dxy, :px, :py]


end