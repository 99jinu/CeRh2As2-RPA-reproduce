# 결국 reproduce 해야하는 sym: 
using StaticArrays
using LinearAlgebra

const t = 0.1 # serve as t'; next nearest hopping
const α = 1.0 # serve as RSOC interaction strength

const PI = Float64(π)

const σx = @SMatrix [0.0 1.0;
                    1.0 0.0]
const σy = @SMatrix [0.0 -im;
                    im  0.0]
const σz = @SMatrix [1.0  0.0;
                    0.0 -1.0]
const I2 = @SMatrix [1.0 0.0;
                        0.0 1.0]

function C4v_RashbaHubbard_Ham(k, μ)
    ε_k= (-2.0)*(cos(k[1]) + cos(k[2])) + 4*t*cos(k[1])*cos(k[2]) - μ
    g_k = α * 
    [-2.0*sin(k[2]) + 4*t*cos(k[1])sin(k[2]),
     2.0*sin(k[1])  - 4*t*sin(k[1])cos(k[2]), 0.0 ]
    SOC_term = g_k[1] * σx + g_k[2] * σy
    
     return ε_k * I2 + SOC_term
end

struct HamSym{N, D}
    ops :: NTuple{N,Symbol}
    R   :: NTuple{N, SMatrix{2,2,Int}}          # k-공간 작용(정수 행렬)
    U   :: NTuple{N, SMatrix{D,D,ComplexF64}}   # 스핀 작용(2×2 unitary)
end
function C4v_RashbaHubbard_sym()
    ops = (:E, :C4, :C2, :C4_3, :Mx, :My, :σd_1, :σd_2)

    # k -> R * k  (kx,ky)는 열벡터로 생각
    R = (
        @SMatrix[ 1  0; 0  1],   # E           (kx, ky)
        @SMatrix[ 0 -1; 1  0],   # C4          (-ky, kx)
        @SMatrix[-1  0; 0 -1],   # C2          (-kx, -ky)
        @SMatrix[ 0  1;-1  0],   # C4^3        (ky, -kx)
        @SMatrix[ 1  0; 0 -1],   # Mx (xz)     (kx, -ky)
        @SMatrix[-1  0; 0  1],   # My (yz)     (-kx, ky)
        @SMatrix[ 0  1; 1  0],   # σd1 (x=y)   (ky, kx)
        @SMatrix[ 0 -1;-1  0]    # σd2 (x=-y)  (xk, ky)
    )

    U = (
        I2,
        (I2 - im*σz)/sqrt(2),
        -im*σz,
        (I2 + im*σz)/sqrt(2),
        im*σy,
        im*σx,
        (im*(σx - σy))/sqrt(2),
        (im*(σx + σy))/sqrt(2)
    )

    HamSym{length(ops), size(U[1],1)}(ops, R, U)
end
# test1 at test.ipynb --> clear
struct IBZ_SOA{T<:Real,C<:Complex}
    Nk::Int
    Δk::T
    k_range::Vector{T}          # length Nk+1 # 길이 Nk+1, [-PI..PI]
    kx::Vector{T}               # length N: IBZ 점 갯수 # IBZ의 x좌표들
    ky::Vector{T}               # length N # IBZ의 y좌표들
    ii::Vector{Int32}           # length N # 원래 격자의 x_인덱스
    jj::Vector{Int32}           # length N # 원래 격자의 y_인덱스
    eigvals::Matrix{T}          # (nb, N)  # nb: 밴드 수.
    eigvecs::Union{Array{C,3},Nothing}  # (nb, nb, N) or nothing
end
function C4v_IBZ(Nk::Int, μ::Float64, H; nb::Int, keep_vecs::Bool=true,
                 T=Float64, C=ComplexF64)
    @assert iseven(Nk)
    Δk      = 2π/Nk
    k_range = collect(range(-π, π; length=Nk+1))
    half    = Nk ÷ 2
    m       = half + 1
    N       = m*(m+1) ÷ 2 # IBZ 점 갯수

    kx = Vector{T}(undef, N);  ky = similar(kx)
    ii = Vector{Int32}(undef, N); jj = similar(ii)
    eigvals = Matrix{T}(undef, nb, N)
    eigvecs = keep_vecs ? Array{C}(undef, nb, nb, N) : nothing

    p = 0
    for i in (half+1):(Nk+1)
        kxi = k_range[i]
        for j in (half+1):i
            p += 1
            kx[p] = kxi
            ky[p] = k_range[j]
            ii[p] = Int32(i);  jj[p] = Int32(j)

            E, U = eigen(Hermitian(H(@SVector[kx[p], ky[p]], μ)))  # Hermitian이면 eigen(Hermitian(...))
            @inbounds @views eigvals[:, p] .= E
            if keep_vecs !== nothing
                @inbounds @views eigvecs[:, :, p] .= U
            end
        end
    end
    return IBZ_SOA{T,C}(Nk, Δk, k_range, kx, ky, ii, jj, eigvals, eigvecs)
end
# Memory test2 --> clear

## iBZ 안에 있으면 true 반환
@inline function in_IBZ(kx::Float64, ky::Float64; atol=1e-12)
    return (ky >= -atol) && (kx >= ky - atol) && (kx <= PI + atol)
end

## iBZ 안으로 projection
@inline function fold_BZ(x::Float64)
    # [-PI, PI] 범위로 접기 (PI는 포함시켜야 하므로 round 사용 대신 mod 방식)
    y = mod(x + PI, 2PI) - PI
    # 여기서 -PI와 PI가 같은 점이지만 Simpson grid는 둘 다 쓰므로
    # x가 거의 PI 근처일 때는 강제로 +PI로 보정
    if abs(y + PI) < 1e-12
        return PI
    end
    return y
end
# (i,j) -> p 변환 함수
@inline function ibz_pos(i::Int, j::Int, half::Int)
    I = i - half          # 1..m
    J = j - half          # 1..I
    @inbounds return (I*(I-1)) ÷ 2 + J   # 1..N
end
# x ∈ [-π, π] -> i ∈ [1, Nk+1] 를 반환하는 함수
@inline function grid_index(x::Float64, Δk::Float64, Nk::Int)
    i = round(Int, (x + PI)/Δk) + 1 # x ∈ [-π, π] -> i ∈ [1, Nk+1]
    return clamp(i, 1, Nk+1)
end
# IBZ_SOA 버전: k ∈ BZ --> (p, U) 반환
function map_to_IBZ(k::AbstractVector{<:Real}, IBZ::IBZ_SOA, sym::HamSym)
    Nk, Δk, kR = IBZ.Nk, IBZ.Δk, IBZ.k_range
    half = Nk ÷ 2

    @inbounds for op_idx in eachindex(sym.ops)
        R = sym.R[op_idx]
        kt = R * k
        x = fold_BZ(kt[1])                           # [-π,π]로 접기
        y = fold_BZ(kt[2])
        
        # 격자에 스냅(정확히 grid 위 점)
        i = grid_index(x, Δk, Nk)                    # 1..Nk+1
        j = grid_index(y, Δk, Nk)
        xi = kR[i];  yj = kR[j]

        # IBZ 삼각쐐기(ky≥0, kx≥ky) 체크 후 p 계산
        if in_IBZ(xi, yj) && j <= i && i >= half+1
            p = ibz_pos(i, j, half)                  # 1..N (삼각수 인덱싱)
            return p, sym.U[op_idx]                # 또는 R를 원하시면 R 반환
        end
    end
    error("No IBZ mapping found for k=($kx0,$ky0)")
end

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

function Lindhard_numerator_denominator(e_k, e_kq, ω, T, eta)
    if ω == 0.0 && abs(e_k - e_kq) < eta
        return ComplexF64(derivative_Fermi_Dirac(e_k, T))
    else
        numerator = Fermi_Dirac(e_k, T) - Fermi_Dirac(e_kq, T)
        denominator = e_k - e_kq + ω + im * eta
        return numerator/denominator
    end
end
