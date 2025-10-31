module ChiIBZ

using LinearAlgebra
using StaticArrays
using Distributed
using Base.Threads
include("physics.jl"); using .physics_module
include("types.jl"); using .types_module
BLAS.set_num_threads(1)   # BLAS oversubscription 방지



const PI = Float64(π)

const H = Ref{Function}((k, μ) -> H = ((-2.0)*(cos(k[1]) + cos(k[2])) - μ) * LinearAlgebra.I(2))


const OPS = (:E, :C4, :C2, :C4_3, :sigx, :sigy, :sigd, :sigd_p)

function act_R(kx, ky, R::Symbol)
    if R === :E
        return (kx, ky)
    elseif R === :C4
        return ( ky, -kx)
    elseif R === :C2
        return (-kx, -ky)
    elseif R === :C4_3
        return (-ky,  kx)
    elseif R === :sigx
        return ( kx, -ky)
    elseif R === :sigy
        return (-kx,  ky)
    elseif R === :sigd
        return ( ky,  kx)
    elseif R === :sigd_p
        return (-ky, -kx)
    end
end

struct Irreducible_BZ
    Nk::Int
    Δk::Float64
    k_range::Vector{Float64}                     # 길이 Nk+1, [-PI..PI]
    k_ibz::Vector{SVector{2,Float64}}            # IBZ 좌표들
    index_list::Vector{Tuple{Int,Int}}           # 각 IBZ점의 (i,j) 인덱스(Full-grid 기준)
    eigvals_list::Vector{Vector{Float64}}        # 각 IBZ점의 고유값
    eigvecs_list::Vector{Matrix{ComplexF64}}     # 각 IBZ점의 고유벡터
    idx2pos::Dict{Tuple{Int,Int},Int}            # (i,j) → IBZ 배열 위치
end
function Irreducible_BZ(Nk::Int, μ::Float64)
    @assert iseven(Nk) "Simpson을 쓰려면 구간 수 Nk는 짝수여야 합니다."
    Δk      = 2PI / Nk
    k_range = collect(range(-PI, PI; length=Nk+1))   # 끝점 포함 (Simpson)

    k_ibz        = SVector{2,Float64}[]
    index_list   = Tuple{Int,Int}[]
    eigvals_list = Vector{Float64}[]
    eigvecs_list = Matrix{ComplexF64}[]
    idx2pos      = Dict{Tuple{Int,Int},Int}()

    # 첫사분면 삼각쐐기: i ≥ j, 둘 다 [Nk/2+1 .. Nk+1] (0..PI 영역)
    half = div(Nk, 2)   
    for i in (half+1):(Nk+1)                    # kx ∈ [0..PI]
        kx = k_range[i]
        for j in (half+1):i                     # ky ∈ [0..kx]
            ky = k_range[j]
            e, U = eigen(H[](@SVector[kx,ky], μ))
            push!(k_ibz,        @SVector [kx,ky])
            push!(index_list,   (i,j))
            push!(eigvals_list, e)
            push!(eigvecs_list, U)
            idx2pos[(i,j)] = length(k_ibz)      # O(1) lookup용
        end
    end
    return Irreducible_BZ(Nk, Δk, k_range, k_ibz, index_list, eigvals_list, eigvecs_list, idx2pos)
end

@inline function grid_index(x::Float64, Δk::Float64, Nk::Int)
    # Simpson 그리드: [-PI, PI], Nk는 '구간 수'(짝수), 점은 Nk+1(홀수)
    i = round(Int, (x + PI) / Δk) + 1
    return clamp(i, 1, Nk+1)
end

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

@inline function in_IBZ(kx::Float64, ky::Float64; atol=1e-12)
    return (ky >= -atol) && (kx >= ky - atol) && (kx <= PI + atol)
end
# (kx,ky)를 IBZ 격자점으로 매핑: (IBZ배열 위치, 사용한 대칭)
function map_to_IBZ(k::AbstractVector{<:Real}, IBZ::Irreducible_BZ)
    kx, ky = Float64(k[1]), Float64(k[2])
    Nk, Δk, kR = IBZ.Nk, IBZ.Δk, IBZ.k_range
    for R in OPS
        x, y = act_R(kx, ky, R)
        x = fold_BZ(x)
        y = fold_BZ(y)
        i = grid_index(x, Δk, Nk)  # 1..Nk+1
        j = grid_index(y, Δk, Nk)
        xi, yj = kR[i], kR[j]      # 정확히 grid 위 좌표
        if in_IBZ(xi, yj) && haskey(IBZ.idx2pos, (i,j))
            return IBZ.idx2pos[(i,j)], R
        end
    end
    error("No IBZ mapPIng found for k=($kx,$ky)")
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

function χ_dd_integrand(k_vec::AbstractVector{<:Real}, q_vec::AbstractVector{<:Real}, μ::Float64, IBZ::Irreducible_BZ, run::NamedTuple, ω=0.0::Float64)
    Hdim = size(H[](k_vec, μ), 1)
    T, eta = run.T, run.soft_eta

    idx_k, R_k = map_to_IBZ(k_vec, IBZ)
    idx_kq, R_kq = map_to_IBZ(k_vec+q_vec, IBZ)
    ## sPIn-orbit coupling이 없을 때는 wave vector도 그대로다.
    eig_k, U_k = IBZ.eigvals_list[idx_k], IBZ.eigvecs_list[idx_k]
    eig_kq, U_kq = IBZ.eigvals_list[idx_kq], IBZ.eigvecs_list[idx_kq]

    integrand = 0.0
    for a in 1:Hdim, b in 1:Hdim
        e_k = eig_k[a]
        uk = U_k[:, a]
        e_kq = eig_kq[b]
        ukq = U_kq[:, b]
        
        overlap_factor = abs2(dot(uk, ukq))
        integrand += Lindhard_numerator_denominator(e_k, e_kq, ω, T, eta) * overlap_factor
    end

    return integrand
end

# Nk = 구간 수(짝수), 점 개수는 Nk+1
function simpson_weights_1d(Nk::Int)
    @assert iseven(Nk)
    Np = Nk + 1
    w = ones(Float64, Np)
    # 1,4,2,4,...,2,4,1
    for i in 2:Np-1
        w[i] = isodd(i) ? 4.0 : 2.0
    end
    return w
end

function χ_dd_simpson(IBZ::Irreducible_BZ, q_vec, μ, run; ω::Float64=0.0, normalize_to_area::Bool=true)
    Nk   = IBZ.Nk
    Δk   = IBZ.Δk
    kR   = IBZ.k_range
    wx   = simpson_weights_1d(Nk)
    wy   = simpson_weights_1d(Nk)

    acc = 0.0
    @inbounds for i in 1:(Nk+1)
        kx = kR[i]; wi = wx[i]
        for j in 1:(Nk+1)
            ky = kR[j]; wj = wy[j]
            acc += wi * wj * χ_dd_integrand(@SVector[kx,ky], q_vec, μ, IBZ, run, ω)
        end
    end

    # Simpson 정규화: (Δk/3)*(Δk/3)
    val = acc * (Δk/3) * (Δk/3)
    return normalize_to_area ? (val / (4PI^2)) : val
end

function χ_dd_iBZ(IBZ::Irreducible_BZ, Nq::Int, μ, run; ω=0.0)
    @assert iseven(Nq)
    Δq = 2π / Nq
    q_range = collect(range(-PI, PI; length = Nq+1))
    half = div(Nq, 2)

    χ_ibz = Dict{Tuple{Int,Int}, ComplexF64}()

    for i in half+1:Nq+1
        qx = q_range[i]
        for j in half+1:i
            qy = q_range[j]
            val = χ_dd_simpson(IBZ, @SVector[qx,qy], μ, run; ω=ω, normalize_to_area=true)
            χ_ibz[(i, j)] = val 
        end
    end

    return (
        Nq = Nq,
        Δq = Δq,
        q_range = q_range,
        χ_ibz = χ_ibz
    )
end

function finding_argmax(χ_iBZ)
    q_range =χ_iBZ.q_range
    χ_ibz = χ_iBZ.χ_ibz
    # ---- argmax Re χ(q) 찾기 ----
    best_k = nothing
    best_re = -Inf
    best_val = 0.0 + 0.0im
    for (k, v) in χ_ibz
        r = real(v)
        if isfinite(r) && r > best_re
            best_re = r
            best_k = k
            best_val = v
        end
    end

    # 비어있을 가능성은 없지만, 혹시 모를 안전장치
    if best_k === nothing
        return (
            q_argmax = @SVector[NaN, NaN],
            χ_argmax = NaN + 0im,
            idx_argmax = (0, 0),
            Re_argmax = NaN
        )
    end

    i_max, j_max = best_k
    q_argmax = @SVector[q_range[i_max], q_range[j_max]]

    return (
        q_argmax = q_argmax,
        χ_argmax = best_val,
        idx_argmax = best_k,
        Re_argmax = best_re
    )
end

function χ_dd_map(χ_iBZ)
    Nq, Δq, q_range, χ_ibz = χ_iBZ.Nq, χ_iBZ.Δq, χ_iBZ.q_range, χ_iBZ.χ_ibz 
    χ_full = fill(NaN + 0im, Nq+1, Nq+1)
    for i in 1:Nq+1, j in 1:Nq+1
        qx, qy = q_range[i], q_range[j]
        # 대칭 연산 적용 → IBZ 대응 인덱스 찾기
        for R in OPS
            x,y = act_R(qx,qy,R)
            ii = grid_index(x, Δq, Nq)
            jj = grid_index(y, Δq, Nq)
            xi, yj = q_range[ii], q_range[jj]
            if in_IBZ(xi, yj) && haskey(χ_ibz, (ii,jj))
                χ_full[i,j] = χ_ibz[(ii,jj)]
                break
            end
        end
    end
    return χ_full, q_range
end


# ---------------------------
# (C) pmap용 유틸
# ---------------------------

"i ≥ j인 iBZ 삼각영역 인덱스 벡터"
function triangular_indices(Nq::Int)
    @assert iseven(Nq)
    half = div(Nq, 2)
    idxs = Tuple{Int,Int}[]
    @inbounds for i in half+1:Nq+1, j in half+1:i
        push!(idxs, (i, j))
    end
    return idxs
end

"벡터를 길이 chunk로 분할"
function _chunk(v::Vector{T}, chunk::Int) where {T}
    n = length(v)
    chunk <= 0 && error("chunk must be ≥ 1")
    out = Vector{Vector{T}}()
    i = 1
    while i <= n
        j = min(i + chunk - 1, n)
        push!(out, collect(v[i:j])) # 뷰를 원격으로 보내기 어려우니 collect
        i = j + 1
    end
    return out
end

# 워커에 큰 상수들을 한 번만 배치하기 위한 Ref
const _IBZ_ref = Ref{Any}()
const _qr_ref  = Ref{Vector{Float64}}()
const _μ_ref   = Ref{Any}()
const _run_ref = Ref{Any}()
const _ω_ref   = Ref{Float64}()

"워커에 IBZ, q_range, μ, run, ω를 배치"
function setup_workers!(IBZ, q_range::Vector{Float64}, μ, run; ω::Float64=0.0)
    @sync for p in workers()
        @async remotecall_wait(p) do
            _IBZ_ref[] = IBZ
            _qr_ref[]  = q_range
            _μ_ref[]   = μ
            _run_ref[] = run
            _ω_ref[]   = ω
            nothing
        end
    end
    return nothing
end

"청크(여러 (i,j))를 받아 해당 χ를 계산해 벡터[(i,j), χ]로 반환"
function χ_dd_iBZ_chunk(idxs::Vector{Tuple{Int,Int}})
    qr  = _qr_ref[]; IBZ = _IBZ_ref[]; μ = _μ_ref[]; run = _run_ref[]; ω = _ω_ref[]
    out = Vector{Tuple{Tuple{Int,Int}, ComplexF64}}(undef, length(idxs))
    for t in eachindex(idxs)
        i, j = idxs[t]
        qx = qr[i]; qy = qr[j]
        val = χ_dd_simpson(IBZ, @SVector[qx, qy], μ, run; ω=ω, normalize_to_area=true)
        out[t] = (idxs[t], val)
    end
    return out
end

"pmap 드라이버: Dict로 모으고 argmax까지 계산"
function χ_dd_iBZ_pmap(IBZ::Irreducible_BZ, Nq::Int, μ, run;
                       ω::Float64=0.0, chunk::Int=256)

    @assert iseven(Nq)
    Δq   = 2π / Nq
    idxs = triangular_indices(Nq)
    q_range = collect(range(-PI, PI; length = Nq+1))
    setup_workers!(IBZ, q_range, μ, run; ω=ω)

    parts = pmap(χ_dd_iBZ_chunk, _chunk(idxs, chunk))  # Vector of parts
    χ_ibz = Dict{Tuple{Int,Int}, ComplexF64}()

    # 모으기 + argmax 계산을 한 번에 (메모리/패스 절약)
    best_i::Int = 0; best_j::Int = 0
    best_re     = -Inf
    best_val    = 0.0 + 0.0im

    for part in parts               # part :: Vector{( (i,j), val )}
        for (k, v) in part
            χ_ibz[k] = v
            r = real(v)
            if isfinite(r) && r > best_re
                best_re  = r
                best_i, best_j = k
                best_val = v
            end
        end
    end

    q_argmax = (best_i == 0) ? @SVector[NaN, NaN] : @SVector[q_range[best_i], q_range[best_j]]

    return (Nq=Nq,
            Δq=Δq,
            q_range=q_range,
            χ_ibz=χ_ibz,
            q_argmax=q_argmax,
            χ_argmax=best_val,
            idx_argmax=(best_i, best_j),
            Re_argmax=best_re)
end

end # module