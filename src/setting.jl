module setting_module
## HAMILTONIAN part
using LinearAlgebra, StaticArrays
using ..physics_module

export hamiltonian_from_json2, parametrize_from_json2

abstract type HamParams end
struct cubicTBParams <: HamParams
    t::Float64
    kdim::Int         # 기본 3
    Hdim::Int         # 기본 2
    extras::NamedTuple
end
"""
    cubicTBParams(d::NamedTuple)

TOML에서 읽은 NamedTuple로부터 cubicTBParams를 생성합니다.
`t` 외의 모든 키는 `extras`로 묶입니다.
"""
cubicTBParams(d::NamedTuple) = begin
    base = (
        t = get(d, :t, 1.0),
        kdim = get(d, :kdim, 3),
        Hdim = get(d, :Hdim, 2),
    )
    extra_pairs = filter(p -> !(p.first in (:t, :kdim, :Hdim)), pairs(d))
    cubicTBParams(base.t, base.kdim, base.Hdim, NamedTuple(extra_pairs))
end
struct squareTBParams <: HamParams
    t::Float64
    t2::Float64
    kdim::Int         # 기본 2
    Hdim::Int         # 기본 2
    extras::NamedTuple
end
"""
    squareTBParams(d::NamedTuple)

TOML에서 읽은 NamedTuple로부터 squareTBParams를 생성합니다.
`t`, `t2` 외의 모든 키는 `extras`로 묶입니다.
"""
squareTBParams(d::NamedTuple) = begin
    base = (
        t = get(d, :t, 1.0),
        t2 = get(d, :t2, 0.0),
        kdim = get(d, :kdim, 2),
        Hdim = get(d, :Hdim, 2),
    )
    extra_pairs = filter(p -> !(p.first in (:t, :t2, :kdim, :Hdim)), pairs(d))
    squareTBParams(base.t, base.t2, base.kdim, base.Hdim, NamedTuple(extra_pairs))
end

function hamiltonian(k::SVector{3,Float64}, mu::Float64, params::cubicTBParams)
    # 기본 TB dispersion
    @assert length(k) == params.kdim "k-dimension mismatch!"

    ek = (-2.0)*params.t*(cos(k[1]) + cos(k[2]) + cos(k[3])) - mu
    H = ek * LinearAlgebra.I(params.Hdim)

    return H
end

function hamiltonian(k::SVector{2,Float64}, mu::Float64, params::squareTBParams)
    # 기본 TB dispersion
    @assert length(k) == params.kdim "k-dimension mismatch!"

    ek = (-2.0)*params.t*(cos(k[1]) + cos(k[2])) + params.t2 * 4.0 * cos(k[1]) * cos(k[2]) - mu
    H = ek * I2

    # 추가 옵션: Rashba_strength
    Rashba_alpha = get(params.extras, :Rashba_strength, 0.0)
    if Rashba_alpha != 0.0
        g1 = 2 * params.t * sin(k[2]) - 4 * params.t2 * cos(k[1]) * sin(k[2])
        g2 = -2 * params.t * sin(k[1]) + 4 * params.t2 * cos(k[2]) * sin(k[1])
        H += Rashba_alpha * (g1 * σx + g2 * σy)
    end

    return H
end




using JSON3
function parametrize_from_json2(hparams_json::String, np_json::String)
    h_nt = parse_namedtuple_from_json(hparams_json)
    np_nt = parse_namedtuple_from_json(np_json)

    # model_name이 없을 경우 squreTB로 설정
    model_name = get(h_nt, :model, "squareTB")

    # np_nt 전체를 그냥 RunParams
    return (model_name, h_nt, np_nt)
end


function hamiltonian_from_json2(hparams_json::String)
    h_nt = parse_namedtuple_from_json(hparams_json)

    # model_name이 없을 경우 squreTB로 설정
    model_name = get(h_nt, :model, "squareTB")
    params = _make_params(model_name, h_nt)

    H = (k_vec, mu) -> hamiltonian(k_vec, mu, params)

    return H
end

#JSON 문자열을 NamedTuple로 변환
function parse_namedtuple_from_json(json_str::String)
    dict = JSON3.read(json_str, Dict{String,Any})
    pairs = [Symbol(k) => v for (k, v) in dict]
    return NamedTuple(pairs)
end

#모델에 맞는 params struct 생성
function _make_params(model_name::String, nt::NamedTuple)
    if model_name == "squareTB"
        return squareTBParams(nt)
    elseif model_name == "cubicTB"
        return cubicTBParams(nt)
    else
        error("Unknown model: $model_name")
    end
end

end