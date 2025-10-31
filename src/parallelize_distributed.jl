module parallelize_distributed

using Distributed
export parallelize_array_distributed

"""
    parallelize_array_distributed(array::AbstractArray, f::Function, args...; dim::Int=1)

`array`를 dim 방향으로 나눠서 worker에게 분배.
각 worker는 chunk의 각 element에 대해 f(element, args...)를 적용하고,
결과 chunk를 반환.
최종적으로 다시 합쳐서 원래 크기의 배열을 반환.
"""
function parallelize_array_distributed(array::AbstractArray, f::Function, args...; dim::Int=1)
    nd = ndims(array)
    @assert 1 ≤ dim ≤ nd "dim must be between 1 and $nd"

    wids = workers()
    nworkers = length(wids)
    nworkers > 0 || error("No workers available. Use addprocs().")

    total_len = size(array, dim)
    base_size = div(total_len, nworkers)
    remainder = total_len % nworkers

    # 쪼갤 인덱스 범위
    ranges = Vector{UnitRange{Int}}()
    start_idx = 1
    for i in 1:nworkers
        extra = i <= remainder ? 1 : 0
        stop_idx = start_idx + base_size + extra - 1
        push!(ranges, start_idx:stop_idx)
        start_idx = stop_idx + 1
        if start_idx > total_len
            break
        end
    end

    # 각 chunk view 만들기
    chunks = map(r -> view(array,
        ntuple(d -> d == dim ? r : (:), nd)...),
        ranges)

    # 각 worker에 비동기로 던지기
    futures = Vector{Future}(undef, length(chunks))
    for (i, chunk) in enumerate(chunks)
        wid = wids[(i-1) % nworkers + 1]
        # 각 chunk의 각 element에 f 적용
        futures[i] = @spawnat wid begin
            firstval = f(array[begin], args...)
            out = similar(chunk, typeof(firstval))
            for idx in eachindex(chunk)
                out[idx] = f(chunk[idx], args...)
            end
            out
        end
    end

    # 결과 모으기
    results = map(fetch, futures)
    return cat(results...; dims=dim)
end

end # module
