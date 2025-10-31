using JLD2, Distributed
cpu_number = Sys.CPU_THREADS
addprocs(cpu_number)

@everywhere include("../src/symmetry_suscep.jl"); using .ChiIBZ

begin
    Nk = 256
    Nq = 32
    run = (T=0.01, soft_eta=1e-6)
end
    
μ = 3.9 

IBZ = ChiIBZ.Irreducible_BZ(Nk, μ)
χ_iBZ = ChiIBZ.χ_dd_iBZ(IBZ, Nq, μ, run)
χ_full, q_range =  ChiIBZ.χ_dd_map(χ_iBZ)
using Plots
heatmap(q_range, q_range, -real(χ_full))

q_argmax, χ_argmax = χ_iBZ.q_argmax, χ_iBZ.χ_argmax
push!(max_χq_list, (q_argmax, χ_argmax, χ_iBZ))


@save "test.jld2" max_χq_list