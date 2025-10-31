using Plots
using JLD2

ver_name = "ver4"
mu_list = [0.5]
gap_type_list = [:s_const, :s_ext, :dx2y2, :dxy, :px, :py]
label_color_list = [:black, :purple, :red, :green, :blue, :blue]
k_range = LinRange(-pi, pi , 65)[1:end-1]

p_tot = []
for mu in mu_list
    p1 = plot(title= "λ(t' = 0.0, square TB, μ=$mu),"*ver_name, ylim = (-1.0, 1.0))
    p2 = plot(title= "λ(t' = 0.0, square TB, μ=$mu),"*ver_name, ylim = (-0.05, 0.05))

    file_name = pwd() * "/analysis/raw_data/lambda_U_projection_$mu."*ver_name*".jld2"
    data = JLD2.load(file_name)
    U_range = data["U_range"]
    N = length(U_range)

    lambda_gap_U = data["lambda_gap_U"]
    for (idx, gap_type) in enumerate(gap_type_list)
        lambda_per_type = [row[idx] for row in lambda_gap_U]
        println(idx)
        color = label_color_list[idx]
        plot!(p1, U_range, lambda_per_type, label="$gap_type", color = color)
        plot!(p2, U_range, lambda_per_type, label="$gap_type", color = color)
    end

    p3 = plot(title= "Eigen vecs")
    file_name = pwd() *"/analysis/raw_data/lambda_U_diagonalization_$mu."*ver_name*".jld2"
    data = JLD2.load(file_name)
    diag_result = data["lambda_diag_U"]
    println("A")
    lambda_diag_U = [row[1] for row in diag_result]
    println(size(lambda_diag_U))
    eigvec_diag_U = real(diag_result[Int(N/2)][2])

    plot!(p1, U_range, real(lambda_diag_U), label="direct_diagonal", color = :black, style = :dash)
    plot!(p2, U_range, real(lambda_diag_U), label="direct_diagonal", color = :black, style = :dash)
    heatmap!(p3, k_range, k_range, eigvec_diag_U)
    xticks!(p3, [-pi, 0, pi*63/64],["-π","0","π"])
    yticks!(p3, [-pi, 0, pi*63/64],["-π","0","π"])

    println("A")
    push!(p_tot, p1)
    push!(p_tot, p2)
    push!(p_tot, p3)
end
plt = plot(p_tot..., size = (1400, 400), layout = (1,3))
savefig(plt, "lambda_U_proj_vs_diag_"*ver_name*".png")