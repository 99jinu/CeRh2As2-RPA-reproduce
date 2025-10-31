using Plots
using JLD2

ver_name = ".ver4"
mu = 0.5
gap_type_list = [:s_const, :s_ext, :dx2y2, :dxy, :px, :py]
label_color_list = [:black, :purple, :red, :green, :blue, :blue]
k_range = LinRange(-pi, pi , 65)[1:end-1]

p3 = plot(title= "Eigen vecs");
file_name = pwd() *"/analysis/raw_data/lambda_U_diagonalization_$mu"*ver_name*".jld2"
data = JLD2.load(file_name)
U_range = data["U_range"]
N = length(U_range)
half = Int(N/2)
diag_result = data["lambda_diag_U"]
lambda_diag_U = real([row[1] for row in diag_result])
eigvec_diag_U = real(diag_result[half][2])

U_range[half]
heatmap!(p3, k_range, k_range, eigvec_diag_U)
xticks!(p3, [-pi, 0, pi*63/64],["-π","0","π"])
yticks!(p3, [-pi, 0, pi*63/64],["-π","0","π"])

println("A")
push!(p_tot, p1)
push!(p_tot, p2)
push!(p_tot, p3)
plt = plot(p_tot..., size = (1400, 800))
savefig(plt, "lambda_U_proj_vs_diag_"*ver_name*".png")
