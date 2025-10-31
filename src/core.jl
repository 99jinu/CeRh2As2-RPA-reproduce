include("./physics.jl"); using .physics_module
include("./types.jl"); using .types_module
include("./compute.jl"); using .compute_module
init_H("""{"model":"squareTB","t":1.0}""")
compute_module.link_submodules()