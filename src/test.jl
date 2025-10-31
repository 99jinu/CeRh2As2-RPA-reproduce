HPARAMS_JSON="""
    {"model": "squareTB", "kdim": 2, "Hdim": 2, "t": 1.0, "t2": 0.0, "mu": 0, "Rashba_strength": 0.0
}"""

RPARAMS_JSON="""{
  "soft_eta": 1e-6, "hard_eta": 1e-6, "Num_E_range": 16, "kgrid_number": 64, "hard_energy_cutoff": 20.0, "rtol": 1e-6, "atol":1e-8, "maxevals":1e7, "T": 0.01
}"""

ARGS = [HPARAMS_JSON, RPARAMS_JSON]
include("./core.jl")