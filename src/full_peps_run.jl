using Pkg
Pkg.activate("..")
using TimerOutputs, Statistics, ArgParse, CUDAnative
include("peps.jl")

s = ArgParseSettings()
@add_arg_table! s begin
    "--prefix"
        help = "Prefix string to store magnetization data"
        arg_type = String 
    "--do_mag"
        help = "Measure magnetization"
        action = :store_true
    "--Nx"
        help = "Number of columns in the lattice"
        arg_type = Int
    "--Ny"
        help = "Number of rows in the lattice"
        arg_type = Int
    "--D"
        help = "PEPS bond dimension"
        arg_type = Int
        default = 3
    "--chi"
        help = "Environment bond dimension"
        arg_type = Int
        default = 3
    "--sweep_count"
        help = "Total number of sweeps to perform"
        arg_type = Int
        default = 50
    "--simple_update_cutoff"
        help = "Number of sweeps to perform simple updates"
        arg_type = Int
        default = 3
    "--device"
        help = "Which GPU to use"
        arg_type = Int
        default = 1
end

# get basic simulation parameters 
parsed_args = parse_args(s)
# log file which keeps track of more detailed info about the simulation, not super exciting
io = open(parsed_args["prefix"]*".txt", "w+")
logger = SimpleLogger(io)
global_logger(logger)
Nx = parsed_args["Nx"]
Ny = parsed_args["Ny"]
χ  = parsed_args["chi"]
D  = parsed_args["D"]

parsed_args["device"] != 1 && device!(parsed_args["device"])

# Hamiltonian parameters
J  = 1.0
sites = siteinds("S=1/2",Nx*Ny)

# disallow scalar indexing on GPU, which is very slow 
CuArrays.allowscalar(false)
A = checkerboardPEPS(sites, Nx, Ny, mindim=D)
# to the user, these appear as normal ITensors, but they have on-device storage
# Julia can detect this at runtime and appropriately dispatch to CUTENSOR
cA = cuPEPS(A)
H  = makeCuH_XXZ(Nx, Ny, J)
@info "Built cA and H"
# run heaviest functions one time to make Julia compile everything
@info "Built cA and H"
Ls = buildLs(cA, H; mindim=D, maxdim=D)
@info "Built first Ls"
#Rs = buildRs(cA, H; mindim=D, maxdim=D)
Rs = Vector{Environments}(undef, Nx)
@info "Built first Rs"

# actual profiling run
#prefix = "magvar/$(Nx)_$(env_add)_$(chi)_magvar"
cA, tS, bytes, gctime, memallocs = @timed doSweeps(cA, Ls, Rs, H; mindim=D, maxdim=D, simple_update_cutoff=parsed_args["simple_update_cutoff"], sweep_count=parsed_args["sweep_count"], cutoff=0.0, env_maxdim=χ, do_mag=parsed_args["do_mag"], prefix=parsed_args["prefix"])
println("Done sweeping GPU $tS")
flush(stdout)
flush(io)
