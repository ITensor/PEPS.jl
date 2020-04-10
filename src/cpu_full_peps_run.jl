using Pkg
Pkg.activate("..")
using TimerOutputs, Statistics, ArgParse, Distributions, Logging
using PEPS, ITensors, ITensorsGPU
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
    "--model"
        help = "Which Hamiltonian to use"
        arg_type = String
        default = "XXZ"
    "--J"
        help = "J-parameter"
        arg_type = Float64
        default = 1.0
    "--hz"
        help = "Field in the z parameter, or random # bound"
        arg_type = Float64
        default = 0.0
    "--hx"
        help = "Field in the x parameter, or random # bound"
        arg_type = Float64
        default = 0.0
    "--random_x"
        help = "Use random fields in the x direction"
        action = :store_true
    "--random_z"
        help = "Use random fields in the z direction"
        action = :store_true
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
J     = parsed_args["J"] 
sites = siteinds("S=1/2",Nx*Ny)

# disallow scalar indexing on GPU, which is very slow 
#CuArrays.allowscalar(false)
A = PEPS.checkerboardfPEPS(sites, Nx, Ny, mindim=D)
# to the user, these appear as normal ITensors, but they have on-device storage
# Julia can detect this at runtime and appropriately dispatch to CUTENSOR
H  = nothing
if parsed_args["model"] == "XXZ"
    H  = PEPS.makeH_XXZ(Nx, Ny, J)
elseif parsed_args["model"] == "Ising"
    hz = parsed_args["hz"]
    hx = parsed_args["hx"]
    random_z = parsed_args["random_z"]
    random_x = parsed_args["random_x"]
    if random_x || random_z
        seed = abs(rand(Int))
        @show seed
        Random.seed!(seed)
    end
    if random_x
        dx = Uniform(-hx, hx)
        hx = rand(dx, Ny, Nx)
    end
    if random_z
        dz = Uniform(-hz, hz)
        hz = rand(dz, Ny, Nx)
    end
    H  = PEPS.makeH_Ising(Nx, Ny, J, hx, hz)
end
@info "Built A and H"
# run heaviest functions one time to make Julia compile everything
@info "Built A and H"
Ls = buildLs(A, H; mindim=D, maxdim=D)
@info "Built first Ls"
#Rs = buildRs(A, H; mindim=D, maxdim=D)
Rs = Vector{PEPS.Environments}(undef, Nx)
@info "Built first Rs"

# actual profiling run
A, tS, bytes, gctime, memallocs = @timed PEPS.doSweeps(A, Ls, Rs, H; mindim=D, maxdim=D, simple_update_cutoff=parsed_args["simple_update_cutoff"], sweep_count=parsed_args["sweep_count"], cutoff=0.0, env_maxdim=χ, do_mag=parsed_args["do_mag"], prefix=parsed_args["prefix"])
println("Done sweeping GPU $tS")
flush(stdout)
flush(io)
