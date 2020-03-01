using Pkg
Pkg.activate("..")
using Profile, TimerOutputs, StatProfilerHTML
include("peps.jl")

Nx  = 4 
Ny  = 4
mdim = 3
io = open("prof_$(string(Nx))_$mdim.txt", "w+")
logger = SimpleLogger(io)
global_logger(logger)
J  = 1.0
sites = siteinds("S=1/2",Nx*Ny)
println("Beginning A")
A = checkerboardPEPS(sites, Nx, Ny, mindim=mdim)
cA = cuPEPS(A)
H  = makeCuH_XXZ(Nx, Ny, J)
@info "Built cA and H"
Ls = buildLs(cA, H; mindim=mdim, maxdim=mdim)
@info "Built first Ls"
Rs = buildRs(cA, H; mindim=mdim, maxdim=mdim)
@info "Built first Rs"
cA, Ls, Rs = rightwardSweep(cA, Ls, Rs, H; mindim=mdim, maxdim=mdim)
cA, Ls, Rs = leftwardSweep(cA, Ls, Rs, H; mindim=mdim, maxdim=mdim)
Profile.init(n=10_000_000)
@profile doSweeps(cA, Ls, Rs, H; mindim=mdim, maxdim=mdim)
statprofilehtml()
