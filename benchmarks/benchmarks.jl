using BenchmarkTools, PEPS, ITensors, ITensorsGPU

suite = BenchmarkGroup()
Nx = 6
Ny = 6
χ  = 4
D  = 4
sites = siteinds("S=1/2",Nx*Ny)
A     = cufPEPS(randomfPEPS(sites, Nx, Ny, mindim=D))
H     = PEPS.makeCuH_XXZ(Nx, Ny, 1.0)
is_cu = PEPS.is_gpu(A)
include("environments.jl")
#include("hamiltonian.jl")
#include("observables.jl")

tune!(suite)
results = run(suite, verbose=true, seconds=20)
display(results)
