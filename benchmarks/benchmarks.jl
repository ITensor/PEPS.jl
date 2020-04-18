using BenchmarkTools, PEPS, ITensors, ITensorsGPU

suite = BenchmarkGroup()
Nx = 4
Ny = 4
χ  = 6
D  = 6
sites = siteinds("S=1/2",Nx*Ny)
A     = randomfPEPS(sites, Nx, Ny, mindim=D)
H     = PEPS.makeH_XXZ(Nx, Ny, 1.0)
is_cu = PEPS.is_gpu(A)
include("environments.jl")
include("hamiltonian.jl")
include("ancillaries.jl")
include("gauge.jl")

tune!(suite)
results = run(suite, verbose=true, seconds=20)
display(results)
