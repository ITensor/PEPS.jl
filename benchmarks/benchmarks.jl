using BenchmarkTools, PEPS, ITensors, ITensorsGPU

suite = BenchmarkGroup()
include("environments.jl")
#include("hamiltonian.jl")
#include("observables.jl")

tune!(suite)
results = run(suite, verbose=true, seconds=20)
display(results)
