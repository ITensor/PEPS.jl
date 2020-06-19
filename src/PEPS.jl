module PEPS

using ITensors, ITensorsGPU
using CUDA, TimerOutputs
import ITensors: data, store
import ITensors.NDTensors: data
using Random, Logging, LinearAlgebra, DelimitedFiles

const q_dict = Dict{Pair{Int, Symbol}, MPO}()

include("peps_util.jl")

export fPEPS, randomfPEPS, checkerboardfPEPS, cufPEPS, randomCufPEPS, buildEdgeEnvironment, buildNextEnvironment, buildLs, buildRs

end # module
