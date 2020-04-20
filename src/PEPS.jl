module PEPS

using ITensors, ITensorsGPU
using CuArrays, TimerOutputs
import ITensors: data, store
import ITensors.NDTensors: data
using Random, Logging, LinearAlgebra, DelimitedFiles


include("peps_util.jl")

export fPEPS, randomfPEPS, checkerboardfPEPS, cufPEPS, randomCufPEPS, buildEdgeEnvironment, buildNextEnvironment, buildLs, buildRs

end # module
