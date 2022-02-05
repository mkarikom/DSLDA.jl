module DSLDA

using Distributions, StatsBase, Statistics, SpecialFunctions
using LinearAlgebra, Distributed
using SharedArrays,DataFrames,CSV
using FileIO,MCMCChains,ProgressMeter
using Random: Random

export Word,getCorp,getSynthData
export getBitarrays
export initShared
export mat2docind, vec2procind, initterms!,initcorp,initcorpBeta,initcorpChain
export getZ,getJ,getW
export nj2token,nw2token
export chunkSharedSV!,sampleSharedSV!,runChainSharedSV
export splitApprox,epochIndex,filterTokens,getIndRand
export findGenePrefix,filterCells,filterGenes,getParamDict,filterPbulkparams,dsLDA_E_step,procsN,getTheta!

include("corpus.jl") # corpus data and methods to get bit arrays from
include("bitarrays.jl") # methods on bit arrays
include("partition.jl") # partitioning and allocation
include("sampling.jl") # sampler code
include("chains.jl") # sampling diagnostics
include("durian.jl") # dslda stuff for durian
end
