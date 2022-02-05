using DSLDA
using Test, Statistics, StatsBase, Random, Distributions, CSV, DataFrames
using Distributed, CUDA, SharedArrays

nwork = 4
while length(workers()) < nwork
    addprocs(1)
end

@everywhere using DistributedArrays,dsLDA

function fillpb(scMeta,scReads,trueP;ncells=100,celltypes=["alpha","beta","gamma","delta"])
  pbulk = zeros(Int,size(trueP)[1],size(scReads)[2]-1)
  for i in 1:size(trueP)[1]
    for j in 1:size(trueP)[2]
      # @infiltrate
      allids = scMeta[scMeta.cellType.==celltypes[j],:].cellID
      inds = StatsBase.sample(1:length(allids),Int(ncells*trueP[i,j]),replace=true)
      ids = allids[inds]
      pbulk[i,:] .= vec(sum(Matrix(filter(x -> x.cellID in ids,scReads)[:,2:end]),dims=1))
    end
  end
  pbulk
end

@testset "deconvolution: baron" begin
  data_dir = "test/data/baron"
  
  C_fn = joinpath(data_dir,"BaronSC.H.isletVST_C.csv")
  pDataC_fn = joinpath(data_dir,"BaronSC.H.isletVST_pDataC.csv")
  thetabenchmark=nothing
  
  # sc metadata
  scMeta = DataFrame(CSV.File(pDataC_fn))[:,["cellID","cellType","sampleID"]]
  scMeta = filter(x -> x.cellType in ["alpha","beta","gamma","delta"],scMeta)
  
  # sc reads
  scReads = DataFrame(CSV.File(C_fn,transpose=true)) # this is an R data frame with "row names"
  rename!(scReads,Symbol.(["cellID",names(scReads)[2:end]...]))
  scReads = filter(x -> x.cellID in scMeta.cellID,scReads)
  
  # true proportions
  trueP = [
    0.1 0.3 0.5 0.1;
    0.5 0.1 0.3 0.1;
    0.3 0.1 0.1 0.5]
  
  # bulk reads
  pbulk = fillpb(scMeta,scReads,trueP,ncells=50,celltypes=["alpha","beta","gamma","delta"])
  bulkReads = DataFrame(pbulk,:auto)
  rename!(bulkReads,names(scReads[:,2:end]))
  insertcols!(bulkReads, 1, :bulkID => ["Bulk1","Bulk2","Bulk3"] )
    
  trueP = DataFrame(trueP,:auto)
  rename!(trueP,["alpha","beta","gamma","delta"])
  insertcols!(trueP, 1, :bulkID => ["Bulk1","Bulk2","Bulk3"] )
  
  jl_output = dsLDA_E_step(
      Matrix(scReads[:,2:end]),
      names(scReads[:,2:end]),
      scReads.cellID,
      scMeta,
      Matrix(bulkReads[:,2:end]),
      names(bulkReads[:,2:end]),
      bulkReads.bulkID,
      "",
      nparts=3,
      runqc=true,
      ldagenethresh=0.1,
          minCellsTopicCorp=1,
      scalebulk="log1p",
      bulkfactor=1000,
      scalesc="ident",
      betapseudo=0.0,
      scfactor=1.0,
      betaeps=0.01,
      nchains=2,
      alpha=0.1,
      philatent=2,
      blocksize=5,
      niter=1000,
      initflavor="unif",
      verbose=false,
      burn=0.5,
      thinning=5,
      rmchains=true
    )
  @test sqrt(mean((jl_output[1].-Matrix(trueP[:,2:end])).^2)) < .2
end
