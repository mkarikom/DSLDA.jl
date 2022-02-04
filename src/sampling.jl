"""
  chunkSharedSV!(f,philatent,Y,T,A,D,nj,nwinc,ninc,nw,n,alpha,beta,indt,indnw,indn; verbose=false)

Iterate over chunk `p` of data on a SharedArray, updating copies of sufficient stats: `nj`,`nwinc`,`n`, and assignments: `A` for all tokens. 

Sample the topic index `z ~ Z` from the sufficient stats of token from document `j` corresponding to lexicon term index `i`.
```math
p(z=k) \\propto \\frac{nw^{\\lnot ij}+\\beta[i]}{n^{\\lnot ij} + W\\beta[i]} (nj^{\\lnot ij} + \\alpha)
``` 
The function \$ f(A_{\\lnot i},Y_{\\lnot j})\$ is applied to the assignments (`A`) and response data `Y` during sampling of \$ a_{i,j}\$ for term \$i\$ of document \$j\$. 

The function `philatent(\beta,nw)` denotes prior knowledge of the word/topic associations.

# Examples
```
chunkSharedSV!(f,philatent,Y,T,A,D,nj,nwinc,ninc,nw,n,alpha,beta,indt,indnw,indn; verbose=false)
```
"""
function chunkSharedSV!(philatent,T,A,D,nj,nwinc,nw,alpha,beta,indt,indnw,nlex; verbose=false)    
  indnw_linear = LinearIndices(nw)[indnw...]
  for i in 1:length(indt)
    t = T[indt[i]]
    a = A[indt[i]]
    d = D[indt[i]]

    # take current doc/term complement of nj,nw
    njI = LinearIndices(nj)
    nji = njI[CartesianIndex(a,d)] 
    nj_old = nj[nji] # copy old for debug    
    if verbose
      println("attempting to access a:$a, t:$t of LinearIndices(nw)[indnw...]:")
    end
    nwi = indnw_linear[CartesianIndex(a,t)]
    nw_old = nw[nwi] # copy old for debug
    if verbose
      println("\n \n")
      println("sampling chunk token $i of ",length(indt)," from doc $d")
      println("nji=$nji,nwi=$nwi")
      println("nj[nji]=",nj[nji])
      println("nw[nwi]=",nw[nwi])
      println("nj[:,d]=",nj[:,d])
      println("nw[:,t]=",nw[:,t])
      println("indnw:\n",indnw)
      println("size(nw[indnw...]):",size(nw[indnw...]))
      println("size(nw):",size(nw))
      println("beta[:,t]:",beta[:,t])
    end
    @assert nj_old > 0 "nji must be positive"
    @assert nw_old > 0 "nwi must be positive"

    nj[nji] -= 1
    nw[nwi] -= 1

    # sample the new assignment
    @assert philatent in [2,1,0] "latent phi needs to be specified" 
    if philatent == 0
      wprob = beta[:,t]
    elseif philatent == 1
      wprob = beta[:,t] ./ (beta[:,t] .* nlex)
    elseif philatent == 2
      wprob = (nw[indnw...][:,t] .+ beta[:,t]) ./ sum(nw[indnw...] .+ beta[:,t],dims=2)
    end

    dprob = (nj[:,d] .+ vec(alpha)) ./ sum(nj[:,d] .+ vec(alpha))
    z = dprob .* wprob

    if verbose
      println("\n \n")
      println("z:",z)
      println("wprob:",wprob)
      println("dprob:",dprob)
    end


    samp = rand()*sum(z)
    a_new = findfirst(x->x>0,vec(cumsum(z,dims=1)) .- samp)

    if verbose
      println("\n \n")
      println("a_new:",a_new)
      println("a:",a)
      println("d:",d)
      println("njI:",njI)
      println("CartesianIndex(a_new,d):",CartesianIndex(a_new,d))
    end

    if isnothing(a_new)
      a_new = a
    end


    # update nj, nw
    nji_new = njI[CartesianIndex(a_new,d)]
    nwi_new = indnw_linear[CartesianIndex(a_new,t)]
    nj[nji_new] += 1
    nw[nwi_new] += 1
    A[indt[i]] = a_new

    # record assignment nwinc
    nwinc_old = nwinc[nwi_new]
    nwinc[nwi_new] += 1

    if verbose
      println("\n \n")
      println("A=$a->$a_new")
      println("nwi=$nwi,size(nw[indnw...])=",size(nw[indnw...]))
      println("nwinc update $nwi_new: $nwinc_old->",nwinc[nwi_new])
    end

    if verbose
      println("nj update: doc=$d, topic $nj_old -> ",nj[nji])
      println("nw update: term=$t, topic $nw_old -> ",nw[nwi])
      # sleep(0.2)
    end
  end
end


"""
  sampleSharedSV!(f,philatent,Y,T,A,D,nj,nwinc,ninc,nw,n,alpha,beta,indt,indnw,indn,k,l,wids; verbose=false)

Parallel loop over SharedArray data, iterating locally on each worker in `wids`. 
See `initShared` for parameter definitions: `T`,`A`,`D`,`nj`,`nwinc`,`ninc`,`nw`,`n`,`alpha`,`beta`.
Additional parameters: number of topics `k`, number of lexicon terms `l`.

The function \$ f(A_{\\lnot i},Y_{\\lnot j})\$ is applied to the assignments (`A`) and orthogonal data `Y` during sampling of \$ a_{i,j}\$ for term \$i\$ of document \$j\$. 

The function `philatent(\beta,nw)` denotes prior knowledge of the word/topic associations.

# Examples
```
sampleSharedSV!(f,philatent,Y,T,A,D,nj,nwinc,ninc,nw,n,alpha,beta,indt,indnw,indn,k,l,wids; verbose=false)
```
"""
function sampleSharedSV!(philatent,T,A,D,nj,nwinc,nw,alpha,beta,indt,indnw,k,nlex,wids; verbose=false)
  nw_global = zeros(k,nlex)
  # sample the assignments

  # test
  if verbose
    for i in 1:length(wids)
      println("i:",i)
      println("indt[i]:",indt[i])
      println("indnw[i]:",indnw[i])
      chunkSharedSV!(philatent,T,A,D,nj,nwinc,nw,alpha,beta,indt[i],indnw[i],nlex,verbose=verbose)
    end
  else
    @sync begin
      for i in 1:length(wids)
        @async remotecall_wait(chunkSharedSV!,wids[i],philatent,T,A,D,nj,nwinc,nw,alpha,beta,indt[i],indnw[i],nlex,verbose=verbose)
      end
    end  
  end

  # synchronize global nw,n
  for i in 1:length(wids)
    nw_global += nwinc[indnw[i]...]
  end      
  for i in 1:length(wids)
      nw[indnw[i]...] .= nw_global
  end
end


"""
  runChainSharedSV(corpus,partition,partmap,betalocal,alphalocal,k,l,drnm,nlex,stateint,statefloat,philatent,f; verbose=false)

Parallel loop over the data, iterating locally on each worker. 
See `initDistributed` for parameter definitions: `T`,`A`,`D`,`nj`,`nwinc`,`ninc`,`nw`,`n`,`alpha`,`beta`.
Additional parameters: number of topics `k`, number of lexicon terms `l`.

# Examples
```
runChainSharedSV(corpus,partition,partmap,betalocal,alphalocal,k,l,drnm,nlex,stateint,statefloat,philatent,f; verbose=false)
```
"""

function runChainSharedSV(corpus0,partition,partmap,betalocal,alphalocal,
                            k,nlex,philatent,
                            stateint,statefloat,nworkers_local,chainblockn,n_iter,
                            chain_i,drnm,initcorp,Thetas,indthin; verbose=false)
  
  ccopy = deepcopy(corpus0)
  if initcorp == "beta"
    corpus = initcorpBeta(corpus0,betalocal)
  elseif initcorp == "chain"
    corpus = initcorpChain(corpus0,chain_i,k)
  elseif initcorp == "unif"
    corpus = initcorpUnif(corpus0,k)
  end

  @assert ccopy != corpus "re-initialization failed"
  println("starting chain $chain_i")

  println("begin initializing state arrays: T,A,D,Z,J,W")
  T,A,D,Z,J,W,nj,nwinc,nw,alpha,beta,indt,indnw = initShared(corpus,partition,k,nlex,alphalocal,betalocal,sstatint=Int64)

  @assert size(nw) == (k,nlex*nworkers_local) "nw: incorrect shape or number of distributed copies, topics"
  @assert size(nj) == (k,length(corpus)) "nj: incorrect shape or number of documents, topics"
  
  wids = workers()[1:nworkers_local]

  thin_ind = 1 # the index of the saved thinned samples
  @showprogress 1 "Running $n_iter samples..." for i in 1:Int(n_iter)
      # println("iteration $i")
      sampleSharedSV!(philatent,T,A,D,nj,nwinc,nw,alpha,beta,indt,indnw,k,nlex,wids; verbose=verbose)
      if i in indthin
        @sync @distributed for doc_id = 1:length(partmap)
          thetaIter!(Thetas,A,J,doc_id,alpha,thin_ind,chain_i,k)
        end  
        thin_ind+=1
      end
  end
  println("sampling: chain $chain_i mcmc has finished")                        
end