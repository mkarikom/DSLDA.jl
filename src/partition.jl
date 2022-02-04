"""
getIndRand(cc)

Get the row of the indicator for column `cc`, else return a random row.

# Examples

```
getIndRand(cc)
```
"""
function getIndRand(cc)
    if all(cc .== 0)
        println("zeros encountered")
        return rand(1:length(cc))
    else
        return findfirst(x->x==1,cc)
    end
end


# """
# invEpInd(epochinds,Z)

# Invert the partitioned vector of indicator matrices `Z` and return an array similar to `getA()`.

# # Examples

# ```
# invEpInd(epochinds,Z)
# ```
# """
# function invEpInd(epochinds,Z)
#     inds = reduce(vcat,[reduce(vcat,epi) for epi in epochinds])
#     a = reduce(hcat,convert.(Array,Z))[:,sortperm(inds)]
#     map(t->getIndRand(t), eachcol(a))
# end

"""
    initShared(corpus,partition,k,l,alphalocal,betalocal)

Initialize:
1) shared data vectors `T`,`D`
2) shared assignment vector `Z`
3) shared indicator matrices `Z`,`J`,`W`
4) shared sufficient statistics `nj`
5) global sufficient statistics `nwglobal`,`nglobal`
6) shared increment matrices `nwinc`,`ninc`
7) vectors of 'local indices': `indt`,`inda`,`indd`,`indnj`,`indnw` corresponding to each worker id in `wkids` 

# Examples

```
using Distributions
corp = map(x->Word.(rand(1:100,x),Categorical([0.5,0.3,0.2])),[10,50,100])
dist = [[1,2],[3]]
initShared(corpus,partition,3,100,fill(0.1,3),fill(0.1,100))
```
"""
function initShared(corpus,partition,k,l,alphalocal,betalocal;statefloat=Float32,stateint=Int32,sstatfloat=Float32,sstatint=Int32)
    Z = SharedArray(ones(Bool,k,sum(map(x->length(x),corpus)))) # size Z is k by # corpus tokens
    J = SharedArray(ones(Bool,sum(map(x->length(x),corpus)),length(partition))) # size J is # corpus tokens by # docs
    W = SharedArray(ones(Bool,sum(map(x->length(x),corpus)),l)) # size W is # corpus tokens by # terms
    T = SharedVector(ones(stateint,sum(map(x->length(x),corpus))))
    A = SharedVector(ones(stateint,sum(map(x->length(x),corpus))))
    D = SharedVector(ones(stateint,sum(map(x->length(x),corpus))))
    ends = cumsum(map(x->length(x),corpus)) # ends of the segments for each doc
    starts = cumsum(map(x->length(x),corpus)) .- map(x->length(x),corpus) .+ 1 # the starts of the segments for each doc
    nwstarts = map(x->x-l,cumsum(fill(l,length(partition)))).+1
    nwends = cumsum(fill(l,length(partition)))
    nj = SharedArray(zeros(sstatint,k,length(partition)))
    nw = SharedArray(zeros(sstatint,k,l,length(partition)))
    nwinc = SharedArray(zeros(sstatint,k,l*length(partition)))
    indnw = [(1:k,nwstarts[1]:nwends[1])]
    indt = [starts[1]:ends[1]]
    for i in 2:length(partition)
        push!(indnw,(1:k,nwstarts[i]:nwends[i]))
        push!(indt,starts[i]:ends[i])
    end
    indt=SharedArray(indt)
    indnw=SharedArray(indnw)

    @sync @distributed for i in 1:length(partition)
        ba = getBitarrays(corpus,partition[i],ttype=stateint,atype=stateint,dtype=stateint)
        T[starts[i]:ends[i]] .= vec(ba[:T])
        A[starts[i]:ends[i]] .= vec(ba[:A])
        D[starts[i]:ends[i]] .= vec(ba[:D])
        Z[1:k,starts[i]:ends[i]] = getZ(ba[:A],k)
        J[starts[i]:ends[i],1:length(partition)] = getJ(corpus[partition[i]],partition[i],length(partition))  
        W[starts[i]:ends[i],1:l] = getW(corpus[partition[i]],l)
        nw[:,:,i] .= Z[1:k,starts[i]:ends[i]] * W[starts[i]:ends[i],1:l]
        nj[1:k,i] .= vec(Z[1:k,starts[i]:ends[i]]*J[starts[i]:ends[i],1:length(partition)][:,partition[i]])
    end
    nw = SharedArray{sstatint,2}(sum(nw,dims=3)[:,:,1])

    println("initialization complete for:")
    println("Z{",typeof(Z),"}: ",Base.summarysize(Z)," bytes")
    println("J{",typeof(J),"}: ",Base.summarysize(J)," bytes")
    println("W{",typeof(W),"}: ",Base.summarysize(W)," bytes")
    println("T{",typeof(T),"}: ",Base.summarysize(T)," bytes")
    println("A{",typeof(A),"}: ",Base.summarysize(A)," bytes")
    println("D{",typeof(D),"}: ",Base.summarysize(D)," bytes")
    println("nwinc{",typeof(nwinc),"}: ",Base.summarysize(nwinc)," bytes")
    # copy the global counts to separate sampling vectors
    if length(partition) > 1
        for p in 2:length(partition)
            nw = hcat(nw,nw[:,1:l])
        end
        println("allocated nw{",typeof(nw),"}: ",sizeof(nw)," bytes")
    end

    return (T=T,A=A,D=D,Z=Z,J=J,W=W,
            nj=nj,nwinc=nwinc,nw=nw,
            alpha=SharedArray{sstatfloat,1}(alphalocal),beta=SharedArray{sstatfloat,2}(betalocal),
            indt=indt,indnw=indnw)
end



# """
#   getwids(arr)

# Get the worker ids of a DArray or SharedArray. 

# # Examples
# ```
# getwids(arr)
# ```
# """
# function getwids(arr)
#     if typeof(arr) <: DArray
#         ids = [(i,@fetchfrom i DistributedArrays.localindices(arr)) for i in workers()]
#     elseif typeof(arr) <: SharedArray
#         ids = [(i,@fetchfrom i SharedArrays.localindices(arr)) for i in workers()]
#     end    
#     ids = filter(!isnothing,[length(i[2][1]) > 0 ? i[1] : nothing for i in ids])
# end
  

"""
  splitApprox(corpus,n)

Re-order `corpus` into `n` chunks each containing approximately the same number of documents.
Return the shuffled view `c_view`, a vector `partition` of unit-ranges for each chunk, and vector `partmap`, where `c_view[i] = corpus[partmap[i]]`.  

# Examples
```
splitApprox(corpus,n)
```
"""
function splitApprox(corpus,n)
    parts = [findall(x->x==i,mod.([1:length(corpus);],n) .+ 1) for i in 1:n]
    partmap = reduce(vcat,parts)
    ind1 = [0,cumsum([length(i) for i in parts])[1:end-1]...] .+ 1
    ind2 = cumsum([length(i) for i in parts])
    partition = UnitRange.(ind1,ind2)
    c_perm = corpus[partmap]
    c_perm,partition,partmap
end


"""
  epochIndex(corpus,n,l)

Make random, approximately equal-length partition of docs and terms.

# Examples
```
epochIndex(corpus,n,l)
```
"""
function epochIndex(corpus,n,l)
    # partition the documents
    docindex = [findall(x->x==i,mod.([1:length(corpus);],n) .+ 1) for i in 1:n]
    # partition the terms
    termindex = [findall(x->x==i,mod.([1:l;],n) .+ 1) for i in 1:n]
    return (docindex=docindex,termindex=termindex)
end

"""
  filterTokens(n,D,T,docindex,termindex)

Generate global token epoch indices by filtering concatenated tokens.

# Examples
```
filterTokens(n,D,T,docindex,termindex)
```
"""
function filterTokens(n,D,T,docindex,termindex)
    epochs = []
    for i in 1:n
        termindexShft = circshift(termindex,i-1)
        chunks = []
        for j in 1:n
            docs = reduce(vcat,[findall(d->d==di,vec(D)) for di in docindex[j]])
            terms = reduce(vcat,[findall(t->t==ti,vec(T)) for ti in termindexShft[j]])
            inds = intersect(docs,terms)
            push!(chunks,inds)
        end
        push!(epochs,chunks)
    end
    epochs
end

