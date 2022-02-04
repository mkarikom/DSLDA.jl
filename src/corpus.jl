"""
    Word{T,A}

Word corresponding to token `t` and assignment `a`.

# Examples
```
# Generate single words
Word(1,Int16(2))
Word("a",Int16(2))

# Generate a corpus (Int64,full loop)
c = []
k = 3 # number of topics
ndoc = 10 # number of documents
nlex = 100 # number of terms in lexicon
nread = 1000 # number of tokens per document
lexicon = [1:nlex;]
topics = [1:k;]
for i in 1:ndoc
    x = rand(lexicon,1000)
    y = rand(topics,1000)
    arr = Word.(x,y)
    push!(c,arr)
end

# Generate a corpus (Int8 topic, map)
corp = map(x->Word.(rand(1:100,x),Int8.(rand(Categorical([0.5,0.3,0.2]),x))),[10,50,100])
```
"""
struct Word{T,A}
  t::T # the index of the token in the lexicon
  a::A # the latent topic associated with the token
end


"""
  getCorp(corp)

Get the parametric representation of a corpus.

# Examples
```
# Generate a corpus (Int8 topic, map)
corp = map(x->Word.(rand(1:100,x),Int8.(rand(Categorical([0.5,0.3,0.2]),x))),[10,50,100])
c = getCorp(c)
```
"""
getCorp(corp) = Array{Array{Word{typeof(corp[1][1].t),typeof(corp[1][1].a)},1},1}(corp)


"""
  mat2docind(mat)

Given a matrix of counts, get the indices of the resulting row-wise doc vector for each lexicon term.

# Examples
```
mat2docind(mat)
```
"""
function mat2docind(mat)
  ends = cumsum(mat,dims=2)
  starts = hcat(ones(Int,size(mat)[1]),(cumsum(mat,dims=2) .+ 1)[:,1:end-1])
  [[starts[i],ends[i]] for i in CartesianIndices(mat)]
end

"""
  vec2procind(vlen,nproc)

Partition the indices of a vector of length `vlen` into `nproc` approximately equal groups.

# Examples
```
vec2procind(vlen,nproc)
```
"""
function vec2procind(vlen,nproc)
  pvec = mod.(1:vlen,nproc).+1
  [findall(x->x==i,pvec) for i in unique(pvec)]
end

"""
  initterms!(data,Tdoc,Adoc,datind,docind,k)

For each term corresponding to `datind` in a row (`data`) from a doc x terms matrix (as in gene expression data), draw initial assignments from `[1:k;]` for tokens (corresponding to `docind`) as `Adoc` and update the token indices in `Tdoc`.

# Examples
```
initterms!(data,Tdoc,Adoc,datind,docind,k)
```
"""
function initterms!(data,Tdoc,Adoc,datind,docind,k)
  for i in 1:length(datind)
    # println("initializing term:",datind[i])
    Tdoc[docind[i][1]:docind[i][2]] .= datind[i]
    Adoc[docind[i][1]:docind[i][2]] .= rand([1:k;],data[datind[i]])
  end
  # println("init complete")
end

"""
  initcorp(counts,inds,procinds,doclengths,k,nwork)

Initialize a corpus over the matrix `counts` using `nwork` available workers.

# Examples
```
initcorp(counts,inds,procinds,doclengths,k,nwork)
```
"""
function initcorp(counts,inds,procinds,doclengths,k,nwork)
  corp = []
  for d in 1:size(counts)[1]
    println("initializing doc $d")
    Tdoc = SharedArray(zeros(Int32,doclengths[d]))
    Adoc = SharedArray(zeros(Int32,doclengths[d]))
    # @sync begin
    for i in 1:nwork
        # p = workers()[i]
        # initterms!(counts[d,:],Tdoc,Adoc,procinds[i],inds[d,procinds[i]],k)
        initterms!(counts[d,:],Tdoc,Adoc,procinds[i],inds[d,procinds[i]],k)
        # @async remotecall_wait(initterms!,p,counts[d,:],Tdoc,Adoc,procinds[i],inds[d,procinds[i]],k)
    end
    # end
    push!(corp,[Word(Tdoc[i],Adoc[i]) for i in 1:doclengths[d]])
  end
  corp
end


"""
  initcorpBeta!(corp,beta)

Randomly initialize the word assignments in a corpus, based on the topic word prior

# Examples
```
initcorpBeta!(corp,beta)
```
"""
function initcorpBeta(corp,beta)
  newcorp = deepcopy(corp)
  for i in 1:length(corp)
    for j in 1:length(corp[i])
      atype = typeof(corp[i][j].a)
      betanorm = beta[corp[i][j].t,:] ./ sum(beta[corp[i][j].t,:])
      global newcorp[i][j] = Word(corp[i][j].t,atype(rand(Categorical(betanorm))))
    end
  end
  newcorp
end

"""
  initcorpChain!(corp,chain_i)

Initialize the word assignments in a corpus, based on the chain id

# Examples
```
initcorpChain!(corp,chain_i)
```
"""
function initcorpChain(corp,chain_i,k)
  newcorp = deepcopy(corp)
  for i in 1:length(corp)
    for j in 1:length(corp[i])
      atype = typeof(corp[i][j].a)
      global newcorp[i][j] = Word(corp[i][j].t,atype(mod(chain_i,k)+1))
    end
  end
  newcorp
end

"""
  initcorpChain!(corp,chain_i)

Uniform random initialization

# Examples
```
initcorpUnif!(corp,chain_i)
```
"""
function initcorpUnif(corp,k)
  newcorp = deepcopy(corp)
  for i in 1:length(corp)
    for j in 1:length(corp[i])
      atype = typeof(corp[i][j].a)
      global newcorp[i][j] = Word(corp[i][j].t,atype(rand([1:k;])))
    end
  end
  newcorp
end


"""
  getSynthData(nbulk,ncells,nreads,zeta,K)

Create `nbulk` pseudobulk samples from `ncells` simulated cells.
Generate n x `ncells` reads where n ~ Poisson(λ = `nreads`).
Allocate these bulk reads based to a queue based on fidelity matrix `zeta`.   

# Examples
```
getSynthData(nbulk,ncells,nreads,zeta,K)
```
"""
function getSynthData(nbulk,ncells,nreads,zeta,K)
  println("done")
end
