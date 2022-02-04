"""
  getBitarrays(corpus,inds;ttype=Int64,atype=Int64,dtype=Int64)

Get the bit array representation of the corpus data.
Return initial topic indices `T`, lexicon index `A`, and the document indices `D` of the tokens. 
Shape is `Array{type,2}(data,1,length(data))`.
Optionally specify numeric types for the output.

# Examples
```
corp = map(x->Word.(rand(1:100,x),Int8.(rand(Categorical([0.5,0.3,0.2]),x))),[10,50,100])
T,A = getBitArrays(corp,1:10,Int16,Int16)
```
"""
function getBitarrays(corpus,inds;ttype=Int64,atype=Int64,dtype=Int64)
  tokens = [map(w->w.t,d) for d in corpus[inds]]
  topics = [map(w->w.a,d) for d in corpus[inds]]
  docinds = [map(i->d,corpus[d]) for d in inds]
  T = Array{ttype}(reduce(vcat,tokens))
  A = Array{atype}(reduce(vcat,topics))
  D = Array{dtype}(reduce(vcat,docinds))
  T = reshape(T,(1,length(T)))
  A = reshape(A,(1,length(A)))
  D = reshape(D,(1,length(D)))
  (T=T,A=A,D=D)
end


"""
    getZ(topics,k;inds=nothing)

Generate an indicator-representation of the topic-assignment matrix.
`Z_{i,j} = 1` when token `j` is generated from latent topic `i`

Optionally specify a term index filter `inds` to sample a distributed epoch. 

# Examples
```
getZ(StatsBase.sample(1:3,10),3))
```
"""
function getZ(topics,k;inds=nothing)
  if !isnothing(inds)
    topics = topics[inds]
  end
  z = zeros(Bool,k,length(topics))
  for t in 1:length(topics)
    z[topics[t],t] += 1
  end
  z
end


"""
    getJ(corpus;inds=nothing)

Generate the indicator document/topic loadings `J` s.t. `Z*J=nj`.
`J_{i,j} = 1` when token `i` is in document `j`. 

Optionally specify a term index filter `inds` to sample a distributed epoch. 

# Examples
```
corp = map(x->Word.(rand(1:100,x),Int8.(rand(Categorical([0.5,0.3,0.2]),x))),[10,50,100])
c = getCorp(c)
A,T = getBitarrays(c)
J = getJ(T)
```
"""
function getJ(corpus;inds=nothing)
  tokens = reduce(vcat,[map(w->w.t,d) for d in corpus])
  if !isnothing(inds)
    tokens = filter(!isnothing,indexin(tokens,inds))
  end
  j = zeros(Bool,length(tokens),length(corpus))
  ct = 0
  for d in 1:length(corpus)
    for t in 1:length(corpus[d])
      ct+=1
      j[ct,d]+=1
    end
  end
  j
end


"""
    getJ(corpus,drange,ndocs;inds=nothing)

See `getJ(corpus;inds=nothing)`.  Inserts padding columns for vertical concatenation.  

Optionally specify a term index filter `inds` to sample a distributed epoch. 

# Examples
```
corp = map(x->Word.(rand(1:100,x),Int8.(rand(Categorical([0.5,0.3,0.2]),x))),[10,50,100])
c = getCorp(c)
Z = getZ(c,3)
J = getJ(c[1:2],1:2,3)
```
"""
function getJ(corpus,drange,ndocs;inds=nothing)
  tokens = reduce(vcat,[map(w->w.t,d) for d in corpus])
  if !isnothing(inds)
    tokens = filter(!isnothing,indexin(tokens,inds))
  end
  j = zeros(Bool,length(tokens),ndocs)
  ct = 0
  for d in 1:length(corpus)
    for t in 1:length(corpus[d])
      ct+=1
      j[ct,drange[d]]+=1
    end
  end
  j
end


"""
    getW(corpus,l;inds=nothing)

Generate the indicator term/topic loadings `W` s.t. `Z*W=nw`.
`W_{i,j} = 1` when token `i` corresponds to term `j` in the lexicon with length `l`. 

Optionally specify a term index filter `inds` to sample a distributed epoch.

# Examples
```
corp = map(x->Word.(rand(1:100,x),Int8.(rand(Categorical([0.5,0.3,0.2]),x))),[10,50,100])
c = getCorp(c)
Z = getZ(c,3)
W = getW(c)
```
"""
function getW(corpus,l;inds=nothing)
  tokens = reduce(vcat,[map(w->w.t,d) for d in corpus])
  if !isnothing(inds)
    tokens = filter(!isnothing,indexin(tokens,inds))
  end
  w = zeros(Bool,length(tokens),l)
  for t in 1:l
    w[:,t] .= .!isnothing.(indexin(tokens,[t]))
  end
  w  
end

"""
  nj2token(nj,J)

Expand `nj` (the topic x document counts) over tokens.
`NJK_{i,j} = n` where `n` is the number of times topic `j` is assigned to some token in the parent document of token `i`. 

# Examples
```
corp = map(x->Word.(rand(1:100,x),Int8.(rand(Categorical([0.5,0.3,0.2]),x))),[10,50,100])
c = getCorp(c)
Z = getZ(c,3)
J = getJ(c)
nj2token(Z*J,J)
```
"""
function nj2token(nj,J)
  d = map(i->i[2],findmax(J,dims=2)[2])
  NJK = nj[:,vec(d)]
end

"""
  nw2token(nw,W)

Expand `nw` (the topic x term counts) over tokens.
`NWK_{i,j} = n` where `n` is the number of times topic `j` is assigned to some token in the parent document of token `i` corresponding to the same term in the lexicon. 

# Examples
```
corp = map(x->Word.(rand(1:100,x),Int8.(rand(Categorical([0.5,0.3,0.2]),x))),[10,50,100])
c = getCorp(c)
Z = getZ(c,3)
W = getW(c)
nj2token(Z*W,W)
```
"""
function nw2token(nw,W)
  t = map(i->i[2],findmax(W,dims=2)[2])
  NWK = nw[:,vec(t)]
end
