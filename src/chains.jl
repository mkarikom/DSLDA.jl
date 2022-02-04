"""
    getPhi(Zchain,Beta,W,niter,nchains,k)
get the topic/word posterior for `niter` samples
# Examples
```
getPhi(Zchain,Beta,W,niter,nchains,k)
```
"""
function getPhi(Zchain,Beta,W,niter,nchains,k,nlex;verbose=false)
    println("niter:",niter)
    Phi = SharedArray(zeros(niter,nlex,k,nchains))
    println("Phi size: ",size(Phi))
    for j in 1:nchains
        Threads.@threads for i in 1:niter    
            verbose ? println("getting chain $j, iter $i") : nothing
            nw = permutedims(getZ(Zchain[i,:,j],k)*W)
            Phi[i,:,:,j] .= (nw .+ Beta) ./ sum(nw .+ Beta)
        end
    end
    Phi
end

"""
    getTheta(Zchain,J,doc_id,niter,nchains,k)
get the thetas for `niter` samples of document `doc_id`
# Examples
```
getTheta(Zchain,J,doc_id,niter,nchains,k)
```
"""
function getTheta!(Thetas,Zchain,J,Alpha,doc_id,niter,chain_id,k;verbose=false)
    @sync @distributed for i = 1:niter
        thetaIter!(Thetas,Zchain[i,:,chain_id],J,doc_id,Alpha,i,chain_id,k)
    end  
end

function thetaIter!(Theta,Z,J,doc_id,Alpha,iter,chain_id,k)
    nj = getZ(Z,k)*J
    Theta[iter,:,chain_id,doc_id] .= vec((nj[:,doc_id] .+ Alpha) ./ sum(nj[:,doc_id] .+ Alpha))
end


"""
    getTheta(Zchain,J,doc_id,niter,nchains,k)
get the thetas for `niter` samples of document `doc_id`
# Examples
```
getTheta(Zchain,J,doc_id,niter,nchains,k)
```
"""
function getTheta(Zchain,J,Alpha,doc_id,niter,nchains,k;verbose=false)
    Theta = SharedArray(zeros(niter,k,nchains))
    for j in 1:nchains
        @sync @distributed for i = 1:niter
            thetaIterOld!(Theta,Zchain,J,doc_id,Alpha,i,j,k)
        end  
    end
    Theta
end

function thetaIterOld!(Theta,Zchain,J,doc_id,Alpha,i,j,k)
    nj = getZ(Zchain[i,:,j],k)*J
    Theta[i,:,j] .= vec((nj[:,doc_id] .+ Alpha) ./ sum(nj[:,doc_id] .+ Alpha))
end


"""
    getThetaHat(thetas,ndocs,ntopic)
Get the estimate of theta by averaging samples across chains
# Examples
```
getThetaHat(thetas,ndocs,ntopic)
```
"""
function getThetaHat(thetas,ndocs,ntopic)
    thetahat0 = mean(thetas,dims=[1,3])
    thetahat = zeros(ndocs,ntopic)
    for i in 1:size(thetahat0)[4]
        thetahat[i,:] .= vec(thetahat0[:,:,:,i])
    end
    thetahat
end

"""
    getPhiHat(phis,nlex,ntopic)
Get the estimate of phi by averaging samples across chains
# Examples
```
getPhiHat(phis,nlex,ntopic)
```
"""
function getPhiHat(phis,nlex,ntopic)
    phihat0 = mean(phis,dims=[1,4])
    phihat = zeros(nlex,ntopic)
    for i in 1:size(phihat0)[3]
        phihat[:,i] .= phihat0[1,:,i,1]
    end
    phihat
end

"""
    getHPI(x; alpha=0.05)
Get the highest-probability interval
# Examples
```
getHPI(rand(Beta(6,2),10000))
```
"""
function getHPI(x; alpha=0.05)
    # this is MCMCChains._hpd @ master "2ff0a26beda7223552e47d25d948d6c7e44baa95"
    n = length(x)
    m = max(1, ceil(Int, alpha * n))

    y = sort(x)
    a = y[1:m]
    b = y[(n - m + 1):n]
    _, i = findmin(b - a)

    return [a[i], b[i]]
end


"""
    getETI(x; alpha=0.05)
Get the equal-tailed interval (quantiles)
# Examples
```
getETI(rand(Beta(6,2),10000))
```
"""
function getETI(x; alpha=0.05)
    a=quantile!(x,alpha/2)
    b=quantile!(x,1-alpha/2)
    return [a,b]
end