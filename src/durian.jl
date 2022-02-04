function procsN(n)
    if nprocs() < n
        addprocs(n-nprocs())
    end
end

function getParamDict(paramarray)
    paramnames = []
    paramvalues = []
    paramtypes = Vector{DataType}()
    for i in 1:length(paramarray)
        n,v = split(paramarray[i],".")
        push!(paramnames,n)
        if !isnothing(tryparse(Int,v))
            push!(paramtypes,Int64)
            push!(paramvalues,parse(Int64,v))
        elseif !isnothing(tryparse(Float64,v))
            push!(paramtypes,Float64)
            push!(paramvalues,parse(Float64,v))
        else
            push!(paramtypes,String)
            push!(paramvalues,v)
        end
    end
    Dict{Symbol,Any}(Symbol.(paramnames) .=> paramvalues)
end

function filterPbulkparams(pbulkparams,dfrow)
    matched = []
    ikeys = collect(keys(pbulkparams))
    ivals = collect(values(pbulkparams))
    for i in 1:length(ikeys)
        push!(matched,dfrow[ikeys[i]] == ivals[i])
    end 
    all(matched)        
end

"""
    findGenePrefix(prefs,genes)
Get the gene names starting with `prefs[i]`
# Examples
```
findGenePrefix(prefs,genes)
```
"""
function findGenePrefix(prefs,genes)
    inds = []
    for i in 1:length(genes)
        for j in 1:length(prefs)
            if startswith(genes[i],prefs[j])
                push!(inds,i)
            end
        end
    end
    genes[inds]
end


"""
    filterCells!(data,param)
Filter cells based on mAD stats
# Examples
```
filterCells!(data,param)
```
"""
function filterCells(data,param,thresh=3.0)
    stats = combine(groupby(data,:cellType), [param] => 
        ((s) -> (
            lower=median(s) - thresh*std(s),
            upper=median(s) + thresh*std(s))) => AsTable)
    datathresh = innerjoin(data,stats,on=:cellType)

    inbounds = datathresh[:,:lower] .< datathresh[:,param] .< datathresh[:,:upper]
    delete!(datathresh,.!inbounds)
    select(datathresh,Not([:lower,:upper]))[:,:cellID]
end

"""
    filterGenes!(data,minratio)
Filter genes based on minratio percent of cells expressing
# Examples
```
filterGenes!(data,minratio)
```
"""
function filterGenes(bulkdata,scdata,genes;
    minratio=0.05,protectedgenes=[])

    protected_expressed = intersect(genes,protectedgenes)
    ncells = size(scdata)[1]
    n_sample_exp_bulk = vec(sum(Array(bulkdata[:,genes]) .> 0,dims=1))
    n_sample_exp_sc = vec(sum(Array(scdata[:,genes]) .> 0,dims=1))
    
    # find genes expressed in at least minratio of cells
    genes_thresh = genes[findall(n_sample_exp_sc .>= ncells*minratio)]

    protected_plus_thresh = sort(unique([protected_expressed...,genes_thresh...]))
end

"""
    dsLDA_E_step()
Perform deconvolution for DURIAN.  Function parameters described in comments.  
"""
function dsLDA_E_step(
    scReads, # the sc reads (samples x genes)
    scGenes, # the sc genes
    scIDs, # the sc cell ids
    scMeta, # the sc metadata
    bulkReads, # the bulk reads (samples x genes)
    bulkGenes, # the bulk genes
    bulkIDs, # the bulk sample ids
    outputdir; # "where the output should be stored"
    minCellsTopicCorp = 0, # the number of cells that must exist in a population of the topic corpus after qc
    ldamodelname="dsLDA", # "name of the model, to prepend sampler output dir"
    niter=100, # "number of mcmc iterations"
    nparts=4, # "number of partitions for distributed inference, must be ≤ nworkers"
    nchains=2, # "number of chains to run"
    blocksize=10, # "number of iterations to store per block"
    betapseudo=0.0, # "pseudocount to add to beta prior to scaling"
    betaeps=0.000001, # "small value to add to allow well defined prob at each term"
    alpha=0.1, # "symmetric concentration parameter for cell types theta"
    ldanscthresh=3.0, # "filter cells outside nscthresh standard deviations for library size, deconvolution."
    scrnscthresh=3.0, # "filter cells outside nscthresh standard deviations for library size, imputation."
    ldagenethresh=0.01, # "filter genes expressed in less than genethresh*#cells for each cell type, deconvolution."
    scrgenethresh=0.3, # "filter genes expressed in less than genethresh*#cells for each cell type, imputation."
    protectedgenes=[], # array of gene names to protect from qc
    scalesc="column", # "how to scale the single cell reference."
    scalebulk="lognorm", # "how to scale the bulk data."
    bulkfactor=10000, # "scale factor for bulk data"
    scfactor=1.0, # "scale factor for sc data"
    initflavor="unif", # "how to initialize the latent assignments."
    verbose=false, # "produce diagnostic output"
    philatent=0, # "infer phi, single cell data will be used as beta concentration parameter"
    ldageneinds=nothing,
    ldascinds=nothing,
    thinning=2, # "keep mod(i,thinning) for i=1:niter"
    rmchains=true, # "remove saved chains after analysis"
    burn=0.5, # "burn in rate (eg 0.5 means burn for n/2).  0 means no burn in, start at n=1"
    runqc=false) # "if not set, genethresh and ncthresh will be ignored, intended for real data, not benchmarking already clean pseudobulk"

    println("detected $nparts partition")
    println("running $ldamodelname with nworkers:",nworkers())
    @assert nworkers() >= nparts "must have at least as many workers as partitions"

    modelparams = Dict(
        "ldamodelname"=>ldamodelname,
        "philatent"=>philatent,
        "scalebulk"=>scalebulk,
        "scalesc"=>scalesc,
        "scfactor"=>scfactor,
        "bulkfactor"=>bulkfactor,
        "betapseudo"=>betapseudo,
        "betaeps"=>betaeps,
        "alpha"=>alpha)
    params = Dict(
        "ldanscthresh"=>ldanscthresh,
        "scrnscthresh"=>scrnscthresh,
        "ldagenethresh"=>ldagenethresh,
        "scrgenethresh"=>scrgenethresh,
        "mincelltc"=>minCellsTopicCorp,
        "runqc"=>runqc,
        "ldamodelname"=>ldamodelname,
        "philatent"=>philatent,
        "scalebulk"=>scalebulk,
        "scalesc"=>scalesc,
        "scfactor"=>scfactor,
        "bulkfactor"=>bulkfactor,
        "betapseudo"=>betapseudo,
        "betaeps"=>betaeps,
        "alpha"=>alpha,
        "niter"=>niter,
        "nchains"=>nchains,
        "initflavor"=>initflavor,
        "blocksize"=>blocksize,
        "verbose"=>verbose,
        "rmchains"=>rmchains,
        "thinning"=>thinning,
        "burn"=>burn)

    modelparamvec = join([string(k,".",get(modelparams,k,nothing)) for k in keys(modelparams)],"-")
    paramvec = join([string(k,".",get(params,k,nothing)) for k in keys(params)],"-")

    println("dsLDA: loading bulk data")
    bulkdata = DataFrame(bulkReads,:auto)
    insertcols!(bulkdata, 1, :bulkID=>bulkIDs)
    rename!(bulkdata,["bulkID",bulkGenes...])

    println("dsLDA: loading sc data")
    scdata = DataFrame(scReads,:auto)
    insertcols!(scdata, 1, :scID=>scIDs)
    rename!(scdata,["cellID",scGenes...])

    println("dsLDA: loading meta data")
    rename!(scMeta,["cellID","cellType","sampleID"])

    scdata = innerjoin(scMeta,scdata,on="cellID")

    # ensure all genes are expressed in at least 1 sample
    ind_exp_bulk = findall(vec(sum(Array(bulkdata[:,bulkGenes]),dims=1)) .> 0)
    bulkdata = bulkdata[:,["bulkID",bulkGenes[ind_exp_bulk]...]]

    ind_exp_sc = findall(vec(sum(Array(scdata[:,scGenes]),dims=1)) .> 0)
    scdata = scdata[:,["cellID","cellType",scGenes[ind_exp_sc]...]]

    # intersect and reorder *expressed* genes for sc and bulk
    genes = sort(intersect(names(bulkdata),names(scdata)))
        
    if !isnothing(ldageneinds) && any(ldageneinds)
        ldagenes = scGenes[ldageneids]
        ldacells = ids_sc[ldascinds]
    elseif runqc
        #######################################################################################
        # # begin QC for co-expressed genes and single cells
        #######################################################################################

        # run the qc normally done in cobos simulations
        println("pre-qc size of sc data= ",size(scdata[:,genes]))
        println("pre-qc size of bulk data= ",size(bulkdata[:,genes]))

        # # filter genes        
        ldagenes = filterGenes(bulkdata,scdata,genes;minratio=ldagenethresh,protectedgenes=protectedgenes);
        scrgenes = filterGenes(bulkdata,scdata,genes;minratio=scrgenethresh,protectedgenes=protectedgenes);

        # # filter cells
        # add the library size to each sc sample
        ldalibsize = vec(sum(Array(scdata[:,ldagenes]),dims=2))
        scrlibsize = vec(sum(Array(scdata[:,scrgenes]),dims=2))

        insertcols!(scdata,1,:ldalibsize=>ldalibsize)
        insertcols!(scdata,1,:scrlibsize=>scrlibsize)
        
        ldacells = filterCells(scdata,:ldalibsize,ldanscthresh)
        scrcells = filterCells(scdata,:scrlibsize,scrnscthresh)

        celltypes = sort(unique(filter(row -> row.cellID in ldacells, scdata)[:,"cellType"]))
        topic_corpus = DataFrame(zeros(length(ldagenes),length(celltypes)),:auto)
        celltype_counts = []
        qc_cells = filter(row -> row.cellID in ldacells, scdata) # get the data for the qc cells
        for i in 1:length(celltypes)
            t = celltypes[i]
            topic_corpus[:,i] .= vec(mean(Matrix(filter(row -> row.cellID in ldacells, scdata)[findall(x->x==t,qc_cells[:,"cellType"]),ldagenes]),dims=1)) 
            push!(celltype_counts,findall(x->x==t,qc_cells.cellType)) # keep track of the celltypes that pass
        end
        rename!(topic_corpus,celltypes)
        insertcols!(topic_corpus, 1, :gene => ldagenes )
    
        qc_topics = [ct>minCellsTopicCorp for ct in map(x->length(x),celltype_counts)] # find the celltypes that have enough cells after qc
        celltypes = celltypes[qc_topics]
        topic_corpus = topic_corpus[:,["gene",celltypes...]]
    
        resize_tc_genes = findall(x->x>0,vec(sum(Matrix(topic_corpus[:,2:end]),dims=2)))
        ngenes_found = length(resize_tc_genes)
        println("found $ngenes_found genes expressed in remaining cell types after removing cell types with < $minCellsTopicCorp cells post-qc")
        topic_corpus = topic_corpus[resize_tc_genes,:]
        ldagenes = ldagenes[resize_tc_genes]    

        # save the qc data
        if length(outputdir) > 1
            qct_fn = joinpath(outputdir,"scldaPostqcT.csv")
            qcc_sclda_fn = joinpath(outputdir,"qc_sclda_C.csv")
            qcc_scr_fn = joinpath(outputdir,"qc_scr_C.csv")
            
            CSV.write(qct_fn, bulkdata[:,Symbol.(ldagenes)])
            CSV.write(qcc_sclda_fn, filter(row -> row.cellID in scrcells, scdata)[:,["cellID","cellType",ldagenes...]])
            CSV.write(qcc_scr_fn, filter(row -> row.cellID in scrcells, scdata)[:,["cellID","cellType",scrgenes...]])    
        end

        println("deconvolution post-cell qc size of sc data= ",size(filter(row -> row.cellID in ldacells, scdata)[:,Symbol.(ldagenes)]))
        println("imputation post-cell qc size of sc data= ",size(filter(row -> row.cellID in scrcells, scdata)[:,Symbol.(scrgenes)]))
        println("post-qc size of bulk data= ",size(bulkdata[:,Symbol.(ldagenes)]))
    else
        ldacells = scdata[:,"cellID"]
        ldagenes = genes
        scrcells = ldacells
        scrgenes = ldagenes
        celltypes = sort(unique(filter(row -> row.cellID in ldacells, scdata)[:,"cellType"]))
        topic_corpus = DataFrame(zeros(length(ldagenes),length(celltypes)),:auto)
        celltype_counts = []
        qc_cells = filter(row -> row.cellID in ldacells, scdata) # get the data for the qc cells
        for i in 1:length(celltypes)
            t = celltypes[i]
            topic_corpus[:,i] .= vec(mean(Matrix(filter(row -> row.cellID in ldacells, scdata)[findall(x->x==t,qc_cells[:,"cellType"]),ldagenes]),dims=1)) 
            push!(celltype_counts,findall(x->x==t,qc_cells.cellType)) # keep track of the celltypes that pass
        end
        rename!(topic_corpus,celltypes)
        insertcols!(topic_corpus, 1, :gene => ldagenes )
    
        qc_topics = [ct>minCellsTopicCorp for ct in map(x->length(x),celltype_counts)] # find the celltypes that have enough cells after qc
        celltypes = celltypes[qc_topics]
        topic_corpus = topic_corpus[:,["gene",celltypes...]]
    
        println("post filter size(topic_corpus)=$(size(topic_corpus))")
        println("post filter unique(celltypes)=$(unique(celltypes))")
        resize_tc_genes = findall(x->x>0,vec(sum(Matrix(topic_corpus[:,2:end]),dims=2)))
        ngenes_found = length(resize_tc_genes)
        println("found $ngenes_found genes expressed in remaining cell types after removing cell types with < $minCellsTopicCorp cells post-qc")
        topic_corpus = topic_corpus[resize_tc_genes,:]
        ldagenes = ldagenes[resize_tc_genes]    
    end

    #######################################################################################
    # # end QC
    #######################################################################################


    
    cmat = Matrix{Float64}(bulkdata[:,ldagenes])

    if scalebulk == "column"
        println("bulk column scaling selected, divide the counts by the total counts for each cell type, multiply by bulkfactor($bulkfactor)") 
        cmat .= cmat ./ sum(cmat,dims=1)
        cmat .= cmat .* bulkfactor
    elseif scalebulk == "ident"
        println("bulk ident selected, no change") 
        cmat .= cmat
    elseif scalebulk == "lognorm"
        println("bulk lognorm scaling selected, divide the counts by the total counts for each sample, multiply by bulkfactor($bulkfactor), add pseudocount of 1, then log transform") 
        cmat .= cmat ./ sum(cmat,dims=1)
        cmat .= cmat .* bulkfactor
        cmat .= log.(cmat .+ 1)
    elseif scalebulk == "log1p"
        println("bulk log1p scaling selected, add pseudocount of 1, then log transform") 
        cmat .= log.(cmat .+ 1)
    end        

    cmat = SharedArray(Int32.(round.(cmat)))
    @assert size(cmat)[2] == length(ldagenes)

    topicnames = celltypes
    k = length(celltypes)
    topic_corpus = permutedims(Matrix(topic_corpus[:,2:end]))

    inds = mat2docind(cmat)
    procinds = vec2procind(size(cmat)[2],nworkers())
    doclengths = [i[2] for i in inds[:,end]]
    corpus = initcorp(cmat,inds,procinds,doclengths,k,nworkers())
    
    k,nlex = size(topic_corpus)
    
    println("checking env....")
    
    println("set $nparts for corpus of length $(length(corpus))")
    nparts = minimum([nparts,length(corpus)])
    if nparts > length(corpus)
        println("partition count > length(corpus), using workers()[1:$nparts]")
    end

    ###########################################################
    # # sampling parameters
    ###########################################################
        
    c_perm,partition,partmap = splitApprox(corpus,nparts)

    @assert c_perm == corpus[partmap] "partition failed"    
    @assert size(topic_corpus) == (k,nlex)

    if scalesc == "column"
        println("sc column scaling selected, divide the counts by the total counts for each cell type, add betapsueudo($betapseudo), multiply by scfactor($scfactor)") 
        betalocal = topic_corpus ./ sum(topic_corpus,dims=2)
        betalocal = (betalocal .+ betapseudo) .* scfactor
    elseif scalesc == "ident"
        println("sc ident selected, add betapseudo($betapseudo),betaeps($betaeps)") 
        betalocal = topic_corpus .+ betapseudo .+ betaeps
    elseif scalesc == "logpseudo"
        if betapseudo < 1
            println("sc logpseudo scaling selected but betapseudo < 1 (negative values will be created), updating betapseudo = 1")
            betapseudo = 1.0
        end
        println("sc logpseudo scaling selected, add betapseudo($betapseudo), log transform, multiply by scfactor($scfactor), add betaeps($betaeps)") 
        betalocal = log.(topic_corpus .+ betapseudo) .* scfactor .+ betaeps
    elseif scalesc == "symmetric" # debug full lda with dirichlet prior on beta
        betalocal = zeros(size(topic_corpus)) .+ betapseudo .+ betaeps
    end        

    @assert size(betalocal) == (k,nlex)

    @assert all(betalocal .>= 0) "beta: negative values detected, make sure that betapseudo is large enough to cover log transform"
    @assert all(sum(betalocal,dims=1) .> 0) "beta: some genes have zero expression across all cell types"

    ###########################################################
    # # run chains
    ###########################################################
    # @everywhere Random.seed!(myid())

    meanrhat_high = [true]
    niter_inc = [0]
    # thetahat=[nothing]
    thetahat = Vector{Matrix{Float64}}(undef,1)
    meanrhat=[0.0]

    # get thinning inds
    if burn > 0
        nstart = Int64(round(niter*burn))        
    else
        nstart = 1        
    end

    indthin = [mod(i,thinning)==0 ? i : nothing for i in 1:niter]
    indthin = indthin[indthin.!=nothing]
    indthin = indthin[indthin .> nstart]
    Thetas = SharedArray(zeros(Float64,length(indthin),k,nchains,length(partmap)))
    
    try_niter!(
        alpha,betalocal,blocksize,c_perm,initflavor,k,meanrhat,meanrhat_high,niter,
        nlex,nparts,outputdir,partition,partmap,philatent,thetahat,
        burn,nchains,thinning,topic_corpus,topicnames,Thetas,indthin;
        rmchains=rmchains,verbose=verbose)

    # back up thetahat to a separate file (useful to debug EM/E-step)
    # P_fn = joinpath(outputdir,"P.csv")
    # CSV.write(P_fn, DataFrame(thetahat[1],:auto))
    
    Thetachains = []
    for i in 1:length(partmap)
        push!(Thetachains,Chains(Thetas[:,:,:,i],topicnames))
    end

    thetaresult = getThetaHat(Thetas,length(partmap),k)[sortperm(partmap),:]
    meanrhatresult = mean([mean(summarystats(Thetachains[i])[:,:rhat]) for i in 1:length(Thetachains)])
    resultarr = [thetaresult,meanrhatresult,ldagenes,scrgenes,ldacells,scrcells,celltypes]

    return resultarr
end

# increase niter until mean r-hat is below threshold, otherwise this will cause 
# imputation iterations to halt
function try_niter!(
    alpha,betalocal,blocksize,c_perm,initflavor,k,meanrhat,meanrhat_high,niter,
    nlex,nparts,outputdir,partition,partmap,philatent,thetahat,
    burn,nchains,thinning,topic_corpus,topicnames,Thetas,indthin;
    rhatthresh=1.1,rmchains=rmchains,verbose=verbose)

    while meanrhat_high[1]
        for i in 1:nchains
            println("starting MCMC chain $i with n = $niter")

            # @everywhere Random.seed!($i*myid())
            runChainSharedSV(c_perm,partition,partmap,betalocal,fill(alpha,k),
                k,nlex,philatent,
                Int64,Float64,nparts,blocksize,niter,i,
                outputdir,initflavor,Thetas,indthin,verbose=verbose)
        end            

        
        println("analysis: begin convergence diagnostics")

        if meanrhat[1] < rhatthresh
            meanrhat_high[1] = false
            println("\n r-hat < $rhatthresh, deconvolution succeeds \n")
        else
            niter = niter*2
            println("\n r-hat above threshold=$rhatthresh, MCMC restart with n = ",niter,"\n")
        end
    end
end