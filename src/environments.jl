struct Environments
    I::MPS
    H::MPS
    InProgress::Matrix{ITensor}
end

function fitPEPSMPO(A::fPEPS, prev_mps::Vector{<:ITensor}, ops::Vector{ITensor}, col::Int, chi::Int)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    # need to figure out index structure of guess!
    up_inds = [Index(chi, "Link,c$col,r$row,u") for row in 1:Ny-1]
    # guess should have the indices of A going left/right that prev_mps does not have
    A_prev_common  = [commonind(A[row, col], prev_mps[row]) for row in 1:Ny]
    hori_A_inds    = [inds(A[row, col], "Link, r") for row in 1:Ny]
    hori_prev_inds = [inds(prev_mps[row], "Link, r") for row in 1:Ny]
    double_hori_A  = [IndexSet(hori_A_inds[row], prime(hori_A_inds[row])) for row in 1:Ny]
    A_prev_unique  = [setdiff(double_hori_A[row], hori_prev_inds[row]) for row in 1:Ny]
    hori_cmbs      = Vector{ITensor}(undef, Ny)
    hori_cis       = Vector{Index}(undef, Ny)
    for row in 1:Ny
        cmb, ci = combiner(A_prev_unique[row]..., tags="r$row,CMB,Site")
        hori_cmbs[row] = cmb 
        hori_cis[row]  = ci
    end
    guess = randomMPS(hori_cis, chi)
    for row in 1:Ny
        guess[row] *= hori_cmbs[row]
    end
    #=if col != 1 && col != Nx
        guess = deepcopy(prev_mps)
        for row in 1:Ny
            replaceinds!(guess, hori_prev_inds[row], A_prev_unique[row]) 
        end
    end=#
    #=guess          = Vector{ITensor}(undef, Ny)
    guess[1]       = randomITensor(IndexSet(A_prev_unique[1]..., up_inds[1] ) ) 
    guess[2:Ny-1]  = [randomITensor(IndexSet(A_prev_unique[row]..., up_inds[row], up_inds[row-1] )) for row in 2:Ny-1]
    guess[Ny]      = randomITensor(IndexSet(A_prev_unique[Ny]..., up_inds[Ny- 1] ) ) 
    for row in 1:Ny
        normalize!(guess[row])
        @show guess[row]
    end
    if is_cu
        guess = cuITensor.(guess)
    end=#
    norm = is_cu ? cuITensor(1.0) : ITensor(1.0)
    for row in 1:Ny
        norm *= guess[row] * dag(prime(guess[row], "Link,u"))
    end
    for sweep in 1:1
        #guess_mps = MPS(Ny, guess, 0, Ny+1)
        #orthogonalize!(guess_mps, 1; mindim=chi)
        #guess = store(guess_mps)
        order = isodd(sweep) ? (1:Ny) : reverse(1:Ny)
        for row in order
            # construct environment for the row, have to do this every time
            Env_above = is_cu ? cuITensor(1.0) : ITensor(1.0)
            Env_below = is_cu ? cuITensor(1.0) : ITensor(1.0)
            for env_row in 1:row-1
                Env_below *= prev_mps[env_row]
                Env_below *= A[env_row, col]
                Env_below *= ops[env_row]
                Env_below *= dag(prime(A[env_row, col]))
                Env_below *= guess[env_row]
            end
            for env_row in reverse(row+1:Ny)
                Env_above *= prev_mps[env_row]
                Env_above *= A[env_row, col]
                Env_above *= ops[env_row]
                Env_above *= dag(prime(A[env_row, col]))
                Env_above *= guess[env_row]
            end
            Env  = Env_below
            Env *= prev_mps[row]
            Env *= A[row, col]
            Env *= ops[row]
            Env *= dag(prime(A[row, col]))
            Env *= Env_above
            # update guess at row
            guess[row] = Env
        end
    end
    return guess
end

function buildEdgeEnvironment(A::fPEPS, H, left_H_terms, side::Symbol, col::Int; kwargs...)::Environments
    Ny, Nx          = size(A)
    is_cu::Bool     = is_gpu(A)
    chi::Int        = get(kwargs, :env_maxdim, 1)
    dummy = Vector{ITensor}(undef, Ny)
    for row in 1:Ny
        dummy[row] = is_cu ? cuITensor(1.0) : ITensor(1.0) 
    end
    dummy_mps       = MPS(Ny, dummy, 0, Ny+1)
    I_mps           = buildNewI(A, dummy_mps, col, chi)
    field_H_terms   = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms    = getDirectional(vcat(H[:, col]...), Vertical)
    vHs             = [buildNewVerticals(A, vert_H_terms[vert_op], dummy_mps, col, chi) for vert_op in 1:length(vert_H_terms)]
    @debug "Built new Vs"
    fHs             = [buildNewFields(A, field_H_terms[field_op], dummy_mps, col, chi) for field_op in 1:length(field_H_terms)]
    Hs              = vcat(vHs, fHs)
    maxdim::Int     = get(kwargs, :maxdim, 1)
    cutoff::Float64 = get(kwargs, :cutoff, 0.0)
    hori_inds = Vector{IndexSet}(undef, Ny)
    hori_cmbs = Vector{ITensor}(undef, Ny)
    up_inds   = [Index(chi, "Link,c$col,r$row,u") for row in 1:Ny-1]
    for row in 1:Ny
        hori_inds[row] = intersect(map(x->inds(x[row], "Link,r"), Hs))[1]
        cmb, ci = combiner(hori_inds[row], tags="r$row,CMB,Site")
        hori_cmbs[row] = cmb 
        for ii in 1:length(Hs)
            Hs[ii][row] *= hori_cmbs[row]
        end
    end
    for ii in 1:length(Hs)
        for row in 1:Ny-1
            bad_ind = commonind(Hs[ii][row], Hs[ii][row+1])
            replaceind!(Hs[ii][row], bad_ind, up_inds[row])
            replaceind!(Hs[ii][row+1], bad_ind, up_inds[row])
        end
        @show Hs[ii][1]
        orthogonalize!(Hs[ii], 1)
    end
    H_overall       = sum(Hs; cutoff=cutoff, maxdim=chi)
    for row in 1:Ny
        H_overall[row] *= hori_cmbs[row]
    end
    @show H_overall[1]
    flush(stdout)
    @debug "Summed Hs, maxdim=$maxdim"
    side_H          = side == :left ? H[:, col] : H[:, col - 1]
    side_H_terms    = getDirectional(vcat(side_H...), Horizontal)
    @debug "Trying to alloc $(Ny*length(side_H_terms))"
    in_progress     = Matrix{ITensor}(undef, Ny, length(side_H_terms))
    @inbounds for side_term in 1:length(side_H_terms)
        @debug "Generating edge bonds for term $side_term"
        in_progress[1:Ny, side_term] = generateEdgeDanglingBonds(A, side_H_terms[side_term], side, col, chi)
    end
    @debug "Generated edge bonds"
    return Environments(I_mps, H_overall, in_progress)
end

function buildNextEnvironment(A::fPEPS, prev_Env::Environments, H,
                              side::Symbol,
                              col::Int;
                              kwargs...)
    Ny, Nx        = size(A)
    chi::Int       = get(kwargs, :env_maxdim, 1)
    println("A row 1:")
    #@show A[1, col]
    @timeit "build new_I and new_H" begin
        new_I = buildNewI(A, prev_Env.I, col, chi)
        new_H = buildNewI(A, prev_Env.H, col, chi)
        println("result in new_H 1:")
        #@show new_H[1]
    end
    @debug "Built new I and H"
    field_H_terms = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]...), Vertical)
    hori_H_terms  = getDirectional(vcat(H[:, col]...), Horizontal)
    side_H        = side == :left ? H[:, col] : H[:, col - 1]
    side_H_terms  = getDirectional(vcat(side_H...), Horizontal)
    H_term_count  = 1 + length(field_H_terms) + length(vert_H_terms)
    H_term_count += (side == :left ? length(side_H_terms) : length(hori_H_terms))
    new_H_mps     = Vector{MPS}(undef, H_term_count)
    new_H_mps[1]  = deepcopy(new_H)
    @timeit "build new verts" begin
        vHs = [buildNewVerticals(A, vert_H_terms[vert_op], prev_Env.I, col, chi) for vert_op in 1:length(vert_H_terms)]
    end
    @timeit "build new fields" begin
        fHs = [buildNewFields(A, field_H_terms[field_op], prev_Env.I, col, chi) for field_op in 1:length(field_H_terms)]
    end
    # yuck improve this
    @timeit "build new H array" begin
        for (vv, vH) in enumerate(vHs)
            println("result in buildNewVerticals 1:")
            #@show vHs[vv][1]
            new_H_mps[1+vv] = vHs[vv]
        end
        for (ff, fH) in enumerate(fHs)
            println("result in buildNewFields 1:")
            #@show fHs[ff][1]
            new_H_mps[1+length(vHs)+ff] = fHs[ff]
        end
    end
    connect_H    = side == :left ? side_H_terms : hori_H_terms
    @timeit "connect dangling bonds" begin
        for (cc, cH) in enumerate(connect_H)
            new_H = connectDanglingBonds(A, cH, prev_Env.InProgress[:, cc], side, -1, col; kwargs...)
            println("result in connect dangling bonds 1:")
            #@show new_H[1]
            new_H_mps[length(vert_H_terms) + length(field_H_terms) + 1 + cc] = MPS(Ny, new_H, 0, Ny+1)
        end
    end
    @debug "Connected dangling bonds"

    maxdim::Int     = get(kwargs, :maxdim, 1)
    cutoff::Float64 = get(kwargs, :cutoff, 0.0)
    @timeit "sum H mps" begin
        hori_inds = Vector{IndexSet}(undef, Ny)
        hori_cmbs = Vector{ITensor}(undef, Ny)
        up_inds   = [Index(chi, "Link,c$col,r$row,u") for row in 1:Ny-1]
        for row in 1:Ny
            hori_inds[row] = intersect(map(x->inds(x[row], "Link,r"), new_H_mps))[1]
            cmb, ci = combiner(hori_inds[row], tags="r$row,CMB,Site")
            hori_cmbs[row] = cmb 
            for ii in 1:length(new_H_mps)
                new_H_mps[ii][row] *= hori_cmbs[row]
            end
        end
        for ii in 1:length(new_H_mps)
            for row in 1:Ny-1
                bad_ind = commonind(new_H_mps[ii][row], new_H_mps[ii][row+1])
                replaceind!(new_H_mps[ii][row], bad_ind, up_inds[row])
                replaceind!(new_H_mps[ii][row+1], bad_ind, up_inds[row])
            end
        end
        H_overall    = sum(new_H_mps; cutoff=cutoff, maxdim=chi)
        for row in 1:Ny
            H_overall[row] *= hori_cmbs[row]
        end
    end
    @debug "Summed Hs"
    gen_H_terms  = side == :left ? hori_H_terms : side_H_terms
    in_progress  = Matrix{ITensor}(undef, Ny, length(side_H_terms))
    @timeit "gen dangling bonds" begin
        @inbounds for side_term in 1:length(gen_H_terms)
            in_progress[1:Ny, side_term] = generateNextDanglingBonds(A, gen_H_terms[side_term], prev_Env.I, side, col; kwargs...)
        end
    end
    @debug "Generated next dangling bonds"
    return Environments(new_I, H_overall, in_progress)
end

function buildNewVerticals(A::fPEPS, H, prev_I::MPS, col::Int, chi::Int)::MPS
    Ny, Nx              = size(A)
    is_cu               = is_gpu(A)
    col_site_inds       = [firstind(A[row, col], "Site") for row in 1:Ny]
    ops                 = ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in col_site_inds]
    vertical_row_a      = H.sites[1][1]
    vertical_row_b      = H.sites[2][1]
    ops[vertical_row_a] = replaceind!(copy(H.ops[1]), H.site_ind, col_site_inds[vertical_row_a])
    ops[vertical_row_a] = replaceind!(ops[vertical_row_a], H.site_ind', col_site_inds[vertical_row_a]')
    ops[vertical_row_b] = replaceind!(copy(H.ops[2]), H.site_ind, col_site_inds[vertical_row_b])
    ops[vertical_row_b] = replaceind!(ops[vertical_row_b], H.site_ind', col_site_inds[vertical_row_b]')
    return fitPEPSMPO(A, store(prev_I), ops, col, chi)
end

function buildNewFields(A::fPEPS, H, prev_I::MPS, col::Int, chi::Int)::MPS
    Ny, Nx         = size(A)
    is_cu          = is_gpu(A)
    col_site_inds  = [firstind(A[row, col], "Site") for row in 1:Ny]
    ops            = ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in col_site_inds]
    field_row      = H.sites[1][1]
    ops[field_row] = replaceind!(copy(H.ops[1]), H.site_ind, col_site_inds[field_row])
    ops[field_row] = replaceind!(ops[field_row], H.site_ind', col_site_inds[field_row]')
    return fitPEPSMPO(A, store(prev_I), ops, col, chi)
end

function buildNewI(A::fPEPS, prev_I::MPS, col::Int, chi::Int)::MPS
    Ny, Nx         = size(A)
    is_cu          = is_gpu(A)
    col_site_inds  = [firstind(A[row, col], "Site") for row in 1:Ny]
    ops            = ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in col_site_inds] 
    return fitPEPSMPO(A, store(prev_I), ops, col, chi)
end

function generateEdgeDanglingBonds(A::fPEPS, H, side::Symbol, col::Int, chi::Int)::Vector{ITensor}
    Ny, Nx         = size(A)
    is_cu          = is_gpu(A)
    dummy          = [is_cu ? cuITensor(1.0) : ITensor(1.0) for row in 1:Ny]
    op_row         = side == :left ? H.sites[1][1] : H.sites[2][1]
    H_op           = side == :left ? H.ops[1]      : H.ops[2]
    col_site_inds  = [firstind(A[row, col], "Site") for row in 1:Ny]
    ops            = ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in col_site_inds] 
    ops[op_row]    = replaceind!(copy(H_op), H.site_ind, col_site_inds[op_row]) 
    ops[op_row]    = replaceind!(ops[op_row], H.site_ind', col_site_inds[op_row]') 
    return store(fitPEPSMPO(A, dummy, ops, col, chi))
end

function generateNextDanglingBonds(A::fPEPS, 
                                   H, 
                                   Ident::MPS, 
                                   side::Symbol, 
                                   col::Int; 
                                   kwargs...)::Vector{ITensor}
    Ny, Nx          = size(A)
    is_cu           = is_gpu(A)
    chi::Int         = get(kwargs, :env_maxdim, 1)
    op_row          = side == :left ? H.sites[1][1] : H.sites[2][1]
    H_op            = side == :left ? H.ops[1]      : H.ops[2]
    col_site_inds   = [firstind(A[row, col], "Site") for row in 1:Ny]
    ops             = ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in col_site_inds] 
    ops[op_row]     = replaceind!(copy(H_op), H.site_ind, col_site_inds[op_row]) 
    ops[op_row]     = replaceind!(ops[op_row], H.site_ind', col_site_inds[op_row]')
    return fitPEPSMPO(A, store(Ident), ops, col, chi)
end

function connectDanglingBonds(A::fPEPS,
                              oldH,
                              in_progress::Vector{ITensor},
                              side::Symbol,
                              work_row::Int, 
                              col::Int;
                              kwargs...)::Vector{ITensor}
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A)
    chi::Int = get(kwargs, :env_maxdim, 1)
    op_row_a = oldH.sites[1][1]
    op_row_b = oldH.sites[2][1]
    op       = side == :left ? oldH.ops[2] : oldH.ops[1]
    application_row = side == :left ? op_row_b : op_row_a
    col_site_inds  = [firstind(A[row, col], "Site") for row in 1:Ny]
    ops            = ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in col_site_inds] 
    ops[application_row] = replaceind!(copy(op), oldH.site_ind, col_site_inds[application_row])
    ops[application_row] = replaceind!(ops[application_row], oldH.site_ind', col_site_inds[application_row]')
    in_prog_mps    = MPS(Ny, in_progress, 0, Ny + 1)
    if work_row == -1
        return store(fitPEPSMPO(A, in_progress, ops, col, chi))
    else
        this_IP[work_row] = ops[work_row]
        completed_H = MPO(Ny, this_IP, 0, Ny+1)
        @inbounds for row in 1:Ny-1
            ci = linkIndex(completed_H, row)
            ni = Index(ITensors.dim(ci), "u,Link,c$col,r$row")
            replaceIndex!(completed_H[row],   ci, ni)
            replaceIndex!(completed_H[row+1], ci, ni)
        end
        return store(completed_H) .* store(in_prog_mps)
    end
end

function buildLs(A::fPEPS, H; kwargs...)
    Ny, Nx         = size(A)
    Ls             = Vector{Environments}(undef, Nx)
    start_col::Int = get(kwargs, :start_col, 1)
    if start_col == 1
        left_H_terms = getDirectional(H[1], Horizontal)
        @debug "Building left col $start_col"
        Ls[1] = buildEdgeEnvironment(A, H, left_H_terms, :left, 1; kwargs...)
    end
    loop_col = start_col == 1 ? 2 : start_col
    @inbounds for col in loop_col:(Nx-1)
        @debug "Building left col $col"
        Ls[col] = buildNextEnvironment(A, Ls[col-1], H, :left, col; kwargs...)
    end
    return Ls
end

function buildRs(A::fPEPS, H; kwargs...)
    Ny, Nx         = size(A)
    start_col::Int = get(kwargs, :start_col, Nx)
    Rs             = Vector{Environments}(undef, Nx)
    if start_col == Nx
        right_H_terms = getDirectional(H[Nx-1], Horizontal)
        Rs[Nx]        = buildEdgeEnvironment(A, H, right_H_terms, :right, Nx; kwargs...)
    end
    loop_col = start_col == Nx ? Nx - 1 : start_col
    @inbounds for col in reverse(2:loop_col)
        @debug "Building right col $col"
        Rs[col] = buildNextEnvironment(A, Rs[col+1], H, :right, col; kwargs...)
    end
    return Rs
end
