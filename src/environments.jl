struct Environments
    I::MPS
    H::MPS
    InProgress::Matrix{ITensor}
    DiagInProgress::Matrix{ITensor}
end

function Base.copy(E::Environments)
    return Environments(copy(E.I), copy(E.H), copy(E.InProgress))
end

function getGuessInds(A::fPEPS, prev_mps, col::Int)
    # guess should have the indices of A going left/right that prev_mps does not have
    Ny, Nx = size(A)
    A_prev_common  = [commonind(A[row, col], prev_mps[row]) for row in 1:Ny]
    hori_A_inds    = [inds(A[row, col], "Link, r") for row in 1:Ny]
    hori_prev_inds = [inds(prev_mps[row], "Link, r") for row in 1:Ny]
    double_hori_A  = [IndexSet(hori_A_inds[row]..., prime(hori_A_inds[row])...) for row in 1:Ny]
    A_prev_unique  = [setdiff(double_hori_A[row], hori_prev_inds[row]) for row in 1:Ny]
    return A_prev_unique
end

function initGuessRandomMPS(A::fPEPS, prev_mps, col::Int, chi::Int)
    @timeit "randomMPS" begin
        Ny, Nx = size(A)
        A_prev_unique  = getGuessInds(A, prev_mps, col)
        hori_cmbs      = Vector{ITensor}(undef, Ny)
        hori_cis       = Vector{Index}(undef, Ny)
        @inbounds for row in 1:Ny
            cmb            = combiner(A_prev_unique[row]..., tags="r$row,CMB,Site")
            hori_cmbs[row] = cmb 
            hori_cis[row]  = combinedind(cmb) 
        end
        guess = randomMPS(hori_cis, chi)
        @inbounds for row in 1:Ny
            guess[row] *= hori_cmbs[row]
        end
    end
    return guess
end

function initGuessRandomITensor(A::fPEPS, prev_mps, col::Int, chi::Int)
    @timeit "randomITensor" begin
        Ny, Nx = size(A)
        A_prev_unique  = getGuessInds(A, prev_mps, col)
        guess = MPS(Ny)
        up_inds        = [Index(chi, "Link,u,c$col,r$row") for row in 1:Ny-1]
        @inbounds for row in 1:Ny
            row_inds = nothing
            if row == 1
                row_inds = IndexSet(A_prev_unique[row]..., up_inds[row])
            elseif 1 < row < Ny
                row_inds = IndexSet(A_prev_unique[row]..., up_inds[row-1], up_inds[row])
            else
                row_inds = IndexSet(A_prev_unique[row]..., up_inds[row-1])
            end
            #guess[row] = is_cu ? randomCuITensor(row_inds) : randomITensor(row_inds) 
            guess[row] = randomITensor(row_inds) 
        end
    end
    return guess
end

function constructEnvsAbove(A::fPEPS, guess::MPS, prev_mps::Vector{<:ITensor}, ops::Vector{ITensor}, col::Int, sweep::Int)
    is_cu  = is_gpu(A)
    Ny, Nx = size(A)
    order  = isodd(sweep) ? (1:Ny) : reverse(1:Ny)
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0)
    Envs_above = Vector{ITensor}(undef, Ny+1)
    Envs_above[end]  = is_cu ? cuITensor(1.0) : ITensor(1.0)
    @inbounds for (ii, env_row) in enumerate(reverse(order))
        tmp  = A[env_row, col]
        tmp *= ops[env_row]
        tmp *= prev_mps[env_row]
        next_row   = isodd(sweep) ? env_row + 1 : env_row - 1
        Env_above  = next_row > 0 ? copy(Envs_above[next_row]) : dummy 
        Env_above *= tmp
        Env_above *= dag(prime(A[env_row, col]))
        Env_above *= guess[env_row]
        Envs_above[env_row] = Env_above 
    end
    return Envs_above
end

function fitPEPSMPOold(A::fPEPS, prev_mps::Vector{<:ITensor}, ops::Vector{ITensor}, col::Int, chi::Int)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    @timeit "make guess" begin
        guess::MPS = initGuessRandomMPS(A, prev_mps, col, chi)
        if is_cu
            guess = cuMPS(guess)
        end
    end
    #@show inner(guess, guess)
    dummy = is_cu ? cuITensor(1.0) : ITensor(1.0)
    for sweep in 1:2
        order      = isodd(sweep) ? (1:Ny) : reverse(1:Ny)
        Envs_above = constructEnvsAbove(A, guess, prev_mps, ops, col, sweep)
        Env_below  = is_cu ? cuITensor(1.0) : ITensor(1.0)
        @inbounds for row in order
            # construct environment for the row, have to do this every time
            @timeit "build row Env" begin
                Env  = copy(Env_below)
                tmp  = A[row, col]
                tmp *= ops[row]
                tmp *= prev_mps[row]
                Env *= dag(prime(A[row, col]))
                Env *= tmp
                Env_below  = copy(Env)
                next_row = isodd(sweep) ? row+1 : row -1
                Env *= next_row > 0 ? copy(Envs_above[next_row]) : dummy 
                #guess[row] = copy(Env)
                #Env_below  *= guess[row] 
            end
            # get l-inds for svd
            @timeit "do SVD" begin
                if isodd(sweep) && row < Ny
                    svd_linds     = setdiff(inds(guess[row]),IndexSet(commonind(guess[row], guess[row+1])))
                    tsvd          = svd(Env, svd_linds, mindim=chi, maxdim=chi)
                    # update guess at row
                    guess[row]    = tsvd.U
                    guess[row+1] *= tsvd.S*tsvd.V
                elseif !isodd(sweep) && row > 1
                    svd_linds     = setdiff(inds(guess[row]),IndexSet(commonind(guess[row], guess[row-1])))
                    tsvd          = svd(Env, svd_linds, mindim=chi, maxdim=chi)
                    # update guess at row
                    guess[row]    = tsvd.U
                    guess[row-1] *= tsvd.S*tsvd.V
                else
                    # update guess at row
                    guess[row]   = copy(Env) 
                end
                Env_below    *= guess[row]
            end
        end
    end
    return guess
end

# now with two-site
function fitPEPSMPO(A::fPEPS, prev_mps::Vector{<:ITensor}, ops::Vector{ITensor}, col::Int, chi::Int)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    @timeit "make guess" begin
        guess::MPS = initGuessRandomMPS(A, prev_mps, col, chi)
        if is_cu
            guess = cuMPS(guess)
        end
    end
    for sweep in 1:1
        @timeit "make Envs_above" begin
            Envs_above = constructEnvsAbove(A, guess, prev_mps, ops, col, sweep)
        end
        order     = isodd(sweep) ? (1:Ny-1) : reverse(1:Ny-1)
        Env_below = is_cu ? cuITensor(1.0) : ITensor(1.0)
        @inbounds for row in order
            # construct environment for the row, have to do this every time
            @timeit "build row Env" begin
                Env  = copy(Env_below)
                for row_ in (row, row+1)
                    @timeit "build tmp" begin
                        tmp = A[row_, col]
                        tmp *= ops[row_]
                        tmp *= prev_mps[row_]
                    end
                    @timeit "tmp into Env" begin
                        Env *= tmp
                    end
                    Env *= dag(prime(A[row_, col]))
                    if row_ == row
                        Env_below = copy(Env)
                    end
                end
                Env *= Envs_above[row+2]
            end
            # get l-inds for svd
            @timeit "do SVD" begin
                svd_linds    = setdiff(inds(guess[row]),IndexSet(commonind(guess[row], guess[row+1])))
                tsvd         = svd(Env, svd_linds, mindim=chi, maxdim=chi)
                # update guess at row
                guess[row]   = tsvd.U
                guess[row+1] = tsvd.S*tsvd.V
                Env_below   *= guess[row]
            end
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
        dummy[row]  = is_cu ? cuITensor(1.0) : ITensor(1.0) 
    end
    dummy_mps       = MPS(dummy, 0, Ny+1)
    I_mps           = buildNewI(A, dummy_mps, col, chi)
    field_H_terms   = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms    = getDirectional(vcat(H[:, col]...), Vertical)
    vHs             = [buildNewVerticals(A, vert_H_terms[vert_op], dummy_mps, col, chi) for vert_op in 1:length(vert_H_terms)]
    fHs             = [buildNewFields(A, field_H_terms[field_op], dummy_mps, col, chi) for field_op in 1:length(field_H_terms)]
    Hs              = vcat(vHs, fHs)
    maxdim::Int     = get(kwargs, :maxdim, 1)
    cutoff::Float64 = get(kwargs, :cutoff, 0.0)
    hori_inds = Vector{IndexSet}(undef, Ny)
    hori_cmbs = Vector{ITensor}(undef, Ny)
    up_inds   = [Index(chi, "Link,c$col,r$row,u") for row in 1:Ny-1]
    @inbounds for row in 1:Ny
        hori_inds[row] = intersect(map(x->inds(x[row], "Link,r"), Hs))[1]
        hori_cmbs[row] = combiner(hori_inds[row], tags="r$row,CMB,Site")
        for ii in 1:length(Hs)
            Hs[ii][row] *= hori_cmbs[row]
        end
    end
    H_overall       = sum(Hs; cutoff=cutoff, maxdim=chi)
    @inbounds for row in 1:Ny
        H_overall[row] *= hori_cmbs[row]
    end
    side_H          = side == :left ? H[:, col] : H[:, col - 1]
    side_H_terms    = getDirectional(vcat(side_H...), Horizontal)
    in_progress     = Matrix{ITensor}(undef, Ny, length(side_H_terms))
    @inbounds for side_term in 1:length(side_H_terms)
        in_progress[1:Ny, side_term] = generateEdgeDanglingBonds(A, side_H_terms[side_term], side, col, chi)
    end
    diag_H_terms     = getDirectional(vcat(side_H...), Diag)
    diag_in_progress = Matrix{ITensor}(undef, Ny, length(diag_H_terms))
    @inbounds for diag_term in 1:length(diag_H_terms)
        diag_in_progress[1:Ny, diag_term] = generateEdgeDanglingBonds(A, diag_H_terms[diag_term], side, col, chi)
    end
    return Environments(I_mps, H_overall, in_progress, diag_in_progress)
end

function buildNextEnvironment(A::fPEPS, prev_Env::Environments, H,
                              side::Symbol,
                              col::Int;
                              kwargs...)
    Ny, Nx        = size(A)
    chi::Int      = get(kwargs, :env_maxdim, 1)
    @timeit "build new_I and new_H" begin
        new_I = buildNewI(A, prev_Env.I, col, chi)
        new_H = buildNewI(A, prev_Env.H, col, chi)
    end
    field_H_terms = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]...), Vertical)
    hori_H_terms  = getDirectional(vcat(H[:, col]...), Horizontal)
    diag_H_terms  = getDirectional(vcat(H[:, col]...), Diag)
    side_H        = side == :left ? H[:, col] : H[:, col - 1]
    side_H_terms  = getDirectional(vcat(side_H...), Horizontal)
    diag_side_H_terms = getDirectional(vcat(side_H...), Diag)
    H_term_count  = 1 + length(field_H_terms) + length(vert_H_terms)
    H_term_count += (side == :left ? length(side_H_terms) : length(hori_H_terms))
    H_term_count += (side == :left ? length(diag_side_H_terms) : length(diag_H_terms))
    new_H_mps     = Vector{MPS}(undef, H_term_count)
    new_H_mps[1]  = deepcopy(new_H)
    @timeit "build new verts" begin
        new_H_mps[2:1+length(vert_H_terms)] = [buildNewVerticals(A, vert_H_terms[vert_op], prev_Env.I, col, chi) for vert_op in 1:length(vert_H_terms)]
    end
    @timeit "build new fields" begin
        new_H_mps[2+length(vert_H_terms):1+length(vert_H_terms)+length(field_H_terms)]  = [buildNewFields(A, field_H_terms[field_op], prev_Env.I, col, chi) for field_op in 1:length(field_H_terms)]
    end
    connect_H    = side == :left ? side_H_terms : hori_H_terms
    @timeit "connect dangling bonds" begin
        @inbounds for (cc, cH) in enumerate(connect_H)
            new_H = connectDanglingBonds(A, cH, prev_Env.InProgress[:, cc], side, col; kwargs...)
            new_H_mps[length(vert_H_terms) + length(field_H_terms) + 1 + cc] = MPS(new_H, 0, Ny+1)
        end
    end
    diag_connect_H = side == :left ? diag_side_H_terms : diag_H_terms
    @timeit "connect dangling bonds" begin
        @inbounds for (cc, cH) in enumerate(diag_connect_H)
            new_H = connectDanglingBonds(A, cH, prev_Env.DiagInProgress[:, cc], side, col; kwargs...)
            new_H_mps[length(vert_H_terms) + length(field_H_terms) + length(connect_H) + 1 + cc] = MPS(new_H, 0, Ny+1)
        end
    end
    maxdim::Int     = get(kwargs, :maxdim, 1)
    cutoff::Float64 = get(kwargs, :cutoff, 0.0)
    @timeit "sum H mps" begin
        hori_inds = Vector{IndexSet}(undef, Ny)
        hori_cmbs = Vector{ITensor}(undef, Ny)
        up_inds   = [Index(chi, "Link,c$col,r$row,u") for row in 1:Ny-1]
        for row in 1:Ny
            hori_inds[row] = intersect(map(x->inds(x[row], "Link,r"), new_H_mps))[1]
            hori_cmbs[row] = combiner(hori_inds[row], tags="r$row,CMB,Site")
            for ii in 1:length(new_H_mps)
                new_H_mps[ii][row] *= hori_cmbs[row]
            end
        end
        H_overall    = sum(new_H_mps; cutoff=cutoff, maxdim=chi)
        for row in 1:Ny
            H_overall[row] *= hori_cmbs[row]
        end
    end
    gen_H_terms  = side == :left ? hori_H_terms : side_H_terms
    in_progress  = Matrix{ITensor}(undef, Ny, length(gen_H_terms))
    @timeit "gen dangling bonds" begin
        @inbounds for side_term in 1:length(gen_H_terms)
            in_progress[1:Ny, side_term] = generateNextDanglingBonds(A, gen_H_terms[side_term], prev_Env.I, side, col; kwargs...)
        end
    end
    diag_gen_H_terms  = side == :left ? diag_H_terms : diag_side_H_terms
    diag_in_progress  = Matrix{ITensor}(undef, Ny, length(diag_gen_H_terms))
    @timeit "diag gen dangling bonds" begin
        @inbounds for side_term in 1:length(diag_gen_H_terms)
            diag_in_progress[1:Ny, side_term] = generateNextDanglingBonds(A, diag_gen_H_terms[side_term], prev_Env.I, side, col; kwargs...)
        end
    end
    return Environments(new_I, H_overall, in_progress, diag_in_progress)
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
    #return fitPEPSMPOold(A, data(prev_I), ops, col, chi)
    return fitPEPSMPO(A, data(prev_I), ops, col, chi)
end

function buildNewFields(A::fPEPS, H, prev_I::MPS, col::Int, chi::Int)::MPS
    Ny, Nx         = size(A)
    is_cu          = is_gpu(A)
    col_site_inds  = [firstind(A[row, col], "Site") for row in 1:Ny]
    ops            = ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in col_site_inds]
    field_row      = H.sites[1][1]
    ops[field_row] = replaceind!(copy(H.ops[1]), H.site_ind, col_site_inds[field_row])
    ops[field_row] = replaceind!(ops[field_row], H.site_ind', col_site_inds[field_row]')
    #return fitPEPSMPOold(A, data(prev_I), ops, col, chi)
    return fitPEPSMPO(A, data(prev_I), ops, col, chi)
end

function buildNewI(A::fPEPS, prev_I::MPS, col::Int, chi::Int)::MPS
    Ny, Nx         = size(A)
    is_cu          = is_gpu(A)
    col_site_inds  = [firstind(A[row, col], "Site") for row in 1:Ny]
    ops            = ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in col_site_inds] 
    #return fitPEPSMPOold(A, data(prev_I), ops, col, chi)
    return fitPEPSMPO(A, data(prev_I), ops, col, chi)
end

function generateNextDanglingBonds(A::fPEPS,
                                   H,
                                   Ident::MPS,
                                   side::Symbol,
                                   col::Int;
                                   kwargs...)::Vector{ITensor}
    Ny, Nx          = size(A)
    is_cu           = is_gpu(A)
    chi::Int        = get(kwargs, :env_maxdim, 1)
    op_row          = side == :left ? H.sites[1][1] : H.sites[2][1]
    H_op            = side == :left ? H.ops[1]      : H.ops[2]
    col_site_inds   = [firstind(A[row, col], "Site") for row in 1:Ny]
    ops             = ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in col_site_inds] 
    ops[op_row]     = replaceind!(copy(H_op), H.site_ind, col_site_inds[op_row]) 
    ops[op_row]     = replaceind!(ops[op_row], H.site_ind', col_site_inds[op_row]')
    #return data(fitPEPSMPOold(A, data(Ident), ops, col, chi))
    return data(fitPEPSMPO(A, data(Ident), ops, col, chi))
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
    #return data(fitPEPSMPOold(A, dummy, ops, col, chi))
    return data(fitPEPSMPO(A, dummy, ops, col, chi))
end

function connectDanglingBonds(A::fPEPS,
                              oldH,
                              in_progress::Vector{ITensor},
                              side::Symbol,
                              col::Int;
                              kwargs...)::Vector{ITensor}
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A)
    chi::Int = get(kwargs, :env_maxdim, 1)
    op_row_a = oldH.sites[1][1]
    op_row_b = oldH.sites[2][1]
    op       = side == :left ? oldH.ops[2] : oldH.ops[1]
    application_row      = side == :left ? op_row_b : op_row_a
    col_site_inds        = [firstind(A[row, col], "Site") for row in 1:Ny]
    ops                  = ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in col_site_inds]
    ops[application_row] = replaceind!(copy(op), oldH.site_ind, col_site_inds[application_row])
    ops[application_row] = replaceind!(ops[application_row], oldH.site_ind', col_site_inds[application_row]')
    in_prog_mps          = MPS(in_progress, 0, Ny + 1)
    #return data(fitPEPSMPOold(A, in_progress, ops, col, chi))
    return data(fitPEPSMPO(A, in_progress, ops, col, chi))
end

function buildLs(A::fPEPS, H; kwargs...)
    Ny, Nx         = size(A)
    Ls             = Vector{Environments}(undef, Nx)
    start_col::Int = get(kwargs, :start_col, 1)
    if start_col == 1
        left_H_terms = getDirectional(vcat(H[:, 1]...), Horizontal)
        Ls[1]        = buildEdgeEnvironment(A, H, left_H_terms, :left, 1; kwargs...)
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
        right_H_terms = getDirectional(vcat(H[:, Nx-1]...), Horizontal)
        Rs[Nx]        = buildEdgeEnvironment(A, H, right_H_terms, :right, Nx; kwargs...)
    end
    loop_col = start_col == Nx ? Nx - 1 : start_col
    @inbounds for col in reverse(2:loop_col)
        @debug "Building right col $col"
        Rs[col] = buildNextEnvironment(A, Rs[col+1], H, :right, col; kwargs...)
    end
    return Rs
end
