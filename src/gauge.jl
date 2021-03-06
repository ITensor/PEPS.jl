using ITensors, ITensorsGPU

function my_polar( T::ITensor, inds...; kwargs... )
    U, S, V, spec, u, v = svd(T, inds; kwargs...)
    replaceind!(U, u, v)
    return U*V
end


function initQs( A::fPEPS, col::Int, next_col::Int; kwargs...)
    Ny, Nx = size(A)
    maxdim::Int = get(kwargs, :maxdim, 1)
    Q         = MPO(deepcopy(A[:, col]), 0, Ny+1)
    prev_col  = next_col > col ? col - 1 : col + 1
    A_r_inds  = [commonind(A[row, col], A[row, next_col]) for row in 1:Ny]
    QR_inds   = [Index(ITensors.dim(A_r_inds[row]), "Site,QR,c$col,r$row") for row in 1:Ny]
    A_up_inds = [commonind(A[row, col], A[row+1, col]) for row in 1:Ny-1]
    Q_up_inds = [Index(ITensors.dim(A_up_inds[row]), "Link,u,Qup$row") for row in 1:Ny-1]
    next_col_inds = [commonind(A[row, col], A[row, next_col]) for row in 1:Ny]
    prev_col_inds = 0 < prev_col < Nx ? [commonind(A[row, col], A[row, prev_col]) for row in 1:Ny] : Vector{Index}(undef, Ny)
    for row in 1:Ny
        row < Ny && replaceind!(Q[row], A_up_inds[row], Q_up_inds[row])
        row > 1  && replaceind!(Q[row], A_up_inds[row-1], Q_up_inds[row-1])
        replaceind!(Q[row], next_col_inds[row], QR_inds[row])
    end
    return Q, QR_inds, next_col_inds
end

function initQs( A::fPEPS, Q::MPO, col::Int, next_col::Int; kwargs...)
    Ny, Nx = size(A)
    maxdim::Int = get(kwargs, :maxdim, 1)
    Q         = MPO(data(Q), 0, Ny+1)
    prev_col  = next_col > col ? col - 1 : col + 1
    A_r_inds  = [commonind(A[row, col], A[row, next_col]) for row in 1:Ny]
    QR_inds   = [Index(ITensors.dim(A_r_inds[row]), "Site,QR,c$col,r$row") for row in 1:Ny]
    A_up_inds = [commonind(A[row, col], A[row+1, col]) for row in 1:Ny-1]
    Q_up_inds = [Index(ITensors.dim(A_up_inds[row]), "Link,u,Qup$row") for row in 1:Ny-1]
    next_col_inds = [commonind(A[row, col], A[row, next_col]) for row in 1:Ny]
    prev_col_inds = 0 < prev_col < Nx ? [commonind(A[row, col], A[row, prev_col]) for row in 1:Ny] : Vector{Index}(undef, Ny)
    for row in 1:Ny
        #row < Ny && replaceind!(Q[row], A_up_inds[row], Q_up_inds[row])
        #row > 1  && replaceind!(Q[row], A_up_inds[row-1], Q_up_inds[row-1])
        #replaceind!(Q[row], next_col_inds[row], QR_inds[row])
        QR_inds[row] = firstind(Q[row], "QR,Site")
        if ((row == 1 || row == Ny) && ndims(Q[row]) == 4) || ndims(Q[row]) == 5
            prev_ind = firstind(Q[row], "Link,r")
            prev_ind_A = firstind(A[row, col], tags(prev_ind))
            replaceind!(Q[row], prev_ind, prev_ind_A)
        end
    end
    return Q, QR_inds, next_col_inds
end

function initQsHorizontal( A::fPEPS, col::Int, next_col::Int; kwargs...)
    maxdim::Int = get(kwargs, :maxdim, 1)
    Ny        = length(A)
    Q         = MPO(deepcopy(A), 0, Ny+1)
    prev_col  = next_col > col ? col - 1 : col + 1
    A_r_inds  = [commonind(A[row, col], A[row, next_col]) for row in 1:Ny]
    QR_inds   = [Index(ITensors.dim(A_r_inds[row]), "Site,QR,c$col,r$row") for row in 1:Ny]
    A_up_inds = [commonind(A[row, col], A[row+1, col]) for row in 1:Ny-1]
    Q_up_inds = [Index(ITensors.dim(A_up_inds[row]), "Link,u,Qup$row") for row in 1:Ny-1]
    next_col_inds = [commonind(A[row, col], A[row, next_col]) for row in 1:Ny]
    prev_col_inds = 0 < prev_col < Nx ? [commonind(A[row, col], A[row, prev_col]) for row in 1:Ny] : Vector{Index}(undef, Ny)
    for row in 1:Ny
        row < Ny && replaceind!(Q[row], A_up_inds[row], Q_up_inds[row])
        row > 1  && replaceind!(Q[row], A_up_inds[row-1], Q_up_inds[row-1])
        replaceind!(Q[row], next_col_inds[row], QR_inds[row])
    end
    return Q, QR_inds, next_col_inds
end

function buildEnvs(A::fPEPS, Q::MPO, next_col_inds, QR_inds, col)
    Ny, Ny = size(A)
    thisTerm  = Vector{ITensor}(undef, Ny)
    thisfTerm = Vector{ITensor}(undef, Ny)
    for row in 1:Ny
        Ap = dag(deepcopy(A[row, col]))'
        Ap = setprime(Ap, 0, next_col_inds[row]')
        Qp = dag(deepcopy(Q[row]))'
        Qp = setprime(Qp, 0, QR_inds[row]')
        thisTerm[row]  = A[row, col] * Ap * Qp
        thisfTerm[row] = thisTerm[row] * Q[row]
    end
    Envs = thisTerm
    fF   = cumprod(thisfTerm)
    rF   = reverse(cumprod(reverse(thisfTerm)))
    for row in 1:Ny
        if row > 1
            Envs[row] *= fF[row - 1]
        end
        if row < Ny
            Envs[row] *= rF[row + 1]
        end
    end
    return Envs
end

function buildEnvsHorizontal(A::Vector{ITensor}, Q::MPO, next_col_inds, QR_inds, col, ncol)
    Ny = length(A)
    thisTerm  = Vector{ITensor}(undef, Ny)
    thisfTerm = Vector{ITensor}(undef, Ny)
    for row in 1:Ny
        Ap = dag(deepcopy(A[row]))'
        Ap = setprime(Ap, 0, next_col_inds[row]')
        Ap = setprime(Ap, 0, firstind(A[row], "Site,c$ncol,r$row")')
        Qp = dag(deepcopy(Q[row]))'
        Qp = setprime(Qp, 0, QR_inds[row]')
        thisTerm[row]  = A[row] * Ap * Qp
        thisfTerm[row] = thisTerm[row] * Q[row]
    end
    Envs = thisTerm
    fF   = cumprod(thisfTerm)
    rF   = reverse(cumprod(reverse(thisfTerm)))
    for row in 1:Ny
        if row > 1
            Envs[row] *= fF[row - 1]
        end
        if row < Ny
            Envs[row] *= rF[row + 1]
        end
    end
    return Envs
end

function buildEnvsHorizontalVariant(A::Vector{ITensor}, Q::MPO, next_col_inds, QR_inds, col, ncol)
    Ny    = length(A)
    is_cu = is_gpu(A)
    Envs  = Vector{ITensor}(undef, Ny)
    Env   = is_cu ? cuITensor(1.0) : ITensor(1.0)
    for row in reverse(1:Ny)
        Ap = dag(deepcopy(A[row]))'
        Ap = setprime(Ap, 0, next_col_inds[row]')
        Ap = setprime(Ap, 0, firstind(A[row], "Site,c$ncol,r$row")')
        Qp = dag(deepcopy(Q[row]))'
        Qp = setprime(Qp, 0, QR_inds[row]')
        Env *= A[row]
        Env *= Ap
        Env *= Qp
        if row > 1
            Env *= Q[row]
        end
        Envs[row] = Env
    end
    return Envs
end


function compute_overlap(Ampo, Q, R, is_cu)
    Ny = length(Q)
    aqr_overlap = is_cu ? cuITensor(1.0) : ITensor(1.0)
    a_norm      = is_cu ? cuITensor(1.0) : ITensor(1.0)
    cis         = [commonind(Ampo[row], Ampo[row+1]) for row in 1:Ny-1]
    Ampo_       = deepcopy(Ampo)
    for row in 1:Ny
        if row < Ny
            Ampo_[row] = prime(Ampo_[row], cis[row])
        end
        if row > 1
            Ampo_[row] = prime(Ampo_[row], cis[row-1])
        end
        aqr_overlap *= Ampo[row] * Q[row] * R[row]
        a_norm      *= dag(Ampo_[row]) * Ampo[row]
    end
    ratio = abs(scalar(collect(aqr_overlap)))/abs(scalar(collect(a_norm)))
    return ratio
end

function orthogonalize_Q!(Envs, Q, A, QR_inds, next_col_inds, dummy_nexts, col, two_sided; kwargs...)
    Ny    = length(Q)
    Ampo  = MPO(Ny)
    cmb_l = Vector{ITensor}(undef, Ny)
    for row in reverse(1:Ny)
        if row < Ny
            Q_ = my_polar(Envs[row], QR_inds[row], commonind(Q[row], Q[row+1]); kwargs...)
            Q[row] = deepcopy(noprime(Q_))
        else
            Q_ = my_polar(Envs[row], QR_inds[row]; kwargs...)
            Q[row] = deepcopy(noprime(Q_))
        end
        AQinds     = IndexSet(firstind(A[row, col], "Site")) 
        if two_sided
            AQinds = IndexSet(AQinds..., commonind(inds(A[row, col], "Link"), inds(Q[row], "Link"))) # prev_col_ind
        end
        cmb_l[row] = combiner(AQinds, tags="Site,AQ,r$row")
        Ampo[row]  = A[row, col] * cmb_l[row]
        Ampo[row]  = replaceind!(Ampo[row], next_col_inds[row], dummy_nexts[row])
        Q[row]    *= cmb_l[row]
    end
    return Q, Ampo, cmb_l
end

function orthogonalize_QHorizontal!(Envs, Q, A, Af, QR_inds, next_col_inds, dummy_nexts, col, ncol, two_sided; kwargs...)
    Ny, Ny = size(Af)
    Ampo  = MPO(Ny)
    cmb_l = Vector{ITensor}(undef, Ny)
    for row in 1:Ny
        if row < Ny
            Q_ = my_polar(Envs[row], QR_inds[row], commonind(Q[row], Q[row+1]); kwargs...)
            Q[row] = deepcopy(noprime(Q_))
        else
            Q_ = my_polar(Envs[row], QR_inds[row]; kwargs...)
            Q[row] = deepcopy(noprime(Q_))
        end
        AQinds     = IndexSet(firstind(A[row], "Site,c$col")) 
        if two_sided
            AQinds = IndexSet(AQinds..., commonind(inds(A[row], "Link"), inds(Q[row], "Link"))) # prev_col_ind
        end
        cmb_l[row] = combiner(AQinds, tags="Site,AQ,r$row")
        Ampo[row]  = A[row] * cmb_l[row]
        Ampo[row] *= dummy_nexts[row] 
        Q[row]    *= cmb_l[row]
    end
    return Q, Ampo, cmb_l
end

function orthogonalize_QHorizontalRebuild!(Envs, Q, A, Af, QR_inds, next_col_inds, dummy_nexts, col, ncol, two_sided; kwargs...)
    Ny, Ny = size(Af)
    is_cu  = is_gpu(Af)
    Ampo   = MPO(Ny)
    cmb_l  = Vector{ITensor}(undef, Ny)
    for row in 1:Ny
        if row < Ny
            Q_ = my_polar(Envs[row], QR_inds[row], commonind(Q[row], Q[row+1]); kwargs...)
            Q[row] = deepcopy(noprime(Q_))
        else
            Q_ = my_polar(Envs[row], QR_inds[row]; kwargs...)
            Q[row] = deepcopy(noprime(Q_))
        end
        Ap = dag(deepcopy(A[row]))'
        Ap = setprime(Ap, 0, next_col_inds[row]')
        Ap = setprime(Ap, 0, firstind(A[row], "Site,c$ncol,r$row")')
        Qp = dag(deepcopy(Q[row]))'
        Qp = setprime(Qp, 0, QR_inds[row]')
        thisTerm  = is_cu ? cuITensor(1.0) : ITensor(1.0)
        if row > 1 
            thisTerm *= Envs[row-1]
        end
        thisTerm = A[row] * Ap * Qp * Q[row]
        Envs[row] = row > 1 ? thisTerm*Envs[row-1] : thisTerm
        if row < Ny
            Ap = dag(deepcopy(A[row+1]))'
            Ap = setprime(Ap, 0, next_col_inds[row+1]')
            Ap = setprime(Ap, 0, firstind(A[row+1], "Site,c$ncol,r$(row+1)")')
            Qp = dag(deepcopy(Q[row+1]))'
            Qp = setprime(Qp, 0, QR_inds[row+1]')
            thisTerm  = Envs[row] 
            thisTerm *= A[row+1] * Ap * Qp
            Envs[row+1] = thisTerm
            if row + 1 < Ny
                Envs[row+1] *= Envs[row+2]
            end
        end
        AQinds     = IndexSet(firstind(A[row], "Site,c$col")) 
        if two_sided
            AQinds = IndexSet(AQinds..., commonind(inds(A[row], "Link"), inds(Q[row], "Link"))) # prev_col_ind
        end
        cmb_l[row] = combiner(AQinds, tags="Site,AQ,r$row")
        Ampo[row]  = A[row] * cmb_l[row]
        Ampo[row] *= dummy_nexts[row]
        Q[row]    *= cmb_l[row]
    end
    return Q, Ampo, cmb_l
end

function gaugeQR(A::fPEPS, col::Int, side::Symbol; kwargs...)
    overlap_cutoff::Real = get(kwargs, :overlap_cutoff, 1e-4)
    max_gauge_iter::Int  = get(kwargs, :max_gauge_iter, 100)
    Ny, Nx   = size(A)
    prev_col_inds = Vector{Index}(undef, Ny)
    is_cu    = is_gpu(A)
    next_col = side == :left ? col - 1 : col + 1
    prev_col = side == :left ? col + 1 : col - 1
    Q, QR_inds, next_col_inds = initQs(A, col, next_col; kwargs...)
    left_edge     = col == 1
    right_edge    = col == Nx
    ratio_history = Vector{Float64}() 
    ratio         = 0.0
    best_overlap  = 0.0
    best_Q        = MPO(Ny)
    best_R        = MPO(Ny)
    iter          = 1
    dummy_nexts   = [Index(ITensors.dim(next_col_inds[row]), "DM,Site,r$row") for row in 1:Ny]
    iter          = 0
    while best_overlap < overlap_cutoff 
        @timeit "build envs" begin
            Envs = buildEnvs(A, Q, next_col_inds, QR_inds, col)
        end
        @timeit "polar decomp" begin
            two_sided      = (side == :left && !right_edge) || (side == :right && !left_edge)
            Q, Ampo, cmb_l = orthogonalize_Q!(Envs, Q, A, QR_inds, next_col_inds, dummy_nexts, col, two_sided; kwargs...)
        end
        @timeit "Q*A -> R" begin
            R = multMPO(dag(Q), Ampo; kwargs...)
        end
        @timeit "compute overlap" begin
            ratio = compute_overlap(Ampo, Q, R, is_cu)
        end
        for row in 1:Ny
            Q[row]      *= cmb_l[row]
        end
        push!(ratio_history, ratio)
        if ratio > best_overlap || iter == 0
            best_Q = deepcopy(Q)
            best_R = deepcopy(R)
            best_overlap = ratio
        end
        ratio > overlap_cutoff && break
        iter += 1
        iter > max_gauge_iter && break
        if (iter > 10 && best_overlap < 0.5) || (iter > 20 && mod(iter, 20) == 0)
            Q_, QR_inds_, next_col_inds_ = initQs(A, col, next_col; kwargs...)
            for row in 1:Ny
                if row < Ny
                    old_u = commonind(Q[row], Q[row+1])
                    new_u = commonind(Q_[row], Q_[row+1])
                    replaceind!(Q_[row], new_u, old_u)
                    replaceind!(Q_[row+1], new_u, old_u)
                end
                replaceind!(Q_[row], QR_inds_[row], QR_inds[row])
            end
            for row in 1:Ny
                Q[row]  = Q_[row] 
                salt    = is_cu ? cuITensor(randomITensor(inds(Q[row])))/100.0 : randomITensor(inds(Q[row]))/100.0
                salt_d  = ratio < 0.5 ? norm(salt) : 10.0*norm(salt)
                Q[row] += salt/salt_d
                Q[row] /= sqrt(norm(Q[row])) 
            end
        end
    end
    println( "\tbest overlap: ", best_overlap)#, "\nratio_history: $ratio_history")
    return best_Q, best_R, next_col_inds, QR_inds, dummy_nexts
end

function gaugeQRHorizontal(A::fPEPS, Aj::Vector{ITensor}, col::Int, ncol::Int, side::Symbol; kwargs...)
    overlap_cutoff::Real = get(kwargs, :overlap_cutoff, 1e-4)
    max_gauge_iter::Int  = get(kwargs, :max_gauge_iter, 100)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    prev_col_inds = Vector{Index}(undef, Ny)
    next_col = side == :left ? col - 1 : col + 1
    prev_col = side == :left ? col + 1 : col - 1
    #Q, QR_inds, next_col_inds = initQs(A, col, ncol; kwargs...)
    Q = MPO()
    QR_inds = Vector{Index}(undef, Ny)
    next_col_inds = Vector{Index}(undef, Ny)
    #=if haskey(q_dict, col=>side)
        Q = q_dict[col=>side]
        Q, QR_inds, next_col_inds = initQs(A, Q, col, ncol; kwargs...)
    else=#
    old_Q, old_R, old_ncis, old_QRinds, old_dns = gaugeQR(A, col, side; max_gauge_iter=10, overlap_cutoff=0.999)
    Q = old_Q 
    QR_inds = old_QRinds
    next_col_inds = old_ncis
    #end
    left_edge     = col == 1
    right_edge    = col == Nx
    ratio_history = Vector{Float64}()
    ratio         = 0.0
    best_overlap  = 0.0
    best_Q        = MPO(Ny)
    best_R        = MPO(Ny)
    iter          = 1
    dummy_nexts   = Vector{ITensor}(undef, Ny) #[Index(ITensors.dim(next_col_inds[row]), "DM,Site,r$row") for row in 1:Ny]
    for row in 1:Ny
        cis = IndexSet(firstind(Aj[row], "Site,c$ncol"))
        if ncol > col && ncol <= Nx - 1
            anci = commonindex(A[row, ncol], A[row, ncol+1])
            next_col_inds[row] = anci
            cis = IndexSet(cis..., anci)
        elseif ncol < col && ncol >= 2
            anci = commonindex(A[row, ncol], A[row, ncol-1])
            next_col_inds[row] = anci
            cis = IndexSet(cis..., anci)
        end
        dummy_nexts[row] = combiner(cis, tags="DM,Site,r$row")
    end
    iter          = 0
    downward      = 0
    below         = 0
    while best_overlap < overlap_cutoff
        @timeit "build envs" begin
            #Envs = buildEnvsHorizontal(Aj, Q, next_col_inds, QR_inds, col, ncol)
            Envs = buildEnvsHorizontalVariant(Aj, Q, next_col_inds, QR_inds, col, ncol)
        end
        @timeit "polar decomp" begin
            two_sided      = (side == :left && !right_edge) || (side == :right && !left_edge)
            Q, Ampo, cmb_l = orthogonalize_QHorizontalRebuild!(Envs, Q, Aj, A, QR_inds, next_col_inds, dummy_nexts, col, ncol, two_sided; kwargs...)
        end
        @timeit "Q*A -> R" begin
            R = multMPO(dag(Q), Ampo; kwargs...)
        end
        @timeit "compute overlap" begin
            ratio = compute_overlap(Ampo, Q, R, is_cu)
        end
        for row in 1:Ny
            Q[row]      *= cmb_l[row]
        end
        push!(ratio_history, ratio)
        if iter > 1
            if ratio < ratio_history[end-1]
                downward += 1
            else
                downward = 0
            end
            if ratio < 0.5
                below += 1
            else
                below = 0
            end
        end
        if ratio > best_overlap || iter == 0
            best_Q = deepcopy(Q)
            best_R = deepcopy(R)
            best_overlap = ratio
            #q_dict[col=>side] = Q
        end
        ratio > overlap_cutoff && break
        iter += 1
        iter > max_gauge_iter && break
        #if (iter > 10 && best_overlap < 0.5) || (iter > 20 && mod(iter, 40) == 0)
        if downward > 20 || below > 20
            Q_, QR_inds_, next_col_inds_ = initQs(A, col, next_col; kwargs...)
            for row in 1:Ny
                if row < Ny
                    old_u = commonind(Q[row], Q[row+1])
                    new_u = commonind(Q_[row], Q_[row+1])
                    replaceind!(Q_[row], new_u, old_u)
                    replaceind!(Q_[row+1], new_u, old_u)
                end
                replaceind!(Q_[row], QR_inds_[row], QR_inds[row])
            end
            for row in 1:Ny
                Q[row]  = Q_[row] 
                salt    = is_cu ? cuITensor(randomITensor(inds(Q[row])))/100.0 : randomITensor(inds(Q[row]))/100.0
                salt_d  = ratio < 0.5 ? norm(salt) : 10.0*norm(salt)
                Q[row] += salt/salt_d
                Q[row] /= sqrt(norm(Q[row])) 
            end
            push!(ratio_history, -1000.0)
            downward = 0
            below = 0
        end
    end
    println( "best overlap: ", best_overlap)
    println( "ratio_history:")
    display(ratio_history)
    println()
    println()
    #=for row in 1:Ny
        A_r_inds  = [commonind(A[row, col], A[row, next_col]) for row in 1:Ny]
        QR_inds  .= [Index(ITensors.dim(A_r_inds[row]), "Site,QR,c$col,r$row") for row in 1:Ny]
        replaceind!(best_Q[row], next_col_inds[row], QR_inds[row])
    end
    q_dict[col=>side] = best_Q=#
    return best_Q, best_R, next_col_inds, QR_inds, dummy_nexts
end


function gaugeColumn( A::fPEPS, col::Int, side::Symbol; kwargs...)
    Ny, Nx = size(A)

    prev_col_inds = Vector{Index}(undef, Ny)
    next_col_inds = Vector{Index}(undef, Ny)

    next_col   = side == :left ? col - 1 : col + 1
    prev_col   = side == :left ? col + 1 : col - 1
    left_edge  = col == 1
    right_edge = col == Nx
    is_cu      = is_gpu(A)
    
    @timeit "gauge QR" begin
        Q, R, next_col_inds, QR_inds, dummy_next_inds = gaugeQR(A, col, side; kwargs...)
    end
    @timeit "update next col" begin
        cmb_r = Vector{ITensor}(undef, Ny)
        cmb_u = Vector{ITensor}(undef, Ny - 1)
        if (side == :left && col > 1 ) || (side == :right && col < Nx)
            next_col_As = MPO(deepcopy(A[:, next_col]), 0, Ny+1)
            nn_col = side == :left ? next_col - 1 : next_col + 1
            cmb_inds_ = [IndexSet(firstind(A[row, next_col], "Site")) for row in 1:Ny]
            for row in 1:Ny
                cmb_inds_r = cmb_inds_[row] 
                if 0 < nn_col < Nx + 1
                    cmb_inds_r = IndexSet(cmb_inds_[row]..., commonind(A[row, next_col], A[row, nn_col]))
                end
                cmb_r[row] = combiner(cmb_inds_r, tags="Site,CMB,c$col,r$row")
                next_col_As[row] *= cmb_r[row]
                if (side == :left && !left_edge) || (side == :right && !right_edge)
                    next_col_As[row] = replaceind!(next_col_As[row], next_col_inds[row], dummy_next_inds[row])
                end
            end
        end
        maxdim::Int  = get(kwargs, :maxdim, 1)
        result       = multMPO(R, next_col_As; kwargs...)
        a_norm      = is_cu ? cuITensor(1.0) : ITensor(1.0)
        q_norm      = is_cu ? cuITensor(1.0) : ITensor(1.0)
        r_norm      = is_cu ? cuITensor(1.0) : ITensor(1.0)
        for row in 1:Ny
            a_norm      *= dag(result[row]) * result[row]
            q_norm      *= dag(Q[row]) * Q[row]
            r_norm      *= dag(R[row]) * R[row]
        end
        true_QR_inds = [Index(ITensors.dim(QR_inds[row]), "Link,r,r$row" * (side == :left ? ",c$(col-1)" : ",c$col")) for row in 1:Ny]
        for row in 1:Ny
            A[row, col] = Q[row]
        end
        cUs = [commonind(A[row, col], A[row+1, col]) for row in 1:Ny-1]
        true_U_inds = [Index(ITensors.dim(cUs[row]), "Link,u,r$row,c$col") for row in 1:Ny-1]
        A[:, col] = [replaceind!(A[row, col], QR_inds[row], true_QR_inds[row]) for row in 1:Ny]
        A[:, col] = vcat([replaceind!(A[row, col], cUs[row], true_U_inds[row]) for row in 1:Ny-1], A[Ny, col])
        A[:, col] = vcat(A[1, col], [replaceind!(A[row, col], cUs[row-1], true_U_inds[row-1]) for row in 2:Ny])

        for row in 1:Ny
            A[row, next_col] = result[row]
        end
        A[:, next_col] = [replaceind!(A[row, next_col], QR_inds[row], true_QR_inds[row]) for row in 1:Ny]
        cUs            = [commonind(A[row, next_col], A[row+1, next_col]) for row in 1:Ny-1]
        true_nU_inds   = [Index(ITensors.dim(cUs[row]), "Link,u,r$row,c" * string(next_col)) for row in 1:Ny-1]
        A[:, next_col] = vcat([replaceind!(A[row, next_col], cUs[row], true_nU_inds[row]) for row in 1:Ny-1], A[Ny, next_col])
        A[:, next_col] = vcat(A[1, next_col], [replaceind!(A[row, next_col], cUs[row-1], true_nU_inds[row-1]) for row in 2:Ny])
        A[:, next_col] = [A[row, next_col] * cmb_r[row] for row in 1:Ny]
    end
    return A
end

function gaugeColumnHorizontal( A::fPEPS, Aj::Vector{ITensor}, col::Int, ncol::Int, side::Symbol; kwargs...)
    Ny, Nx = size(A)

    prev_col_inds = Vector{Index}(undef, Ny)
    next_col_inds = Vector{Index}(undef, Ny)

    next_col   = side == :left ? col - 1 : col + 1
    prev_col   = side == :left ? col + 1 : col - 1
    left_edge  = col == 1
    right_edge = col == Nx
    is_cu      = is_gpu(A)
    
    @timeit "gauge QR" begin
        Q, R, next_col_inds, QR_inds, dummy_next_inds = gaugeQRHorizontal(A, Aj, col, ncol, side; kwargs...)
    end
    @timeit "update next col" begin
        cmb_r = dummy_next_inds  
        cmb_u = Vector{ITensor}(undef, Ny - 1)
        maxdim::Int  = get(kwargs, :maxdim, 1)
        true_QR_inds = [Index(ITensors.dim(QR_inds[row]), "Link,r,r$row" * (side == :left ? ",c$(col-1)" : ",c$col")) for row in 1:Ny]
        for row in 1:Ny
            A[row, col] = Q[row]
        end
        cUs = [commonind(A[row, col], A[row+1, col]) for row in 1:Ny-1]
        true_U_inds = [Index(ITensors.dim(cUs[row]), "Link,u,r$row,c$col") for row in 1:Ny-1]
        A[:, col] = [replaceind!(A[row, col], QR_inds[row], true_QR_inds[row]) for row in 1:Ny]
        A[:, col] = vcat([replaceind!(A[row, col], cUs[row], true_U_inds[row]) for row in 1:Ny-1], A[Ny, col])
        A[:, col] = vcat(A[1, col], [replaceind!(A[row, col], cUs[row-1], true_U_inds[row-1]) for row in 2:Ny])

        for row in 1:Ny
            A[row, next_col] = R[row]
        end
        A[:, next_col] = [replaceind!(A[row, next_col], QR_inds[row], true_QR_inds[row]) for row in 1:Ny]
        cUs            = [commonind(A[row, next_col], A[row+1, next_col]) for row in 1:Ny-1]
        true_nU_inds   = [Index(ITensors.dim(cUs[row]), "Link,u,r$row,c" * string(next_col)) for row in 1:Ny-1]
        A[:, next_col] = vcat([replaceind!(A[row, next_col], cUs[row], true_nU_inds[row]) for row in 1:Ny-1], A[Ny, next_col])
        A[:, next_col] = vcat(A[1, next_col], [replaceind!(A[row, next_col], cUs[row-1], true_nU_inds[row-1]) for row in 2:Ny])
        A[:, next_col] = [A[row, next_col] * cmb_r[row] for row in 1:Ny]
    end
    return A
end


function gaugeColumnForInsert( A::fPEPS, col::Int, side::Symbol; kwargs...)
    Ny, Nx = size(A)

    prev_col_inds = Vector{Index}(undef, Ny)
    next_col_inds = Vector{Index}(undef, Ny)

    next_col   = side == :left ? col - 1 : col + 1
    prev_col   = side == :left ? col + 1 : col - 1
    left_edge  = col == 1
    right_edge = col == Nx
    is_cu      = is_gpu(A)
    
    @timeit "gauge QR" begin
        Q, R, next_col_inds, QR_inds, dummy_next_inds = gaugeQR(A, col, side; kwargs...)
    end
    return A, Q, R, dummy_next_inds
end
