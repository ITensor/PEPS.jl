function measureXmag(A::fPEPS, 
                     Ls::Vector{Environments}, 
                     Rs::Vector{Environments}, 
                     col; 
                     kwargs...)
    s = Index(2, "Site,SpinInd")
    X = ITensor(s, s')
    is_cu         = is_gpu(A) 
    X[s(1), s'(2)] = 0.5
    X[s(2), s'(1)] = 0.5
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    dummyI = is_cu ? MPS([cuITensor(1.0) for ii in 1:Ny], 0, Ny+1) : MPS([ITensor(1.0) for ii in 1:Ny], 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny), fill(ITensor(), 1, Ny))
    measuredX = zeros(Ny)
    op = is_cu ? cuITensor(X) : X
    Xs = [Operator([row=>col], [op], s, Field) for row in 1:Ny]
    tR = col == Nx ? dummyEnv : Rs[col+1]
    tL = col == 1  ? dummyEnv : Ls[col-1]
    AI = makeAncillaryIs(A, tL, tR, col)
    AF = makeAncillaryFs(A, tL, tR, Xs, col)
    fT = fieldTerms(A, tL, tR, (above=AI,), (above=AF,), Xs, 1, col, A[1, col])
    N  = buildN(A, tL, tR, (above=AI,), 1, col, A[1, col])
    for row in 1:Ny
        measuredX[row] = scalar(fT[row] * dag(A[1, col])')/scalar(N * dag(A[1, col]'))
    end
    return measuredX
end

function measureZmag(A::fPEPS, 
                     Ls::Vector{Environments}, 
                     Rs::Vector{Environments}, 
                     col; 
                     kwargs...)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    is_cu         = is_gpu(A) 
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    Nx, Ny = size(A)
    is_cu  = is_gpu(A)
    dummyI = is_cu ? MPS([cuITensor(1.0) for ii in 1:Ny], 0, Ny+1) : MPS([ITensor(1.0) for ii in 1:Ny], 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny), fill(ITensor(), 1, Ny))
    measuredZ = zeros(Ny)
    op = is_cu ? cuITensor(Z) : Z 
    Zs = [Operator([row=>col], [op], s, Field) for row in 1:Ny]
    #A  = intraColumnGauge(A, col; kwargs...)
    tR = col == Nx ? dummyEnv : Rs[col+1]
    tL = col == 1  ? dummyEnv : Ls[col-1]
    AI = makeAncillaryIs(A, tL, tR, col)
    AF = makeAncillaryFs(A, tL, tR, Zs, col)
    fT = fieldTerms(A, tL, tR, (above=AI,), (above=AF,), Zs, 1, col, A[1, col])
    N  = buildN(A, tL, tR, (above=AI,), 1, col, A[1, col])
    for row in 1:Ny
        measuredZ[row] = scalar(fT[row] * dag(A[1, col]'))/scalar(N * dag(A[1, col]'))
    end
    return measuredZ
end

function measureSmagVertical(A::fPEPS, 
                             Ls::Vector{Environments}, 
                             Rs::Vector{Environments}, 
                             col; 
                             kwargs...)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    P = ITensor(s, s')
    M = ITensor(s, s')
    P[s(1), s'(2)] = 1.0
    M[s(2), s'(1)] = 1.0
    Nx, Ny     = size(A)
    is_cu  = is_gpu(A)
    dummyI = is_cu ? MPS([cuITensor(1.0) for ii in 1:Ny], 0, Ny+1) : MPS([ITensor(1.0) for ii in 1:Ny], 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny), fill(ITensor(), 1, Ny))
    measuredSV = zeros(Ny)
    is_cu     = is_gpu(A) 
    Z = is_cu ? cuITensor(Z) : Z
    P = is_cu ? cuITensor(P) : P
    M = is_cu ? cuITensor(M) : M
    SVs = Operator[]
    for row in 1:Ny-1
        push!(SVs, Operator([row=>col, row+1=>col], [0.5*P, M], s, Vertical))
        push!(SVs, Operator([row=>col, row+1=>col], [0.5*M, P], s, Vertical))
        push!(SVs, Operator([row=>col, row+1=>col], [Z, Z], s, Vertical))
    end
    tR = col == Nx ? dummyEnv : Rs[col+1]
    tL = col == 1  ? dummyEnv : Ls[col-1]
    AI = makeAncillaryIs(A, tL, tR, col)
    AV = makeAncillaryVs(A, tL, tR, SVs, col)
    vTs = verticalTerms(A, tL, tR, (above=AI,), (above=AV,), SVs, 1, col, A[1, col])
    N   = buildN(A, tL, tR, (above=AI,), 1, col, A[1, col])
    nrm = scalar(N * dag(A[1, col]'))
    for (vi, vT) in enumerate(vTs)
        row = SVs[vi].sites[1][1]
        measuredSV[row] += scalar(vT * dag(A[1, col]'))/nrm
    end
    return measuredSV
end

function measureSmagHorizontal(A::fPEPS, 
                               Ls::Vector{Environments}, 
                               Rs::Vector{Environments}, 
                               col; 
                               kwargs...)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    P = ITensor(s, s')
    M = ITensor(s, s')
    P[s(1), s'(2)] = 1.0
    M[s(2), s'(1)] = 1.0
    is_cu     = is_gpu(A) 
    Z = is_cu ? cuITensor(Z) : Z
    P = is_cu ? cuITensor(P) : P
    M = is_cu ? cuITensor(M) : M
    Nx, Ny = size(A)
    is_cu  = is_gpu(A)
    dummyI = is_cu ? MPS([cuITensor(1.0) for ii in 1:Ny], 0, Ny+1) : MPS([ITensor(1.0) for ii in 1:Ny], 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny), fill(ITensor(), 1, Ny))
    measuredSH = zeros(Ny)
    SHs = Operator[]
    for row in 1:Ny
        push!(SHs, Operator([row=>col, row=>col+1], [0.5*P, M], s, Horizontal))
        push!(SHs, Operator([row=>col, row=>col+1], [0.5*M, P], s, Horizontal))
        push!(SHs, Operator([row=>col, row=>col+1], [Z, Z], s, Horizontal))
    end
    tR = Rs[col+1]
    tL = col == 1 ? dummyEnv : Ls[col-1]
    AI = makeAncillaryIs(A, tL, tR, col)
    AS = makeAncillarySide(A, tR, tL, SHs, col, :right)
    hTs = connectRightTerms(A, tL, tR, (above=AI,), (above=AS,), SHs, 1, col, A[1, col])
    N  = buildN(A, tL, tR, (above=AI,), 1, col, A[1, col])
    nrm = scalar(N * dag(A[1, col]'))
    for (hi, hT) in enumerate(hTs)
        row = SHs[hi].sites[1][1]
        measuredSH[row] += scalar(hT * dag(A[1, col]'))/nrm
    end
    return measuredSH
end

function measure_correlators_heisenberg(A::fPEPS, H, Ls, Rs, sweep::Int; kwargs...)
    Ny, Nx = size(A)
    x_mag = zeros(Ny, Nx)
    z_mag = zeros(Ny, Nx)
    v_mag = zeros(Ny, Nx)
    h_mag = zeros(Ny, Nx)
    env_maxdim::Int = get(kwargs, :env_maxdim, 1)
    maxdim::Int     = get(kwargs, :maxdim, 1)
    mindim::Int     = get(kwargs, :mindim, 1)
    cutoff::Float64 = get(kwargs, :cutoff, 0.0)
    prefix::String  = get(kwargs, :prefix, "")
    max_gauge_iter::Int = get(kwargs, :max_gauge_iter, 50)
    if iseven(sweep)
        #R_s = buildRs(A, H; mindim=mindim, maxdim=maxdim, env_maxdim=env_maxdim)
        R_s = Vector{PEPS.Environments}(undef, Nx)
        for col in reverse(1:Nx)
            A  = intraColumnGauge(A, col; mindim=mindim, maxdim=maxdim)
            x_mag[:, col] = measureXmag(A, Ls, R_s, col; mindim=mindim, maxdim=maxdim)
            z_mag[:, col] = measureZmag(A, Ls, R_s, col; mindim=mindim, maxdim=maxdim)
            v_mag[:, col] = measureSmagVertical(A, Ls, R_s, col; mindim=mindim, maxdim=maxdim)
            if col < Nx
                h_mag[:, col] = measureSmagHorizontal(A, Ls, R_s, col; mindim=mindim, maxdim=maxdim)
            end
            if col > 1
                A  = gaugeColumn(A, col, :left; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim, overlap_cutoff=0.999, max_gauge_iter=max_gauge_iter)
                if col < Nx
                    R_s[col] = buildNextEnvironment(A, R_s[col+1], H, :right, col; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim)
                else
                    right_H_terms = getDirectional(H[Nx-1], Horizontal)
                    R_s[col] = buildEdgeEnvironment(A, H, right_H_terms, :right, col; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim)
                end
            end
        end
    else
        #L_s = buildLs(A, H; mindim=mindim, maxdim=maxdim, env_maxdim=env_maxdim)
        L_s = Vector{PEPS.Environments}(undef, Nx)
        for col in 1:Nx
            A  = intraColumnGauge(A, col; mindim=mindim, maxdim=maxdim)
            x_mag[:, col] = measureXmag(A, L_s, Rs, col; mindim=mindim, maxdim=maxdim)
            z_mag[:, col] = measureZmag(A, L_s, Rs, col; mindim=mindim, maxdim=maxdim)
            v_mag[:, col] = measureSmagVertical(A, L_s, Rs, col; mindim=mindim, maxdim=maxdim)
            if col < Nx
                h_mag[:, col] = measureSmagHorizontal(A, L_s, Rs, col; mindim=mindim, maxdim=maxdim)
            end
            if col < Nx
                A  = gaugeColumn(A, col, :right; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim, overlap_cutoff=0.999, max_gauge_iter=max_gauge_iter)
                if col > 1
                    L_s[col] = buildNextEnvironment(A, L_s[col-1], H, :left, col; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim)
                else
                    left_H_terms = getDirectional(H[1], Horizontal)
                    L_s[col] = buildEdgeEnvironment(A, H, left_H_terms, :left, col; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim)
                end
            end
        end
    end
    writedlm(prefix*"_$(sweep)_x", x_mag)
    writedlm(prefix*"_$(sweep)_z", z_mag)
    writedlm(prefix*"_$(sweep)_v", v_mag)
    writedlm(prefix*"_$(sweep)_h", h_mag)
end

function measure_correlators_ising(A::fPEPS, H, Ls, Rs, sweep::Int; kwargs...)
    Ny, Nx = size(A)
    x_mag = zeros(Ny, Nx)
    z_mag = zeros(Ny, Nx)
    env_maxdim::Int = get(kwargs, :env_maxdim, 1)
    maxdim::Int     = get(kwargs, :maxdim, 1)
    mindim::Int     = get(kwargs, :mindim, 1)
    cutoff::Float64 = get(kwargs, :cutoff, 0.0)
    prefix::String  = get(kwargs, :prefix, "")
    max_gauge_iter::Int = get(kwargs, :max_gauge_iter, 50)
    if iseven(sweep)
        R_s = Vector{PEPS.Environments}(undef, Nx)
        for col in reverse(1:Nx)
            A  = intraColumnGauge(A, col; mindim=mindim, maxdim=maxdim)
            x_mag[:, col] = measureXmag(A, Ls, R_s, col; mindim=mindim, maxdim=maxdim)
            z_mag[:, col] = measureZmag(A, Ls, R_s, col; mindim=mindim, maxdim=maxdim)
            if col > 1 
                A  = gaugeColumn(A, col, :left; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim, overlap_cutoff=0.999, max_gauge_iter=max_gauge_iter)
                if col < Nx
                    R_s[col] = buildNextEnvironment(A, R_s[col+1], H, :right, col; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim)
                else
                    right_H_terms = getDirectional(vcat(H[:, Nx-1]), Horizontal)
                    R_s[col] = buildEdgeEnvironment(A, H, right_H_terms, :right, col; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim)
                end
            end
        end
    else
        L_s = Vector{PEPS.Environments}(undef, Nx)
        for col in 1:Nx
            A  = intraColumnGauge(A, col; mindim=mindim, maxdim=maxdim)
            x_mag[:, col] = measureXmag(A, L_s, Rs, col; mindim=mindim, maxdim=maxdim)
            z_mag[:, col] = measureZmag(A, L_s, Rs, col; mindim=mindim, maxdim=maxdim)
            if col < Nx 
                A  = gaugeColumn(A, col, :right; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim, overlap_cutoff=0.999, max_gauge_iter=max_gauge_iter)
                if col > 1
                    L_s[col] = buildNextEnvironment(A, L_s[col-1], H, :left, col; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim)
                else
                    left_H_terms = getDirectional(vcat(H[:, 1]), Horizontal)
                    L_s[col] = buildEdgeEnvironment(A, H, left_H_terms, :left, col; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim)
                end
            end
        end
    end
    writedlm(prefix*"_$(sweep)_x", x_mag)
    writedlm(prefix*"_$(sweep)_z", z_mag)
end
