function buildN(A::fPEPS, 
                L::Environments, 
                R::Environments, 
                IEnvs, 
                row::Int, 
                col::Int, 
                ϕ::ITensor)::ITensor
    Ny, Nx   = size(A)
    N        = spinI(firstind(A[row, col], "Site"); is_gpu=is_gpu(A))
    workingN = N
    if row > 1
        workingN *= IEnvs[:below][row - 1]
    end
    workingN *= L.I[row]
    workingN *= ϕ
    workingN *= R.I[row]
    if row < Ny
        workingN *= IEnvs[:above][end - row]
    end
    return workingN
end

function buildHIedge(A::fPEPS, 
                     E::Environments, 
                     row::Int, 
                     col::Int, 
                     side::Symbol, 
                     ϕ::ITensor )
    Ny, Nx = size(A)
    is_cu  = is_gpu(A) 
    IH_a   = is_cu ? cuITensor(1.0) : ITensor(1.0)
    IH_b   = is_cu ? cuITensor(1.0) : ITensor(1.0)
    IH     = is_cu ? cuITensor(1.0) : ITensor(1.0)
    next_col = side == :left ? 2 : Nx - 1
    @inbounds for work_row in 1:row-1
        IH_b *= E.H[work_row] * A[work_row, col]
        IH_b *= dag(prime(A[work_row, col], "Link"))
    end
    @inbounds for work_row in row+1:Ny
        IH_a *= E.H[work_row] * A[work_row, col]
        IH_a *= dag(prime(A[work_row, col], "Link"))
    end
    IH *= E.H[row]
    IH *= IH_b
    op  = spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
    op  = is_cu ? cuITensor(op) : op
    IH *= ϕ
    IH *= op
    IH *= IH_a
    AAinds = inds(prime(ϕ))
    @assert hasinds(inds(IH), AAinds)
    @assert hasinds(AAinds, inds(IH))
    return (IH,)
end

function buildHIs(A::fPEPS,
                  L::Environments, 
                  R::Environments, 
                  row::Int, 
                  col::Int, 
                  ϕ::ITensor)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A) 
    @timeit "buildHIedge" begin
        if col == 1
            return buildHIedge(A, R, row, col, :left, ϕ)
        end
        if col == Nx
            return buildHIedge(A, L, row, col, :right, ϕ)
        end
    end
    HLI_a = is_cu ? cuITensor(1.0) : ITensor(1.0)
    HLI_b = is_cu ? cuITensor(1.0) : ITensor(1.0)
    IHR_a = is_cu ? cuITensor(1.0) : ITensor(1.0)
    IHR_b = is_cu ? cuITensor(1.0) : ITensor(1.0)
    HLI   = is_cu ? cuITensor(1.0) : ITensor(1.0)
    IHR   = is_cu ? cuITensor(1.0) : ITensor(1.0)
    @timeit "build HI lower" begin
        @inbounds for work_row in 1:row-1
            tL     = A[work_row, col]*L.H[work_row]
            HLI_b *= tL 
            HLI_b *= R.I[work_row]
            HLI_b *= dag(prime(A[work_row, col], "Link"))
            tR     = A[work_row, col]*R.H[work_row]
            IHR_b *= tR 
            IHR_b *= L.I[work_row]
            IHR_b *= dag(prime(A[work_row, col], "Link"))
        end
    end
    @timeit "build HI upper" begin
        @inbounds for work_row in reverse(row+1:Ny)
            tL     = A[work_row, col]*L.H[work_row]
            HLI_a *= tL 
            HLI_a *= R.I[work_row]
            HLI_a *= dag(prime(A[work_row, col], "Link"))
            tR     = A[work_row, col]*R.H[work_row]
            IHR_a *= tR 
            IHR_a *= L.I[work_row]
            IHR_a *= dag(prime(A[work_row, col], "Link"))
        end
    end
    @timeit "join up HIs" begin
        HLI  *= L.H[row]
        HLI  *= HLI_b
        HLI  *= ϕ
        HLI  *= R.I[row]
        IHR  *= L.I[row]
        IHR  *= IHR_b
        IHR  *= ϕ
        IHR  *= R.H[row]
        HLI *= HLI_a
        IHR *= IHR_a
        op   = spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
        HLI *= op
        IHR *= op
    end
    AAinds = inds(prime(ϕ))
    @assert hasinds(inds(IHR), AAinds)
    @assert hasinds(inds(HLI), AAinds)
    @assert hasinds(AAinds, inds(IHR))
    @assert hasinds(AAinds, inds(HLI))
    return (HLI, IHR)
end

function verticalTerms(A::fPEPS,
                       L::Environments,
                       R::Environments,
                       AI,
                       AV,
                       H,
                       row::Int,
                       col::Int,
                       ϕ::ITensor)::Vector{ITensor}
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    vTerms = ITensor[]
    AAinds = inds(prime(ϕ))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0) 
    @inbounds for opcode in 1:length(H)
        thisVert = dummy 
        op_row_a = H[opcode].sites[1][1]
        op_row_b = H[opcode].sites[2][1]
        if op_row_b < row || row < op_row_a
            local V, I
            if op_row_a > row
                V = AV[:above][opcode][end - row]
                I = row > 1 ? AI[:below][row - 1] : dummy 
            elseif op_row_b < row
                V = AV[:below][opcode][row - op_row_b]
                I = row < Ny ? AI[:above][end - row] : dummy
            end
            thisVert *= V
            thisVert *= L.I[row]
            thisVert *= ϕ
            thisVert *= R.I[row]
            thisVert *= I
            thisVert *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
        elseif row == op_row_a
            low_row  = op_row_a - 1
            high_row = op_row_b
            AIL = low_row > 0 ? AI[:below][low_row] : dummy 
            AIH = high_row < Ny ? AI[:above][end - high_row] : dummy 
            sA   = firstind(A[op_row_a, col], "Site")
            op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
            op_a = replaceind!(op_a, H[opcode].site_ind', sA')
            sB   = firstind(A[op_row_b, col], "Site")
            op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
            op_b = replaceind!(op_b, H[opcode].site_ind', sB')
            thisVert  = AIH
            thisVert *= L.I[op_row_b] 
            thisVert *= A[op_row_b, col] * op_b
            thisVert *= R.I[op_row_b] 
            thisVert *= dag(A[op_row_b, col])'
            thisVert *= L.I[op_row_a] 
            thisVert *= ϕ
            thisVert *= R.I[op_row_a] 
            thisVert *= AIL 
            thisVert *= op_a
        elseif row == op_row_b
            low_row  = op_row_a - 1
            high_row = op_row_b
            AIL = low_row > 0 ? AI[:below][low_row] : dummy 
            AIH = high_row < Ny ? AI[:above][end - high_row] : dummy 
            sA = firstind(A[op_row_a, col], "Site")
            op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
            op_a = replaceind!(op_a, H[opcode].site_ind', sA')
            sB = firstind(A[op_row_b, col], "Site")
            op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
            op_b = replaceind!(op_b, H[opcode].site_ind', sB')
            thisVert  = AIL
            thisVert *= L.I[op_row_a] 
            thisVert *= A[op_row_a, col] * op_a 
            thisVert *= R.I[op_row_a] 
            thisVert *= dag(A[op_row_a, col])'
            thisVert *= L.I[op_row_b] 
            thisVert *= ϕ
            thisVert *= R.I[op_row_b] 
            thisVert *= AIH 
            thisVert *= op_b
        end
        @assert hasinds(inds(thisVert), AAinds) "inds of thisVert and AAinds differ!\n$(inds(thisVert))\n$AAinds\n"
        @assert hasinds(AAinds, inds(thisVert)) "inds of thisVert and AAinds differ!\n$(inds(thisVert))\n$AAinds\n"
        if hasinds(AAinds, inds(thisVert)) && hasinds(inds(thisVert), AAinds)
            push!(vTerms, thisVert)
        end
    end
    return vTerms
end

function diagonalTerms(A::fPEPS,
                       L::Environments,
                       R::Environments,
                       AI,
                       AD,
                       H,
                       row::Int,
                       col::Int,
                       ϕ::ITensor)::Vector{ITensor} 
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    dTerms = Vector{ITensor}(undef, length(H))
    AAinds = inds(prime(ϕ))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0)
    for opcode in 1:length(H)
        thisDiag = dummy
        op_row_a = H[opcode].sites[1][1]
        op_row_b = H[opcode].sites[2][1]
        op_col_a = H[opcode].sites[1][2]
        op_col_b = H[opcode].sites[2][2]
        side     = op_col_a < col  
        max_op_row = max(op_row_a, op_row_b)
        min_op_row = min(op_row_a, op_row_b)
        if max_op_row < row || row < min_op_row
            local D, I
            if min_op_row > row
                D = AD[:above][opcode][end - row]
                I = row > 1 ? AD[:below][opcode][row - 1] : dummy 
            elseif max_op_row < row
                D = AD[:below][opcode][row - 1]
                I = row < Ny ? AD[:above][opcode][end - row] : dummy
            end
            thisDiag *= D
            thisDiag *= side ? L.DiagInProgress[row, opcode] : L.I[row]
            thisDiag *= ϕ
            thisDiag *= side ? R.I[row] : R.DiagInProgress[row, opcode]
            thisDiag *= I
            thisDiag *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
            @assert hasinds(inds(thisDiag), AAinds) "inds of thisDiag and AAinds differ!\n$(inds(thisDiag))\n$AAinds\n"
            @assert hasinds(AAinds, inds(thisDiag)) "inds of thisDiag and AAinds differ!\n$(inds(thisDiag))\n$AAinds\n"
        else
            low_row  = min(op_row_a, op_row_b) - 1
            high_row = max(op_row_a, op_row_b)
            AIL      = low_row  >= 1 ? AD[:below][opcode][low_row]        : dummy
            AIH      = high_row < Ny ? AD[:above][opcode][end - high_row] : dummy
            sA       = firstind(A[op_row_a, col], "Site")
            sB       = firstind(A[op_row_b, col], "Site")
            local op_a, op_b
            if row == op_row_a
                if col == op_col_a
                    op_a     = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
                    op_a     = replaceind!(op_a, H[opcode].site_ind', sA')
                    op_b     = spinI(sB; is_gpu=is_cu)
                elseif col == op_col_b
                    op_b     = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, sB)
                    op_b     = replaceind!(op_b, H[opcode].site_ind', sB')
                    op_a     = spinI(sA; is_gpu=is_cu)
                end
                thisDiag  = op_row_b > op_row_a ? AIH : AIL
                thisDiag *= side ? L.DiagInProgress[op_row_b, opcode] : L.I[op_row_b]
                thisDiag *= A[op_row_b, col] * op_b
                thisDiag *= side ? R.I[op_row_b] : R.DiagInProgress[op_row_b, opcode]
                thisDiag *= dag(A[op_row_b, col])'
                thisDiag *= side ? L.DiagInProgress[op_row_a, opcode] : L.I[op_row_a]
                thisDiag *= ϕ
                thisDiag *= side ? R.I[op_row_a] : R.DiagInProgress[op_row_a, opcode]
                thisDiag *= op_row_b > op_row_a ? AIL : AIH
                thisDiag *= op_a
            elseif row == op_row_b
                if col == op_col_b
                    op_a     = spinI(sA; is_gpu=is_cu)
                    op_b     = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
                    op_b     = replaceind!(op_b, H[opcode].site_ind', sB')
                elseif col == op_col_a
                    op_a     = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
                    op_a     = replaceind!(op_a, H[opcode].site_ind', sA')
                    op_b     = spinI(sB; is_gpu=is_cu)
                end
                thisDiag  = op_row_b > op_row_a ? AIL : AIH
                thisDiag *= side ? L.DiagInProgress[op_row_a, opcode] : L.I[op_row_a]
                thisDiag *= A[op_row_a, col] * op_a
                thisDiag *= side ? R.I[op_row_a] : R.DiagInProgress[op_row_a, opcode]
                thisDiag *= dag(A[op_row_a, col])'
                thisDiag *= side ? L.DiagInProgress[op_row_b, opcode] : L.I[op_row_b]
                thisDiag *= ϕ
                thisDiag *= side ? R.I[op_row_b] : R.DiagInProgress[op_row_b, opcode]
                thisDiag *= op_row_b > op_row_a ? AIH : AIL
                thisDiag *= op_b
            end
            @assert hasinds(inds(thisDiag), AAinds) "inds of thisDiag and AAinds differ!\n$(inds(thisDiag))\n$AAinds\n"
            @assert hasinds(AAinds, inds(thisDiag)) "inds of thisDiag and AAinds differ!\n$(inds(thisDiag))\n$AAinds\n"
        end
        if hasinds(AAinds, inds(thisDiag)) && hasinds(inds(thisDiag), AAinds)
            dTerms[opcode] = thisDiag
        end
    end
    return dTerms
end

function fieldTerms(A::fPEPS,
                    L::Environments,
                    R::Environments,
                    AI,
                    AF,
                    H,
                    row::Int,
                    col::Int,
                    ϕ::ITensor)::Vector{ITensor} 
    Ny, Nx = size(A)
    is_cu  = is_gpu(A) 
    fTerms = Vector{ITensor}(undef, length(H))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0) 
    AAinds = inds(prime(ϕ))
    @inbounds for opcode in 1:length(H)
        thisField = copy(dummy)
        op_row    = H[opcode].sites[1][1]
        if op_row != row
            local F, I
            if op_row > row
                F = AF[:above][opcode][end - row]
                I = row > 1 ? AI[:below][row - 1] : dummy
            else
                F = AF[:below][opcode][row - 1]
                I = row < Ny ? AI[:above][end - row] : dummy
            end
            thisField *= F
            thisField *= L.I[row]
            thisField *= ϕ
            thisField *= R.I[row]
            thisField *= I
            thisField *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
        else
            low_row  = op_row - 1
            high_row = op_row
            AIL = low_row > 0   ? AI[:below][low_row]        : dummy
            AIH = high_row < Ny ? AI[:above][end - high_row] : dummy
            thisField  = AIL
            thisField *= L.I[row]
            thisField *= ϕ
            thisField *= R.I[row]
            thisField *= AIH
            sA = firstind(A[row, col], "Site")
            op = copy(H[opcode].ops[1])
            op = replaceind!(op, H[opcode].site_ind, sA) 
            op = replaceind!(op, H[opcode].site_ind', sA')
            thisField *= op
        end
        @assert hasinds(inds(thisField), AAinds)
        @assert hasinds(AAinds, inds(thisField))
        fTerms[opcode] = thisField
    end
    return fTerms
end

function connectLeftTerms(A::fPEPS,
                          L::Environments,
                          R::Environments,
                          AI, AL, H,
                          row::Int, col::Int,
                          ϕ::ITensor)::Vector{ITensor}
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    lTerms = Vector{ITensor}(undef, length(H))
    AAinds = inds(prime(ϕ))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0) 
    @inbounds for opcode in 1:length(H)
        op_row_b = H[opcode].sites[2][1]
        op_b = copy(H[opcode].ops[2])
        as   = firstind(A[op_row_b, col], "Site")
        op_b = replaceind!(op_b, H[opcode].site_ind, as)
        op_b = replaceind!(op_b, H[opcode].site_ind', as')
        thisHori = is_cu ? cuITensor(1.0) : ITensor(1.0)
        if op_row_b != row
            local ancL, I
            if op_row_b > row
                ancL = AL[:above][opcode][end - row]
                I = row > 1 ? AL[:below][opcode][row - 1] : dummy 
            else
                ancL = AL[:below][opcode][row - 1]
                I = row < Ny ? AL[:above][opcode][end - row] : dummy
            end
            thisHori  = ancL
            thisHori *= L.InProgress[row, opcode]
            thisHori *= ϕ
           thisHori *= R.I[row]
            thisHori *= I
            thisHori *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
        else
            low_row = (op_row_b <= row) ? op_row_b - 1 : row - 1;
            high_row = (op_row_b >= row) ? op_row_b + 1 : row + 1;
            if low_row >= 1
                thisHori *= AL[:below][opcode][low_row]
            end
            thisHori *= R.I[row]
            thisHori *= ϕ
            thisHori *= L.InProgress[row, opcode]
            thisHori *= op_b
            if high_row <= Ny
                thisHori *= AL[:above][opcode][end - high_row + 1]
            end
        end
        @assert hasinds(inds(thisHori), AAinds)
        @assert hasinds(AAinds, inds(thisHori))
        lTerms[opcode] = thisHori
    end
    return lTerms
end

function connectRightTerms(A::fPEPS,
                           L::Environments,
                           R::Environments,
                           AI, AR, H,
                           row::Int, col::Int,
                           ϕ::ITensor)::Vector{ITensor}
    Ny, Nx = size(A)
    is_cu  = is_gpu(A) 
    rTerms = Vector{ITensor}(undef, length(H))
    AAinds = inds(prime(ϕ))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0) 
    @inbounds for opcode in 1:length(H)
        op_row_a = H[opcode].sites[1][1]
        op_a = copy(H[opcode].ops[1])
        as   = firstind(A[op_row_a, col], "Site")
        op_a = replaceind!(op_a, H[opcode].site_ind, as)
        op_a = replaceind!(op_a, H[opcode].site_ind', as')
        thisHori = is_cu ? cuITensor(1.0) : ITensor(1.0)
        if op_row_a != row
            local ancR, I
            if op_row_a > row
                ancR = AR[:above][opcode][end - row]
                I = row > 1 ? AR[:below][opcode][row - 1] : dummy 
            else
                ancR = AR[:below][opcode][row - 1]
                I = row < Ny ? AR[:above][opcode][end - row] : dummy 
            end
            thisHori = ancR
            thisHori *= R.InProgress[row, opcode]
            thisHori *= ϕ
            thisHori *= L.I[row]
            thisHori *= I
            thisHori *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
        else
            low_row = (op_row_a <= row) ? op_row_a - 1 : row - 1;
            high_row = (op_row_a >= row) ? op_row_a + 1 : row + 1;
            if low_row >= 1
                thisHori *= AR[:below][opcode][low_row]
            end
            thisHori *= L.I[row]
            thisHori *= R.InProgress[row, opcode]
            thisHori *= ϕ
            thisHori *= op_a 
            if high_row <= Ny
                thisHori *= AR[:above][opcode][end - high_row + 1]
            end
        end
        @assert hasinds(inds(thisHori), AAinds)
        @assert hasinds(AAinds, inds(thisHori))
        rTerms[opcode] = thisHori
    end
    return rTerms
end

function buildLocalH(A::fPEPS, 
                     L::Environments, R::Environments, 
                     AncEnvs, H, 
                     row::Int, col::Int, 
                     ϕ::ITensor; 
                     verbose::Bool=false)
    field_H_terms = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]...), Vertical)
    term_count    = 1 + length(field_H_terms) + length(vert_H_terms)
    Ny, Nx        = size(A)
    @timeit "build Ns" begin
        N   = buildN(A, L, R, AncEnvs[:I], row, col, ϕ)
    end
    den = scalar(collect(N * dag(ϕ)'))
    if verbose
        println("--- NORM ---")
        println(den)
    end
    local left_H_terms, right_H_terms, diag_left_H_terms, diag_right_H_terms
    if col > 1
        left_H_terms = getDirectional(vcat(H[:, col - 1]...), Horizontal)
        term_count += length(left_H_terms)
        diag_left_H_terms = getDirectional(vcat(H[:, col - 1]...), Diag)
        term_count += length(diag_left_H_terms)
    end
    if col < Nx
        right_H_terms = getDirectional(vcat(H[:, col]...), Horizontal)
        term_count += length(right_H_terms)
        diag_right_H_terms = getDirectional(vcat(H[:, col]...), Diag)
        term_count += length(diag_right_H_terms)
    end
    if 1 < col < Nx
        term_count += 1
    end
    Hs = Vector{ITensor}(undef, term_count)
    term_counter = 1
    @debug "\t\tBuilding I*H and H*I row $row col $col"
    @timeit "build HIs" begin
        HIs = buildHIs(A, L, R, row, col, ϕ)
        Hs[term_counter] = HIs[1]
        if length(HIs) == 2
            Hs[term_counter+1] = HIs[2]
        end
        term_counter += length(HIs)
        if verbose
            println("--- HI TERMS ---")
            for HI in HIs
                println(scalar(HI * dag(ϕ)'))
            end
        end
    end
    @debug "\t\tBuilding vertical H terms row $row col $col"
    @timeit "build vertical terms" begin
        vTs = verticalTerms(A, L, R, AncEnvs[:I], AncEnvs[:V], vert_H_terms, row, col, ϕ) 
        Hs[term_counter:term_counter+length(vTs) - 1] = vTs
        term_counter += length(vTs)
        if verbose
            println( "--- vT TERMS col $col row $row ---")
            for (vv, vT) in enumerate(vTs)
                println(vert_H_terms[vv])
                println(scalar(vT * dag(ϕ)')/den)
            end
        end
    end
    @debug "\t\tBuilding field H terms row $row col $col"
    @timeit "build field terms" begin
        fTs = fieldTerms(A, L, R, AncEnvs[:I], AncEnvs[:F], field_H_terms, row, col, ϕ)
        Hs[term_counter:term_counter+length(fTs) - 1] = fTs[:]
        term_counter += length(fTs)
        if verbose
            println( "--- fT TERMS col $col row $row ---")
            for (ff, fT) in enumerate(fTs)
                println(field_H_terms[ff])
                println(scalar(fT * dag(ϕ)')/den)
            end
        end
    end
    if col > 1
        @debug "\t\tBuilding left H terms row $row col $col"
        @timeit "build left terms" begin
            lTs = connectLeftTerms(A, L, R, AncEnvs[:I], AncEnvs[:L], left_H_terms, row, col, ϕ)
            Hs[term_counter:term_counter+length(lTs) - 1] = lTs[:]
            term_counter += length(lTs)
            if verbose
                println( "--- lT TERMS col $col row $row ---")
                for (ll, lT) in enumerate(lTs)
                    println(left_H_terms[ll])
                    println(scalar(lT * dag(ϕ)')/den)
                end
            end
        end
        @debug "\t\tBuilt left terms"
        @debug "\t\tBuilding left diag H terms row $row col $col"
        @timeit "build diag left terms" begin
            lTs = diagonalTerms(A, L, R, AncEnvs[:I], AncEnvs[:DL], diag_left_H_terms, row, col, ϕ)
            Hs[term_counter:term_counter+length(lTs) - 1] = lTs[:]
            term_counter += length(lTs)
            if verbose
                println( "--- diag lT TERMS col $col row $row ---")
                for (ll, lT) in enumerate(lTs)
                    println(left_H_terms[ll])
                    println(scalar(lT * dag(ϕ)')/den)
                end
            end
        end
        @debug "\t\tBuilt diag left terms"
    end
    if col < Nx
        @debug "\t\tBuilding right H terms row $row col $col"
        @timeit "build right terms" begin
            rTs = connectRightTerms(A, L, R, AncEnvs[:I], AncEnvs[:R], right_H_terms, row, col, ϕ)
            Hs[term_counter:term_counter+length(rTs) - 1] = rTs[:]
            term_counter += length(rTs)
            if verbose
                println( "--- rT TERMS col $col row $row ---")
                for (rr, rT) in enumerate(rTs)
                    println(right_H_terms[rr])
                    println(scalar(rT * dag(ϕ)')/den)
                end
            end
        end
        @debug "\t\tBuilding right diag H terms row $row col $col"
        @timeit "build diag right terms" begin
            rTs = diagonalTerms(A, L, R, AncEnvs[:I], AncEnvs[:DR], diag_right_H_terms, row, col, ϕ)
            Hs[term_counter:term_counter+length(rTs) - 1] = rTs[:]
            term_counter += length(rTs)
            if verbose
                println( "--- diag rT TERMS col $col row $row ---")
                for (rr, rT) in enumerate(rTs)
                    println(right_H_terms[rr])
                    println(scalar(rT * dag(ϕ)')/den)
                end
            end
        end
        @debug "\t\tBuilt right terms"
    end
    return Hs, N
end

function buildAncs(A::fPEPS, L::Environments, R::Environments, H, col::Int)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A) 
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0) 
    @timeit "makeAncIs" begin
        Ia = makeAncillaryIs(A, L, R, col)
        Ib = Vector{ITensor}(undef, Ny)
        Is = (above=Ia, below=Ib)
    end
    @timeit "makeAncVs" begin
        vH = getDirectional(vcat(H[:, col]...), Vertical)
        Va = makeAncillaryVs(A, L, R, vH, col)
        Vb = [Vector{ITensor}() for ii in 1:length(Va)]
        Vs = (above=Va, below=Vb)
    end
    @timeit "makeAncFs" begin
        fH = getDirectional(vcat(H[:, col]...), Field)
        Fa = makeAncillaryFs(A, L, R, fH, col)
        Fb = [Vector{ITensor}() for ii in 1:length(Fa)]
        Fs = (above=Fa, below=Fb)
    end
    Ls = (above=Vector{ITensor}(), below=Vector{ITensor}()) 
    Rs = (above=Vector{ITensor}(), below=Vector{ITensor}()) 
    DLs = (above=Vector{ITensor}(), below=Vector{ITensor}()) 
    DRs = (above=Vector{ITensor}(), below=Vector{ITensor}()) 
    if col > 1
        @timeit "makeAncLs" begin
            lH = getDirectional(vcat(H[:, col-1]...), Horizontal)
            La = makeAncillarySide(A, L, R, lH, col, :left)
            Lb = [Vector{ITensor}() for ii in  1:length(La)]
            Ls = (above=La, below=Lb)
        end
        @timeit "makeAncDLs" begin
            lH = getDirectional(vcat(H[:, col-1]...), Diag)
            La = makeAncillarySideDiag(A, L, R, lH, col, :left)
            Lb = [Vector{ITensor}() for ii in  1:length(La)]
            DLs = (above=La, below=Lb)
        end
    end
    if col < Nx
        @timeit "makeAncRs" begin
            rH = getDirectional(vcat(H[:, col]...), Horizontal)
            Ra = makeAncillarySide(A, R, L, rH, col, :right)
            Rb = [Vector{ITensor}() for ii in  1:length(Ra)]
            Rs = (above=Ra, below=Rb)
        end
        @timeit "makeAncRs" begin
            rH = getDirectional(vcat(H[:, col]...), Diag)
            Ra = makeAncillarySideDiag(A, R, L, rH, col, :right)
            Rb = [Vector{ITensor}() for ii in  1:length(Ra)]
            DRs = (above=Ra, below=Rb)
        end
    end
    Ancs = (I=Is, V=Vs, F=Fs, L=Ls, R=Rs, DL=DLs, DR=DRs)
    return Ancs
end

function updateAncs(A::fPEPS, 
                    L::Environments, R::Environments, 
                    AncEnvs, H, 
                    row::Int, col::Int)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A) 
   
    Is, Vs, Fs, Ls, Rs, DLs, DRs = AncEnvs
    @debug "\tUpdating ancillary identity terms for col $col row $row"
    Ib = updateAncillaryIs(A, Is[:below], L, R, row, col)
    Is = (above=Is[:above], below=Ib)

    @debug "\tUpdating ancillary vertical terms for col $col row $row"
    vH = getDirectional(vcat(H[:, col]...), Vertical)
    Vb = updateAncillaryVs(A, Vs[:below], Ib, L, R, vH, row, col)  
    Vs = (above=Vs[:above], below=Vb)

    @debug "\tUpdating ancillary field terms for col $col row $row"
    fH = getDirectional(vcat(H[:, col]...), Field)
    Fb = updateAncillaryFs(A, Fs[:below], Ib, L, R, fH, row, col)  
    Fs = (above=Fs[:above], below=Fb)

    if col > 1
        @debug "\tUpdating ancillary left terms for col $col row $row"
        lH = getDirectional(vcat(H[:, col-1]...), Horizontal)
        Lb = updateAncillarySide(A, Ls[:below], Ib, L, R, lH, row, col, :left)
        Ls = (above=Ls[:above], below=Lb)
        @debug "\tUpdating ancillary diag left terms for col $col row $row"
        lH = getDirectional(vcat(H[:, col-1]...), Diag)
        DLb = updateAncillarySideDiag(A, DLs[:below], Ib, L, R, lH, row, col, :left)
        DLs = (above=DLs[:above], below=DLb)
    end
    if col < Nx
        @debug "\tUpdating ancillary right terms for col $col row $row"
        rH = getDirectional(vcat(H[:, col]...), Horizontal)
        Rb = updateAncillarySide(A, Rs[:below], Ib, R, L, rH, row, col, :right)
        Rs = (above=Rs[:above], below=Rb)
        @debug "\tUpdating ancillary diag right terms for col $col row $row"
        rH = getDirectional(vcat(H[:, col]...), Diag)
        DRb = updateAncillarySideDiag(A, DRs[:below], Ib, R, L, rH, row, col, :right)
        DRs = (above=DRs[:above], below=DRb)
    end
    Ancs = (I=Is, V=Vs, F=Fs, L=Ls, R=Rs, DL=DLs, DR=DRs)
    return Ancs
end

struct ITensorMap
  A::fPEPS
  H::Matrix{Vector{Operator}}
  L::Environments
  R::Environments
  AncEnvs::NamedTuple
  row::Int
  col::Int
end
Base.eltype(M::ITensorMap)  = eltype(M.A)
Base.size(M::ITensorMap)    = ITensors.dim(M.A[M.row, M.col])
function (M::ITensorMap)(v::ITensor) 
    Hs, N  = buildLocalH(M.A, M.L, M.R, M.AncEnvs, M.H, M.row, M.col, v)
    localH = sum(Hs)
    return noprime(localH)
end

function optimizeLocalH(A::fPEPS,
                        L::Environments, R::Environments, 
                        AncEnvs, H, 
                        row::Int, col::Int; 
                        kwargs...)
    Ny, Nx        = size(A)
    is_cu         = is_gpu(A) 
    field_H_terms = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]...), Vertical)
    @debug "\tBuilding H for col $col row $row"
    @timeit "build H" begin
        Hs, N = buildLocalH(A, L, R, AncEnvs, H, row, col, A[row, col])
    end
    initial_N = real(scalar(collect(N * dag(A[row, col])')))
    @timeit "sum H terms" begin
        localH = sum(Hs)
    end
    initial_E = real(scalar(collect(deepcopy(localH) * dag(A[row, col])')))
    @info "Initial energy at row $row col $col : $(initial_E/(initial_N*Nx*Ny)) and norm : $initial_N"
    @debug "\tBeginning davidson for col $col row $row"
    mapper   = ITensorMap(A, H, L, R, AncEnvs, row, col)
    λ, new_A = davidson(mapper, A[row, col]; maxiter=1, kwargs...)
    new_E    = λ #real(scalar(collect(new_A * localH * dag(new_A)')))
    N        = buildN(A, L, R, AncEnvs[:I], row, col, new_A)
    new_N    = real(scalar(collect(N * dag(new_A)')))
    @info "Optimized energy at row $row col $col : $(new_E/(new_N*Nx*Ny)) and norm : $new_N"
    #println("Optimized energy at row $row col $col : $(new_E/(new_N*Nx*Ny)) and norm : $new_N")
    @timeit "restore intraColumnGauge" begin
        if row < Ny
            @debug "\tRestoring intraColumnGauge for col $col row $row"
            cmb_is     = IndexSet(firstind(A[row, col], "Site"))
            if col > 1
                cmb_is = IndexSet(cmb_is..., commonind(A[row, col], A[row, col - 1]))
            end
            if col < Nx
                cmb_is = IndexSet(cmb_is..., commonind(A[row, col], A[row, col + 1]))
            end
            cmb     = combiner(cmb_is, tags="CMB")
            ci      = combinedind(cmb)
            Lis     = IndexSet(ci) #cmb_is 
            if row > 1
                Lis = IndexSet(Lis..., commonind(A[row, col], A[row - 1, col]))
            end
            old_ci  = commonind(A[row, col], A[row+1, col])
            svdA    = new_A*cmb
            Ris     = uniqueinds(inds(svdA), Lis) 
            U, S, V = svd(svdA, Ris; kwargs...)
            new_ci  = commonind(V, S)
            replaceind!(V, new_ci, old_ci)
            A[row, col]    = V*cmb 
            newU = S*U*A[row+1, col]
            replaceind!(newU, new_ci, old_ci)
            A[row+1, col] = newU
            if row < Ny - 1
                nI    = spinI(firstind(A[row+1, col], "Site"); is_gpu=is_cu)
                newAA = AncEnvs[:I][:above][end - row - 1]
                newAA *= L.I[row+1]
                newAA *= A[row+1, col] * nI
                newAA *= R.I[row+1]
                newAA *= dag(A[row+1, col])'
                AncEnvs[:I][:above][end - row] = newAA
            end
        else
            A[row, col] = initial_N > 0 ? new_A/√new_N : new_A
        end
    end
    return A, AncEnvs
end

function sweepColumn(A::fPEPS, 
                     L::Environments, R::Environments, 
                     H, 
                     col::Int; 
                     kwargs...)
    Ny, Nx = size(A)
    @timeit "intraColumnGauge" begin
        A = intraColumnGauge(A, col; kwargs...)
    end
    if col == div(Nx,2)
        L_s = buildLs(A, H; kwargs...)
        R_s = buildRs(A, H; kwargs...)
        EAncEnvs = buildAncs(A, L_s[col - 1], R_s[col + 1], H, col)
        N, E = measureEnergy(A, L_s[col - 1], R_s[col + 1], EAncEnvs, H, 1, col)
        println("Energy at MID: ", E/(Nx*Ny))
        println("Nx: ", Nx)
    end
    @debug "Beginning buildAncs for col $col"
    two_site_vertical::Bool   = get(kwargs, :two_site_vertical, false)
    if two_site_vertical
        AncEnvs = buildAncs(A, L, R, H, col)
        @inbounds for row in 1:Ny-1
            if row > 1
                @timeit "updateAncs" begin
                    AncEnvs = updateAncs(A, L, R, AncEnvs, H, row-1, col)
                end
            end
            @debug "Beginning optimizing H for col $col"
            A, AncEnvs = optimizeLocalHTwoSiteVertical(A, L, R, AncEnvs, H, row, col; kwargs...)
        end
    else
        AncEnvs = buildAncs(A, L, R, H, col)
        @inbounds for row in 1:Ny
            if row > 1
                @timeit "updateAncs" begin
                    AncEnvs = updateAncs(A, L, R, AncEnvs, H, row-1, col)
                end
            end
            @debug "Beginning optimizing H for col $col"
            A, AncEnvs = optimizeLocalH(A, L, R, AncEnvs, H, row, col; kwargs...)
        end
    end
    return A
end

