function buildNTwoSiteHorizontal(Aj::Vector{ITensor},
                                 L::Environments,
                                 R::Environments,
                                 IEnvs,
                                 row::Int,
                                 col::Int,
                                 ncol::Int,
                                 ϕ::ITensor)::ITensor
    Ny       = length(Aj) 
    is_cu    = is_gpu(Aj)
    N        = is_cu ? cuITensor(1.0) : ITensor(1.0)
    for si in inds(Aj[row], "Site")
        N   *= spinI(si; is_gpu=is_cu)
    end
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

function buildHIedgeTwoSiteHorizontal(Aj::Vector{ITensor},
                                      E::Environments,
                                      row::Int,
                                      col::Int,
                                      ncol::Int,
                                      side::Symbol,
                                      ϕ::ITensor)
    Ny     = length(Aj)
    is_cu  = is_gpu(Aj)
    IH_a   = is_cu ? cuITensor(1.0) : ITensor(1.0)
    IH_b   = is_cu ? cuITensor(1.0) : ITensor(1.0)
    IH     = is_cu ? cuITensor(1.0) : ITensor(1.0)
    @inbounds for work_row in 1:row-1
        IH_b *= Aj[work_row] * E.H[work_row]
        IH_b *= dag(prime(Aj[work_row], "Link"))
    end
    @inbounds for work_row in row+1:Ny
        IH_a *= Aj[work_row] * E.H[work_row]
        IH_a *= dag(prime(Aj[work_row], "Link"))
    end
    IH *= E.H[row]
    IH *= IH_b
    op  = is_cu ? cuITensor(1.0) : ITensor(1.0)
    for si in inds(Aj[row], "Site")
        op *= spinI(si; is_gpu=is_cu)
    end
    IH *= ϕ
    IH *= op
    IH *= IH_a
    AAinds = inds(prime(ϕ))
    @assert hasinds(inds(IH), AAinds) "inds(IH): $(inds(IH)),\n inds(AA): $AAinds"
    @assert hasinds(AAinds, inds(IH)) "inds(IH): $(inds(IH)),\n inds(AA): $AAinds"
    return (IH,)
end

function buildHIsTwoSiteHorizontal(Aj::Vector{ITensor},
                                   L::Environments,
                                   R::Environments,
                                   row::Int,
                                   col::Int,
                                   ncol::Int,
                                   ϕ::ITensor;
                                   edge::Symbol=:none)
    Ny     = length(Aj)
    is_cu  = is_gpu(Aj) 
    @timeit "buildHIedge" begin
        if edge == :left 
            return buildHIedgeTwoSiteHorizontal(Aj, R, row, col, ncol, :left, ϕ)
        end
        if edge == :right 
            return buildHIedgeTwoSiteHorizontal(Aj, L, row, col, ncol, :right, ϕ)
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
            tL     = Aj[work_row]*L.H[work_row]
            HLI_b *= tL
            HLI_b *= R.I[work_row]
            HLI_b *= dag(prime(Aj[work_row], "Link"))
            tR     = Aj[work_row]*R.H[work_row]
            IHR_b *= tR 
            IHR_b *= L.I[work_row]
            IHR_b *= dag(prime(Aj[work_row], "Link"))
        end
    end
    @timeit "build HI upper" begin
        @inbounds for work_row in reverse(row+1:Ny)
            tL     = Aj[work_row]*L.H[work_row]
            HLI_a *= tL 
            HLI_a *= R.I[work_row]
            HLI_a *= dag(prime(Aj[work_row], "Link"))
            tR     = Aj[work_row]*R.H[work_row]
            IHR_a *= tR 
            IHR_a *= L.I[work_row]
            IHR_a *= dag(prime(Aj[work_row], "Link"))
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
        HLI  *= HLI_a
        IHR  *= IHR_a
        op    = is_cu ? cuITensor(1.0) : ITensor(1.0)
        for si in inds(Aj[row], "Site")
            op   *= spinI(si; is_gpu=is_cu)
        end
        HLI  *= op
        IHR  *= op
    end
    AAinds = inds(prime(ϕ))
    @assert hasinds(inds(IHR), AAinds)
    @assert hasinds(inds(HLI), AAinds)
    @assert hasinds(AAinds, inds(IHR))
    @assert hasinds(AAinds, inds(HLI))
    return (HLI, IHR)
end

function verticalTermsTwoSiteHorizontal(Aj::Vector{ITensor},
                                        L::Environments,
                                        R::Environments,
                                        AI,
                                        AV,
                                        H,
                                        row::Int,
                                        col::Int,
                                        ncol::Int,
                                        ϕ::ITensor)::Vector{ITensor}
    Ny     = length(Aj)
    is_cu  = is_gpu(Aj)
    vTerms = Vector{ITensor}(undef, length(H))
    AAinds = inds(prime(ϕ))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0)
    @inbounds for opcode in 1:length(H)
        thisVert = dummy
        op_row_a = H[opcode].sites[1][1]
        op_col_a = H[opcode].sites[1][2]
        op_row_b = H[opcode].sites[2][1]
        op_col_b = H[opcode].sites[2][2]
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
            for si in inds(Aj[row], "Site")
                thisVert *= spinI(si; is_gpu=is_cu)
            end
        elseif row == op_row_a
            low_row  = op_row_a - 1
            high_row = op_row_b
            AIL = low_row > 0   ? AI[:below][low_row]        : dummy
            AIH = high_row < Ny ? AI[:above][end - high_row] : dummy
            if col == op_col_a
                sA   = firstind(Aj[op_row_a], "Site,c$col")
                op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
                op_a = replaceind!(op_a, H[opcode].site_ind', sA')
                sB   = firstind(Aj[op_row_b], "Site,c$col")
                op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
                op_b = replaceind!(op_b, H[opcode].site_ind', sB')
                op_a *= spinI(firstind(Aj[op_row_a], "Site,c$ncol"); is_gpu=is_cu)
                op_b *= spinI(firstind(Aj[op_row_b], "Site,c$ncol"); is_gpu=is_cu)
            else
                sA   = firstind(Aj[op_row_a], "Site,c$ncol")
                op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
                op_a = replaceind!(op_a, H[opcode].site_ind', sA')
                sB   = firstind(Aj[op_row_b], "Site,c$ncol")
                op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
                op_b = replaceind!(op_b, H[opcode].site_ind', sB')
                op_a *= spinI(firstind(Aj[op_row_a], "Site,c$col"); is_gpu=is_cu)
                op_b *= spinI(firstind(Aj[op_row_b], "Site,c$col"); is_gpu=is_cu)
            end
            thisVert  = AIH
            thisVert *= L.I[op_row_b]
            thisVert *= Aj[op_row_b] * op_b
            thisVert *= R.I[op_row_b]
            thisVert *= dag(Aj[op_row_b])'
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
            if op_col_a == col
                sA   = firstind(Aj[op_row_a], "Site,c$col")
                op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
                op_a = replaceind!(op_a, H[opcode].site_ind', sA')
                sB   = firstind(Aj[op_row_b], "Site,c$col")
                op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
                op_b = replaceind!(op_b, H[opcode].site_ind', sB')
                op_a *= spinI(firstind(Aj[op_row_a], "Site,c$ncol"); is_gpu=is_cu)
                op_b *= spinI(firstind(Aj[op_row_b], "Site,c$ncol"); is_gpu=is_cu)
            else
                sA   = firstind(Aj[op_row_a], "Site,c$ncol")
                op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
                op_a = replaceind!(op_a, H[opcode].site_ind', sA')
                sB   = firstind(Aj[op_row_b], "Site,c$ncol")
                op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
                op_b = replaceind!(op_b, H[opcode].site_ind', sB')
                op_a *= spinI(firstind(Aj[op_row_a], "Site,c$col"); is_gpu=is_cu)
                op_b *= spinI(firstind(Aj[op_row_b], "Site,c$col"); is_gpu=is_cu)
            end
            thisVert  = AIL
            thisVert *= L.I[op_row_a]
            thisVert *= Aj[op_row_a] * op_a
            thisVert *= R.I[op_row_a]
            thisVert *= dag(Aj[op_row_a])'
            thisVert *= L.I[op_row_b]
            thisVert *= ϕ
            thisVert *= R.I[op_row_b]
            thisVert *= AIH
            thisVert *= op_b
        end
        @assert hasinds(inds(thisVert), AAinds) "inds of thisVert and AAinds differ!\n$(inds(thisVert))\n$AAinds\n"
        @assert hasinds(AAinds, inds(thisVert)) "inds of thisVert and AAinds differ!\n$(inds(thisVert))\n$AAinds\n"
        if hasinds(AAinds, inds(thisVert)) && hasinds(inds(thisVert), AAinds)
            vTerms[opcode] = thisVert
        end
    end
    return vTerms
end

function interiorTermsTwoSiteHorizontal(Aj::Vector{ITensor},
                                        L::Environments,
                                        R::Environments,
                                        AI,
                                        AInt,
                                        H,
                                        row::Int,
                                        col::Int,
                                        ncol::Int,
                                        ϕ::ITensor)::Vector{ITensor}
    Ny     = length(Aj)
    is_cu  = is_gpu(Aj) 
    intTerms = Vector{ITensor}(undef, length(H))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0) 
    AAinds = inds(prime(ϕ))
    @inbounds for opcode in 1:length(H)
        thisInter = copy(dummy)
        op_row    = H[opcode].sites[1][1]
        op_col_a  = H[opcode].sites[1][2]
        op_col_b  = H[opcode].sites[2][2]
        if op_row != row
            local IT, I
            if op_row > row
                IT = AInt[:above][opcode][end - row]
                I  = row > 1 ? AI[:below][row - 1] : dummy
            else
                IT = AInt[:below][opcode][row - 1]
                I = row < Ny ? AI[:above][end - row] : dummy
            end
            thisInter *= IT 
            thisInter *= L.I[row]
            thisInter *= ϕ
            thisInter *= R.I[row]
            thisInter *= I
            for si in inds(Aj[row], "Site")
                thisInter *= spinI(si; is_gpu=is_cu)
            end
            @assert hasinds(inds(thisInter), AAinds)
            @assert hasinds(AAinds, inds(thisInter))
        else
            low_row  = op_row - 1
            high_row = op_row
            AIL = low_row > 0   ? AI[:below][low_row]        : dummy
            AIH = high_row < Ny ? AI[:above][end - high_row] : dummy
            thisInter  = AIL
            thisInter *= L.I[row]
            thisInter *= ϕ
            thisInter *= R.I[row]
            thisInter *= AIH
            if op_col_a == col
                sA   = firstind(Aj[row], "Site,c$col")
                op_a = copy(H[opcode].ops[1])
                op_a = replaceind!(op_a, H[opcode].site_ind, sA) 
                op_a = replaceind!(op_a, H[opcode].site_ind', sA')
                sB   = firstind(Aj[row], "Site,c$ncol")
                op_b = copy(H[opcode].ops[2])
                op_b = replaceind!(op_b, H[opcode].site_ind, sB) 
                op_b = replaceind!(op_b, H[opcode].site_ind', sB')
                op = op_a * op_b
            else
                sA   = firstind(Aj[row], "Site,c$col")
                op_a = copy(H[opcode].ops[2])
                op_a = replaceind!(op_a, H[opcode].site_ind, sA) 
                op_a = replaceind!(op_a, H[opcode].site_ind', sA')
                sB   = firstind(Aj[row], "Site,c$ncol")
                op_b = copy(H[opcode].ops[1])
                op_b = replaceind!(op_b, H[opcode].site_ind, sB) 
                op_b = replaceind!(op_b, H[opcode].site_ind', sB')
                op = op_a * op_b
            end
            thisInter *= op
            @assert hasinds(inds(thisInter), AAinds)
            @assert hasinds(AAinds, inds(thisInter))
        end
        intTerms[opcode] = thisInter
    end
    return intTerms
end

function fieldTermsTwoSiteHorizontal(Aj::Vector{ITensor},
                                     L::Environments,
                                     R::Environments,
                                     AI,
                                     AF,
                                     H,
                                     row::Int,
                                     col::Int,
                                     ncol::Int,
                                     ϕ::ITensor)::Vector{ITensor}
    Ny     = length(Aj)
    is_cu  = is_gpu(Aj)
    fTerms = Vector{ITensor}(undef, length(H))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0) 
    AAinds = inds(prime(ϕ))
    @inbounds for opcode in 1:length(H)
        thisField = copy(dummy)
        op_row    = H[opcode].sites[1][1]
        op_col    = H[opcode].sites[1][2]
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
            for si in inds(Aj[row], "Site")
                thisField *= spinI(si; is_gpu=is_cu)
            end
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
            if op_col == col
                sA = firstind(Aj[row], "Site,c$col")
                op = copy(H[opcode].ops[1])
                op = replaceind!(op, H[opcode].site_ind, sA) 
                op = replaceind!(op, H[opcode].site_ind', sA')
                op *= spinI(firstind(Aj[row], "Site,c$ncol"); is_gpu=is_cu)
            else
                sA = firstind(Aj[row], "Site,c$ncol")
                op = copy(H[opcode].ops[1])
                op = replaceind!(op, H[opcode].site_ind, sA) 
                op = replaceind!(op, H[opcode].site_ind', sA')
                op *= spinI(firstind(Aj[row], "Site,c$col"); is_gpu=is_cu)
            end
            thisField *= op
        end
        @assert hasinds(inds(thisField), AAinds)
        @assert hasinds(AAinds, inds(thisField))
        fTerms[opcode] = thisField
    end
    return fTerms
end

function connectLeftTermsTwoSiteHorizontal(Aj::Vector{ITensor},
                                           L::Environments,
                                           R::Environments,
                                           AI, AL, H,
                                           row::Int, col::Int, ncol::Int,
                                           ϕ::ITensor)::Vector{ITensor}
    Ny     = length(Aj)
    is_cu  = is_gpu(Aj)
    lTerms = Vector{ITensor}(undef, length(H))
    AAinds = inds(prime(ϕ))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0)
    lcol, rcol = extrema([col, ncol])
    @inbounds for opcode in 1:length(H)
        op_row_b = H[opcode].sites[2][1]
        op_b     = copy(H[opcode].ops[2])
        as       = firstind(Aj[op_row_b], "Site,c$lcol")
        op_b     = replaceind!(op_b, H[opcode].site_ind, as)
        op_b     = replaceind!(op_b, H[opcode].site_ind', as')
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
            for si in inds(Aj[row], "Site")
                thisHori *= spinI(si; is_gpu=is_cu)
            end
        else
            low_row  = (op_row_b <= row) ? op_row_b - 1 : row - 1;
            high_row = (op_row_b >= row) ? op_row_b + 1 : row + 1;
            if low_row >= 1
                thisHori *= AL[:below][opcode][low_row]
            end
            thisHori *= R.I[row]
            thisHori *= ϕ
            thisHori *= L.InProgress[row, opcode]
            thisHori *= op_b
            thisHori *= spinI(firstind(Aj[row], "Site,c$rcol"); is_gpu=is_cu)
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

function connectRightTermsTwoSiteHorizontal(Aj::Vector{ITensor},
                                            L::Environments,
                                            R::Environments,
                                            AI, AR, H,
                                            row::Int, col::Int, ncol::Int,
                                            ϕ::ITensor)::Vector{ITensor}
    Ny     = length(Aj)
    is_cu  = is_gpu(Aj) 
    rTerms = Vector{ITensor}(undef, length(H))
    AAinds = inds(prime(ϕ))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0)
    lcol, rcol = extrema([col, ncol])
    @inbounds for opcode in 1:length(H)
        op_row_a = H[opcode].sites[1][1]
        op_a     = copy(H[opcode].ops[1])
        as       = firstind(Aj[op_row_a], "Site,c$rcol")
        op_a     = replaceind!(op_a, H[opcode].site_ind, as)
        op_a     = replaceind!(op_a, H[opcode].site_ind', as')
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
            for si in inds(Aj[row], "Site")
                thisHori *= spinI(si; is_gpu=is_cu)
            end
        else
            low_row  = (op_row_a <= row) ? op_row_a - 1 : row - 1;
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
            thisHori *= spinI(firstind(Aj[row], "Site,c$lcol"); is_gpu=is_cu)
        end
        @assert hasinds(inds(thisHori), AAinds)
        @assert hasinds(AAinds, inds(thisHori))
        rTerms[opcode] = thisHori
    end
    return rTerms
end

function buildLocalHTwoSiteHorizontal(A::Vector{ITensor},
                                      L::Environments, R::Environments,
                                      AncEnvs, H,
                                      row::Int, col::Int, ncol::Int,
                                      ϕ::ITensor;
                                      verbose::Bool=false)
    field_H_terms = getDirectional(vcat(H[:, col]..., H[:, ncol]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]..., H[:, ncol]...), Vertical)
    term_count    = 1 + length(field_H_terms) + length(vert_H_terms)
    Ny            = length(A)
    Nx            = size(H, 2)
    @timeit "build Ns" begin
        N   = buildNTwoSiteHorizontal(A, L, R, AncEnvs[:I], row, col, ncol, ϕ)
    end
    den = scalar(collect(N * dag(ϕ)'))
    if verbose
        println("--- NORM ---")
        println(den)
    end
    local left_H_terms, right_H_terms
    mincol, maxcol    = extrema([col, ncol])
    interior_H_terms  = getDirectional(vcat(H[:, mincol]...), Horizontal)
    term_count       += length(interior_H_terms)
    if mincol > 1
        left_H_terms  = getDirectional(vcat(H[:, mincol - 1]...), Horizontal)
        term_count   += length(left_H_terms)
    end
    if maxcol < Nx
        right_H_terms = getDirectional(vcat(H[:, maxcol]...), Horizontal)
        term_count   += length(right_H_terms)
    end
    if 1 < col < Nx && 1 < ncol < Nx
        term_count += 1
    end
    Hs = Vector{ITensor}(undef, term_count)
    term_counter = 1
    @debug "\t\tBuilding I*H and H*I row $row col $col"
    @timeit "build HIs" begin
        edge = (1 < col < Nx && 1 < ncol < Nx) ? :none : ((col == 1 || ncol == 1) ? :left : :right) 
        HIs = buildHIsTwoSiteHorizontal(A, L, R, row, col, ncol, ϕ; edge=edge)
        Hs[term_counter] = HIs[1]
        if length(HIs) == 2
            Hs[term_counter+1] = HIs[2]
        end
        term_counter += length(HIs)
        if verbose
            println("--- HI TERMS ---")
            for HI in HIs
                println(scalar(HI * dag(ϕ)')/den)
            end
        end
    end
    @debug "\t\tBuilding vertical H terms row $row col $col"
    @timeit "build vertical terms" begin
        vTs = verticalTermsTwoSiteHorizontal(A, L, R, AncEnvs[:I], AncEnvs[:V], vert_H_terms, row, col, ncol, ϕ) 
        Hs[term_counter:term_counter+length(vTs) - 1] = vTs
        term_counter += length(vTs)
        if verbose
            println( "--- vT TERMS ---")
            for vT in vTs
                println(scalar(vT * dag(ϕ)')/den)
            end
        end
    end
    @debug "\t\tBuilding field H terms row $row col $col"
    @timeit "build field terms" begin
        fTs = fieldTermsTwoSiteHorizontal(A, L, R, AncEnvs[:I], AncEnvs[:F], field_H_terms, row, col, ncol, ϕ)
        Hs[term_counter:term_counter+length(fTs) - 1] = fTs[:]
        term_counter += length(fTs)
        if verbose
            println( "--- fT TERMS ---")
            for fT in fTs
                println(scalar(fT * dag(ϕ)')/den)
            end
        end
    end
    @debug "\t\tBuilding interior H terms row $row col $col"
    @timeit "build interior terms" begin
        iTs = interiorTermsTwoSiteHorizontal(A, L, R, AncEnvs[:I], AncEnvs[:IT], interior_H_terms, row, col, ncol, ϕ)
        Hs[term_counter:term_counter+length(iTs) - 1] = iTs[:]
        term_counter += length(iTs)
        if verbose
            println( "--- interiorT TERMS ---")
            for iT in iTs
                println(scalar(iT * dag(ϕ)')/den)
            end
        end
    end
    if mincol > 1
        @debug "\t\tBuilding left H terms row $row col $col"
        @timeit "build left terms" begin
            lTs = connectLeftTermsTwoSiteHorizontal(A, L, R, AncEnvs[:I], AncEnvs[:L], left_H_terms, row, col, ncol, ϕ)
            Hs[term_counter:term_counter+length(lTs) - 1] = lTs[:]
            term_counter += length(lTs)
            if verbose
                println( "--- lT TERMS ---")
                for lT in lTs
                    println(scalar(lT * dag(ϕ)')/den)
                end
            end
        end
        @debug "\t\tBuilt left terms"
    end
    if maxcol < Nx
        @debug "\t\tBuilding right H terms row $row col $col"
        @timeit "build right terms" begin
            rTs = connectRightTermsTwoSiteHorizontal(A, L, R, AncEnvs[:I], AncEnvs[:R], right_H_terms, row, col, ncol, ϕ)
            Hs[term_counter:term_counter+length(rTs) - 1] = rTs[:]
            term_counter += length(rTs)
            if verbose
                println( "--- rT TERMS ---")
                for rT in rTs
                    println(scalar(rT * dag(ϕ)')/den)
                end
            end
        end
        @debug "\t\tBuilt right terms"
    end
    return Hs, N
end

function intraColumnGaugeHorizontal(A::Vector{ITensor}; kwargs...)::Vector{ITensor}
    Ny = length(A)
    @inbounds for row in reverse(2:Ny)
        @debug "\tBeginning intraColumnGauge for col $col row $row"
        cmb_is     = IndexSet(inds(A[row], "Site"))
        cmb_is     = IndexSet(cmb_is..., inds(A[row], "Link,r")...)
        cmb = combiner(cmb_is, tags="CMB")
        Lis = IndexSet(combinedind(cmb)) #cmb_is
        if row < Ny
            Lis = IndexSet(Lis..., commonind(A[row], A[row + 1]))
        end
        old_ci    = commonind(A[row], A[row-1])
        Ac        = A[row]*cmb
        U, S, V   = svd(Ac, Lis; mindim=dim(old_ci), maxdim=dim(old_ci), cutoff=0.0, utags=tags(old_ci))
        A[row]    = U*cmb
        A[row-1] *= (S*V)
        #new_ci    = commonind(A[row], A[row-1])
        #A[row]    = replaceind!(A[row], new_ci, old_ci)
        #A[row-1]  = replaceind!(A[row-1], new_ci, old_ci)
    end
    return A
end

struct ITensorMapTwoSiteHorizontal
  A::Vector{ITensor}
  H::Matrix{Vector{Operator}}
  L::Environments
  R::Environments
  AncEnvs::NamedTuple
  row::Int
  col::Int
  ncol::Int
end
Base.eltype(M::ITensorMapTwoSiteHorizontal)  = eltype(M.A[M.row])
Base.size(M::ITensorMapTwoSiteHorizontal)    = ITensors.dim(M.A[M.row])
function (M::ITensorMapTwoSiteHorizontal)(v::ITensor) 
    Hs, N  = buildLocalHTwoSiteHorizontal(M.A, M.L, M.R, M.AncEnvs, M.H, M.row, M.col, M.ncol, v)
    localH = sum(Hs)
    return noprime(localH)
end

function optimizeLocalHTwoSiteHorizontal(A::Vector{ITensor},
                                         L::Environments, R::Environments, 
                                         AncEnvs, H, 
                                         row::Int, col::Int, ncol::Int;
                                         kwargs...)
    Ny            = length(A)
    Nx            = size(H, 2)
    is_cu         = is_gpu(A)
    field_H_terms = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]...), Vertical)
    @debug "\tBuilding H for col $col row $row"
    @timeit "build H" begin
        Hs, N = buildLocalHTwoSiteHorizontal(A, L, R, AncEnvs, H, row, col, ncol, A[row])
    end
    initial_N = real(scalar(collect(N * dag(A[row])')))
    @timeit "sum H terms" begin
        localH = sum(Hs)
    end
    initial_E = real(scalar(collect(deepcopy(localH) * dag(A[row])')))
    @info "Initial energy at row $row col $col : $(initial_E/(initial_N*Nx*Ny)) and norm : $initial_N"
    #println("Initial energy at row $row col $col : $(initial_E/(initial_N*Nx*Ny)) and norm : $initial_N")
    @debug "\tBeginning davidson for col $col row $row"
    mapper   = ITensorMapTwoSiteHorizontal(A, H, L, R, AncEnvs, row, col, ncol)
    λ, new_A = davidson(mapper, A[row]; maxiter=2, kwargs...)
    new_E    = λ
    N        = buildNTwoSiteHorizontal(A, L, R, AncEnvs[:I], row, col, ncol, new_A)
    new_N    = real(scalar(collect(N * dag(new_A)')))
    @info "Optimized energy at row $row col $col : $(new_E/(new_N*Nx*Ny)) and norm : $new_N"
    #println("Optimized energy at row $row col $col : $(new_E/(new_N*Nx*Ny)) and norm : $new_N")
    @timeit "restore intraColumnGauge" begin
        if row < Ny
            @debug "\tRestoring intraColumnGauge for col $col row $row"
            cmb_is  = IndexSet(inds(A[row], "Site"))
            cmb_is  = IndexSet(cmb_is..., inds(A[row], "Link,r")...)
            cmb     = combiner(cmb_is, tags="CMB")
            ci      = combinedind(cmb)
            Lis     = IndexSet(ci)
            if row > 1
                Lis = IndexSet(Lis..., commonind(A[row], A[row - 1]))
            end
            old_ci  = commonind(A[row], A[row+1])
            svdA    = new_A*cmb
            Ris     = uniqueinds(inds(svdA), Lis)
            U, S, V = svd(svdA, Ris; mindim=dim(old_ci), maxdim=dim(old_ci), vtags=tags(old_ci))
            new_ci  = commonind(V, S)
            #replaceind!(V, new_ci, old_ci)
            A[row]  = V*cmb
            newU    = S*U*A[row+1]
            #replaceind!(newU, new_ci, old_ci)
            A[row+1] = newU
            if row < Ny - 1
                nI    = is_cu ? cuITensor(1.0) : ITensor(1.0)
                for si in inds(A[row+1], "Site")
                    nI    *= spinI(si; is_gpu=is_cu)
                end
                newAA  = AncEnvs[:I][:above][end - row - 1]
                newAA *= L.I[row+1]
                newAA *= A[row+1] * nI
                newAA *= R.I[row+1]
                newAA *= dag(A[row+1])'
                AncEnvs[:I][:above][end - row] = newAA
            end
        else
            A[row] = initial_N > 0 ? new_A/√new_N : new_A
        end
    end
    return A, AncEnvs
end

function buildAncsHorizontal(A::fPEPS, Aj::Vector{ITensor}, L::Environments, R::Environments, H, col::Int, ncol::Int)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0)
    @timeit "makeAncIs" begin
        Ia = makeAncillaryIsHorizontal(A, Aj, L, R, col, ncol)
        Ib = Vector{ITensor}(undef, Ny)
        Is = (above=Ia, below=Ib)
    end
    @timeit "makeAncVs" begin
        vH = getDirectional(vcat(H[:, col]..., H[:, ncol]...), Vertical)
        Va = makeAncillaryVsHorizontal(A, Aj, L, R, vH, col, ncol)
        Vb = [Vector{ITensor}() for ii in 1:length(Va)]
        Vs = (above=Va, below=Vb)
    end
    @timeit "makeAncFs" begin
        fH = getDirectional(vcat(H[:, col]..., H[:, ncol]...), Field)
        Fa = makeAncillaryFsHorizontal(A, Aj, L, R, fH, col, ncol)
        Fb = [Vector{ITensor}() for ii in 1:length(Fa)]
        Fs = (above=Fa, below=Fb)
    end
    mincol, maxcol = extrema([col, ncol])
    @timeit "makeAncInts" begin
        iH = getDirectional(vcat(H[:, mincol]...), Horizontal)
        ITa = makeAncillaryIntsHorizontal(A, Aj, L, R, iH, col, ncol)
        ITb = [Vector{ITensor}() for ii in 1:length(ITa)]
        ITs = (above=ITa, below=ITb)
    end
    Ls = (above=Vector{ITensor}(), below=Vector{ITensor}())
    Rs = (above=Vector{ITensor}(), below=Vector{ITensor}())
    if mincol > 1
        @timeit "makeAncLs" begin
            lH = getDirectional(vcat(H[:, mincol-1]...), Horizontal)
            La = makeAncillarySideHorizontal(A, Aj, L, R, lH, col, ncol, :left)
            Lb = [Vector{ITensor}() for ii in  1:length(La)]
            Ls = (above=La, below=Lb)
        end
    end
    if maxcol < Nx
        @timeit "makeAncRs" begin
            rH = getDirectional(vcat(H[:, maxcol]...), Horizontal)
            Ra = makeAncillarySideHorizontal(A, Aj, R, L, rH, col, ncol, :right)
            Rb = [Vector{ITensor}() for ii in  1:length(Ra)]
            Rs = (above=Ra, below=Rb)
        end
    end
    Ancs = (I=Is, V=Vs, F=Fs, L=Ls, R=Rs, IT=ITs)
    return Ancs
end

function updateAncsHorizontal(A::fPEPS, Aj::Vector{ITensor}, 
                              L::Environments, R::Environments, 
                              AncEnvs, H, 
                              row::Int, col::Int, ncol::Int)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
   
    Is, Vs, Fs, Ls, Rs, ITs = AncEnvs
    @debug "\tUpdating ancillary identity terms for col $col row $row"
    Ib = updateAncillaryIsHorizontal(A, Aj, Is[:below], L, R, row, col, ncol)
    Is = (above=Is[:above], below=Ib)

    @debug "\tUpdating ancillary vertical terms for col $col row $row"
    vH = getDirectional(vcat(H[:, col]..., H[:, ncol]...), Vertical)
    Vb = updateAncillaryVsHorizontal(A, Aj, Vs[:below], Ib, L, R, vH, row, col, ncol)
    Vs = (above=Vs[:above], below=Vb)

    @debug "\tUpdating ancillary field terms for col $col row $row"
    fH = getDirectional(vcat(H[:, col]..., H[:, ncol]...), Field)
    Fb = updateAncillaryFsHorizontal(A, Aj, Fs[:below], Ib, L, R, fH, row, col, ncol)  
    Fs = (above=Fs[:above], below=Fb)
    
    mincol, maxcol    = extrema([col, ncol])
    @debug "\tUpdating interior field terms for col $col row $row"
    itH = getDirectional(vcat(H[:, mincol]...), Horizontal)
    ITb = updateAncillaryIntsHorizontal(A, Aj, ITs[:below], Ib, L, R, itH, row, col, ncol)  
    ITs = (above=ITs[:above], below=ITb)

    if mincol > 1
        @debug "\tUpdating ancillary left terms for col $col row $row"
        lH = getDirectional(vcat(H[:, mincol-1]...), Horizontal)
        Lb = updateAncillarySideHorizontal(A, Aj, Ls[:below], Ib, L, R, lH, row, col, ncol, :left)
        Ls = (above=Ls[:above], below=Lb)
    end
    if maxcol < Nx
        @debug "\tUpdating ancillary right terms for col $col row $row"
        rH = getDirectional(vcat(H[:, maxcol]...), Horizontal)
        Rb = updateAncillarySideHorizontal(A, Aj, Rs[:below], Ib, R, L, rH, row, col, ncol, :right)
        Rs = (above=Rs[:above], below=Rb)
    end
    Ancs = (I=Is, V=Vs, F=Fs, L=Ls, R=Rs, IT=ITs)
    return Ancs
end

function joincols(A::fPEPS, col::Int, ncol::Int)::Tuple{Vector{ITensor}, Vector{ITensor}}
    Ny, Nx = size(A)
    Aj = Vector{ITensor}(undef, Ny)
    cis = Vector{ITensor}(undef, Ny-1)
    for row in 1:Ny
        Aj[row] = A[row, col]*A[row, ncol]
        if row > 1
            Aj[row] *= cis[row-1]
        end
        if row < Ny
            ci = combiner(commonindex(A[row, col], A[row+1,col]), commonindex(A[row, ncol], A[row+1, ncol]), tags="Link,u,ji$row")
            Aj[row] *= ci
            cis[row] = ci
        end
    end
    return Aj, cis
end

function unjoincols(A::fPEPS, Aj::Vector{ITensor}, cis::Vector{ITensor}, col::Int, ncol::Int)::fPEPS
    Ny, Nx = size(A)
    for row in 1:Ny
        if row > 1
            Aj[row] *= cis[row-1]
        end
        if row < Ny
            Aj[row] *= cis[row]
        end
    end
    for row in 1:Ny
        col_is       = commoninds(Aj[row], A[row, col])
        cnc_ci       = commonind(A[row, col], A[row, ncol])
        U, S, V      = svd(Aj[row], col_is; mindim=dim(cnc_ci), maxdim=dim(cnc_ci), lefttags=tags(cnc_ci), cutoff=0.)
        A[row, col]  = U
        A[row, ncol] = S*V
    end
    D = dim(commonind(A[1, col], A[1, ncol]))
    return A
end

function sweepColumnHorizontal(A::fPEPS, 
                               L::Environments, R::Environments, 
                               H, col::Int, ncol::Int; 
                               kwargs...)
    Ny, Nx = size(A)
    #println("Sweeping column $col with next column $ncol")
    is_cu  = is_gpu(A)
    dummyI = is_cu ? MPS([cuITensor(1.0) for ii in 1:Ny], 0, Ny+1) : MPS([ITensor(1.0) for ii in 1:Ny], 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny), fill(ITensor(), 1, Ny)) 
    @debug "Beginning buildAncs for col $col"
    Aj, jis = joincols(A, col, ncol)
    @timeit "intraColumnGauge Aj" begin
        Aj = intraColumnGaugeHorizontal(Aj; kwargs...)
    end
    AncEnvs = buildAncsHorizontal(A, Aj, L, R, H, col, ncol)
    @inbounds for row in 1:Ny
        if row > 1
            @timeit "updateAncs" begin
                AncEnvs = updateAncsHorizontal(A, Aj, L, R, AncEnvs, H, row-1, col, ncol)
            end
        end
        @debug "Beginning optimizing H for col $col"
        Aj, AncEnvs = optimizeLocalHTwoSiteHorizontal(Aj, L, R, AncEnvs, H, row, col, ncol; kwargs...)
    end
    #A = unjoincols(A, Aj, jis, col, ncol)
    #return A
    return A, Aj
end

