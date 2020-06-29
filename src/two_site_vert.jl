function buildNTwoSiteVertical(A::fPEPS, 
                               L::Environments, 
                               R::Environments, 
                               IEnvs, 
                               row::Int, 
                               col::Int, 
                               ϕ::NamedTuple)::ITensor
    Ny, Nx   = size(A)
    up_row   = row + 1
    Nbelow   = L.I[row]
    if row > 1
        Nbelow *= IEnvs[:below][row - 1]
    end
    Nbelow *= ϕ[:Ab]
    Nbelow *= R.I[row]
    Nbelow *= dag(ϕ[:Ab]')
    Nabove = L.I[up_row]
    if up_row < Ny
        Nabove *= IEnvs[:above][end - up_row]
    end
    Nabove *= ϕ[:Aa]
    Nabove *= R.I[up_row]
    Nabove *= dag(ϕ[:Aa]')
    workingN  = ϕ[:sab]
    workingN *= spinI(firstind(A[up_row, col], "Site"); is_gpu=is_gpu(A))
    workingN *= spinI(firstind(A[row, col], "Site"); is_gpu=is_gpu(A))
    workingN *= Nbelow
    workingN *= Nabove
    AAinds    = inds(prime(ϕ[:sab]))
    @assert hasinds(inds(workingN), AAinds) "row $row\nIndsN: $(inds(workingN))\nIndsAA: $AAinds"
    @assert hasinds(AAinds, inds(workingN)) "row $row\nIndsN: $(inds(workingN))\nIndsAA: $AAinds"
    return workingN
end

function buildHIedgeTwoSiteVertical(A::fPEPS, 
                                    E::Environments, 
                                    row::Int, 
                                    col::Int, 
                                    side::Symbol, 
                                    ϕ::NamedTuple)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    up_row = row + 1
    IH_a   = is_cu ? cuITensor(1.0) : ITensor(1.0)
    IH_b   = is_cu ? cuITensor(1.0) : ITensor(1.0)
    IH     = is_cu ? cuITensor(1.0) : ITensor(1.0)
    next_col = side == :left ? 2 : Nx - 1
    @inbounds for work_row in 1:row-1
        IH_b *= E.H[work_row] * A[work_row, col]
        IH_b *= dag(prime(A[work_row, col], "Link"))
    end
    @inbounds for work_row in up_row+1:Ny
        IH_a *= E.H[work_row] * A[work_row, col]
        IH_a *= dag(prime(A[work_row, col], "Link"))
    end
    IH_b *= E.H[row]
    IH_a *= E.H[up_row]
    IH_b *= ϕ[:Ab]
    IH_b *= dag(ϕ[:Ab]')
    IH_a *= ϕ[:Aa]
    IH_a *= dag(ϕ[:Aa]')
    
    IH = ϕ[:sab]
    IH *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
    IH *= spinI(firstind(A[up_row, col], "Site"); is_gpu=is_cu)
    IH *= IH_b
    IH *= IH_a
    AAinds    = inds(prime(ϕ[:sab]))
    @assert hasinds(inds(IH), AAinds)
    @assert hasinds(AAinds, inds(IH))
    return (IH,)
end

function buildHIsTwoSiteVertical(A::fPEPS,
                                 L::Environments, 
                                 R::Environments, 
                                 row::Int, 
                                 col::Int, 
                                 ϕ::NamedTuple)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    up_row = row + 1
    @timeit "buildHIedge" begin
        if col == 1
            return buildHIedgeTwoSiteVertical(A, R, row, col, :left, ϕ)
        end
        if col == Nx
            return buildHIedgeTwoSiteVertical(A, L, row, col, :right, ϕ)
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
        @inbounds for work_row in reverse(up_row+1:Ny)
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
        HLI_b *= L.H[row]
        HLI_a *= L.H[up_row]
        HLI_b *= ϕ[:Ab]
        HLI_a *= ϕ[:Aa]
        HLI_b *= R.I[row]
        HLI_a *= R.I[up_row]
        HLI_b *= dag(ϕ[:Ab]')
        HLI_a *= dag(ϕ[:Aa]')
        IHR_b *= R.H[row]
        IHR_a *= R.H[up_row]
        IHR_b *= ϕ[:Ab]
        IHR_a *= ϕ[:Aa]
        IHR_b *= L.I[row]
        IHR_a *= L.I[up_row]
        IHR_b *= dag(ϕ[:Ab]')
        IHR_a *= dag(ϕ[:Aa]')
        HLI *= ϕ[:sab]
        IHR *= ϕ[:sab]
        HLI *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
        HLI *= spinI(firstind(A[up_row, col], "Site"); is_gpu=is_cu)
        IHR *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
        IHR *= spinI(firstind(A[up_row, col], "Site"); is_gpu=is_cu)
        HLI *= HLI_a
        IHR *= IHR_a
        HLI *= HLI_b
        IHR *= IHR_b
    end
    AAinds    = inds(prime(ϕ[:sab]))
    @assert hasinds(inds(IHR), AAinds)
    @assert hasinds(inds(HLI), AAinds)
    @assert hasinds(AAinds, inds(IHR))
    @assert hasinds(AAinds, inds(HLI))
    return (HLI, IHR)
end

function verticalTermsTwoSiteVertical(A::fPEPS, 
                                      L::Environments, 
                                      R::Environments, 
                                      AI, 
                                      AV, 
                                      H, 
                                      row::Int, 
                                      col::Int, 
                                      ϕ::NamedTuple)::Vector{ITensor}
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    vTerms = Vector{ITensor}(undef, length(H))
    AAinds = inds(prime(ϕ[:sab]))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0)
    up_row = row + 1
    @inbounds for opcode in 1:length(H)
        thisVert = dummy
        op_row_a = H[opcode].sites[1][1]
        op_row_b = H[opcode].sites[2][1]
        if op_row_b < row || up_row < op_row_a
            local Va, Vb
            if op_row_a > up_row
                Va = AV[:above][opcode][end - up_row]
                Vb = row > 1 ? AI[:below][row - 1] : dummy
            elseif op_row_b < row
                Vb = AV[:below][opcode][row - op_row_b]
                Va = up_row < Ny ? AI[:above][end - up_row] : dummy
            end
            Vb *= L.I[row]
            Vb *= ϕ[:Ab]
            Vb *= R.I[row]
            Vb *= dag(ϕ[:Ab]')
            Va *= L.I[up_row]
            Va *= ϕ[:Aa]
            Va *= R.I[up_row]
            Va *= dag(ϕ[:Aa]')
            thisVert = ϕ[:sab]
            thisVert *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
            thisVert *= spinI(firstind(A[up_row, col], "Site"); is_gpu=is_cu)
            thisVert *= Vb
            thisVert *= Va
        elseif row == op_row_a
            low_row  = op_row_a - 1
            high_row = op_row_b
            AIL  = low_row > 0   ? AI[:below][low_row]        : dummy 
            AIH  = high_row < Ny ? AI[:above][end - high_row] : dummy 
            sA   = firstind(A[op_row_a, col], "Site")
            op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
            op_a = replaceind!(op_a, H[opcode].site_ind', sA')
            sB   = firstind(A[op_row_b, col], "Site")
            op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
            op_b = replaceind!(op_b, H[opcode].site_ind', sB')
            AIH *= L.I[op_row_b]
            AIH *= ϕ[:Aa]
            AIH *= R.I[op_row_b]
            AIH *= dag(ϕ[:Aa]')
            AIL *= L.I[op_row_a]
            AIL *= ϕ[:Ab]
            AIL *= R.I[op_row_a]
            AIL *= dag(ϕ[:Ab]')
            thisVert = ϕ[:sab]
            thisVert *= op_a
            thisVert *= op_b
            thisVert *= AIL
            thisVert *= AIH
        elseif row == op_row_b
            low_row  = op_row_a - 1
            high_row = op_row_b + 1
            AIL  = low_row > 0   ? AI[:below][low_row]        : dummy 
            AIH  = high_row < Ny ? AI[:above][end - high_row] : dummy 
            sA   = firstind(A[op_row_a, col], "Site")
            op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
            op_a = replaceind!(op_a, H[opcode].site_ind', sA')
            sB   = firstind(A[op_row_b, col], "Site")
            op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
            op_b = replaceind!(op_b, H[opcode].site_ind', sB')
            AIL *= L.I[op_row_a] 
            AIL *= A[op_row_a, col] * op_a 
            AIL *= R.I[op_row_a] 
            AIL *= dag(A[op_row_a, col])'
            AIL *= L.I[op_row_b] 
            AIL *= ϕ[:Ab]
            AIL *= R.I[op_row_b] 
            AIL *= dag(ϕ[:Ab]')
            AIH *= L.I[up_row] 
            AIH *= ϕ[:Aa]
            AIH *= R.I[up_row] 
            AIH *= dag(ϕ[:Aa]')
            thisVert *= ϕ[:sab] 
            thisVert *= op_b
            thisVert *= spinI(firstind(A[up_row, col], "Site"); is_gpu=is_cu)
            thisVert *= AIL
            thisVert *= AIH
        elseif up_row == op_row_a
            low_row  = op_row_a - 2
            high_row = op_row_b
            AIL  = low_row > 0   ? AI[:below][low_row]        : dummy 
            AIH  = high_row < Ny ? AI[:above][end - high_row] : dummy 
            sA   = firstind(A[op_row_a, col], "Site")
            op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, sA)
            op_a = replaceind!(op_a, H[opcode].site_ind', sA')
            sB   = firstind(A[op_row_b, col], "Site")
            op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, sB)
            op_b = replaceind!(op_b, H[opcode].site_ind', sB')
            AIH *= L.I[op_row_b] 
            AIH *= A[op_row_b, col] * op_b
            AIH *= R.I[op_row_b] 
            AIH *= dag(A[op_row_b, col])'
            AIH *= L.I[op_row_a]
            AIH *= ϕ[:Aa]
            AIH *= R.I[op_row_a]
            AIH *= dag(ϕ[:Aa]')
            AIL *= L.I[row]
            AIL *= ϕ[:Ab]
            AIL *= R.I[row]
            AIL *= dag(ϕ[:Ab]')
            thisVert  = ϕ[:sab]
            thisVert *= op_a
            thisVert *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
            thisVert *= AIL 
            thisVert *= AIH
        end
        @assert hasinds(inds(thisVert), AAinds) "inds of thisVert and AAinds differ!\n$(inds(thisVert))\n$AAinds\n"
        @assert hasinds(AAinds, inds(thisVert)) "inds of thisVert and AAinds differ!\n$(inds(thisVert))\n$AAinds\n"
        if hasinds(AAinds, inds(thisVert)) && hasinds(inds(thisVert), AAinds)
            vTerms[opcode] = thisVert
        end
    end
    return vTerms
end

function fieldTermsTwoSiteVertical(A::fPEPS,
                                   L::Environments,
                                   R::Environments,
                                   AI,
                                   AF,
                                   H,
                                   row::Int,
                                   col::Int,
                                   ϕ::NamedTuple)::Vector{ITensor} 
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    fTerms = Vector{ITensor}(undef, length(H))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0)
    AAinds = inds(prime(ϕ[:sab]))
    up_row = row + 1
    @inbounds for opcode in 1:length(H)
        thisField = dummy
        op_row    = H[opcode].sites[1][1]
        if op_row != row && up_row != op_row
            local Fa, Fb
            if op_row > up_row
                Fa = AF[:above][opcode][end - up_row]
                Fb = row > 1 ? AI[:below][row - 1] : dummy
            else
                Fb = AF[:below][opcode][row - 1]
                Fa = up_row < Ny ? AI[:above][end - up_row] : dummy
            end
            Fb *= L.I[row]
            Fb *= ϕ[:Ab]
            Fb *= R.I[row]
            Fb *= dag(ϕ[:Ab]')
            Fa *= L.I[up_row]
            Fa *= ϕ[:Aa]
            Fa *= R.I[up_row]
            Fa *= dag(ϕ[:Aa]')
            thisField = ϕ[:sab]
            thisField *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
            thisField *= spinI(firstind(A[up_row, col], "Site"); is_gpu=is_cu)
            thisField *= Fa
            thisField *= Fb
        else
            low_row  = row - 1
            high_row = up_row
            AIL = low_row > 0   ? AI[:below][low_row]        : dummy 
            AIH = high_row < Ny ? AI[:above][end - high_row] : dummy 
            thisField  = AIL
            AIL *= L.I[row]
            AIH *= L.I[up_row]
            AIL *= ϕ[:Ab]
            AIH *= ϕ[:Aa]
            AIL *= R.I[row]
            AIH *= R.I[up_row]
            AIL *= dag(ϕ[:Ab]')
            AIH *= dag(ϕ[:Aa]')
            thisField = ϕ[:sab]
            sA = firstind(A[row, col], "Site")
            sB = firstind(A[up_row, col], "Site")
            if op_row == row
                op = copy(H[opcode].ops[1])
                op = replaceind!(op, H[opcode].site_ind, sA)
                op = replaceind!(op, H[opcode].site_ind', sA')
                thisField *= op
                thisField *= spinI(sB; is_gpu=is_cu)
            elseif op_row == up_row
                op = copy(H[opcode].ops[1])
                op = replaceind!(op, H[opcode].site_ind, sB)
                op = replaceind!(op, H[opcode].site_ind', sB')
                thisField *= op
                thisField *= spinI(sA; is_gpu=is_cu)
            end
            thisField *= AIH
            thisField *= AIL
        end
        @assert hasinds(inds(thisField), AAinds) "row $row\ntf: $(inds(thisField))\naa: $AAinds"
        @assert hasinds(AAinds, inds(thisField)) "row $row\ntf: $(inds(thisField))\naa: $AAinds"
        fTerms[opcode] = thisField
    end
    return fTerms
end

function connectLeftTermsTwoSiteVertical(A::fPEPS,
                                         L::Environments,
                                         R::Environments,
                                         AI, AL, H,
                                         row::Int, col::Int,
                                         ϕ::NamedTuple)::Vector{ITensor}
    Ny, Nx = size(A)
    up_row = row + 1
    is_cu  = is_gpu(A)
    lTerms = Vector{ITensor}(undef, length(H))
    AAinds = inds(prime(ϕ[:sab]))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0) 
    @inbounds for opcode in 1:length(H)
        op_row_b = H[opcode].sites[2][1]
        op_b = copy(H[opcode].ops[2])
        as   = firstind(A[op_row_b, col], "Site")
        op_b = replaceind!(op_b, H[opcode].site_ind, as)
        op_b = replaceind!(op_b, H[opcode].site_ind', as')
        thisHori = is_cu ? cuITensor(1.0) : ITensor(1.0)
        if op_row_b != row && op_row_b != up_row
            if op_row_b > up_row
                La = AL[:above][opcode][end - up_row]
                Lb = row > 1 ? AL[:below][opcode][row - 1] : dummy 
            else
                Lb = AL[:below][opcode][row - 1]
                La = up_row < Ny ? AL[:above][opcode][end - up_row] : dummy
            end
            Lb *= L.InProgress[row, opcode]
            La *= L.InProgress[up_row, opcode]
            Lb *= ϕ[:Ab]
            La *= ϕ[:Aa]
            Lb *= R.I[row]
            La *= R.I[up_row]
            Lb *= dag(ϕ[:Ab]')
            La *= dag(ϕ[:Aa]')
            thisHori  = ϕ[:sab]
            thisHori *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
            thisHori *= spinI(firstind(A[up_row, col], "Site"); is_gpu=is_cu)
            thisHori *= La
            thisHori *= Lb
        else
            low_row  = (op_row_b <= row)    ? op_row_b - 1 : row - 1
            high_row = (op_row_b >= up_row) ? op_row_b + 1 : up_row + 1
            THB  = low_row >= 1   ? AL[:below][opcode][low_row]            : dummy
            THB *= R.I[row]
            THB *= ϕ[:Ab]
            THB *= L.InProgress[row, opcode]
            THB *= dag(ϕ[:Ab]')
            thisHori = ϕ[:sab]
            if row == op_row_b
                thisHori *= op_b
                thisHori *= spinI(firstind(A[up_row, col], "Site"); is_gpu=is_cu)
            elseif up_row == op_row_b
                thisHori *= op_b
                thisHori *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
            end
            thisHori *= THB
            THA  = high_row <= Ny ? AL[:above][opcode][end - high_row + 1] : dummy
            THA *= R.I[up_row]
            THA *= ϕ[:Aa]
            THA *= L.InProgress[up_row, opcode]
            THA *= dag(ϕ[:Aa]')
            thisHori *= THA
        end
        @assert hasinds(inds(thisHori), AAinds) "row $row\ntH: $(inds(thisHori))\nAA: $AAinds"
        @assert hasinds(AAinds, inds(thisHori)) "row $row\ntH: $(inds(thisHori))\nAA: $AAinds"
        lTerms[opcode] = thisHori
    end
    return lTerms
end

function connectRightTermsTwoSiteVertical(A::fPEPS,
                                          L::Environments,
                                          R::Environments,
                                          AI, AR, H,
                                          row::Int, col::Int,
                                          ϕ::NamedTuple)::Vector{ITensor}
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    up_row = row + 1
    rTerms = Vector{ITensor}(undef, length(H))
    AAinds = inds(prime(ϕ[:sab]))
    dummy  = is_cu ? cuITensor(1.0) : ITensor(1.0) 
    @inbounds for opcode in 1:length(H)
        op_row_a = H[opcode].sites[1][1]
        op_a = copy(H[opcode].ops[1])
        as   = firstind(A[op_row_a, col], "Site")
        op_a = replaceind!(op_a, H[opcode].site_ind, as)
        op_a = replaceind!(op_a, H[opcode].site_ind', as')
        thisHori = is_cu ? cuITensor(1.0) : ITensor(1.0)
        if op_row_a != row && op_row_a != up_row
            if op_row_a > up_row
                Ra = AR[:above][opcode][end - up_row]
                Rb = row > 1 ? AR[:below][opcode][row - 1] : dummy 
            else
                Rb = AR[:below][opcode][row - 1]
                Ra = up_row < Ny ? AR[:above][opcode][end - up_row] : dummy 
            end
            Rb *= R.InProgress[row, opcode]
            Ra *= R.InProgress[up_row, opcode]
            Rb *= ϕ[:Ab]
            Ra *= ϕ[:Aa]
            Rb *= L.I[row]
            Ra *= L.I[up_row]
            Rb *= dag(ϕ[:Ab]')
            Ra *= dag(ϕ[:Aa]')
            thisHori *= ϕ[:sab]
            thisHori *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
            thisHori *= spinI(firstind(A[up_row, col], "Site"); is_gpu=is_cu)
            thisHori *= Ra
            thisHori *= Rb
        else
            low_row  = (op_row_a <= row)    ? op_row_a - 1 : row - 1;
            high_row = (op_row_a >= up_row) ? op_row_a + 1 : up_row + 1;
            THB  = low_row >= 1   ? AR[:below][opcode][low_row]            : dummy
            THA  = high_row <= Ny ? AR[:above][opcode][end - high_row + 1] : dummy
            THB *= L.I[row]
            THA *= L.I[up_row]
            THB *= ϕ[:Ab]
            THA *= ϕ[:Aa]
            THB *= R.InProgress[row, opcode]
            THA *= R.InProgress[up_row, opcode]
            THB *= dag(ϕ[:Ab]')
            THA *= dag(ϕ[:Aa]')
            thisHori = ϕ[:sab]
            thisHori *= op_a
            if op_row_a == row
                thisHori *= spinI(firstind(A[up_row, col], "Site"); is_gpu=is_cu)
            elseif op_row_a == up_row
                thisHori *= spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
            end
            thisHori *= THB
            thisHori *= THA
        end
        @assert hasinds(inds(thisHori), AAinds)
        @assert hasinds(AAinds, inds(thisHori))
        rTerms[opcode] = thisHori
    end
    return rTerms
end

function buildLocalHTwoSiteVertical(A::fPEPS,
                                    L::Environments, R::Environments,
                                    AncEnvs, H,
                                    row::Int, col::Int,
                                    ϕ::NamedTuple;
                                    verbose::Bool=false)
    field_H_terms = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]...), Vertical)
    term_count    = 1 + length(field_H_terms) + length(vert_H_terms)
    Ny, Nx        = size(A)
    @timeit "build Ns" begin
        N   = buildNTwoSiteVertical(A, L, R, AncEnvs[:I], row, col, ϕ)
    end
    expϕ = ϕ[:Ab] * ϕ[:sab] * ϕ[:Aa]
    den = scalar(collect(N * dag(ϕ[:sab])'))
    local left_H_terms, right_H_terms
    if col > 1
        left_H_terms = getDirectional(vcat(H[:, col - 1]...), Horizontal)
        term_count  += length(left_H_terms)
    end
    if col < Nx
        right_H_terms = getDirectional(vcat(H[:, col]...), Horizontal)
        term_count   += length(right_H_terms)
    end
    if 1 < col < Nx
        term_count += 1
    end
    Hs = Vector{ITensor}(undef, term_count)
    term_counter = 1
    @debug "\t\tBuilding I*H and H*I row $row col $col"
    @timeit "build HIs" begin
        HIs = buildHIsTwoSiteVertical(A, L, R, row, col, ϕ)
        Hs[term_counter] = HIs[1]
        if length(HIs) == 2
            Hs[term_counter+1] = HIs[2]
        end
        term_counter += length(HIs)
        if verbose
            println("--- HI TERMS ---")
            for HI in HIs
                println(scalar(HI * dag(ϕ[:sab])'))
            end
        end
    end
    @debug "\t\tBuilding vertical H terms row $row col $col"
    @timeit "build vertical terms" begin
        vTs = verticalTermsTwoSiteVertical(A, L, R, AncEnvs[:I], AncEnvs[:V], vert_H_terms, row, col, ϕ) 
        Hs[term_counter:term_counter+length(vTs) - 1] = vTs
        term_counter += length(vTs)
        if verbose
            println( "--- vT TERMS ---")
            for vT in vTs
                println(scalar(vT * dag(ϕ[:sab])'))
            end
        end
    end
    @debug "\t\tBuilding field H terms row $row col $col"
    @timeit "build field terms" begin
        fTs = fieldTermsTwoSiteVertical(A, L, R, AncEnvs[:I], AncEnvs[:F], field_H_terms, row, col, ϕ)
        Hs[term_counter:term_counter+length(fTs) - 1] = fTs[:]
        term_counter += length(fTs)
        if verbose
            println( "--- fT TERMS ---")
            for fT in fTs
                println(scalar(fT * dag(ϕ[:sab])'))
            end
        end
    end
    if col > 1
        @debug "\t\tBuilding left H terms row $row col $col"
        @timeit "build left terms" begin
            lTs = connectLeftTermsTwoSiteVertical(A, L, R, AncEnvs[:I], AncEnvs[:L], left_H_terms, row, col, ϕ)
            Hs[term_counter:term_counter+length(lTs) - 1] = lTs[:]
            term_counter += length(lTs)
            if verbose
                println( "--- lT TERMS ---")
                for lT in lTs
                    println(scalar(lT * dag(ϕ[:sab])'))
                end
            end
        end
        @debug "\t\tBuilt left terms"
    end
    if col < Nx
        @debug "\t\tBuilding right H terms row $row col $col"
        @timeit "build right terms" begin
            rTs = connectRightTermsTwoSiteVertical(A, L, R, AncEnvs[:I], AncEnvs[:R], right_H_terms, row, col, ϕ)
            Hs[term_counter:term_counter+length(rTs) - 1] = rTs[:]
            term_counter += length(rTs)
            if verbose
                println( "--- rT TERMS ---")
                for rT in rTs
                    println(scalar(rT * dag(ϕ[:sab])'))
                end
            end
        end
        @debug "\t\tBuilt right terms"
    end
    return Hs, N
end

struct ITensorMapTwoSiteVertical
  A::fPEPS
  H::Matrix{Vector{Operator}}
  L::Environments
  R::Environments
  AncEnvs::NamedTuple
  row::Int
  col::Int
  ϕa::ITensor
  ϕb::ITensor
end
Base.eltype(M::ITensorMapTwoSiteVertical) = eltype(M.A)
function Base.size(M::ITensorMapTwoSiteVertical)
    vci  = dim(commonindex(M.A[M.row, M.col], M.A[M.row+1, M.col]))
    saci = dim(firstind(M.A[M.row, M.col], "Site"))
    sbci = dim(firstind(M.A[M.row+1, M.col], "Site"))
    return 2*vci*saci*sbci
end

function (M::ITensorMapTwoSiteVertical)(v)
    ϕ = (Aa=M.ϕa, Ab=M.ϕb, sab=v)
    Hs, N  = buildLocalHTwoSiteVertical(M.A, M.L, M.R, M.AncEnvs, M.H, M.row, M.col, ϕ)
    localH = sum(Hs)
    return noprime(localH)
end

function optimizeLocalHTwoSiteVertical(A::fPEPS,
                                       L::Environments, R::Environments, 
                                       AncEnvs, H, 
                                       row::Int, col::Int; 
                                       kwargs...)
    Ny, Nx        = size(A)
    is_cu         = is_gpu(A)
    field_H_terms = getDirectional(vcat(H[:, col]...), Field)
    vert_H_terms  = getDirectional(vcat(H[:, col]...), Vertical)
    Rainds        = IndexSet(firstind(A[row, col], "Site"), commonindex(A[row, col], A[row+1, col]))
    Lainds        = uniqueinds(inds(A[row, col]), IndexSet(firstind(A[row, col], "Site"), commonindex(A[row, col], A[row+1, col])))
    Qa, Ra, qra   = qr(A[row, col], Lainds; tags="sa,Link,r$row,c$col")
    Rbinds        = IndexSet(firstind(A[row+1, col], "Site"), commonindex(A[row, col], A[row+1, col]))
    Lbinds        = uniqueinds(inds(A[row+1, col]), IndexSet(firstind(A[row+1, col], "Site"), commonindex(A[row, col], A[row+1, col])))
    Qb, Rb, qrb   = qr(A[row+1, col], Lbinds; tags="sb,Link,r$(row+1),c$col")
    ϕa            = Qa
    ϕb            = Qb
    ϕs            = Ra * Rb
    ϕ             = (Ab=ϕa, Aa=ϕb, sab=ϕs)
    @debug "\tBuilding H for col $col row $row"
    @timeit "build H" begin
        Hs, N = buildLocalHTwoSiteVertical(A, L, R, AncEnvs, H, row, col, ϕ)
    end
    expϕ = ϕ[:Ab] * ϕ[:sab] * ϕ[:Aa]
    initial_N = real(scalar(collect(N * dag(ϕ[:sab]'))))
    @timeit "sum H terms" begin
        localH = sum(Hs)
    end
    initial_E = real(scalar(collect(deepcopy(localH) * dag(ϕ[:sab])')))
    @info "Initial energy at row $row col $col : $(initial_E/(initial_N*Nx*Ny)) and norm : $initial_N"
    @debug "\tBeginning davidson for col $col row $row"
    mapper   = ITensorMapTwoSiteVertical(A, H, L, R, AncEnvs, row, col, ϕ[:Aa], ϕ[:Ab])
    λ, new_ϕ = davidson(mapper, ϕ[:sab]; maxiter=2, kwargs...)
    new_E    = λ

    # split the result
    Lis      = IndexSet(firstind(A[row, col], "Site"), commonindex(ϕa, new_ϕ))
    U, S, Vt = svd(new_ϕ, Lis; tags=tags(commonindex(A[row, col], A[row+1, col])), kwargs...)
    A[row, col]   = ϕa * U
    A[row+1, col] = ϕb * S * Vt
    #=N        = buildNTwoSite(A, L, R, AncEnvs[:I], row, col, new_ϕ)
    new_N    = real(scalar(collect(N * dag(new_ϕ)')))
    @info "Optimized energy at row $row col $col : $(new_E/(new_N*Nx*Ny)) and norm : $new_N"=#
    @timeit "restore intraColumnGauge" begin
        if row <= Ny - 1
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
            svdA    = A[row, col]*cmb
            Ris     = uniqueinds(inds(svdA), Lis) 
            U, S, V = svd(svdA, Ris; kwargs...)
            new_ci  = commonind(V, S)
            replaceind!(V, new_ci, old_ci)
            A[row, col] = V*cmb
            newU = S*U*A[row+1, col]
            replaceind!(newU, new_ci, old_ci)
            A[row+1, col] = newU
            if row < Ny - 1
                nI    = spinI(firstind(A[row+1, col], "Site"); is_gpu=is_cu)
                newAA = copy(AncEnvs[:I][:above][end - row - 1])
                newAA *= L.I[row+1]
                newAA *= A[row+1, col] * nI
                newAA *= R.I[row+1]
                newAA *= dag(A[row+1, col])'
                AncEnvs[:I][:above][end - row] = newAA
            end
        end
    end
    return A, AncEnvs
end
