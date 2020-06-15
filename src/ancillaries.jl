function makeAncillaryIs(A::fPEPS, L::Environments, R::Environments, col::Int)
    Ny, Nx         = size(A)
    is_cu          = is_gpu(A)
    dummy          = is_cu    ? cuITensor(1.0)  : ITensor(1.0) 
    col_site_inds  = [firstind(x, "Site") for x in A[:, col]]
    ops            = map(x -> spinI(x; is_gpu=is_cu), col_site_inds)
    ancIs          = Vector{ITensor}(undef, Ny)
    ancI           = copy(dummy)
    for row in reverse(1:Ny)
        ancI      *= L.I[row] * A[row, col]
        ancI      *= ops[row] * dag(prime(A[row, col]))
        ancI      *= R.I[row]
        ancIs[row] = copy(ancI)
    end
    reverse!(ancIs)
    return ancIs
end

function makeAncillaryIsHorizontal(A::fPEPS, Aj::Vector{ITensor}, L::Environments, R::Environments, col::Int, ncol::Int)
    Ny, Nx         = size(A)
    is_cu          = is_gpu(A)
    dummy          = is_cu    ? cuITensor(1.0)  : ITensor(1.0)
    col_site_inds  = [firstind(x, "Site") for x in A[:, col]]
    ncol_site_inds = [firstind(x, "Site") for x in A[:, ncol]]
    ops            = map(x -> spinI(x; is_gpu=is_cu), col_site_inds) .* map(x -> spinI(x; is_gpu=is_cu), ncol_site_inds)
    ancIs          = Vector{ITensor}(undef, Ny)
    ancI           = copy(dummy)
    for row in reverse(1:Ny)
        ancI      *= L.I[row]
        ancI      *= Aj[row]
        ancI      *= ops[row] * dag(prime(Aj[row]))
        ancI      *= R.I[row]
        ancIs[row] = copy(ancI)
    end
    reverse!(ancIs)
    return ancIs
end

function updateAncillaryIs(A::fPEPS, Ibelow::Vector{ITensor}, L::Environments, R::Environments, row::Int, col::Int )
    Ny, Nx  = size(A)
    is_cu   = is_gpu(A) 
    dummy   = is_cu   ? cuITensor(1.0)  : ITensor(1.0) 
    op      = spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
    AA      = row > 1 ? copy(Ibelow[row - 1]) : dummy 
    AA     *= L.I[row] * A[row, col]
    AA     *= op * dag(prime(A[row, col]))
    AA     *= R.I[row]
    Ibelow[row] = AA
    return Ibelow
end

function updateAncillaryIsHorizontal(A::fPEPS, Aj::Vector{ITensor}, Ibelow::Vector{ITensor}, L::Environments, R::Environments, row::Int, col::Int, ncol::Int )
    Ny, Nx  = size(A)
    is_cu   = is_gpu(A)
    dummy   = is_cu   ? cuITensor(1.0)  : ITensor(1.0) 
    op      = spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
    op     *= spinI(firstind(A[row, ncol], "Site"); is_gpu=is_cu)
    AA      = row > 1 ? copy(Ibelow[row - 1]) : dummy 
    AA     *= L.I[row] * Aj[row]
    AA     *= op * dag(prime(Aj[row]))
    AA     *= R.I[row]
    Ibelow[row] = AA
    return Ibelow
end

function makeAncillaryFs(A::fPEPS, L::Environments, R::Environments, H, col::Int)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A) 
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0) 
    Fabove   = [Vector{ITensor}(undef, Ny) for opcode in 1:length(H)]
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    for opcode in 1:length(H)
        op_row      = H[opcode].sites[1][1]
        ops         = map(x -> spinI(x; is_gpu=is_cu), col_site_inds)
        ops[op_row] = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row])
        ops[op_row] = replaceind!(copy(ops[op_row]), H[opcode].site_ind', col_site_inds[op_row]')
        ancFs       = Vector{ITensor}(undef, Ny)
        ancF        = copy(dummy)
        for row in reverse(1:Ny)
            ancF   *= L.I[row] * A[row, col]
            ancF   *= ops[row] * dag(prime(A[row, col]))
            ancF   *= R.I[row]
            ancFs[row] = copy(ancF)
        end
        Fabove[opcode] .= reverse(ancFs)
    end
    return Fabove
end

function makeAncillaryFsHorizontal(A::fPEPS, Aj::Vector{ITensor}, L::Environments, R::Environments, H, col::Int, ncol::Int)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A)
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0)
    Fabove   = [Vector{ITensor}(undef, Ny) for opcode in 1:length(H)]
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    ncol_site_inds = [firstind(x, "Site") for x in A[:, ncol]]
    for opcode in 1:length(H)
        op_row      = H[opcode].sites[1][1]
        op_col      = H[opcode].sites[1][2]
        ops         = map(x -> spinI(x; is_gpu=is_cu), col_site_inds) .* map(x -> spinI(x; is_gpu=is_cu), ncol_site_inds)
        if op_col == col
            op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row])
            op_a = replaceind!(op_a, H[opcode].site_ind', col_site_inds[op_row]')
            ops[op_row] = op_a * spinI(ncol_site_inds[op_row]; is_gpu=is_cu)
        else
            op_b = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, ncol_site_inds[op_row])
            op_b = replaceind!(op_b, H[opcode].site_ind', ncol_site_inds[op_row]')
            ops[op_row] = op_b * spinI(col_site_inds[op_row]; is_gpu=is_cu)
        end
        ancFs       = Vector{ITensor}(undef, Ny)
        ancF        = copy(dummy)
        for row in reverse(1:Ny)
            ancF   *= L.I[row] * Aj[row]
            ancF   *= ops[row] * dag(prime(Aj[row]))
            ancF   *= R.I[row]
            ancFs[row] = copy(ancF)
        end
        Fabove[opcode] .= reverse(ancFs)
    end
    return Fabove
end

function updateAncillaryFs(A::fPEPS, Fbelow, Ibelow::Vector{ITensor}, L::Environments, R::Environments, H, row::Int, col::Int)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A) 
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0) 
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    for opcode in 1:length(H)
        op_row      = H[opcode].sites[1][1]
        ops         = ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in col_site_inds] 
        if op_row == row
            ops[op_row] = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row])
            ops[op_row] = replaceind!(ops[op_row], H[opcode].site_ind', col_site_inds[op_row]')
        end
        ancF  = length(Fbelow[opcode]) > 0 ? copy(Fbelow[opcode][end]) : copy(dummy)
        ancF *= L.I[row]
        ancF *= A[row, col]
        ancF *= ops[row]
        ancF *= prime(dag(A[row, col]))
        ancF *= R.I[row]
        push!(Fbelow[opcode], ancF) 
    end
    return Fbelow
end

function updateAncillaryFsHorizontal(A::fPEPS, Aj::Vector{ITensor}, Fbelow, Ibelow::Vector{ITensor}, L::Environments, R::Environments, H, row::Int, col::Int, ncol::Int)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A)
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0)
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    ncol_site_inds = [firstind(x, "Site") for x in A[:, ncol]]
    for opcode in 1:length(H)
        op_row      = H[opcode].sites[1][1]
        op_col      = H[opcode].sites[1][2]
        ops         = map(x -> spinI(x; is_gpu=is_cu), col_site_inds) .* map(x -> spinI(x; is_gpu=is_cu), ncol_site_inds)
        if op_row == row
            if op_col == col
                op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row])
                op_a = replaceind!(op_a, H[opcode].site_ind', col_site_inds[op_row]')
                ops[op_row] = op_a*spinI(ncol_site_inds[op_row]; is_gpu=is_cu)
            else
                op_b = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, ncol_site_inds[op_row])
                op_b = replaceind!(op_b, H[opcode].site_ind', ncol_site_inds[op_row]')
                ops[op_row] = op_b*spinI(col_site_inds[op_row]; is_gpu=is_cu)
            end
        end
        ancF  = length(Fbelow[opcode]) > 0 ? copy(Fbelow[opcode][end]) : copy(dummy)
        ancF *= L.I[row]
        ancF *= Aj[row]
        ancF *= ops[row]
        ancF *= prime(dag(Aj[row]))
        ancF *= R.I[row]
        push!(Fbelow[opcode], ancF) 
    end
    return Fbelow
end

function makeAncillaryIntsHorizontal(A::fPEPS, Aj::Vector{ITensor}, L::Environments, R::Environments, H, col::Int, ncol::Int)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A)
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0)
    ITabove   = [Vector{ITensor}(undef, Ny) for opcode in 1:length(H)]
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    ncol_site_inds = [firstind(x, "Site") for x in A[:, ncol]]
    for opcode in 1:length(H)
        op_row      = H[opcode].sites[1][1]
        op_col_a    = H[opcode].sites[1][2]
        op_col_b    = H[opcode].sites[2][2]
        ops         = map(x -> spinI(x; is_gpu=is_cu), col_site_inds) .* map(x -> spinI(x; is_gpu=is_cu), ncol_site_inds)
        if op_col_a == col
            op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row])
            op_a = replaceind!(op_a, H[opcode].site_ind', col_site_inds[op_row]')
            op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, ncol_site_inds[op_row])
            op_b = replaceind!(op_b, H[opcode].site_ind', ncol_site_inds[op_row]')
            ops[op_row] = op_a*op_b 
        else
            op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, ncol_site_inds[op_row])
            op_a = replaceind!(op_a, H[opcode].site_ind', ncol_site_inds[op_row]')
            op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, col_site_inds[op_row])
            op_b = replaceind!(op_b, H[opcode].site_ind', col_site_inds[op_row]')
            ops[op_row] = op_a*op_b 
        end
        ancITs       = Vector{ITensor}(undef, Ny)
        ancIT        = copy(dummy)
        for row in reverse(1:Ny)
            ancIT   *= L.I[row] * Aj[row]
            ancIT   *= ops[row] * dag(prime(Aj[row]))
            ancIT   *= R.I[row]
            ancITs[row] = copy(ancIT)
        end
        ITabove[opcode] .= reverse(ancITs)
    end
    return ITabove
end

function updateAncillaryIntsHorizontal(A::fPEPS, Aj::Vector{ITensor}, ITbelow, Ibelow::Vector{ITensor}, L::Environments, R::Environments, H, row::Int, col::Int, ncol::Int)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A)
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0)
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    ncol_site_inds = [firstind(x, "Site") for x in A[:, ncol]]
    for opcode in 1:length(H)
        op_row      = H[opcode].sites[1][1]
        op_col_a    = H[opcode].sites[1][2]
        op_col_b    = H[opcode].sites[2][2]
        ops         = map(x -> spinI(x; is_gpu=is_cu), col_site_inds) .* map(x -> spinI(x; is_gpu=is_cu), ncol_site_inds)
        if op_row == row
            if op_col_a == col
                op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row])
                op_a = replaceind!(op_a, H[opcode].site_ind', col_site_inds[op_row]')
                op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, ncol_site_inds[op_row])
                op_b = replaceind!(op_b, H[opcode].site_ind', ncol_site_inds[op_row]')
                ops[op_row] = op_a*op_b 
            else
                op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, ncol_site_inds[op_row])
                op_a = replaceind!(op_a, H[opcode].site_ind', ncol_site_inds[op_row]')
                op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, col_site_inds[op_row])
                op_b = replaceind!(op_b, H[opcode].site_ind', col_site_inds[op_row]')
                ops[op_row] = op_a*op_b 
            end
        end
        ancIT  = length(ITbelow[opcode]) > 0 ? copy(ITbelow[opcode][end]) : copy(dummy)
        ancIT *= L.I[row]
        ancIT *= Aj[row]
        ancIT *= ops[row]
        ancIT *= prime(dag(Aj[row]))
        ancIT *= R.I[row]
        push!(ITbelow[opcode], ancIT)
    end
    return ITbelow
end

function makeAncillaryVs(A::fPEPS, L::Environments, R::Environments, H, col::Int)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A) 
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0) 
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    Vabove   = [Vector{ITensor}(undef, Ny) for opcode in 1:length(H)]
    for opcode in 1:length(H)
        op_row_a      = H[opcode].sites[1][1]
        op_row_b      = H[opcode].sites[2][1]
        ops           = map(x->spinI(x; is_gpu=is_cu), col_site_inds)
        ops[op_row_a] = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row_a])
        ops[op_row_a] = replaceind!(ops[op_row_a], H[opcode].site_ind', col_site_inds[op_row_a]')
        ops[op_row_b] = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, col_site_inds[op_row_b])
        ops[op_row_b] = replaceind!(ops[op_row_b], H[opcode].site_ind', col_site_inds[op_row_b]')
        ancVs         = Vector{ITensor}(undef, Ny)
        ancV          = copy(dummy)
        for row in reverse(1:Ny)
            ancV   *= L.I[row] * A[row, col]
            ancV   *= ops[row] * dag(prime(A[row, col]))
            ancV   *= R.I[row]
            ancVs[row] = copy(ancV)
        end
        Vabove[opcode] = reverse(ancVs)
    end
    return Vabove
end

function makeAncillaryVsHorizontal(A::fPEPS, Aj::Vector{ITensor}, L::Environments, R::Environments, H, col::Int, ncol::Int)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A)
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0)
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    ncol_site_inds = [firstind(x, "Site") for x in A[:, ncol]]
    Vabove   = [Vector{ITensor}(undef, Ny) for opcode in 1:length(H)]
    for opcode in 1:length(H)
        op_row_a      = H[opcode].sites[1][1]
        op_row_b      = H[opcode].sites[2][1]
        op_col        = H[opcode].sites[1][2]
        ops           = map(x -> spinI(x; is_gpu=is_cu), col_site_inds) .* map(x -> spinI(x; is_gpu=is_cu), ncol_site_inds)
        if op_col == col
            op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row_a])
            op_a = replaceind!(op_a, H[opcode].site_ind', col_site_inds[op_row_a]')
            op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, col_site_inds[op_row_b])
            op_b = replaceind!(op_b, H[opcode].site_ind', col_site_inds[op_row_b]')
            ops[op_row_a] = op_a*spinI(ncol_site_inds[op_row_a]; is_gpu=is_cu)
            ops[op_row_b] = op_b*spinI(ncol_site_inds[op_row_b]; is_gpu=is_cu)
        else
            op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, ncol_site_inds[op_row_a])
            op_a = replaceind!(op_a, H[opcode].site_ind', ncol_site_inds[op_row_a]')
            op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, ncol_site_inds[op_row_b])
            op_b = replaceind!(op_b, H[opcode].site_ind', ncol_site_inds[op_row_b]')
            ops[op_row_a] = op_a*spinI(col_site_inds[op_row_a]; is_gpu=is_cu)
            ops[op_row_b] = op_b*spinI(col_site_inds[op_row_b]; is_gpu=is_cu)
        end
        ancVs         = Vector{ITensor}(undef, Ny)
        ancV          = copy(dummy)
        for row in reverse(1:Ny)
            ancV   *= L.I[row] * Aj[row]
            ancV   *= ops[row] * dag(prime(Aj[row]))
            ancV   *= R.I[row]
            ancVs[row] = copy(ancV)
        end
        Vabove[opcode] = reverse(ancVs)
    end
    return Vabove
end

function updateAncillaryVs(A::fPEPS, Vbelow, Ibelow::Vector{ITensor}, L::Environments, R::Environments, H, row::Int, col::Int)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A) 
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0) 
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    for opcode in 1:length(H)
        op_row_a      = H[opcode].sites[1][1]
        op_row_b      = H[opcode].sites[2][1]
        ops           = map(x->spinI(x; is_gpu=is_cu), col_site_inds)
        ops[op_row_a] = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row_a])
        ops[op_row_a] = replaceind!(ops[op_row_a], H[opcode].site_ind', col_site_inds[op_row_a]')
        ops[op_row_b] = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, col_site_inds[op_row_b])
        ops[op_row_b] = replaceind!(ops[op_row_b], H[opcode].site_ind', col_site_inds[op_row_b]')
        if op_row_b < row
            ancV  = copy(Vbelow[opcode][end])
            ancV *= L.I[row] * A[row, col]
            ancV *= ops[row] * dag(prime(A[row, col]))
            ancV *= R.I[row]
            push!(Vbelow[opcode], ancV)
        elseif op_row_b == row
            ancV  = op_row_a > 1 ? copy(Ibelow[op_row_a - 1]) : dummy
            for row_ in (op_row_a, op_row_b) 
                ancV   *= L.I[row_] * A[row_, col]
                ancV   *= ops[row_] * dag(prime(A[row_, col]))
                ancV   *= R.I[row_]
            end
            push!(Vbelow[opcode], ancV)
        end
    end
    return Vbelow
end

function updateAncillaryVsHorizontal(A::fPEPS, Aj::Vector{ITensor}, Vbelow, Ibelow::Vector{ITensor}, L::Environments, R::Environments, H, row::Int, col::Int, ncol::Int)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A)
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0)
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    ncol_site_inds = [firstind(x, "Site") for x in A[:, ncol]]
    for opcode in 1:length(H)
        op_row_a      = H[opcode].sites[1][1]
        op_row_b      = H[opcode].sites[2][1]
        op_col        = H[opcode].sites[2][2]
        ops           = map(x->spinI(x; is_gpu=is_cu), col_site_inds) .* map(x->spinI(x; is_gpu=is_cu), ncol_site_inds)
        if op_col == col
            op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row_a])
            op_a = replaceind!(op_a, H[opcode].site_ind', col_site_inds[op_row_a]')
            op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, col_site_inds[op_row_b])
            op_b = replaceind!(op_b, H[opcode].site_ind', col_site_inds[op_row_b]')
            ops[op_row_a] = op_a * spinI(ncol_site_inds[op_row_a]; is_gpu=is_cu)
            ops[op_row_b] = op_b * spinI(ncol_site_inds[op_row_b]; is_gpu=is_cu)
        else
            op_a = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, ncol_site_inds[op_row_a])
            op_a = replaceind!(op_a, H[opcode].site_ind', ncol_site_inds[op_row_a]')
            op_b = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, ncol_site_inds[op_row_b])
            op_b = replaceind!(op_b, H[opcode].site_ind', ncol_site_inds[op_row_b]')
            ops[op_row_a] = op_a * spinI(col_site_inds[op_row_a]; is_gpu=is_cu)
            ops[op_row_b] = op_b * spinI(col_site_inds[op_row_b]; is_gpu=is_cu)
        end
        if op_row_b < row
            ancV  = copy(Vbelow[opcode][end])
            ancV *= L.I[row] * Aj[row]
            ancV *= ops[row] * dag(prime(Aj[row]))
            ancV *= R.I[row]
            push!(Vbelow[opcode], ancV)
        elseif op_row_b == row
            ancV  = op_row_a > 1 ? copy(Ibelow[op_row_a - 1]) : dummy
            for row_ in (op_row_a, op_row_b) 
                ancV   *= L.I[row_] * Aj[row_]
                ancV   *= ops[row_] * dag(prime(Aj[row_]))
                ancV   *= R.I[row_]
            end
            push!(Vbelow[opcode], ancV)
        end
    end
    return Vbelow
end

function makeAncillarySide(A::fPEPS, EnvIP::Environments, EnvIdent::Environments, H, col::Int, side::Symbol)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A) 
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    Sabove   = [Vector{ITensor}(undef, Ny) for opcode in 1:length(H)]
    next_col = side == :left ? col + 1 : col - 1
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0) 
    for opcode in 1:length(H)
        op_row      = side == :left ? H[opcode].sites[2][1] : H[opcode].sites[1][1] 
        ops         = map(x->spinI(x; is_gpu=is_cu), col_site_inds)
        this_op     = side == :left ? H[opcode].ops[2] : H[opcode].ops[1]
        ops[op_row] = replaceind!(copy(this_op), H[opcode].site_ind, col_site_inds[op_row])
        ops[op_row] = replaceind!(ops[op_row], H[opcode].site_ind', col_site_inds[op_row]')
        AAs         = Vector{ITensor}(undef, Ny) 
        AA          = copy(dummy)
        for row in reverse(1:Ny)
            AA *= EnvIdent.I[row]
            AA *= A[row, col]
            AA *= ops[row]
            AA *= dag(prime(A[row, col]))
            AA *= EnvIP.InProgress[row, opcode]
            AAs[row] = copy(AA)
        end
        Sabove[opcode] = reverse(AAs)
    end
    return Sabove
end

function makeAncillarySideHorizontal(A::fPEPS, Aj::Vector{ITensor}, EnvIP::Environments, EnvIdent::Environments, H, col::Int, ncol::Int, side::Symbol)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A)
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    ncol_site_inds = [firstind(x, "Site") for x in A[:, ncol]]
    Sabove   = [Vector{ITensor}(undef, Ny) for opcode in 1:length(H)]
    next_col = side == :left ? col + 1 : col - 1
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0)
    for opcode in 1:length(H)
        op_row      = side == :left ? H[opcode].sites[2][1] : H[opcode].sites[1][1] 
        op_col      = side == :left ? H[opcode].sites[2][2] : H[opcode].sites[1][2] 
        ops         = map(x->spinI(x; is_gpu=is_cu), col_site_inds) .* map(x->spinI(x; is_gpu=is_cu), ncol_site_inds)
        this_op     = side == :left ? H[opcode].ops[2] : H[opcode].ops[1]
        if col == op_col
            op_a = replaceind!(copy(this_op), H[opcode].site_ind, col_site_inds[op_row])
            op_a = replaceind!(op_a, H[opcode].site_ind', col_site_inds[op_row]')
            ops[op_row] = op_a * spinI(ncol_site_inds[op_row]; is_gpu=is_cu)
        else
            op_b = replaceind!(copy(this_op), H[opcode].site_ind, ncol_site_inds[op_row])
            op_b = replaceind!(op_b, H[opcode].site_ind', ncol_site_inds[op_row]')
            ops[op_row] = op_b * spinI(col_site_inds[op_row]; is_gpu=is_cu)
        end
        AAs         = Vector{ITensor}(undef, Ny) 
        AA          = copy(dummy)
        for row in reverse(1:Ny)
            AA *= EnvIdent.I[row]
            AA *= Aj[row]
            AA *= ops[row]
            AA *= dag(prime(Aj[row]))
            AA *= EnvIP.InProgress[row, opcode]
            AAs[row] = copy(AA)
        end
        Sabove[opcode] = reverse(AAs)
    end
    return Sabove
end

function updateAncillarySide(A::fPEPS, Sbelow, Ibelow::Vector{ITensor}, EnvIP::Environments, EnvIdent::Environments, H, row::Int, col::Int, side::Symbol)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A) 
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    next_col = side == :left ? col + 1 : col - 1
    prev_col = side == :left ? col + 1 : col - 1
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0) 
    for opcode in 1:length(H)
        op_row      = side == :left ? H[opcode].sites[2][1] : H[opcode].sites[1][1] 
        ops         = ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in col_site_inds] 
        this_op     = side == :left ? H[opcode].ops[2] : H[opcode].ops[1]
        ops[op_row] = replaceind!(copy(this_op), H[opcode].site_ind, col_site_inds[op_row])
        ops[op_row] = replaceind!(ops[op_row], H[opcode].site_ind', col_site_inds[op_row]')
        AA          = row >= 2 ? copy(Sbelow[opcode][row-1]) : copy(dummy)
        AA         *= EnvIdent.I[row]
        AA         *= A[row, col]
        AA         *= ops[row]
        AA         *= prime(dag(A[row, col]))
        AA         *= EnvIP.InProgress[row, opcode]
        push!(Sbelow[opcode], AA)
    end
    return Sbelow
end

function updateAncillarySideHorizontal(A::fPEPS, Aj::Vector{ITensor}, Sbelow, Ibelow::Vector{ITensor}, EnvIP::Environments, EnvIdent::Environments, H, row::Int, col::Int, ncol::Int, side::Symbol)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A) 
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    ncol_site_inds = [firstind(x, "Site") for x in A[:, ncol]]
    next_col = side == :left ? col + 1 : col - 1
    prev_col = side == :left ? col + 1 : col - 1
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0)
    for opcode in 1:length(H)
        op_row      = side == :left ? H[opcode].sites[2][1] : H[opcode].sites[1][1] 
        op_col      = side == :left ? H[opcode].sites[2][2] : H[opcode].sites[1][2] 
        ops         = ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in col_site_inds] .* ITensor[spinI(spin_ind; is_gpu=is_cu) for spin_ind in ncol_site_inds] 
        this_op     = side == :left ? H[opcode].ops[2] : H[opcode].ops[1]
        if op_col == col
            op_a = replaceind!(copy(this_op), H[opcode].site_ind, col_site_inds[op_row])
            op_a = replaceind!(op_a, H[opcode].site_ind', col_site_inds[op_row]')
            ops[op_row] = op_a * spinI(ncol_site_inds[op_row]; is_gpu=is_cu)
        else
            op_b = replaceind!(copy(this_op), H[opcode].site_ind, ncol_site_inds[op_row])
            op_b = replaceind!(op_b, H[opcode].site_ind', ncol_site_inds[op_row]')
            ops[op_row] = op_b * spinI(col_site_inds[op_row]; is_gpu=is_cu)
        end
        AA          = row >= 2 ? copy(Sbelow[opcode][row-1]) : copy(dummy)
        AA         *= EnvIdent.I[row]
        AA         *= Aj[row]
        AA         *= ops[row]
        AA         *= prime(dag(Aj[row]))
        AA         *= EnvIP.InProgress[row, opcode]
        push!(Sbelow[opcode], AA)
    end
    return Sbelow
end

function makeAncillarySideDiag(A::fPEPS, EnvIP::Environments, EnvIdent::Environments, H, col::Int, side::Symbol)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A) 
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    Sabove   = [Vector{ITensor}(undef, Ny) for opcode in 1:length(H)]
    next_col = side == :left ? col + 1 : col - 1
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0) 
    for opcode in 1:length(H)
        op_row      = side == :left ? H[opcode].sites[2][1] : H[opcode].sites[1][1] 
        ops         = map(x->spinI(x; is_gpu=is_cu), col_site_inds)
        this_op     = side == :left ? H[opcode].ops[2] : H[opcode].ops[1]
        ops[op_row] = replaceind!(copy(this_op), H[opcode].site_ind, col_site_inds[op_row])
        ops[op_row] = replaceind!(ops[op_row], H[opcode].site_ind', col_site_inds[op_row]')
        AAs         = Vector{ITensor}(undef, Ny) 
        AA          = copy(dummy)
        for row in reverse(1:Ny)
            AA *= EnvIdent.I[row]
            AA *= A[row, col]
            AA *= ops[row]
            AA *= dag(prime(A[row, col]))
            AA *= EnvIP.DiagInProgress[row, opcode]
            AAs[row] = copy(AA)
        end
        Sabove[opcode] = reverse(AAs)
    end
    return Sabove
end

function updateAncillarySideDiag(A::fPEPS, Dbelow, Ibelow::Vector{ITensor}, EnvIP::Environments, EnvIdent::Environments, H, row::Int, col::Int, side::Symbol)
    Ny, Nx   = size(A)
    is_cu    = is_gpu(A)
    next_col = side == :left ? col + 1 : col - 1
    prev_col = side == :left ? col + 1 : col - 1
    dummy    = is_cu    ? cuITensor(1.0)  : ITensor(1.0) 
    col_site_inds = [firstind(x, "Site") for x in A[:, col]]
    for opcode in 1:length(H)
        op_row_a      = H[opcode].sites[1][1]
        op_row_b      = H[opcode].sites[2][1]
        op_col_a      = H[opcode].sites[1][2]
        op_col_b      = H[opcode].sites[2][2]
        ops           = map(x->spinI(x; is_gpu=is_cu), col_site_inds)
        if op_col_a == col
            ops[op_row_a] = replaceind!(copy(H[opcode].ops[1]), H[opcode].site_ind, col_site_inds[op_row_a])
            ops[op_row_a] = replaceind!(ops[op_row_a], H[opcode].site_ind', col_site_inds[op_row_a]')
        else
            ops[op_row_b] = replaceind!(copy(H[opcode].ops[2]), H[opcode].site_ind, col_site_inds[op_row_b])
            ops[op_row_b] = replaceind!(ops[op_row_b], H[opcode].site_ind', col_site_inds[op_row_b]')
        end
        ancD  = row >= 2 ? copy(Dbelow[opcode][row-1]) : copy(dummy)
        ancD *= EnvIdent.I[row]
        ancD *= A[row, col]
        ancD *= ops[row] * dag(prime(A[row, col]))
        ancD *= EnvIP.DiagInProgress[row, opcode]
        push!(Dbelow[opcode], ancD)
    end
    return Dbelow
end
