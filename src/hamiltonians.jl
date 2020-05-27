
function makeH_Ising(Nx::Int, Ny::Int, J::Real, hx, hz; pinning::Bool=false)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    X = ITensor(s, s')
    X[s(1), s'(2)] = 0.5
    X[s(2), s'(1)] = 0.5

    hxs = hx isa Matrix ? hx : fill(hx, Ny, Nx)
    hzs = hz isa Matrix ? hz : fill(hz, Ny, Nx)

    H = Matrix{Vector{Operator}}(undef, Ny, Nx)
    @inbounds for col in 1:Nx, row in 1:Ny
        H[row, col] = Vector{Operator}()
        if row < Ny
            if J != 0.0
                op_a  = J * copy(Z)
                op_b  = copy(Z)
                sites = [row=>col, row+1=>col]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))
            end
        end
        if col < Nx
            if J != 0.0
                op_a  = J * Z
                op_b  = copy(Z)
                sites = [row=>col, row=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))
            end
        end
        if hzs[ row, col ] != 0.0
            push!(H[row, col], Operator([row=>row], [hzs[row,col]*copy(Z)], s, Field))    
        end
        if hxs[ row, col ] != 0.0
            push!(H[row, col], Operator([row=>row], [hxs[row,col]*copy(X)], s, Field))    
        end
    end
    return H
end

function makeCuH_Ising(Nx::Int, Ny::Int, J::Real, hx, hz; pinning::Bool=false)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    X = ITensor(s, s')
    X[s(1), s'(2)] = 0.5 
    X[s(2), s'(1)] = 0.5

    hxs = hx isa Matrix ? hx : fill(hx, Ny, Nx)
    hzs = hz isa Matrix ? hz : fill(hz, Ny, Nx)

    H = Matrix{Vector{Operator}}(undef, Ny, Nx)
    @inbounds for col in 1:Nx, row in 1:Ny
        H[row, col] = Vector{Operator}()
        if row < Ny
            if J != 0.0
                op_a  = -J * Z
                op_b  = copy(Z)
                sites = [row=>col, row+1=>col]
                push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Vertical))
            end
        end
        if col < Nx
            if J != 0.0
                op_a  = -J * Z
                op_b  = copy(Z)
                sites = [row=>col, row=>col+1]
                push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Horizontal))
            end
        end
        if hzs[ row, col ] != 0.0
            push!(H[row, col], Operator([row=>col], [cuITensor(hzs[row,col]*copy(Z))], s, Field))    
        end
        if hxs[ row, col ] != 0.0
            push!(H[row, col], Operator([row=>col], [cuITensor(hxs[row,col]*copy(X))], s, Field))    
        end
    end
    return H
end

function makeH_XXZ(Nx::Int, Ny::Int, J::Real; pinning::Bool=false)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    P = ITensor(s, s')
    M = ITensor(s, s')
    P[s(1), s'(2)] = 1.0
    M[s(2), s'(1)] = 1.0
    H = Matrix{Vector{Operator}}(undef, Ny, Nx)
    @inbounds for col in 1:Nx, row in 1:Ny
        H[row, col] = Vector{Operator}()
        if row < Ny
            op_a  = 0.5 * P
            op_b  = copy(M)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))

            op_a  = 0.5 * M
            op_b  = copy(P)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))

            if J != 0.0
                op_a  = J * Z
                op_b  = copy(Z)
                sites = [row=>col, row+1=>col]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))
            end
        end
        if col < Nx
            op_a  = 0.5 * P
            op_b  = copy(M)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))

            op_a  = 0.5 * M
            op_b  = copy(P)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))

            if J != 0.0
                op_a  = J * Z
                op_b  = copy(Z)
                sites = [row=>col, row=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))
            end
        end
    end
    # pinning fields
    J_ = 1.;
    @inbounds for row in 1:Ny
        op = isodd(row-1) ? (J_/2.0) * Z : (-J_/2.0) * Z   
        push!(H[row, 1], Operator([row=>1], [op], s, Field))    
        op = isodd(row-1) ? (-J_/2.0) * Z : (J_/2.0) * Z   
        push!(H[row, Nx], Operator([row=>Nx], [op], s, Field))    
    end
    @inbounds for col in 2:Nx-1
        op = isodd(col-1) ? (-J_/2.0) * Z : (J_/2.0) * Z   
        push!(H[Ny, col], Operator([Ny=>col], [op], s, Field))    
        op = isodd(col-1) ? (J_/2.0) * Z : (-J_/2.0) * Z   
        push!(H[1, col], Operator([1=>col], [op], s, Field))    
    end
    return H
end

function makeH_J1J2(Nx::Int, Ny::Int, J1::Real, J2::Real; pinning::Bool=false)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    P = ITensor(s, s')
    M = ITensor(s, s')
    P[s(1), s'(2)] = 1.0
    M[s(2), s'(1)] = 1.0
    H = Matrix{Vector{Operator}}(undef, Ny, Nx)
    @inbounds for col in 1:Nx, row in 1:Ny
        H[row, col] = Vector{Operator}()
        if row < Ny
            op_a  = 0.5 * J1 * P
            op_b  = copy(M)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))

            op_a  = 0.5 * J1 * M
            op_b  = copy(P)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))

            op_a  = J1 * Z
            op_b  = copy(Z)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))
        end
        if col < Nx
            op_a  = 0.5 * J1 * P
            op_b  = copy(M)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))

            op_a  = 0.5 * J1 * M
            op_b  = copy(P)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))

            op_a  = J1 * Z
            op_b  = copy(Z)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))
        end
        # handle J2
        if col < Nx
            if row < Ny
                op_a  = 0.5 * J2 * P
                op_b  = copy(M)
                sites = [row=>col, row+1=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Diag))

                op_a  = 0.5 * J2 * M
                op_b  = copy(P)
                sites = [row=>col, row+1=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Diag))

                op_a  = J2 * Z
                op_b  = copy(Z)
                sites = [row=>col, row+1=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Diag))
            end
            if row > 1
                op_a  = 0.5 * J2 * P
                op_b  = copy(M)
                sites = [row=>col, row-1=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Diag))

                op_a  = 0.5 * J2 * M
                op_b  = copy(P)
                sites = [row=>col, row-1=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Diag))

                op_a  = J2 * Z
                op_b  = copy(Z)
                sites = [row=>col, row-1=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Diag))
            end
        end
    end
    return H
end

function makeCuH_J1J2(Nx::Int, Ny::Int, J1::Real, J2::Real; pinning::Bool=false)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    P = ITensor(s, s')
    M = ITensor(s, s')
    P[s(1), s'(2)] = 1.0
    M[s(2), s'(1)] = 1.0
    H = Matrix{Vector{Operator}}(undef, Ny, Nx)
    @inbounds for col in 1:Nx, row in 1:Ny
        H[row, col] = Vector{Operator}()
        if row < Ny
            op_a  = cuITensor(0.5 * J1 * P)
            op_b  = cuITensor(copy(M))
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))

            op_a  = cuITensor(0.5 * J1 * M)
            op_b  = cuITensor(copy(P))
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))

            op_a  = cuITensor(J1 * Z)
            op_b  = cuITensor(copy(Z))
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Vertical))
        end
        if col < Nx
            op_a  = cuITensor(0.5 * J1 * P)
            op_b  = cuITensor(copy(M))
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))

            op_a  = cuITensor(0.5 * J1 * M)
            op_b  = cuITensor(copy(P))
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))

            op_a  = cuITensor(J1 * Z)
            op_b  = cuITensor(copy(Z))
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [op_a; op_b], s, Horizontal))
        end
        # handle J2
        if col < Nx
            if row < Ny
                op_a  = cuITensor(0.5 * J2 * P)
                op_b  = cuITensor(copy(M))
                sites = [row=>col, row+1=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Diag))

                op_a  = cuITensor(0.5 * J2 * M)
                op_b  = cuITensor(copy(P))
                sites = [row=>col, row+1=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Diag))

                op_a  = cuITensor(J2 * Z)
                op_b  = cuITensor(copy(Z))
                sites = [row=>col, row+1=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Diag))
            end
            if row > 1
                op_a  = cuITensor(0.5 * J2 * P)
                op_b  = cuITensor(copy(M))
                sites = [row=>col, row-1=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Diag))

                op_a  = cuITensor(0.5 * J2 * M)
                op_b  = cuITensor(copy(P))
                sites = [row=>col, row-1=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Diag))

                op_a  = cuITensor(J2 * Z)
                op_b  = cuITensor(copy(Z))
                sites = [row=>col, row-1=>col+1]
                push!(H[row, col], Operator(sites, [op_a; op_b], s, Diag))
            end
        end
    end
    return H
end


function makeCuH_XXZ(Nx::Int, Ny::Int, J::Real; pinning::Bool=false)
    s = Index(2, "Site,SpinInd")
    Z = ITensor(s, s')
    Z[s(1), s'(1)] = 0.5
    Z[s(2), s'(2)] = -0.5
    #Ident = spinI(s)
    P = ITensor(s, s')
    M = ITensor(s, s')
    P[s(1), s'(2)] = 1.0
    M[s(2), s'(1)] = 1.0
    H = Matrix{Vector{Operator}}(undef, Ny, Nx)
    @inbounds for col in 1:Nx, row in 1:Ny
        H[row, col] = Vector{Operator}()
        if row < Ny
            op_a  = 0.5 * P
            op_b  = copy(M)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Vertical))

            op_a  = 0.5 * M
            op_b  = copy(P)
            sites = [row=>col, row+1=>col]
            push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Vertical))

            if J != 0.0
                op_a  = J * Z
                op_b  = copy(Z)
                sites = [row=>col, row+1=>col]
                push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Vertical))
            end
        end
        if col < Nx
            op_a  = 0.5 * P
            op_b  = copy(M)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Horizontal))

            op_a  = 0.5 * M
            op_b  = copy(P)
            sites = [row=>col, row=>col+1]
            push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Horizontal))

            if J != 0.0
                op_a  = J * Z
                op_b  = copy(Z)
                sites = [row=>col, row=>col+1]
                push!(H[row, col], Operator(sites, [cuITensor(op_a); cuITensor(op_b)], s, Horizontal))
            end
        end
    end
    # pinning fields
    J_ = 1.
    @inbounds for row in 1:Ny
        op = isodd(row-1) ? (J_/2.0) * Z : (-J_/2.0) * Z   
        push!(H[row, 1], Operator([row=>1], [cuITensor(op)], s, Field))    
        op = isodd(row-1) ? (-J_/2.0) * Z : (J_/2.0) * Z   
        push!(H[row, Nx], Operator([row=>Nx], [cuITensor(op)], s, Field))    
    end
    @inbounds for col in 2:Nx-1
        op = isodd(col-1) ? (-J_/2.0) * Z : (J_/2.0) * Z   
        push!(H[Ny, col], Operator([Ny=>col], [cuITensor(op)], s, Field))    
        op = isodd(col-1) ? (J_/2.0) * Z : (-J_/2.0) * Z   
        push!(H[1, col], Operator([1=>col], [cuITensor(op)], s, Field))    
    end
    return H
end

