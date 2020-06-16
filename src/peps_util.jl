mutable struct fPEPS
    Nx::Int
    Ny::Int
    A_::AbstractMatrix{ITensor}

    fPEPS() = new(0, 0, Matrix{ITensor}(),0,0)

    fPEPS(Nx::Int, Ny::Int, A::Matrix{ITensor}) = new(Nx, Ny, A)
    function fPEPS(sites, lattice::Lattice, Nx::Int, Ny::Int; mindim::Int=1, is_gpu::Bool=false)
        p           = Matrix{ITensor}(undef, Ny, Nx)
        right_links = [ Index(mindim, "Link,c$j,r$i,r") for i in 1:Ny, j in 1:Nx ]
        up_links    = [ Index(mindim, "Link,c$j,r$i,u") for i in 1:Ny, j in 1:Nx ]
        T           = is_gpu ? cuITensor : ITensor
        @inbounds for ii in eachindex(sites)
            col = div(ii-1, Ny) + 1
            row = mod(ii-1, Ny) + 1
            old_s = sites[ii]
            s = Index(dim(old_s), "Site,r$row,c$col")
            if 1 < row < Ny && 1 < col < Nx 
                p[row, col] = T(right_links[row, col], up_links[row, col], right_links[row, col-1], up_links[row-1, col], s)
            elseif row == 1 && 1 < col < Nx
                p[row, col] = T(right_links[row, col], up_links[row, col], right_links[row, col-1], s)
            elseif 1 < row < Ny && col == 1
                p[row, col] = T(right_links[row, col], up_links[row, col], up_links[row-1, col], s)
            elseif row == Ny && 1 < col < Nx 
                p[row, col] = T(right_links[row, col], right_links[row, col-1], up_links[row-1, col], s)
            elseif 1 < row < Ny && col == Nx 
                p[row, col] = T(up_links[row, col], right_links[row, col-1], up_links[row-1, col], s)
            elseif row == Ny && col == 1 
                p[row, col] = T(right_links[row, col], up_links[row-1, col], s)
            elseif row == Ny && col == Nx 
                p[row, col] = T(right_links[row, col-1], up_links[row-1, col], s)
            elseif row == 1 && col == 1 
                p[row, col] = T(right_links[row, col], up_links[row, col], s)
            elseif row == 1 && col == Nx
                p[row, col] = T(up_links[row, col], right_links[row, col-1], s)
            end
            @assert p[row, col] == p[ii]
        end
        new(Nx, Ny, p)
    end
end
Base.eltype(A::fPEPS) = eltype(A.A_[1,1])

function cudelt(left::Index, right::Index)
    d_data   = CUDA.zeros(Float64, ITensors.dim(left), ITensors.dim(right))
    ddi = diagind(d_data, 0)
    d_data[ddi] = 1.0
    delt = cuITensor(vec(d_data), left, right)
    return delt
end

function mydelt(left::Index, right::Index)
    d_data   = zeros(Float64, ITensors.dim(left), ITensors.dim(right))
    ddi = diagind(d_data, 0)
    d_data[ddi] = ones(Float64, length(ddi))
    delt = ITensor(vec(d_data), left, right)
    return delt
end

function checkerboardfPEPS(sites, Nx::Int, Ny::Int; mindim::Int=1)
    lattice = square_lattice(Nx, Ny,yperiodic=false)
    A = fPEPS(sites, lattice, Nx, Ny, mindim=mindim)
    @inbounds for ii ∈ eachindex(sites)
        row = div(ii-1, Nx) + 1
        col = mod(ii-1, Nx) + 1
        spin_side = isodd(row - 1) ⊻ isodd(col - 1) ? 2 : 1
        si     = firstind(A[ii], "Site") 
        lis = filter(inds(A[ii]), "Link") 
        ivs = [li(1) for li in lis]
        ivs = vcat(ivs, si(spin_side))
        A[ii][ivs...] = 1.0
    end
    for row in 1:Ny, col in 1:Nx
        A[row, col] += randomITensor(inds(A[row, col]))/10.0
    end
    return A
end

function randomfPEPS(sites, Nx::Int, Ny::Int; mindim::Int=1)
    lattice = square_lattice(Nx,Ny,yperiodic=false)
    A = fPEPS(sites, lattice, Nx, Ny, mindim=mindim)
    @inbounds for ii ∈ eachindex(sites)
        randn!(A[ii])
        normalize!(A[ii])
    end
    return A
end

is_gpu(A::fPEPS)    = all(is_gpu.(A[:,:]))
is_gpu(A::Vector{ITensor})    = all(is_gpu.(A[:]))
is_gpu(A::ITensor) = (NDTensors.data(store(A)) isa CuArray)

include("environments.jl")
include("ancillaries.jl")
include("gauge.jl")
include("observables.jl")
include("hamiltonians.jl")

function randomCufPEPS(sites, Nx::Int, Ny::Int; mindim::Int=1)
    lattice = square_lattice(Nx,Ny,yperiodic=false)
    A = fPEPS(sites, lattice, Nx, Ny; mindim=mindim, is_gpu=true)
    @inbounds for ii ∈ eachindex(sites)
        randn!(A[ii])
        normalize!(A[ii])
    end
    return A
end

function cufPEPS(A::fPEPS)
    Ny, Nx = size(A)
    cA     = similar(A)
    @inbounds for i in 1:Ny, j in 1:Nx
        cA[i, j] = cuITensor(A[i, j])
    end
    return cA
end

function Base.collect(cA::fPEPS)
    Ny, Nx = size(A)
    A = similar(cA)
    @inbounds for i in 1:Ny, j in 1:Nx
        A[i, j] = collect(cA[i, j])
    end
    return A
end

store(A::fPEPS)   = A.A_
Base.size(A::fPEPS) = (A.Ny, A.Nx)

Base.getindex(A::fPEPS, i::Integer, j::Integer) = getindex(store(A), i, j)::ITensor
Base.getindex(A::fPEPS, ::Colon,    j::Integer) = getindex(store(A), :, j)::Vector{ITensor}
Base.getindex(A::fPEPS, i::Integer, ::Colon)    = getindex(store(A), i, :)::Vector{ITensor}
Base.getindex(A::fPEPS, ::Colon,    j::UnitRange{Int}) = getindex(store(A), :, j)::Matrix{ITensor}
Base.getindex(A::fPEPS, i::UnitRange{Int}, ::Colon)    = getindex(store(A), i, :)::Matrix{ITensor}
Base.getindex(A::fPEPS, ::Colon, ::Colon)       = getindex(store(A), :, :)::Matrix{ITensor}
Base.getindex(A::fPEPS, i::Integer)             = getindex(store(A), i)::ITensor

Base.setindex!(A::fPEPS, val::ITensor, i::Integer, j::Integer)       = setindex!(store(A), val, i, j)
Base.setindex!(A::fPEPS, vals::Vector{ITensor}, ::Colon, j::Integer) = setindex!(store(A), vals, :, j)
Base.setindex!(A::fPEPS, vals::Vector{ITensor}, i::Integer, ::Colon) = setindex!(store(A), vals, i, :)
Base.setindex!(A::fPEPS, vals::Matrix{ITensor}, ::Colon, j::UnitRange{Int}) = setindex!(store(A), vals, :, j)
Base.setindex!(A::fPEPS, vals::Matrix{ITensor}, i::UnitRange{Int}, ::Colon) = setindex!(store(A), vals, i, :)

Base.copy(A::fPEPS)    = fPEPS(A.Nx, A.Ny, copy(store(A)))
Base.similar(A::fPEPS) = fPEPS(A.Nx, A.Ny, similar(store(A)))

function Base.show(io::IO, A::fPEPS)
  print(io,"fPEPS")
  (size(A)[1] > 0 && size(A)[2] > 0) && print(io,"\n")
  @inbounds for i in 1:A.Nx, j in 1:A.Ny
      println(io,"$i $j $(A[i,j])")
  end
end

@enum Op_Type Field=0 Vertical=1 Horizontal=2 Diag=3
struct Operator
    sites::Vector{Pair{Int,Int}}
    ops::Vector{ITensor}
    site_ind::Index
    dir::Op_Type
end

Base.show(io::IO, o::Operator) = show(io, "Direction: $(o.dir)\nSites [row => col]: $(o.sites)\n")

getDirectional(ops::Vector{Operator}, dir::Op_Type) = collect(filter(x->x.dir==dir, ops))

function spinI(s::Index; is_gpu::Bool=false)::ITensor
    I_data      = is_gpu ? CUDA.zeros(Float64, ITensors.dim(s)*ITensors.dim(s)) : zeros(Float64, ITensors.dim(s), ITensors.dim(s))
    idi         = diagind(reshape(I_data, ITensors.dim(s), ITensors.dim(s)), 0)
    I_data[idi] = is_gpu ? CUDA.ones(Float64, ITensors.dim(s)) : ones(Float64, ITensors.dim(s))
    I           = is_gpu ? cuITensor( I_data, IndexSet(s, s') ) : itensor(I_data, IndexSet(s, s'))
    return I
end

function combine(Aorig::ITensor, Anext::ITensor, tags::String)::ITensor
    ci        = commonind(Aorig, Anext)
    return combiner(IndexSet(ci, prime(ci)), tags=tags)
end

function reconnect(combiner_ind::Index, environment::ITensor)
    environment_combiner = firstind(environment, "Site")
    new_combiner         = combiner(IndexSet(combiner_ind, prime(combiner_ind)), tags="Site")
    combined_ind         = combinedind(new_combiner)
    combiner_transfer    = δ(combined_ind, environment_combiner)
    
    #return new_combiner*combiner_transfer
    replaceind!(new_combiner, combined_ind, environment_combiner)
    return new_combiner
end

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

function nonWorkRow(A::fPEPS, 
                    L::Environments, 
                    R::Environments, 
                    H::Operator, 
                    row::Int, 
                    col::Int)::ITensor
    Ny, Nx  = size(A)
    op_rows = H.sites
    is_cu   = is_gpu(A) 
    ops     = deepcopy(H.ops)
    @inbounds for op_ind in 1:length(ops)
        as = firstind(A[op_rows[op_ind][1][1], col], "Site")
        ops[op_ind] = replaceind!(ops[op_ind], H.site_ind, as)
        ops[op_ind] = replaceind!(ops[op_ind], H.site_ind', as')
    end
    op = spinI(firstind(A[row, col], "Site"); is_gpu=is_cu)
    op_ind = findfirst( x -> x == row, op_rows)
    AA *= L.I[row]
    AA *= A[row, col] * op
    AA *= R.I[row]
    AA *= dag(A[row, col])'
    return AA
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
        IH_b *= E.H[work_row] * Aj[work_row]
        IH_b *= dag(prime(Aj[work_row], "Link"))
    end
    @inbounds for work_row in row+1:Ny
        IH_a *= E.H[work_row] * Aj[work_row]
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

function intraColumnGauge(A::fPEPS, col::Int; kwargs...)::fPEPS
    Ny, Nx = size(A)
    @inbounds for row in reverse(2:Ny)
        @debug "\tBeginning intraColumnGauge for col $col row $row"
        cmb_is     = IndexSet(firstind(A[row, col], "Site"))
        if col > 1
            cmb_is = IndexSet(cmb_is..., commonind(A[row, col], A[row, col - 1]))
        end
        if col < Nx
            cmb_is = IndexSet(cmb_is..., commonind(A[row, col], A[row, col + 1]))
        end
        cmb = combiner(cmb_is, tags="CMB")
        Lis = IndexSet(combinedind(cmb)) #cmb_is
        if row < Ny
            Lis = IndexSet(Lis..., commonind(A[row, col], A[row + 1, col]))
        end
        old_ci         = commonind(A[row, col], A[row-1, col])
        Ac             = A[row, col]*cmb
        U, S, V        = svd(Ac, Lis; kwargs...)
        A[row, col]    = U*cmb
        A[row-1, col] *= (S*V)
        new_ci         = commonind(A[row, col], A[row-1, col])
        A[row, col]    = replaceind!(A[row, col], new_ci, old_ci)
        A[row-1, col]  = replaceind!(A[row-1, col], new_ci, old_ci)
    end
    return A
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

function simpleUpdate(A::fPEPS, col::Int, next_col::Int, H; kwargs...)::fPEPS
    do_side::Bool = get(kwargs, :do_side, true)
    τ::Float64    = get(kwargs, :tau, -0.1)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A) 
    @inbounds for row in Iterators.reverse(1:Ny)
        if do_side
            hori_col   = next_col < col ? next_col : col
            nhori_col  = next_col < col ? col : next_col
            si_a       = firstind(A[row, col], "Site")
            si_b       = firstind(A[row, next_col], "Site")
            ci         = commonind(A[row, col], A[row, next_col])
            min_dim    = ITensors.dim(ci)
            Ua, Sa, Va = svd(A[row, col], si_a, ci; mindim=min_dim, kwargs...)
            Ub, Sb, Vb = svd(A[row, next_col], si_b, ci; mindim=min_dim, kwargs...)
            Hab_hori   = is_cu ? cuITensor() : ITensor()
            horiH      = getDirectional(vcat(H[:, hori_col]...), Horizontal)
            horiH      = filter(x->x.sites[1][1] == row, horiH)
            for hH in horiH
                op_a = replaceind!(copy(hH.ops[1]), hH.site_ind, hori_col == col ? si_a : si_b)
                op_a = replaceind!(op_a, hH.site_ind', hori_col == col ? si_a' : si_b')
                op_b = replaceind!(copy(hH.ops[2]), hH.site_ind, hori_col == col ? si_b : si_a)
                op_b = replaceind!(op_b, hH.site_ind', hori_col == col ? si_b' : si_a')
                Hab_hori = ITensors.dim(Hab_hori) < 2 ? op_a * op_b : Hab_hori + op_a * op_b
            end
            cmb       = combiner(findinds(Hab_hori, 0), tags="hab,Site")
            ci        = combinedind(cmb)
            Hab_hori *= cmb
            Hab_hori *= cmb'
            Hab_mat   = is_cu ? matrix(collect(Hab_hori)) : matrix(Hab_hori)
            expiH_mat = exp(τ*Hab_mat)
            expiH     = is_cu ? cuITensor(vec(expiH_mat), ci, ci') : itensor(expiH_mat, ci, ci')
            expiH *= cmb
            expiH *= cmb'
             
            bond  = noprime(expiH * Ua * Ub)
            Uf, Sf, Vf = svd(bond, si_a, commonind(Ua, Sa); vtags="r,Link,r$row,c$hori_col", mindim=min_dim, kwargs...)
            A[row, col] = Sa * Va * Uf * Sf
            A[row, next_col] = Sb * Vb * Vf
        end
        if row < Ny
            si_a       = firstind(A[row, col], "Site")
            si_b       = firstind(A[row+1, col], "Site")
            ci         = commonind(A[row, col], A[row+1, col])
            min_dim    = ITensors.dim(ci)
            Ua, Sa, Va = svd(A[row, col], si_a, ci; mindim=min_dim, kwargs...)
            Ub, Sb, Vb = svd(A[row+1, col], si_b, ci; mindim=min_dim, kwargs...)
            Hab_vert   = is_cu ? cuITensor() : ITensor()
            vertH      = getDirectional(vcat(H[:, col]...), Vertical)
            vertH      = filter(x->x.sites[1][1] == row, vertH)
            for vH in vertH
                op_a = replaceind!(copy(vH.ops[1]), vH.site_ind, si_a)
                op_a = replaceind!(op_a, vH.site_ind', si_a')
                op_b = replaceind!(copy(vH.ops[2]), vH.site_ind, si_b)
                op_b = replaceind!(op_b, vH.site_ind', si_b')
                Hab_vert = ITensors.dim(Hab_vert) < 2 ? op_a * op_b : Hab_vert + op_a * op_b
            end
            cmb       = combiner(findinds(Hab_vert, 0), tags="hab,Site")
            ci        = combinedind(cmb)
            Hab_vert *= cmb
            Hab_vert *= cmb'
            Hab_mat   = is_cu ? matrix(collect(Hab_vert)) : matrix(Hab_vert)
            expiH_mat = exp(τ*Hab_mat)
            expiH     = is_cu ? cuITensor(vec(expiH_mat), ci, ci') : itensor(expiH_mat, ci, ci')
            expiH *= cmb
            expiH *= cmb'
            bond  = noprime(expiH * Ua * Ub)
            Uf, Sf, Vf = svd(bond, si_a, commonind(Ua, Sa); vtags="u,Link,r$row,c$col", mindim=min_dim, kwargs...)
            A[row, col] = Sa * Va * Uf * Sf
            A[row+1, col] = Sb * Vb * Vf
        end
    end
    return A
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
function measureEnergy(A::fPEPS, 
                       L::Environments, R::Environments, 
                       AncEnvs, H, 
                       row::Int, col::Int)::Tuple{Float64, Float64}
    Ny, Nx    = size(A)
    Hs, N     = buildLocalH(A, L, R, AncEnvs, H, row, col, A[row, col])
    initial_N = collect(N * dag(A[row, col])')
    localH    = sum(Hs)
    initial_E = collect(localH * dag(A[row, col])')
    return real(scalar(initial_N)), real(scalar(initial_E))/real(scalar(initial_N))
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
    two_site_horizontal::Bool = get(kwargs, :two_site_horizontal, false)
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

function sweepColumnHorizontal(A::fPEPS, 
                               L::Environments, R::Environments, 
                               H, col::Int, ncol::Int; 
                               kwargs...)
    Ny, Nx = size(A)
    #println("Sweeping column $col with next column $ncol")
    is_cu  = is_gpu(A)
    dummyI = is_cu ? MPS([cuITensor(1.0) for ii in 1:Ny], 0, Ny+1) : MPS([ITensor(1.0) for ii in 1:Ny], 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny), fill(ITensor(), 1, Ny)) 
    #=@timeit "intraColumnGauge" begin
        A = intraColumnGauge(A, col; kwargs...)
    end
    L_s = buildLs(A, H; kwargs...)
    R_s = buildRs(A, H; kwargs...)
    if 1 < col < Nx
        EAncEnvs = buildAncs(A, L_s[col - 1], R_s[col + 1], H, col)
        N, E = measureEnergy(A, L_s[col - 1], R_s[col + 1], EAncEnvs, H, 1, col)
        println("Energy at col $col: ", E/(Nx*Ny), " and norm: ", N)
    elseif col == 1
        EAncEnvs = buildAncs(A, dummyEnv, R_s[col + 1], H, col)
        N, E = measureEnergy(A, dummyEnv, R_s[col + 1], EAncEnvs, H, 1, col)
        println("Energy at col $col: ", E/(Nx*Ny), " and norm: ", N)
    elseif col == Nx
        EAncEnvs = buildAncs(A, L_s[col - 1], dummyEnv, H, col)
        N, E = measureEnergy(A, L_s[col - 1], dummyEnv, EAncEnvs, H, 1, col)
        println("Energy at col $col: ", E/(Nx*Ny), " and norm: ", N)
    end=#
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

function rightwardSweep(A::fPEPS, 
                        Ls::Vector{Environments}, 
                        Rs::Vector{Environments}, 
                        H; 
                        kwargs...)
    simple_update_cutoff = get(kwargs, :simple_update_cutoff, 4)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    dummyI = is_cu ? MPS([cuITensor(1.0) for ii in 1:Ny], 0, Ny+1) : MPS([ITensor(1.0) for ii in 1:Ny], 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny), fill(ITensor(), 1, Ny)) 
    sweep::Int = get(kwargs, :sweep, 0)
    sweep_width::Int = get(kwargs, :sweep_width, Nx)
    offset    = mod(Nx, 2)
    midpoint  = div(Nx, 2)
    rightmost = sweep_width == Nx ? Nx - 1 : midpoint + div(sweep_width, 2) + offset
    leftmost  = sweep_width == Nx ? 1 : midpoint - div(sweep_width, 2)
    ts_hori::Bool = get(kwargs, :two_site_hori, false)
    if ts_hori
        @inbounds for col in leftmost:rightmost
            next_col = col + 1
            L = col == 1 ? dummyEnv : Ls[col - 1]
            R = next_col == Nx ? dummyEnv : Rs[col + 2]
            @debug "Sweeping col $col"
            if sweep >= simple_update_cutoff
                @timeit "sweep" begin
                    #A = sweepColumnHorizontal(A, L, R, H, col, next_col; kwargs...)
                    A, Aj = sweepColumnHorizontal(A, L, R, H, col, next_col; kwargs...)
                end
            end
            if sweep < simple_update_cutoff
                # Simple update...
                A = simpleUpdate(A, col, col+1, H; do_side=(col < Nx), kwargs...)
            end
            if sweep >= simple_update_cutoff
                # Gauge
                #A = gaugeColumn(A, col, :right; kwargs...)
                A = gaugeColumnHorizontal(A, Aj, col, next_col, :right; kwargs...)
            end
            if col == 1
                left_H_terms = getDirectional(vcat(H[:, 1]...), Horizontal)
                @timeit "left edge env" begin
                    Ls[col] = buildEdgeEnvironment(A, H, left_H_terms, :left, 1; kwargs...)
                end
            else
                @timeit "left next env" begin
                    Ls[col] = buildNextEnvironment(A, Ls[col-1], H, :left, col; kwargs...)
                end
            end
        end
        Ls[Nx-1] = buildNextEnvironment(A, Ls[Nx-2], H, :left, Nx-1; kwargs...)
        @timeit "intraColumnGauge" begin
            A = intraColumnGauge(A, Nx; kwargs...)
        end
        EAncEnvs = buildAncs(A, Ls[Nx - 1], dummyEnv, H, Nx)
        N, E = measureEnergy(A, Ls[Nx - 1], dummyEnv, EAncEnvs, H, 1, Nx)
        println("Energy at Nx: ", E/(Nx*Ny), " and norm: ", N)
        println()
        println()
    else
        @inbounds for col in leftmost:rightmost
            L = col == 1 ? dummyEnv : Ls[col - 1]
            @debug "Sweeping col $col"
            if sweep >= simple_update_cutoff
                @timeit "sweep" begin
                    A = sweepColumn(A, L, Rs[col+1], H, col; kwargs...)
                end
            end
            if sweep < simple_update_cutoff
                # Simple update...
                A = simpleUpdate(A, col, col+1, H; do_side=(col < Nx), kwargs...)
            end
            if sweep >= simple_update_cutoff
                # Gauge
                A = gaugeColumn(A, col, :right; kwargs...)
            end
            if col == 1
                left_H_terms = getDirectional(vcat(H[:, 1]...), Horizontal)
                @timeit "left edge env" begin
                    Ls[col] = buildEdgeEnvironment(A, H, left_H_terms, :left, 1; kwargs...)
                end
            else
                @timeit "left next env" begin
                    Ls[col] = buildNextEnvironment(A, Ls[col-1], H, :left, col; kwargs...)
                end
            end
        end
    end
    return A, Ls, Rs
end

function leftwardSweep(A::fPEPS, 
                       Ls::Vector{Environments}, 
                       Rs::Vector{Environments}, 
                       H; 
                       kwargs...)
    simple_update_cutoff = get(kwargs, :simple_update_cutoff, 4)
    Ny, Nx = size(A)
    is_cu  = is_gpu(A)
    dummyI = is_cu ? MPS([cuITensor(1.0) for ii in 1:Ny], 0, Ny+1) : MPS([ITensor(1.0) for ii in 1:Ny], 0, Ny+1)
    dummyEnv = Environments(dummyI, dummyI, fill(ITensor(), 1, Ny), fill(ITensor(), 1, Ny)) 
    sweep::Int = get(kwargs, :sweep, 0)
    sweep_width::Int = get(kwargs, :sweep_width, Nx)
    offset    = mod(Nx, 2)
    midpoint  = div(Nx, 2)
    rightmost = midpoint + div(sweep_width, 2) + offset
    leftmost  = sweep_width == Nx ? 2 : midpoint - div(sweep_width, 2)
    ts_hori::Bool = get(kwargs, :two_site_hori, false)
    if ts_hori
        @inbounds for col in reverse(leftmost:rightmost)
            next_col = col - 1
            R = col == Nx    ? dummyEnv    : Rs[col + 1]
            L = next_col > 1 ? Ls[col - 2] : dummyEnv
            @debug "Sweeping col $col and next col $(col - 1)"
            if sweep >= simple_update_cutoff
                @timeit "sweep" begin
                    A, Aj = sweepColumnHorizontal(A, L, R, H, col, next_col; kwargs...)
                    #A = sweepColumnHorizontal(A, L, R, H, col, next_col; kwargs...)
                end
            end
            if sweep < simple_update_cutoff
                # Simple update...
                A = simpleUpdate(A, col, col-1, H; do_side=(col>1), kwargs...)
            end
            if sweep >= simple_update_cutoff
                # Gauge
                A = gaugeColumnHorizontal(A, Aj, col, next_col, :left; kwargs...)
                #A = gaugeColumn(A, col, :left; kwargs...)
            end
            if col == Nx
                right_H_terms  = getDirectional(vcat(H[:, Nx - 1]...), Horizontal)
                @timeit "right edge env" begin
                    Rs[col] = buildEdgeEnvironment(A, H, right_H_terms, :right, Nx; kwargs...)
                end
            else
                @timeit "right next env" begin
                    Rs[col] = buildNextEnvironment(A, Rs[col+1], H, :right, col; kwargs...)
                end
            end
        end
        Rs[2] = buildNextEnvironment(A, Rs[3], H, :right, 2; kwargs...)
        @timeit "intraColumnGauge" begin
            A = intraColumnGauge(A, 1; kwargs...)
        end
        EAncEnvs = buildAncs(A, dummyEnv, Rs[2], H, 1)
        N, E = measureEnergy(A, dummyEnv, Rs[2], EAncEnvs, H, 1, 1)
        println("Energy at 1: ", E/(Nx*Ny), " and norm: ", N)
        println()
        println()
    else
        @inbounds for col in reverse(leftmost:rightmost)
            R = col == Nx ? dummyEnv : Rs[col + 1]
            @debug "Sweeping col $col"
            if sweep >= simple_update_cutoff
                @timeit "sweep" begin
                    A = sweepColumn(A, Ls[col - 1], R, H, col; kwargs...)
                end
            end
            if sweep < simple_update_cutoff
                # Simple update...
                A = simpleUpdate(A, col, col-1, H; do_side=(col>1), kwargs...)
            end
            if sweep >= simple_update_cutoff
                # Gauge
                A = gaugeColumn(A, col, :left; kwargs...)
            end
            if col == Nx
                right_H_terms  = getDirectional(vcat(H[:, Nx - 1]...), Horizontal)
                @timeit "right edge env" begin
                    Rs[col] = buildEdgeEnvironment(A, H, right_H_terms, :right, Nx; kwargs...)
                end
            else
                @timeit "right next env" begin
                    Rs[col] = buildNextEnvironment(A, Rs[col+1], H, :right, col; kwargs...)
                end
            end
        end
    end
    return A, Ls, Rs
end

function doSweeps(A::fPEPS, 
                  Ls::Vector{Environments}, 
                  Rs::Vector{Environments},
                  H; 
                  mindim::Int=1, 
                  maxdim::Int=1, 
                  simple_update_cutoff::Int=4, 
                  sweep_start::Int=1, 
                  sweep_count::Int=10, 
                  cutoff::Float64=0., 
                  env_maxdim=2maxdim, 
                  do_mag::Bool=false, 
                  prefix="mag", 
                  max_gauge_iter::Int=200,
                  model::Symbol=:XXZ,
                  two_site_vert::Bool=false,
                  two_site_hori::Bool=false,
                  rotation::Bool=false)
    Ny, Nx = size(A) 
    for sweep in sweep_start:sweep_count
        overlap_cutoff = 0.999
        if iseven(sweep)
            (A, Ls, Rs), this_time, bytes, gctime, memallocs = @timed rightwardSweep(A, Ls, Rs, H; sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=overlap_cutoff, cutoff=cutoff, env_maxdim=env_maxdim, max_gauge_iter=max_gauge_iter, two_site_vert=two_site_vert, two_site_hori=two_site_hori)
            println("SWEEP RIGHT $sweep, time $this_time")
        else
            (A, Ls, Rs), this_time, bytes, gctime, memallocs = @timed leftwardSweep(A, Ls, Rs, H; sweep=sweep, mindim=mindim, maxdim=maxdim, simple_update_cutoff=simple_update_cutoff, overlap_cutoff=overlap_cutoff, cutoff=cutoff, env_maxdim=env_maxdim, max_gauge_iter=max_gauge_iter, two_site_vert=two_site_vert, two_site_hori=two_site_hori)
            println("SWEEP LEFT $sweep, time $this_time")
        end
        flush(stdout)
        if sweep == simple_update_cutoff - 1
            for col in reverse(2:Nx)
                A = gaugeColumn(A, col, :left; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim, max_gauge_iter=max_gauge_iter)
            end
            Ls = buildLs(A, H; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim)
            Rs = buildRs(A, H; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim)
        end
        # make this dispatch at some point
        if do_mag && model == :XXZ
            measure_correlators_heisenberg(copy(A), H, Ls, Rs, sweep;  mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim, prefix=prefix)
        elseif do_mag && model == :Ising
            measure_correlators_ising(copy(A), H, Ls, Rs, sweep; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim, prefix=prefix)
        end
        if rotation && mod(sweep, 10) == 0
            new_A = copy(A)
            for row in 1:Ny, col in 1:Nx
                new_A[col, row] = A[row, col]
            end
            Ls = buildLs(A, H; mindim=maxdim, maxdim=maxdim, cutoff=cutoff, env_maxdim=env_maxdim)
        end
    end
    return A
end

