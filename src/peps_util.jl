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

include("single_site.jl")
include("two_site_hori.jl")
include("two_site_vert.jl")

function spinI(s::Index; is_gpu::Bool=false)::ITensor
    I_data      = is_gpu ? CUDA.zeros(Float64, ITensors.dim(s)*ITensors.dim(s)) : zeros(Float64, ITensors.dim(s), ITensors.dim(s))
    idi         = diagind(reshape(I_data, ITensors.dim(s), ITensors.dim(s)), 0)
    I_data[idi] = is_gpu ? CUDA.ones(Float64, ITensors.dim(s)) : ones(Float64, ITensors.dim(s))
    I           = is_gpu ? cuITensor( I_data, IndexSet(s, s') ) : itensor(I_data, IndexSet(s, s'))
    return I
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
                  max_gauge_iter::Int=100,
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

