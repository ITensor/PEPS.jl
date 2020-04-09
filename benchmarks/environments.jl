suite["edge"] = BenchmarkGroup(["identity", "field", "vertical", "horizontal"])
suite["interior"] = BenchmarkGroup(["identity", "field", "vertical", "connect", "generate"])

Nx = 6
Ny = 6
χ  = 4
D  = 4
sites = siteinds("S=1/2",Nx*Ny)
A     = cufPEPS(randomfPEPS(sites, Nx, Ny, mindim=D))
J     = 1.0
H     = PEPS.makeCuH_XXZ(Nx, Ny, J)
next_combiners = Vector{ITensor}(undef, Ny)
fake_next_combiners = Vector{ITensor}(undef, Ny)
fake_prev_combiners = fill(cuITensor(1.0), Ny)
suite["edge"]["identity"] = @benchmarkable PEPS.buildNewI($A, 1, $fake_prev_combiners, :left)
I_mpo, fake_next_combiners, up_combiners = PEPS.buildNewI(A, 1, fake_prev_combiners, :left)
field_H_terms  = PEPS.getDirectional(vcat(H[:, 1]...), PEPS.Field)
vert_H_terms   = PEPS.getDirectional(vcat(H[:, 1]...), PEPS.Vertical)
suite["edge"]["vertical"] = @benchmarkable [PEPS.buildNewVerticals($A, $fake_prev_combiners, $fake_next_combiners, $up_combiners, $vert_H_terms[vert_op], 1) for vert_op in 1:length($vert_H_terms)]
suite["edge"]["field"]    = @benchmarkable [PEPS.buildNewFields($A, $fake_prev_combiners, $fake_next_combiners, $up_combiners, $field_H_terms[field_op], 1) for field_op in 1:length($field_H_terms)]

side_H       = H[:, 1]
side_H_terms = PEPS.getDirectional(vcat(side_H...), PEPS.Horizontal)
in_progress  = Matrix{ITensor}(undef, Ny, length(side_H_terms))
suite["edge"]["horizontal"] = @benchmarkable [PEPS.generateEdgeDanglingBonds($A, $up_combiners, $side_H_terms[side_term], :left, 1) for side_term in 1:length($side_H_terms)]

# interior benchmarks

left_H_terms   = PEPS.getDirectional(H[1], PEPS.Horizontal)
previous_combiners = Vector{ITensor}(undef, Ny)
Ledge = buildEdgeEnvironment(A, H, left_H_terms, previous_combiners, :left, 1; env_maxdim=χ, cutoff=0.0)

working_combiner = Vector{ITensor}(undef, Ny)
up_combiners     = Vector{ITensor}(undef, Ny-1)
col = 2 
cutoff = 0.0 
maxdim     = D 
env_maxdim = χ 
suite["interior"]["identity"] = @benchmarkable PEPS.buildNewI($A, $col, $previous_combiners, :left)
I_mpo, working_combiner, up_combiners = PEPS.buildNewI(A, col, previous_combiners, :left)
copyto!(next_combiners, working_combiner)
@inbounds for row in 1:Ny-1
    ci = linkindex(I_mpo, row)
    ni = Index(ITensors.dim(ci), "u,Link,c$col,r$row")
    replaceind!(I_mpo[row], ci, ni)
    replaceind!(I_mpo[row+1], ci, ni)
end
#new_I     = applyMPO(I_mpo, prev_Env.I; cutoff=cutoff, maxdim=env_maxdim)
#new_H     = applyMPO(I_mpo, prev_Env.H; cutoff=cutoff, maxdim=env_maxdim)
field_H_terms = PEPS.getDirectional(vcat(H[:, col]...), PEPS.Field)
vert_H_terms  = PEPS.getDirectional(vcat(H[:, col]...), PEPS.Vertical)
hori_H_terms  = PEPS.getDirectional(vcat(H[:, col]...), PEPS.Horizontal)
side_H        = H[:, col]
side_H_terms  = PEPS.getDirectional(vcat(side_H...), PEPS.Horizontal)
H_term_count  = 1 + length(field_H_terms) + length(vert_H_terms)
H_term_count += length(side_H_terms)
#final_H       = deepcopy(new_H)
new_H_mps     = Vector{MPS}(undef, H_term_count)
suite["interior"]["vertical"] = @benchmarkable [PEPS.buildNewVerticals($A, $previous_combiners, $next_combiners, $up_combiners, $vert_H_terms[vert_op], $col) for vert_op in 1:length($vert_H_terms)]
suite["interior"]["field"]    = @benchmarkable [PEPS.buildNewFields($A, $previous_combiners, $next_combiners, $up_combiners, $field_H_terms[field_op], $col) for field_op in 1:length($field_H_terms)]
connect_H     = side_H_terms
suite["interior"]["connect"] = @benchmarkable [PEPS.connectDanglingBonds($A, $next_combiners, $up_combiners, $connect_H[cc], $Ledge.InProgress[:, cc], :left, -1, $col; env_maxdim=env_maxdim, cutoff=cutoff) for cc in 1:length($connect_H)]
gen_H_terms  = hori_H_terms
suite["interior"]["generate"] = @benchmarkable [PEPS.generateNextDanglingBonds($A, $previous_combiners, $next_combiners, $up_combiners, $gen_H_terms[side_term], Ledge.I, :left, $col; env_maxdim=env_maxdim, cutoff=cutoff) for side_term in 1:length($gen_H_terms)]
