suite["edge"]     = BenchmarkGroup(["identity", "field", "vertical", "horizontal"])
suite["interior"] = BenchmarkGroup(["identity", "prev_ham", "field", "vertical", "connect", "generate"])
suite["overall"]  = BenchmarkGroup(["left", "right"])


dummy = Vector{ITensor}(undef, Ny)
for row in 1:Ny
    dummy[row]  = is_cu ? cuITensor(1.0) : ITensor(1.0) 
end
dummy_mps       = MPS(Ny, dummy, 0, Ny+1)
suite["edge"]["identity"] = @benchmarkable PEPS.buildNewI($A, $dummy_mps, 1, $χ)
I_mps           = buildNewI(A, dummy_mps, col, χ)
field_H_terms   = PEPS.getDirectional(vcat(H[:, 1]...), PEPS.Field)
vert_H_terms    = PEPS.getDirectional(vcat(H[:, 1]...), PEPS.Vertical)
suite["edge"]["vertical"] = @benchmarkable [PEPS.buildNewVerticals($A, $vert_H_terms[vert_op], $dummy_mps, 1, $χ) for vert_op in 1:length($vert_H_terms)]
suite["edge"]["field"]    = @benchmarkable [PEPS.buildNewFields($A, $field_H_terms[field_op], $dummy_mps, 1, $χ) for field_op in 1:length($field_H_terms)]

side_H       = H[:, 1]
side_H_terms = PEPS.getDirectional(vcat(side_H...), PEPS.Horizontal)
in_progress  = Matrix{ITensor}(undef, Ny, length(side_H_terms))
suite["edge"]["horizontal"] = @benchmarkable [PEPS.generateEdgeDanglingBonds($A, $side_H_terms[side_term], :left, 1, $χ) for side_term in 1:length($side_H_terms)]

# interior benchmarks

left_H_terms   = PEPS.getDirectional(H[1], PEPS.Horizontal)
Ledge          = buildEdgeEnvironment(A, H, left_H_terms, previous_combiners, :left, 1; env_maxdim=χ, cutoff=0.0)

col        = 2 
cutoff     = 0.0 
maxdim     = D 
env_maxdim = χ 
suite["interior"]["identity"] = @benchmarkable PEPS.buildNewI($A, $Ledge.I, $col, $χ, :left)
suite["interior"]["prev_ham"] = @benchmarkable PEPS.buildNewI($A, $Ledge.H, $col, $χ, :left)
I_mps = PEPS.buildNewI(A, Ledge.I, col, χ, :left)
H_mps = PEPS.buildNewI(A, Ledge.H, col, χ, :left)
field_H_terms = PEPS.getDirectional(vcat(H[:, col]...), PEPS.Field)
vert_H_terms  = PEPS.getDirectional(vcat(H[:, col]...), PEPS.Vertical)
hori_H_terms  = PEPS.getDirectional(vcat(H[:, col]...), PEPS.Horizontal)
side_H        = H[:, col]
side_H_terms  = PEPS.getDirectional(vcat(side_H...), PEPS.Horizontal)
H_term_count  = 1 + length(field_H_terms) + length(vert_H_terms)
H_term_count += length(side_H_terms)
new_H_mps     = Vector{MPS}(undef, H_term_count)
suite["interior"]["vertical"] = @benchmarkable [PEPS.buildNewVerticals($A, $Ledge.I, $vert_H_terms[vert_op], $col, $χ) for vert_op in 1:length($vert_H_terms)]
suite["interior"]["field"]    = @benchmarkable [PEPS.buildNewFields($A, $Ledge.I, $field_H_terms[field_op], $col, $χ) for field_op in 1:length($field_H_terms)]
connect_H     = side_H_terms
suite["interior"]["connect"]  = @benchmarkable [PEPS.connectDanglingBonds($A, $connect_H[cc], $Ledge.InProgress[:, cc], :left, $col; env_maxdim=$env_maxdim, cutoff=$cutoff) for cc in 1:length($connect_H)]
gen_H_terms   = hori_H_terms
suite["interior"]["generate"] = @benchmarkable [PEPS.generateNextDanglingBonds($A, $gen_H_terms[side_term], $Ledge.I, :left, $col; env_maxdim=$env_maxdim, cutoff=$cutoff) for side_term in 1:length($gen_H_terms)]

suite["overall"]["left"]  = @benchmarkable buildLs($A, $H; env_maxdim=$env_maxdim, cutoff=$cutoff)
suite["overall"]["right"] = @benchmarkable buildRs($A, $H; env_maxdim=$env_maxdim, cutoff=$cutoff)
