suite["hamiltonian"] = BenchmarkGroup(["intracolgauge", "vertical_terms", "HIs", "field_terms", "left_terms", "right_terms", "davidson", "norm"])
Ls   = buildLs(A, H; env_maxdim=χ, maxdim=D, cutoff=0.0)
Rs   = buildRs(A, H; env_maxdim=χ, maxdim=D, cutoff=0.0)
col  = 3
row  = 1

suite["hamiltonian"]["intracolgauge"] = @benchmarkable PEPS.intraColumnGauge($A, $col; maxdim=D, cutoff=0.0)
A = PEPS.intraColumnGauge(A, col; maxdim=D, cutoff=0.0)
L = Ls[col - 1]
R = Rs[col + 1]
AncEnvs = PEPS.buildAncs(A, L, R, H, col)
ϕ = A[row, col]

field_H_terms  = PEPS.getDirectional(vcat(H[:, col]...), PEPS.Field)
#vert_H_terms   = PEPS.getDirectional(vcat(H[:, col]...), PEPS.Vertical)
#left_H_terms   = PEPS.getDirectional(vcat(H[:, col - 1]...), PEPS.Horizontal)
#right_H_terms  = PEPS.getDirectional(vcat(H[:, col]...), PEPS.Horizontal)
#suite["hamiltonian"]["norm"] = @benchmarkable PEPS.buildN($A, $L, $R, $AncEnvs[:I], $row, $col, $ϕ)
#=suite["hamiltonian"]["HIs"]  = @benchmarkable PEPS.buildHIs($A, $L, $R, $row, $col, $ϕ)
suite["hamiltonian"]["vertical_terms"] = @benchmarkable PEPS.verticalTerms($A, $L, $R, $AncEnvs[:I], $AncEnvs[:V], $vert_H_terms, $row, $col, $ϕ) 
=#
suite["hamiltonian"]["field_terms"] = @benchmarkable PEPS.fieldTerms($A, $L, $R, $AncEnvs[:I], $AncEnvs[:F], $field_H_terms, $row, $col, $ϕ) 
#=suite["hamiltonian"]["left_terms"] = @benchmarkable PEPS.connectLeftTerms($A, $L, $R, $AncEnvs[:I], $AncEnvs[:L], $left_H_terms, $row, $col, $ϕ)
suite["hamiltonian"]["right_terms"] = @benchmarkable PEPS.connectRightTerms($A, $L, $R, $AncEnvs[:I], $AncEnvs[:R], $right_H_terms, $row, $col, $ϕ)
=#
#mapper   = PEPS.ITensorMap(A, H, L, R, AncEnvs, row, col)
#suite["hamiltonian"]["davidson"] = @benchmarkable PEPS.davidson($mapper, $A[$row, $col]; maxiter=1, maxdim=D, cutoff=0.0)
