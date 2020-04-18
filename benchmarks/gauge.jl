suite["gauge"] = BenchmarkGroup(["build_envs", "orthogonalize", "gaugeQR"])

col = 2
next_col = 1

Q, QR_inds, next_col_inds = PEPS.initQs(A, col, next_col; maxdim=D, cutoff=0.0)
dummy_nexts   = [Index(ITensors.dim(next_col_inds[row]), "DM,Site,r$row") for row in 1:Ny]
suite["gauge"]["build_envs"] = @benchmarkable PEPS.buildEnvs(B, $Q, $next_col_inds, $QR_inds, $col) setup=(B=copy($A))
Envs = PEPS.buildEnvs(A, Q, next_col_inds, QR_inds, col)
suite["gauge"]["orthogonalize_q"] = @benchmarkable PEPS.orthogonalize_Q!($Envs, Q_, $A, $QR_inds, $next_col_inds, $dummy_nexts, $col, true; maxdim=D, cutoff=0.0) setup=(Q_=copy($Q))
suite["gauge"]["gaugeQR"] = @benchmarkable PEPS.gaugeQR(B, $col, :left; maxdim=D, cutoff=0.0) setup=(B=copy($A))
