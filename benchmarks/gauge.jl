suite["gauge"] = BenchmarkGroup(["build_envs", "orthogonalize", "gaugeQR"])

Nx = 6
Ny = 6
χ  = 4
D  = 4
sites = siteinds("S=1/2",Nx*Ny)
A     = cufPEPS(randomfPEPS(sites, Nx, Ny, mindim=D))
col = 3
next_col = 2

Q, QR_inds, next_col_inds = PEPS.initQs(A, col, next_col; maxdim=D, cutoff=0.0)
dummy_nexts   = [Index(ITensors.dim(next_col_inds[row]), "DM,Site,r$row") for row in 1:Ny]
#suite["gauge"]["build_envs"] = @benchmarkable PEPS.buildEnvs($A, $Q, $next_col_inds, $QR_inds, $col) breaks for some reason
Envs = PEPS.buildEnvs(A, Q, next_col_inds, QR_inds, col)
suite["gauge"]["orthogonalize_q"] = @benchmarkable PEPS.orthogonalize_Q!($Envs, $Q, $A, $QR_inds, $next_col_inds, $dummy_nexts, $col, true; maxdim=D, cutoff=0.0)
suite["gauge"]["gaugeQR"] = @benchmarkable PEPS.gaugeQR($A, $col, :left; maxdim=D, cutoff=0.0)
