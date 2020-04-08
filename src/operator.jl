@enum Op_Type Field=0 Vertical=1 Horizontal=2
struct Operator
    sites::Vector{Pair{Int,Int}}
    ops::Vector{ITensor}
    site_ind::Index
    dir::Op_Type
end

function getDirectional(ops::Vector{Operator}, dir::Op_Type) 
  return collect(filter(x->x.dir==dir, ops))
end

