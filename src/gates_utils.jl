# convert 2to2 4-dits matrix into tensor 
function gate_to_4dit_tensor(sites, targs, gate_mat)
    # get targ num
    n_in = length(targs)
    # reshape gate_mat
    gate_tens = reshape(gate_mat, [4 for _ in 1:2n_in]...)
    # make ITensor
    gate = ITensor(gate_tens, sites[targs]'..., sites[targs]...)
    # update sites
    # sites[targs] = sites[targs]'
    gate
end;

# convert 2to2 qubits matrix into tensor 
function gate_to_2dit_tensor(sites, targs, gate_mat)
    # get targ num
    n_in = length(targs)
    # reshape gate_mat
    gate_tens = reshape(gate_mat, [2 for _ in 1:2n_in]...)
    # make ITensor
    gate = ITensor(gate_tens, sites[targs]'..., sites[targs]...)
    # update sites
    # sites[targs] = sites[targs]'
    gate
end;

# factorize gate tensor
function gate_to_LR(gate::ITensor; cutoff::Float64=eps())
    # SVD of the gate tensor
    ip, jp, i, j = inds(gate)
    L, R = factorize(gate, (i, ip); cutoff=cutoff)
    return [L, R]
end;

function gate_to_mpo(gate::ITensor; cutoff::Float64=eps())
    # SVD of the gate tensor
    _inds = unique(noprime.(inds(gate)))
    tenss = ITensor[]
    linkv = nothing
    for i in _inds[1:end-1]
        linkv == nothing ? (u,s,v) = svd(gate, i', i; cutoff=cutoff) : (u,s,v) = svd(gate, linkv, i', i; cutoff=cutoff)
        u = u * sqrt.(s)
        v = sqrt.(s) * v
        lu = commonind(u, s)
        lv = commonind(s, v)
        push!(tenss, u)
        replaceind!(v, lv, lu)
        gate = v
        linkv = lu
    end
    push!(tenss, gate)
    return tenss
end;