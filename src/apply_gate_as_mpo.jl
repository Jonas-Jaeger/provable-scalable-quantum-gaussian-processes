# # custom way of applying MPO to MPS
# function apply_mpo_gate(psi::MPS, gate::Vector{ITensor}, targs::Vector{Int64})
#     i, j = targs
#     L, R = gate
    
    
#     # get link indx of mpo
#     link_ind = commonind(L, R)
   
#     # get physical indices
#     ph_inds = siteinds(psi)
    
#     # modify left component
#     ## contract with L
#     psi[i] *= L
#     noprime!(psi[i])
#     ## merge bond and link indices
#     bond_ind_r = findindex(psi[i], "Link,l=$(i)")
#     cr = combiner(bond_ind_r, link_ind ; tags="Link,l=$(i)")
#     psi[i] *= cr
    
#     ## loop over components in between i and j
#     for k in i+1:j-1
#         # use cr as new cl and bond_ind_r as new bond_ind_l
#         cl = cr
#         bond_ind_l = bond_ind_r
        
#         # prepare delta to extend mps component
#         phys_ind = ph_inds[k]
#         delta_phys = delta(phys_ind', phys_ind)
#         delta_link = delta(link_ind', link_ind)
#         delta_tensor = delta_phys * delta_link

#         # Extend psi[k] with the delta tensor
#         psi[k] *= delta_tensor
#         noprime!(psi[k], phys_ind')
        
#         # Merge the link indices
#         bond_ind_r = findindex(psi[k], "Link,l=$(k)")
#         cr = combiner(bond_ind_r, link_ind'; tags="Link,l=$(k)")
#         # cr = combiner(link_inds', bond_ind_r; tags="Link,l=$(k)")
#         psi[k] *= cl
#         psi[k] *= cr
#         #prime the link ind
#         link_ind = link_ind'
#     end
    
#     # modify right component
#     og_link_ind = commonind(L, R)
#     # apply R after priming link
#     R = replaceind(R, og_link_ind, link_ind)
#     psi[j] *= R
#     noprime!(psi[j], ph_inds[j]')
#     # Merge the link indices
#     psi[j] *= cr
    
#     return psi
# end;

function apply_mpo_gate(psi::MPS, gate::Vector{ITensor}, targs::Vector{Int64})
    # targs and inds in gate are assumed to be sorted!
    ## TODO, implement check
    i = targs[1]
    gind = 1
    if length(gate) == 1 == length(targs)
        psi[i] = noprime(gate[gind] * psi[i])
        return psi
    end
    # get link indx of mpo
    link_ind = commonind(gate[1:2]...)
    # get physical indices
    ph_inds = siteinds(psi)
    # modify left component
    ## contract with L
    psi[i] *= gate[gind]
    noprime!(psi[i])
    ## merge bond and link indices
    bond_ind_r = findindex(psi[i], "Link,l=$(i)")
    cr = combiner(bond_ind_r, link_ind ; tags="Link,l=$(i)")
    psi[i] *= cr
    ## loop over components in between i and j
    for j in targs[2:end]
        gind += 1
        # push deltas
        for k in i+1:j-1
            # use cr as new cl and bond_ind_r as new bond_ind_l
            cl = cr
            bond_ind_l = bond_ind_r
            # prepare delta to extend mps component
            phys_ind = ph_inds[k]
            delta_phys = delta(phys_ind', phys_ind)
            delta_link = delta(link_ind', link_ind)
            delta_tensor = delta_phys * delta_link
            # Extend psi[k] with the delta tensor
            psi[k] *= delta_tensor
            noprime!(psi[k], phys_ind')
            # Merge the link indices
            bond_ind_r = findindex(psi[k], "Link,l=$(k)")
            cr = combiner(bond_ind_r, link_ind'; tags="Link,l=$(k)")
            # cr = combiner(link_inds', bond_ind_r; tags="Link,l=$(k)")
            psi[k] *= cl
            psi[k] *= cr
            #prime the link ind
            link_ind = link_ind'
        end
        # apply mpo comp
        og_link_ind = commonind(gate[gind-1:gind]...)
        # replaceind!(gate[gind], og_link_ind, link_ind)
        psi[j] *= replaceind(gate[gind], og_link_ind, link_ind) # gate[gind]
        noprime!(psi[j], ph_inds[j]')
        # merge links left
        psi[j] *= cr
        ## merge bond and link indices right if not at the end
        if j != last(targs)
            # get new linkind
            link_ind = commonind(gate[gind:gind+1]...)
            bond_ind_r = findindex(psi[j], "Link,l=$(j)")
            cr = combiner(bond_ind_r, link_ind ; tags="Link,l=$(j)")
            psi[j] *= cr
        end
        i = j
    end
    return psi
end;