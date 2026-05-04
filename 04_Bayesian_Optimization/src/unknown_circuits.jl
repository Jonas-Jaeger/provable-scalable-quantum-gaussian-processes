using Random 
include("gates_circuit.jl");

function product_rotations(initial_state::MPS, sites::Vector{Index{Int64}}, t::Float64, P_choose = Pz)
    n = length(initial_state)
    new_psi = copy(initial_state)
    gates = [Prot(t, P_choose)]
    for i in 1:n
        indices = [[i]]
        new_psi =  get_state!(gates,new_psi,sites,indices)
        new_psi = truncate(new_psi, cutoff = 10^-16)
    end
    return new_psi
end
