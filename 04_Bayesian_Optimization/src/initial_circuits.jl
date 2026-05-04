using Random 

function random_rotations_with_entanglement_rotations_state_circuit(initial_state::MPS, sites::Vector{Index{Int64}}, t1::Float64, t2::Float64, nlayers::Int64, maxbond::Int64; seed::Int64 = 42)

    n = length(initial_state)
    new_psi = copy(initial_state) #to make sure we do not modify "initial state"
    rng = MersenneTwister(seed)  # create a local RNG with the seed

    for _ in 1:nlayers

        #Apply random rotations on each qubit
        for i in 1:n
            P_choose = rand(rng, (Px, Py, Pz))  # ✅ seeded and deterministic
            angle_i = 2*pi*randn(rng)*t1
            gates = [Prot(angle_i, P_choose)] #defined in "gates_circuit"
            indices = [[i]] #gate on which it acts
            new_psi =  get_state!(gates,new_psi,sites,indices)
        end
        #Apply CNOT after random rotations
        for i in 1:n-1
            angle_i = 2*pi*randn(rng)*t2
            gates = [Cz_controlled_reshaped(angle_i)]
            indices = [[i,i+1]]
            new_psi =  get_state!(gates,new_psi,sites,indices)
        end

        #Get rid of zeros
        new_psi = noprime(truncate(new_psi, cutoff = 10^-16))

        #In case we add many layers, set a threshold in χ

        if maximum(linkdims(new_psi)) > maxbond
            new_psi = truncate(new_psi, maxchi = maxbond)
            break
        end

    end
    return new_psi 
end

function random_rotations_with_entanglement_state_circuit(initial_state::MPS, sites::Vector{Index{Int64}}, t::Float64, nlayers::Int64, maxbond::Int64; seed::Int64 = 42)

    n = length(initial_state)
    new_psi = copy(initial_state) #to make sure we do not modify "initial state"
    rng = MersenneTwister(seed)  # create a local RNG with the seed

    for _ in 1:nlayers

        #Apply random rotations on each qubit
        for i in 1:n
            P_choose = rand(rng, (Px, Py, Pz))  # ✅ seeded and deterministic
            angle_i = 2*pi*randn(rng)*t
            gates = [Prot(angle_i, P_choose)] #defined in "gates_circuit"
            indices = [[i]] #gate on which it acts
            new_psi =  get_state!(gates,new_psi,sites,indices)
        end
        #Apply CNOT after random rotations
        for i in 1:n-1
            gates = [Cz_reshaped]
            indices = [[i,i+1]]
            new_psi =  get_state!(gates,new_psi,sites,indices)
        end

        #Get rid of zeros
        new_psi = noprime(truncate(new_psi, cutoff = 10^-16))

        #In case we add many layers, set a threshold in χ

        if maximum(linkdims(new_psi)) > maxbond
            new_psi = truncate(new_psi, maxchi = maxbond)
            break
        end

    end
    return new_psi 
end

function simple_extent_state_circuit(initial_state::MPS, sites::Vector{Index{Int64}}, t::Float64, nlayers::Int64, maxbond::Int64)
    n = length(initial_state)
    new_psi = copy(initial_state)
    for _ in 1:nlayers
        for i in 1:4:n
            # Construct single-site gates
            gates = [PHad,PS(pi*t),fill(Cnot_reshaped,3)...]            
            indices = [[i], [i], [i,i+1], [i,i+2], [i,i+3]]
            new_psi = get_state!(gates, new_psi, sites,indices)
            new_psi = truncate(new_psi, cutoff = 1e-16)
        end
        if maximum(linkdims(new_psi)) > maxbond
            new_psi = truncate(new_psi, maxchi = maxbond)
            break
        end
    end
    return new_psi
end

function first_qubit_rotation_state_circuit(initial_state::MPS, sites::Vector{Index{Int64}}, t::Float64)
    gates = Rx(2*pi*t)
    indices = [[1]]
    new_psi = get_state!(gates, initial_state,sites,indices)
    return new_psi
end

function random_rotations_state_circuit(initial_state::MPS, sites::Vector{Index{Int64}}, t::Float64; seed::Int = 42)
    rng = MersenneTwister(seed)  # create a local RNG with the seed
    n = length(initial_state)
    new_psi = copy(initial_state)
    for i in 1:n
        P_choose = rand((Px, Py, Pz))
        gates = [Prot(2*pi*randn(rng)*t, P_choose)]
        indices = [[i]]
        new_psi =  get_state!(gates,initial_state,sites,indices)
    end
    new_psi = truncate(new_psi, cutoff = 10^-16)
    return new_psi
end

function random_rotations_state_circuit(initial_state::MPS, sites::Vector{Index{Int64}}, t::Float64; seed::Int = 42)
    rng = MersenneTwister(seed)  # create a local RNG with the seed
    n = length(initial_state)
    new_psi = copy(initial_state)
    for i in 1:n
        P_choose = rand((Px, Py, Pz))
        gates = [Prot(2*pi*randn(rng)*t, P_choose)]
        indices = [[i]]
        new_psi =  get_state!(gates,initial_state,sites,indices)
    end
    new_psi = truncate(new_psi, cutoff = 10^-16)
    return new_psi
end

function random_fermionic_gaussian_circuit(initial_state::MPS, sites::Vector{Index{Int64}}, ngates::Int, maxbond::Int; seed::Int = 42)
    new_psi = copy(initial_state)
    n = length(initial_state)
    for i in 1:ngates
        θ = randn()
            if rand(Bool)
                gates = [Prot(θ, Pz)]
                indices = [[rand(1:n)]]
                new_psi =  get_state!(gates,initial_state,sites,indices)
            else 
                gates = [Prot_xx_reshaped(θ)]
                indi = rand(1:n-1)
                indices = [[indi,indi+1]]
                new_psi =  get_state!(gates,initial_state,sites,indices)
            end
    end
    new_psi = truncate(new_psi, cutoff = 10^-16)
    return new_psi
end


function one_axis_twisting_x(initial_state::MPS, sites::Vector{Index{Int64}}, t::Float64)
    new_psi = initial_state
    n = length(initial_state)
    gates = [Prot_xx_reshaped(t)]
    for k in 2:n
        for j in 1:k-1
            indices = [[j,k]]
            new_psi =  get_state!(gates,new_psi,sites,indices)
            new_psi = truncate(new_psi, cutoff = 10^-16)
        end
    end
    return new_psi
end
