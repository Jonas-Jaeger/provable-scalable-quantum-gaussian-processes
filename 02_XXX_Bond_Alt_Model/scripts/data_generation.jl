using LinearAlgebra, JSON, Combinatorics, ITensors, JLD2, Yao, ITensorMPS, Plots, NPZ

function generate_ground_state(N::Int, sites_vec::Dict{Int,Vector{Index}}, t::Float64, parameterized_hamiltonian::Function)
    Data = Dict("energies" => Float64[], "states" => Vector{MPS}())
    sites = sites_vec[N]
    # println(sites)
    H_opsum = parameterized_hamiltonian(N, t)
    H = MPO(H_opsum, sites)
    nsweeps = 30
    mindim = [4]
    maxdim = [32]
    cutoff = [1e-8]
    psi0 = randomMPS(sites, 4)
    _, psi = dmrg(H, psi0; nsweeps=nsweeps, maxdim=maxdim, cutoff=cutoff)
    return psi
end

function XXZ_hamiltonian(N::Int, delta::Float64)
    J = 1.0
    H_opsum = OpSum()
    for j in 1:(N-1)
        H_opsum += 4*J, "Sx", j, "Sx", j+1
        H_opsum += 4*J, "Sy", j, "Sy", j+1
        H_opsum += 4*J*delta, "Sz", j, "Sz", j+1
    end
    return H_opsum
end
function XXX_hamiltonian(N::Int, J1::Float64)
    J2 = 1.0
    H_opsum = OpSum()
    for j in 1:2:(N-1)
        H_opsum += 4*J1, "Sx", j, "Sx", j+1
        H_opsum += 4*J1, "Sy", j, "Sy", j+1
        H_opsum += 4*J1, "Sz", j, "Sz", j+1
    end
    for j in 2:2:(N-1)
        H_opsum += 4*J2, "Sx", j, "Sx", j+1
        H_opsum += 4*J2, "Sy", j, "Sy", j+1
        H_opsum += 4*J2, "Sz", j, "Sz", j+1
    end
    return H_opsum
end
function make_operator(sites::Vector{Index{Int64}}, op::Symbol, index::Int)
    ITensor(paulis[op], sites[index]', sites[index])
end

function state2vecrho(psi::MPS, newinds::Vector{Index{Int64}})::MPS
    n = length(psi)
    ss = siteinds(psi)
    # create rho MPO
    rho = outer(psi, psi')
    # vectorize
    for(j, s) in enumerate(ss)
        c = combiner(s, s')
        rho[j] *= c
        replaceind!(rho[j], commonind(rho[j], c), newinds[j])
    end
    return MPS([rho...])
end 

# init_empy MPO with custom bond dims
function initialize_empty_mpo(sites, bond_dims)
    # sites is the veotr of physical indices
    # bond_dims is an array of the bond dimensions for each link
    N = length(sites)
    mpo = MPO(N)
    
    if N==1
        mpo[1] = ITensor(zeros(ComplexF64, sites[1].space, sites[1].space), sites[1], sites[1]')
        return mpo
    end
    
    # make left elem
    right_index = Index(bond_dims[1], "Link,l=1")
    mpo[1] = ITensor(zeros(ComplexF64, sites[1].space, sites[1].space, bond_dims[1]), sites[1], sites[1]', right_index)
    # make middle elements
    for n = 2:N-1
        # The previous right index becomes the current left index
        left_index = right_index
        right_index = Index(bond_dims[n], "Link,l=$n")
        # Create the ITensor for site n with the specified indices
        mpo[n] = ITensor(zeros(ComplexF64, bond_dims[n-1], sites[n].space, sites[n].space, bond_dims[n]), left_index, sites[n], sites[n]', right_index)
    end
    #make right elem
    mpo[N] = ITensor(zeros(ComplexF64, bond_dims[N-1], sites[N].space, sites[N].space), right_index, sites[N], sites[N]')
    return mpo
end;


## rotation matrix, moves from computational basis to Pauli baisis
rotate_to_paulis_mat = (1. + 0im) * zeros(2,2,2,2);
rotate_to_paulis_mat[1,1,:,:] = ComplexF64[1 0; 0 1];
rotate_to_paulis_mat[2,1,:,:] = ComplexF64[0 1; 1 0];
rotate_to_paulis_mat[1,2,:,:] = ComplexF64[0 -1im; 1im 0];
rotate_to_paulis_mat[2,2,:,:] = ComplexF64[1 0; 0 -1];
rotate_to_paulis_mat /= sqrt(2)
rotate_to_paulis_mat = reshape(rotate_to_paulis_mat, 4, 4);

function module_proj_majos(sites, k::Int)::MPO
    n = length(sites)    
    # define projectors
    si = LinearAlgebra.diagm([1, 0, 0, 0])
    sz = LinearAlgebra.diagm([0, 0, 0, 1])
    sxy = LinearAlgebra.diagm([0, 1, 1, 0])    
    # check if larger module than needed, if so replace Z and I
    if k > n
        k = 2n-k
        si = LinearAlgebra.diagm([0, 0, 0, 1])
        sz = LinearAlgebra.diagm([1, 0, 0, 0])
    end    
    
    si = rotate_to_paulis_mat * si * rotate_to_paulis_mat'
    sz = rotate_to_paulis_mat * sz * rotate_to_paulis_mat'
    sxy = rotate_to_paulis_mat * sxy * rotate_to_paulis_mat'    
    
    # we compute the needed bond dims
    ## starting from 3, we increase by 2 until we reach k-1
    bond_dims_l = [1+2j for j in 1:k÷2-1]
    ## we then stay at k+1 until we reach m=n-k÷2
    bond_dims_m = [k+1 for _ in 1:n-k+1]
    ## then we decrease until we reach 2
    bond_dims_r = [k+1-2j for j in 1:k÷2-1]    
    bond_dims = [bond_dims_l; bond_dims_m; bond_dims_r]    
    # we can now init our MPO
    b_proj = initialize_empty_mpo(sites, bond_dims)    
    if k==0
        b_proj[1][:,:] = si
        for m in 2:n-1
            b_proj[m][1,:,:,1] = si
        end
        b_proj[n][1,:,:] = si
    else
        #first elem has no left link
        b_proj[1][:,:, 1] = si
        b_proj[1][:,:, 2] = sxy
        b_proj[1][:,:, 3] = sz
        # then we add the rectangular tensors
        for m = 2:k÷2
            for j = 1:2m-1
                if j%2 == 0
                    b_proj[m][j,:,:,j] = sz
                    b_proj[m][j,:,:,j+2] = si
                    b_proj[n-m+1][j,:,:,j] = si
                    b_proj[n-m+1][j+2,:,:,j] = sz
                else
                    b_proj[m][j,:,:,j] = si
                    b_proj[m][j,:,:,j+2] = sz
                    b_proj[n-m+1][j,:,:,j] = sz
                    b_proj[n-m+1][j+2,:,:,j] = si
                end
                b_proj[m][j,:,:,j+1] = sxy;
                b_proj[n-m+1][j+1,:,:,j] = sxy
            end
        end
        # the square ones
        for m = (k÷2+1):(n-k÷2)
            for j = 1:k+1
                j%2 == 0 ? b_proj[m][j,:,:,j] = sz : b_proj[m][j,:,:,j] = si;
            end
            for j = 1:k
                b_proj[m][j,:,:,j+1] = sxy;
            end
            for j = 1:k-1
                j%2 == 0 ? b_proj[m][j,:,:,j+2] = si : b_proj[m][j,:,:,j+2] = sz;
            end
        end
        #last has no right index
        b_proj[n][1,:,:] = sz;
        b_proj[n][2,:,:] = sxy;
        b_proj[n][3,:,:] = si;
    end
    return b_proj
end;

function b_projector(sites, b::Int; mode::Symbol=:all)
    N = length(sites)
    
    Sm = diagm([1,0,0,0])
    if mode == :all  # all Paulis
        Sp = diagm([0,1,1,1]);        
    elseif mode == :X  # all X paulis
        Sp = diagm([0,1,0,0]);
    elseif mode == :Y  # all Y paulis
        Sp = diagm([0,0,1,0]);
    elseif mode == :Z  # all  paulis
        Sp = diagm([0,0,0,1]);
    else
        error("Unsupported mode. Use :all, :X, :Y, or :Z.")
    end
    
    if b > N/2
        Sm, Sp = Sp, Sm
        b = N - b;
    end
    
    Sm = rotate_to_paulis_mat * Sm * rotate_to_paulis_mat'
    Sp = rotate_to_paulis_mat * Sp * rotate_to_paulis_mat'
    
    # we compute the needed bond dims
    ## starting from 2, we increase until we reach b+1
    bond_dims_l = [1+j for j in 1:b]
    ## we then stay at b+1 until we reach n=N-b
    bond_dims_m = [b+1 for j in (b+1):(N-b)]
    ## then we decrease until we reach 2
    bond_dims_r = [b+1-j for j in 1:b-1]
    
    bond_dims = [bond_dims_l; bond_dims_m; bond_dims_r]
    
    # we can now init our MPO
    b_proj = initialize_empty_mpo(sites, bond_dims)
    
    if b==0
        b_proj[1][:,:] = Sm
        for n in 2:N-1
            b_proj[n][1,:,:,1] = Sm
        end
        b_proj[N][1,:,:] = Sm
    else
        #first elem has no left link
        b_proj[1][:,:, 1] = Sm
        b_proj[1][:,:, 2] = Sp
        for n = 2:b
            for k = 1:n
                b_proj[n][k,:,:,k] = Sm;
                b_proj[n][k,:,:,k+1] = Sp;
                b_proj[N-n+1][k,:,:,k] = Sp;
                b_proj[N-n+1][k+1,:,:,k] = Sm;
            end
        end
        for n = (b+1):(N-b)
            for k = 1:b+1
                b_proj[n][k,:,:,k] = Sm;
            end
            for k = 1:b
                b_proj[n][k,:,:,k+1] = Sp;
            end
        end
        # for n = 2:b
        #     for k = 1:n
        #         b_proj[N-n+1][k,:,:,k] = Sp;
        #         b_proj[N-n+1][k+1,:,:,k] = Sm;
        #     end
        # end
        #last has no right index
        b_proj[N][1,:,:] = Sp;
        b_proj[N][2,:,:] = Sm;
    end
    return b_proj
end;

function job()
    
    #number qubits 
    input_file = "../config/initial_variables.json"
    # output_file = "output.json"
    
    variables = JSON.parsefile(input_file)
    n = variables["n"]
    nq = n
    
    sites_maj = siteinds("Qubit", nq)
    
    t_start = variables["t_start"]
    t_stop = variables["t_stop"]
    t_length = variables["t_length"]
    t_vec = [range(t_start, stop=t_stop, length=t_length)...]
    dsize = length(t_vec)
    
    sites_vec = Dict{Int,Vector{Index}}()
    n = nq
    sites_vec[n] = sites_maj
    psi_list = [generate_ground_state(n, sites_vec, t, XXX_hamiltonian) for t in t_vec]
    
    # prepare middle swap measurement: B2 component
    observable_opsum_2 = OpSum()
    observable_opsum_2 += 4.0, "Sx", div(n,2), "Sx", div(n,2)+1
    observable_opsum_2 += 4.0, "Sy", div(n,2), "Sy", div(n,2)+1
    obs_2_mpo = MPO(observable_opsum_2, sites_maj)
    # prepare middle swap measurement: B4 component
    observable_opsum_4 = OpSum()
    observable_opsum_4 += 4.0, "Sz", div(n,2), "Sz", div(n,2)+1
    obs_4_mpo = MPO(observable_opsum_4, sites_maj)
    
    ovps_m2 = zeros(dsize, dsize)
    ovps_m4 = zeros(dsize, dsize)

    ovps_k2 = zeros(dsize, dsize)
    ovps_k4 = zeros(dsize, dsize)
    
    sites_pauli = siteinds(4, nq)
    proj2 = module_proj_majos(sites_pauli, 2)
    proj4 = module_proj_majos(sites_pauli, 4)

    proj_k2 = b_projector(sites_pauli, 2)
    proj_k4 = b_projector(sites_pauli, 4)
    
    observable_vector_2 = zeros(dsize)
    observable_vector_4 = zeros(dsize)
    
    # compute the true module overlaps and true outputs
    for i in 1:dsize
        println("sample $(i) / $(dsize) ... ")  # Print current step
        flush(stdout)
        observable_vector_2[i] = real(inner(psi_list[i]', obs_2_mpo, psi_list[i]))
        observable_vector_4[i] = real(inner(psi_list[i]', obs_4_mpo, psi_list[i]))
        rhovec_i = state2vecrho(psi_list[i], sites_pauli)
        ovps_m2[i,i] = real(inner(rhovec_i', proj2, rhovec_i))
        ovps_m4[i,i] = real(inner(rhovec_i', proj4, rhovec_i))
        ovps_k2[i,i] = real(inner(rhovec_i', proj_k2, rhovec_i))
        ovps_k4[i,i] = real(inner(rhovec_i', proj_k4, rhovec_i))
        for j in i+1:dsize
            rhovec_j = state2vecrho(psi_list[j], sites_pauli)
            ovps_m2[i,j] = real(inner(rhovec_i', proj2, rhovec_j))
            ovps_m4[i,j] = real(inner(rhovec_i', proj4, rhovec_j))
            ovps_k2[i,j] = real(inner(rhovec_i', proj_k2, rhovec_j))
            ovps_k4[i,j] = real(inner(rhovec_i', proj_k4, rhovec_j))
            ovps_m2[j,i] = ovps_m2[i,j]
            ovps_m4[j,i] = ovps_m4[i,j]
            ovps_k2[j,i] = ovps_k2[i,j]
            ovps_k4[j,i] = ovps_k4[i,j]
        end 
    end

    outdir = "../data/"
    isdir(outdir) || mkdir(outdir)

    npzwrite("$(outdir)/true_outputs_m2_$(nq)q.npy", observable_vector_2)
    npzwrite("$(outdir)/true_outputs_m4_$(nq)q.npy", observable_vector_4)
    npzwrite("$(outdir)/true_overlaps_m2_$(nq)q.npy", ovps_m2)
    npzwrite("$(outdir)/true_overlaps_m4_$(nq)q.npy", ovps_m4)
    npzwrite("$(outdir)/true_overlaps_k2_$(nq)q.npy", ovps_k2)
    npzwrite("$(outdir)/true_overlaps_k4_$(nq)q.npy", ovps_k4)
end

job()