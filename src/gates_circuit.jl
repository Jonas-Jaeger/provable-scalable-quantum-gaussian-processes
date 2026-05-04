Pi = ComplexF64[1 0; 0 1]; 

Px = ComplexF64[0 1; 1 0]; 

Py = ComplexF64[0 -1im; 1im 0];

Pz = ComplexF64[1 0; 0 -1];

PS(ϕ) = ComplexF64[1 0; 0 exp(1im*ϕ)];

Sphase = ComplexF64[1 0; 0 1im];

Rx(ϕ) = ComplexF64[cos(ϕ/2) -1im*sin(ϕ/2); -1im*sin(ϕ/2) cos(ϕ/2)];

Prot(ϕ, P) = exp(- 1im * ϕ / 2 * P);

Prot_xx(ϕ) = exp(- 1im * ϕ / 2 * kron(Px,Px));

Prot_xx_reshaped(ϕ) = permutedims(reshape(Prot_xx(ϕ), 2, 2, 2, 2), (2,1,4,3))  # CORRECT

PHad = ComplexF64(1/√2) * ComplexF64[1 1; 1 -1];

Prot_y =  PHad * conj(Sphase)

Cnot = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 0 1; 0 0 1 0];

Cnot_reshaped = permutedims(reshape(Cnot, 2, 2, 2, 2),(2,1,4,3));

Cz = ComplexF64[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 -1];

Cz_reshaped = permutedims(reshape(Cz, 2, 2, 2, 2),(2,1,4,3));

Cz_controlled(θ) = ComplexF64[1 0 0 0; 0 exp(-1im*θ/2) 0 0; 0 0 1 0; 0 0 0 exp(1im*θ/2)];

Cz_controlled_reshaped(θ) = permutedims(reshape(Cz_controlled(θ), 2, 2, 2, 2),(2,1,4,3));

paulis = Dict(:id => Pi, :x => Px, :y => Py, :z => Pz);