using ITensors, ITensorMPS, Random

function generate_OATH_dataset(psi::MPS, s::Vector{Index{Int64}}, ts::Vector{Float64}; dt::Float64=1e-2, std::Float64=1.0, use_two_ax::Bool=false, cutoff=1e-14, verb::Bool=false)::Vector{MPS}
    n = length(s)
    # Make gates: XX on all pairs, then random Ys
    gates = ITensor[]
    gates_xx = ITensor[]
    gates_yy = ITensor[]
    for k in 2:n
        for j in 1:k-1
            s1 = s[j]
            s2 = s[k]
            hj = 2 * op("Sx", s1) * op("Sx", s2)
            # Generate the gate for this pair
            Gj = exp(-im * dt * hj)
            push!(gates_xx, Gj)
            if use_two_ax
                hj = - op("Sy", s1) * op("Sy", s2)
                Gj = exp(-im * dt * hj)
                push!(gates_yy, Gj)
            end
        end
    end
    # add Y noise
    Random.seed!(42)
    gates_y = [exp(-im * std * randn() * dt * op("Sy", sq)) for sq in s]
    # Include gates_xx in reverse order too
    # (N,N-1),(N-1,N-2),...
    gates = [gates_y...; gates_yy...; gates_xx...; reverse(gates_yy)...; reverse(gates_y)...]
    # GENERATE DATASET
    dataset = Vector{MPS}(undef, length(ts))
    # apply evo
    t_prev = 0.0
    for (j, t) in enumerate(ts)
        steps = round(Int, (t - t_prev)/dt)
        for _ in 1:steps
            psi = ITensorMPS.apply(gates, psi; cutoff)
            normalize!(psi)
        end
        if verb
            print("\r - Prepared state $(j)/$(length(ts)) | chi = $(maximum(linkdims(psi))) ... \t")
        end
        dataset[j] = copy(psi)
        t_prev = t
    end

    return dataset
end



function generate_OATH_X_dataset(psi::MPS, s::Vector{Index{Int64}}, ts::Vector{Float64}; cutoff=1e-14, verb::Bool=false)::Vector{MPS}
    n = length(s)
    # GENERATE DATASET
    dataset = Vector{MPS}(undef, length(ts))
    # apply evo
    for (j, t) in enumerate(ts)


        gates_xx = ITensor[]
        for k in 2:n
            for j in 1:k-1
                s1 = s[j]
                s2 = s[k]
                hj = 2 * op("Sx", s1) * op("Sx", s2)
                Gj = exp(-im * t * hj)
                push!(gates_xx, Gj)
            end
        end

        dataset[j] = ITensorMPS.apply(gates_xx, psi; cutoff)
        normalize!(dataset[j])
        if verb
            print("\r - Prepared state $(j)/$(length(ts)) | chi = $(maximum(linkdims(dataset[j]))) ... \t")
        end
    end

    return dataset
end
