using Combinatorics

PAULI_MULTIPLICATION_RULES_BARE = Dict(
    (:x, :x) => (1,:id),
    (:y, :y) => (1,:id),
    (:z, :z) => (1,:id),
    (:x, :y) => (1im,:z),
    (:y, :z) => (1im,:x),
    (:z, :x) => (1im,:y),
    (:y, :x) => (-1im,:z),
    (:z, :y) => (-1im,:x),
    (:x, :z) => (-1im,:y),
    (:x, :id) => (1,:x),
    (:y, :id) => (1,:y),
    (:z, :id) => (1,:z),
    (:id, :x) => (1,:x),
    (:id, :y) => (1,:y),
    (:id, :z) => (1,:z),
    (:id, :id) => (1,:id)
)


# Compute the product of two Pauli operator strings
function pauli_product(op1::Vector{Symbol}, op2::Vector{Symbol})
    length(op1) == length(op2) || error("Incompatible Pauli string lengths.")
    result = fill(:id, length(op1))
    coeff = 1 + 0im

    for (k, (a, b)) in enumerate(zip(op1, op2))
        phase, op = PAULI_MULTIPLICATION_RULES_BARE[(a, b)]
        result[k] = op
        coeff *= phase
    end


    return coeff, result
end

function pauli_commutes(op1::Vector{Symbol}, op2::Vector{Symbol})
    length(op1) == length(op2) || error("Lengths mismatch")
    
    anticommute_count = 0
    
    for (a, b) in zip(op1, op2)
        # Two Paulis anti-commute ONLY if:
        # 1. Neither is Identity
        # 2. They are not the same Pauli
        if a !== :id && b !== :id && a !== b
            anticommute_count += 1
        end
    end
    
    # Commute if anti-commuting positions are even
    return iseven(anticommute_count)
end

# Majorana symbol representation
function majorana_symbol(n::Int, j::Int)::Vector{Symbol}
    majo = fill(:id, n)
    for k in 1:n
        if k < j ÷ 2 + mod(j, 2)
            majo[k] = :z
        elseif k == j ÷ 2 + mod(j, 2)
            majo[k] = mod(j, 2) == 1 ? :x : :y
        end
    end
    return majo
end

# Generate Majorana products of order m over n qubits
function majorana_products(n::Int, m::Int)
    majos = [majorana_symbol(n, j) for j in 1:2n]
    combos = collect(combinations(1:2n, m))

    results = Vector{Tuple{ComplexF64, Vector{Symbol}}}(undef, length(combos))

    for (k, combo) in enumerate(combos)
        coeff = 1 + 0im
        res = fill(:id, n)

        for c in combo
            new_coeff, res_new = pauli_product(res, majos[c])
            coeff *= new_coeff
            res = res_new
        end

        coeff *= (-1im)^((m * (m - 1)) // 2)
        results[k] = (coeff, res)
    end

    return results
end