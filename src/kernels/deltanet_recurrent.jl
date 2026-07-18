# DeltaNet recurrence: S ← decay·S + β(v − Sᵀk)·kᵀ, o = Sᵀq.
# Plain (un-fused) forms — inputs already normalized/gated by the caller.
#
# Grouped heads, attention-GQA inverted: q/k have Hk heads, everything per
# value head (v, β, gate, S, o) has Hv = V_PER_K·Hk. One block per value
# head; the q/k head is the per-block scalar hk = (h-1) ÷ V_PER_K + 1 — no
# host-side repeat_interleave.
#
# state S is de-aliased: S is read-only, the new state goes to the S′ output.
# Alias S′ = S (`output_override`) for in-place recurrence — safe: each tile
# is loaded before stored and a block owns its state slice exclusively.

#=== single step ===#

function deltanet_recurrent_decode_fwd(
    O::TileArray3,     # (Dv, Hv, B) — output
    Q::TileArray3,     # (Dk, Hk, B)
    K::TileArray3,     # (Dk, Hk, B)
    V::TileArray3,     # (Dv, Hv, B)
    Beta::TileMatrix,  # (Hv, B)
    Gate::TileMatrix,  # (Hv, B)
    S::TileArray4,     # (Dk, Dv, Hv, B) — state in, read only
    S′::TileArray4,    # (Dk, Dv, Hv, B) — state out (alias S′ = S in place)
    Dk::Int, Dv::Int,
    V_PER_K::Int,
    TILE_DK::Int,
)
    padding_mode = ct.PaddingMode.Zero
    h, b = ct.bid(1), ct.bid(2)
    hk = (h - 1i32) ÷ Int32(V_PER_K) + 1i32

    g = Gate[h, b] → Float32
    decay = exp(g)
    beta = Beta[h, b] → Float32

    v = ct.load(V, (1, h, b), (Dv,)) → Float32

    num_tiles = cld(Int32(Dk), Int32(TILE_DK))

    # Decay state + accumulate S^T @ k — reads S, writes S′
    acc = zeros(Float32, Dv)
    for i in 1i32:num_tiles
        s = ct.load(S, (i, 1, h, b), (TILE_DK, Dv); padding_mode) → Float32
        k = ct.load(K, (i, hk, b), (TILE_DK,); padding_mode) → Float32

        s = s .* decay
        ct.store(S′, (i, 1, h, b), s → eltype(S′))

        acc = muladd((s)ᵀ, k, acc)
    end

    delta = beta .* (v .- acc)

    # Rank-1 update + output query — entirely within S′
    output = zeros(Float32, Dv)
    for i in 1i32:num_tiles
        s = ct.load(S′, (i, 1, h, b), (TILE_DK, Dv); padding_mode) → Float32
        k = ct.load(K, (i, hk, b), (TILE_DK,); padding_mode) → Float32

        s = s .+ k .* (delta)ᵀ
        ct.store(S′, (i, 1, h, b), s → eltype(S′))

        q = ct.load(Q, (i, hk, b), (TILE_DK,); padding_mode) → Float32
        output = muladd((s)ᵀ, q, output)
    end

    ct.store(O, (1, h, b), output → eltype(O))

    return
end

function deltanet_recurrent_decode!((; O, S′), Q, K, V, Beta, Gate, S)
    @sizes begin
        O    => (Dv, Hv, B)
        Q    => (Dk, Hk, B)
        K    => (Dk, Hk, B)
        V    => (Dv, Hv, B)
        Beta => (Hv, B)
        Gate => (Hv, B)
        S    => (Dk, Dv, Hv, B)
        S′   => (Dk, Dv, Hv, B)
    end
    iszero(Hv % Hk) ||
        throw(DimensionMismatch("value heads ($Hv) is not a multiple of key heads ($Hk)"))

    TILE_DK = 16
    @cutile(blocks=(Hv, B),
        deltanet_recurrent_decode_fwd(
            O, Q, K, V, Beta, Gate, S, S′,
            Constant(Dk), Constant(Dv),
            Constant(Hv ÷ Hk),
            Constant(TILE_DK),
        )
    )
end

Pol.outputs(::typeof(deltanet_recurrent_decode!), Q, K, V, Beta, Gate, S; kws...) =
    (; O = Undef(V), S′ = Undef(S))

const deltanet_recurrent_decode = Allocating(deltanet_recurrent_decode!)


#=== full sequence ===#

# Fused sequential DeltaNet recurrence kernel.
# Grid: (Hv, B) — one block per value head per batch, loops over T.
# Same algorithm as decode kernel but fused over the sequence length.

function deltanet_sequence_fwd(
    Q::TileArray4,      # (Dk, T, Hk, B)
    K::TileArray4,      # (Dk, T, Hk, B)
    V::TileArray4,      # (Dv, T, Hv, B)
    Beta::TileArray3,   # (Hv, T, B)
    Gate::TileArray3,   # (Hv, T, B)
    S::TileArray4,      # (Dk, Dv, Hv, B) — initial state, read only
    S′::TileArray4,     # (Dk, Dv, Hv, B) — state out (alias S′ = S in place)
    O::TileArray4,      # (Dv, T, Hv, B) — output
    Dk::Int, Dv::Int, T_len::Int,
    V_PER_K::Int,
    TILE_DK::Int,
)
    padding_mode = ct.PaddingMode.Zero
    h, b = ct.bid(1), ct.bid(2)
    hk = (h - 1i32) ÷ Int32(V_PER_K) + 1i32
    num_dk_tiles = cld(Int32(Dk), Int32(TILE_DK))

    # Move the initial state into S′ so steps t ≥ 2 read the running state
    # from one place; a same-address copy when aliased.
    for i in 1i32:num_dk_tiles
        s = ct.load(S, (i, 1, h, b), (TILE_DK, Dv); padding_mode)
        ct.store(S′, (i, 1, h, b), s → eltype(S′))
    end

    for t in 1i32:T_len
        g = Gate[h, t, b] → Float32
        decay = exp(g)
        beta = Beta[h, t, b] → Float32

        v = ct.load(V, (1, t, h, b), (Dv,)) → Float32

        # Pass 1: Decay state + accumulate S^T @ k
        acc = zeros(Float32, Dv)
        for i in 1i32:num_dk_tiles
            s = ct.load(S′, (i, 1, h, b), (TILE_DK, Dv); padding_mode) → Float32
            k = ct.load(K, (i, t, hk, b), (TILE_DK,); padding_mode) → Float32

            s = s .* decay
            ct.store(S′, (i, 1, h, b), s → eltype(S′))

            acc = muladd((s)ᵀ, k, acc)
        end

        delta = beta .* (v .- acc)

        # Pass 2: Rank-1 update + output query
        output = zeros(Float32, Dv)
        for i in 1i32:num_dk_tiles
            s = ct.load(S′, (i, 1, h, b), (TILE_DK, Dv); padding_mode) → Float32
            k = ct.load(K, (i, t, hk, b), (TILE_DK,); padding_mode) → Float32

            s = s .+ k .* (delta)ᵀ
            ct.store(S′, (i, 1, h, b), s → eltype(S′))

            q = ct.load(Q, (i, t, hk, b), (TILE_DK,); padding_mode) → Float32
            output = muladd((s)ᵀ, q, output)
        end

        ct.store(O, (1, t, h, b), output → eltype(O))
    end

    return
end

function deltanet_sequence!((; O, S′), Q, K, V, Beta, Gate, S)
    @sizes begin
        O    => (Dv, T, Hv, B)
        Q    => (Dk, T, Hk, B)
        K    => (Dk, T, Hk, B)
        V    => (Dv, T, Hv, B)
        Beta => (Hv, T, B)
        Gate => (Hv, T, B)
        S    => (Dk, Dv, Hv, B)
        S′   => (Dk, Dv, Hv, B)
    end
    iszero(Hv % Hk) ||
        throw(DimensionMismatch("value heads ($Hv) is not a multiple of key heads ($Hk)"))

    TILE_DK = 16
    @cutile(blocks=(Hv, B),
        deltanet_sequence_fwd(
            Q, K, V, Beta, Gate, S, S′, O,
            Constant(Dk), Constant(Dv), Constant(T),
            Constant(Hv ÷ Hk),
            Constant(TILE_DK),
        )
    )
end

Pol.outputs(::typeof(deltanet_sequence!), Q, K, V, Beta, Gate, S; kws...) =
    (; O = Undef(V), S′ = Undef(S))

const deltanet_sequence = Allocating(deltanet_sequence!)
