# Per-tensor QK-RMSNorm + partial RoPE for the attention layers, split into a
# Q kernel (in place) and a K/V kernel (writes through to the cache), because
# Q and K have different head counts (GQA) — fusing them into one launch was
# the source of the query-group confusion, and buys no memory traffic anyway.
#
# knorm_rope_append! is the fusion that does pay: normalized+rotated k goes
# straight to K_cache and v piggybacks in the same block, so neither exists
# post-projection in HBM and attention only ever reads the cache. `positions`
# is the device clock (see kv_cache.jl): rope table column and cache column
# in one buffer, no host-computed offsets — graph-replay safe.
#
# Partial rope is written as two stores: the full normalized head first, the
# rotated 2·HALF_ROT prefix over it. The tail isn't tile-aligned, so one
# store would need an unaligned cat; two overlapping stores to the same fresh
# lines are cheaper than being clever.
#
# Layout (column-major):
#   Q (Dq, Hq, B)                  — mutated in place
#   K_raw (Dk, Hk, B), V_raw (Dv, Hk, B) — this step's projections, dead after
#   K_cache (Dk, Ctx, Hk, B), V_cache (Dv, Ctx, Hk, B) — mutated
#   QW (Dq,), KW (Dk,)             — qk-norm weights, (1 + w) scaling via Offset
#   Cos, Sin (HALF_ROT, Ctx)       — full tables; HALF_ROT = rotary_dim ÷ 2
#   positions (B,) Int32           — 1-based; rope column = cache column

function qnorm_rope_fwd(
    Q::TileArray3,
    QW::TileVector,
    Cos::TileMatrix,
    Sin::TileMatrix,
    Positions::TileVector{Int32},
    D::Int,
    HALF_ROT::Int,
    eps::Float32,
    Offset::Float32,
)
    padding_mode = ct.PaddingMode.Zero
    h, b = ct.bid(1), ct.bid(2)

    q = ct.load(Q, (1, h, b), (D,); padding_mode) → Float32
    qw = ct.load(QW, (1,), (D,); padding_mode) → Float32
    q = q ./ √(sum(q .^ 2) / D + eps) .* (qw .+ Offset)

    ct.store(Q, (1, h, b), q → eltype(Q))

    p = Positions[b]
    cos_v = ct.load(Cos, (1, p), (HALF_ROT,)) → Float32
    sin_v = ct.load(Sin, (1, p), (HALF_ROT,)) → Float32

    q1 = ct.extract(q, (1,), (HALF_ROT,))
    q2 = ct.extract(q, (2,), (HALF_ROT,))
    q_rot = [q1 .* cos_v .- q2 .* sin_v
             q2 .* cos_v .+ q1 .* sin_v]

    ct.store(Q, (1, h, b), q_rot → eltype(Q))

    return
end

function knorm_rope_append_fwd(
    K_raw::TileArray3,
    V_raw::TileArray3,
    KW::TileVector,
    Cos::TileMatrix,
    Sin::TileMatrix,
    Positions::TileVector{Int32},
    K_cache::TileArray4,
    V_cache::TileArray4,
    Dk::Int,
    Dv::Int,
    HALF_ROT::Int,
    eps::Float32,
    Offset::Float32,
)
    padding_mode = ct.PaddingMode.Zero
    hₖ, b = ct.bid(1), ct.bid(2)
    p = Positions[b]

    k = ct.load(K_raw, (1, hₖ, b), (Dk,); padding_mode) → Float32
    kw = ct.load(KW, (1,), (Dk,); padding_mode) → Float32
    k = k ./ √(sum(k .^ 2) / Dk + eps) .* (kw .+ Offset)

    ct.store(K_cache, (1, p, hₖ, b), k → eltype(K_cache))

    cos_v = ct.load(Cos, (1, p), (HALF_ROT,)) → Float32
    sin_v = ct.load(Sin, (1, p), (HALF_ROT,)) → Float32

    k1 = ct.extract(k, (1,), (HALF_ROT,))
    k2 = ct.extract(k, (2,), (HALF_ROT,))
    k_rot = [k1 .* cos_v .- k2 .* sin_v
             k2 .* cos_v .+ k1 .* sin_v]

    ct.store(K_cache, (1, p, hₖ, b), k_rot → eltype(K_cache))

    v = ct.load(V_raw, (1, hₖ, b), (Dv,); padding_mode)
    ct.store(V_cache, (1, p, hₖ, b), v → eltype(V_cache))

    return
end

function qnorm_rope!(Q, QW, Cos, Sin; positions, eps, offset)
    @sizes begin
        Q         => (D, Hq, B)
        QW        => (D,)
        Cos       => (HALF_ROT, Ctx)
        Sin       => (HALF_ROT, Ctx)
        positions => (B,)
    end
    @assert ispow2(D)   # tile shape and norm divisor share D

    @cutile(blocks=(Hq, B),
        qnorm_rope_fwd(
            Q, QW, Cos, Sin, positions,
            Constant(D), Constant(HALF_ROT),
            Constant(Float32(eps)), Constant(Float32(offset)),
        )
    )
end

function knorm_rope_append!(K_cache, V_cache, K_raw, V_raw, KW, Cos, Sin;
    positions, eps, offset
)
    @sizes begin
        K_cache   => (Dk, Ctx, Hk, B)
        V_cache   => (Dv, Ctx, Hk, B)
        K_raw     => (Dk, Hk, B)
        V_raw     => (Dv, Hk, B)
        KW        => (Dk,)
        Cos       => (HALF_ROT, Ctx)
        Sin       => (HALF_ROT, Ctx)
        positions => (B,)
    end
    @assert ispow2(Dk) && ispow2(Dv)

    @cutile(blocks=(Hk, B),
        knorm_rope_append_fwd(
            K_raw, V_raw, KW, Cos, Sin, positions, K_cache, V_cache,
            Constant(Dk), Constant(Dv), Constant(HALF_ROT),
            Constant(Float32(eps)), Constant(Float32(offset)),
        )
    )
end
