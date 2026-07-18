# Split-KV (Flash-Decoding) attention for single-token batched decode.
#
# One new query token per (head, sequence). The GQA query group — the
# QUERY_GROUP_SIZE query heads that share a KV head — is packed into the M
# dimension, so the per-block matmul is M = QUERY_GROUP_SIZE and each K/V tile
# is read once for the whole group.
#
# Decode is bandwidth-bound, so parallelism comes from the grid: each block
# owns one (kv_head, sequence, kv_split). `mha_decode_split_fwd` streams its
# slice of the KV cache with an online softmax and writes a partial result
# (normalized output + running max + running sum). `mha_decode_combine` then
# merges the N_SPLITS partials per (head, sequence) via log-sum-exp.

function mha_decode_split_fwd(
    Q::TileArray3, K::TileArray4, V::TileArray4,
    O_partial::TileArray4,
    M_partial::TileArray3{Float32},
    L_partial::TileArray3{Float32},
    lengths::TileVector{Int32},
    qk_scale::Float32,
    Heads_KV::Int,
    Tc::Type,
    Dk::Int,
    Dv::Int,
    GROUP::Int,
    TILE_N::Int,
    SPLIT_SIZE::Int,
)
    padding_mode = ct.PaddingMode.Zero
    s = ct.bid(1)                       # kv-split index
    hb = ct.bid(2)
    b, hₖ = fldmod1(hb, Heads_KV)       # b = sequence, hₖ = kv-head (fast index)

    len = lengths[b]

    # KV range [kv_start, kv_stop) owned by this split, in tile units.
    # SPLIT_SIZE is a multiple of TILE_N, so kv_start is tile-aligned.
    kv_stop = min(s * SPLIT_SIZE, len)
    j_lo = fld((s - 1i32) * SPLIT_SIZE, TILE_N) + 1i32
    j_hi = cld(kv_stop, TILE_N)
    mask_start = fld(len, TILE_N)     # tiles beyond this need a length mask

    offs_n_tile = ct.arange(TILE_N) .- 1i32

    m_i = fill(-Inf32, (1, GROUP))
    l_i = zeros(Float32, (1, GROUP))
    acc = zeros(Float32, (Dv, GROUP))

    q = ct.load(Q, (1, hₖ, b), (Dk, GROUP); padding_mode)

    for j in j_lo:j_hi
        k = ct.load(K, (1, j, hₖ, b), (Dk, TILE_N); padding_mode, latency=2)

        s_qk = muladd((k)ᵀ → Tc, q → Tc, zeros(Float32, (TILE_N, GROUP)))
        # exp2 softmax: fold log2(e) into the scale (see mha_fwd). M_partial is
        # stored in these log2 units; mha_decode_combine matches them.
        s_qk = s_qk * qk_scale * inv(log(2f0))

        if j > mask_start
            offs_n = (j - 1i32) * TILE_N .+ offs_n_tile
            mask = offs_n .< len
            s_qk = ifelse.(mask, s_qk, -Inf32)
        end

        m_ij = max.(m_i, maximum(s_qk, dims=1))
        ct.@fpmode flush_to_zero=true begin
            p = exp2.(s_qk .- m_ij)
            l_ij = sum(p, dims=1)
            α = exp2.(m_i .- m_ij)
            l_i = l_i .* α .+ l_ij
            acc = acc .* α
        end

        v = ct.load(V, (1, j, hₖ, b), (Dv, TILE_N); padding_mode, latency=4)
        acc = muladd(v → Tc, p → Tc, acc)

        m_i = m_ij
    end

    # Normalized per-split output; guard the empty-split case (l_i == 0).
    ct.@fpmode flush_to_zero=true begin
        o = acc .* ifelse.(l_i .== 0f0, 0f0, 1f0 ./ l_i)
    end
    ct.store(O_partial, (1, hₖ, b, s), o → eltype(O_partial))
    ct.store(M_partial, (hₖ, b, s), reshape(m_i, GROUP))
    ct.store(L_partial, (hₖ, b, s), reshape(l_i, GROUP))

    return
end

function mha_decode_combine(
    O::TileArray3,
    O_partial::TileArray4,
    M_partial::TileArray3{Float32},
    L_partial::TileArray3{Float32},
    Heads_KV::Int,
    Dv::Int,
    GROUP::Int,
    N_SPLITS::Int,
)
    hb = ct.bid(1)
    b, hₖ = fldmod1(hb, Heads_KV)

    m_i = fill(-Inf32, (1, GROUP))
    l_i = zeros(Float32, (1, GROUP))
    acc = zeros(Float32, (Dv, GROUP))

    for s in 1i32:N_SPLITS
        m_s = reshape(ct.load(M_partial, (hₖ, b, s), (GROUP,), latency=1), (1, GROUP))
        l_s = reshape(ct.load(L_partial, (hₖ, b, s), (GROUP,), latency=1), (1, GROUP))
        o_s = ct.load(O_partial, (1, hₖ, b, s), (Dv, GROUP))

        m_new = max.(m_i, m_s)
        # finite guard: m_new is -Inf only while every split seen so far is
        # empty, where the rescale would be exp(-Inf - -Inf) = NaN.
        finite = m_new .> -Inf32
        ct.@fpmode flush_to_zero=true begin
            α = ifelse.(finite, exp2.(m_i .- m_new), 1f0)
            γ = ifelse.(finite, exp2.(m_s .- m_new), 0f0)
            l_i = l_i .* α .+ l_s .* γ
            acc = acc .* α .+ l_s .* γ .* (o_s → Float32)
        end
        m_i = m_new
    end

    ct.@fpmode flush_to_zero=true begin
        o = acc .* ifelse.(l_i .== 0f0, 0f0, 1f0 ./ l_i)
    end
    ct.store(O, (1, hₖ, b), o → eltype(O))

    return
end

function decode_attention!(
    (; O), Q, K, V;
    lengths,
    tensorcore = tensorcore_type(eltype(Q)),
    TILE_N = 64,
    n_splits = 8,
    space = ambientspace(Similar(Q)),
    scratch = alloc(space, Pol.scratch(decode_attention!, Q, K, V; n_splits)),
)
    @sizes begin
        Q => (Dk, H, Batch)
        K => (Dk, L, Hk, Batch)
        V => (Dv, L, Hk, Batch)
        O => (Dv, H, Batch)
    end
    iszero(H % Hk) ||
        throw(DimensionMismatch("Attention heads ($H) is not a multiple of key-value heads ($Hk)"))

    query_group_size = H ÷ Hk
    qk_scale = 1 / √Float32(Dk)
    split_size = cld(cld(L, n_splits), TILE_N) * TILE_N # tile-aligned boundaries

    (; O_partial, M_partial, L_partial) = scratch

    @cutile(blocks=(n_splits, Hk * Batch),
        mha_decode_split_fwd(
            Q, K, V, O_partial, M_partial, L_partial, lengths,
            qk_scale,
            Hk,
            tensorcore,
            Constant(Dk),
            Constant(Dv),
            Constant(query_group_size),
            Constant(TILE_N),
            split_size,
        )
    )

    @cutile(blocks=Hk * Batch,
        mha_decode_combine(
            O, O_partial, M_partial, L_partial,
            Hk,
            Constant(Dv),
            Constant(query_group_size),
            n_splits,
        )
    )
end

Pol.outputs(::typeof(decode_attention!), Q, K, V; kwargs...) =
    (; O = Undef(eltype(Q), (size(V, 1), size(Q, 2), size(Q, 3))))

function Pol.scratch(::typeof(decode_attention!), Q, K, V; n_splits)
    @sizes begin
        Q => (Dk, H, Batch)
        K => (Dk, L, Hk, Batch)
        V => (Dv, L, Hk, Batch)
    end
    return (;
        O_partial = Undef(Float32, (Dv, H, Batch, n_splits)),
        M_partial = Undef(Float32, (H, Batch, n_splits)),
        L_partial = Undef(Float32, (H, Batch, n_splits)),
    )
end

Pol.@takes_space decode_attention!
const decode_attention = Allocating(decode_attention!)
