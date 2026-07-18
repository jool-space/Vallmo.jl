# Flash attention forward for prefill (eager — host scalars are fine here).
#
# Alignment invariant, shared with the decode kernel below: queries are the
# suffix of keys, so the causal offset is derived per sequence as
# k_len − q_len — never passed. Chunked prefill works out automatically
# (cache of P tokens + chunk of C queries → offset P). A caller whose queries
# are not a suffix of K is outside this contract.
#
# Ragged batches: rows past q_lengths[b] compute garbage into padding — the
# caller ignores them. k_len must be ≥ 1 (a fully-masked row divides 0/0).

function mha_fwd(
    Q::TileArray4,
    K::TileArray4,
    V::TileArray4,
    O::TileArray4,
    k_lengths::Optional{TileVector{Int32}},
    q_lengths::Optional{TileVector{Int32}},
    qk_scale::Float32,
    H::Int,
    Tc::Type,
    Dk::Int, Dv::Int,
    TILE_M::Int, TILE_N::Int,
    QUERY_GROUP_SIZE::Int,
    CAUSAL::Bool,
)
    padding_mode = ct.PaddingMode.Zero
    i, hb = ct.bid(1), ct.bid(2)
    b, h = fldmod1(hb, H)
    hₖ = fld1(h, QUERY_GROUP_SIZE)

    q_len = isnothing(q_lengths) ? size(Q, 2) : q_lengths[b]
    (i - 1i32) * TILE_M >= q_len && return

    k_len = isnothing(k_lengths) ? size(K, 2) : k_lengths[b]
    off = Int32(k_len) - Int32(q_len)   # queries are the suffix of keys

    offs_m = (i - 1i32) * TILE_M .+ ct.arange(TILE_M) .- 1i32 .+ off
    offs_n_tile = ct.arange(TILE_N) .- 1i32

    m_i = fill(-Inf32, (1, TILE_M))
    l_i = zeros(Float32, (1, TILE_M))
    acc = zeros(Float32, (Dv, TILE_M))

    q = ct.load(Q, (1, i, h, b), (Dk, TILE_M); padding_mode)

    m_end = off + i * TILE_M

    if CAUSAL
        mask_start = min(fld(off + (i - 1i32) * TILE_M, TILE_N), fld(k_len, TILE_N))
        kv_tiles = cld(min(m_end, Int32(k_len)), TILE_N)
    else
        mask_start = fld(k_len, TILE_N)
        kv_tiles = cld(k_len, TILE_N)
    end

    for j in 1i32:kv_tiles
        k = ct.load(K, (1, j, hₖ, b), (Dk, TILE_N); padding_mode, latency=2)

        s = muladd((k)ᵀ → Tc, q → Tc, zeros(Float32, (TILE_N, TILE_M)))
        s = s * qk_scale * inv(log(2f0))

        if j > mask_start
            offs_n = (j - 1i32) * TILE_N .+ offs_n_tile
            mask = offs_n .< k_len
            CAUSAL && (mask = mask .& (offs_n .<= (offs_m)ᵀ))
            s = ifelse.(mask, s, -Inf32)
        end

        m_ij = max.(m_i, maximum(s, dims=1))
        ct.@fpmode flush_to_zero=true begin
            p = exp2.(s .- m_ij)
            l_ij = sum(p, dims=1)
            alpha = exp2.(m_i .- m_ij)
            l_i = l_i .* alpha .+ l_ij
            acc = acc .* alpha
        end

        v = ct.load(V, (1, j, hₖ, b), (Dv, TILE_N); padding_mode, latency=4)
        acc = muladd(v → Tc, p → Tc, acc)

        m_i = m_ij
    end

    ct.@fpmode flush_to_zero=true begin
        o = acc .* ifelse.(l_i .== 0f0, 0f0, 1f0 ./ l_i)   # fully-masked row guard
    end
    ct.store(O, (1, i, h, b), o → eltype(O))

    return
end

function attention!(
    (; O), Q, K, V;
    causal = false,
    k_lengths = nothing,
    q_lengths = nothing,
    tensorcore = tensorcore_type(eltype(Q)),
    TILE_M = 64,
    TILE_N = 64,
)
    @sizes begin
        Q => (Dk, L, H, B)
        K => (Dk, Lk, Hk, B)
        V => (Dv, Lk, Hk, B)
        O => (Dv, L, H, B)
        k_lengths => (B,)
        q_lengths => (B,)
    end
    iszero(H % Hk) ||
        throw(DimensionMismatch("Attention heads ($H) is not a multiple of key-value heads ($Hk)"))

    query_group_size = H ÷ Hk
    qk_scale = Float32(1 / sqrt(Dk))

    @cutile(blocks=(cld(L, TILE_M), H * B),
        mha_fwd(
            Q, K, V, O, k_lengths, q_lengths,
            qk_scale, H,
            tensorcore,
            Constant(Dk),
            Constant(Dv),
            Constant(TILE_M),
            Constant(TILE_N),
            Constant(query_group_size),
            Constant(causal),
        )
    )
end

Pol.outputs(::typeof(attention!), Q, K, V; kwargs...) =
    (; O = Undef(eltype(Q), (size(V, 1), size(Q, 2), size(Q, 3), size(Q, 4))))

const attention = Allocating(attention!)
