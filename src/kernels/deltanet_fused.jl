# Fused GatedDeltaNet decode step: L2-norm q/k, gate/β, delta-rule recurrence,
# gated RMSNorm + z-gate epilogue — one kernel, one block per (value head, seq).
#
# Head structure (the part everyone forgets): q and k share Hk heads; v — and
# everything else: z, α, β, A_log, dt_bias, the norm, the output — lives per
# *value* head, Hv = V_PER_K · Hk. This is attention-GQA inverted: there, many
# q heads share a kv head; here, many v heads share a qk head. Each value head
# still owns its own state S(Dk, Dv) — states differ within a group because S
# accumulates v and decays with per-value-head gates — so the grid is (Hv, B)
# and only the q/k loads collapse, via h_k = cld(h_v, V_PER_K) (torch's
# repeat_interleave, spelled as an index instead of a copy).
# Qwen3.5-27B: Dk = Dv = 128, Hk = 16, Hv = 48, V_PER_K = 3.
#
# Numerics pinned by the FLA reference: q and k are L2-normalized (Σx² + ε,
# not RMS — no 1/D), q additionally carries the 1/√Dk attention scale;
# decay = exp(-exp(A_log) · softplus(α + dt_bias)); β = sigmoid(β_raw).
#
# Layout (column-major):
#   Q_raw, K_raw (Dk, Hk, B)      — pre-norm, straight from the projection
#   V, Z, O      (Dv, Hv, B)
#   Alpha, BetaRaw (Hv, B)
#   A_log, DtBias (Hv,)
#   NormW (Dv,)                   — gated-RMSNorm weight, shared across heads
#   S (Dk, Dv, Hv, B)             — state in, read only
#   S′ (Dk, Dv, Hv, B)            — state out; alias S′ = S for in-place decode

function fused_deltanet_decode_fwd(
    O::TileArray3,
    Q_raw::TileArray3,
    K_raw::TileArray3,
    V::TileArray3,
    Alpha::TileMatrix,
    BetaRaw::TileMatrix,
    Z::TileArray3,
    A_log::TileVector,
    DtBias::TileVector,
    NormW::TileVector,
    S::TileArray4,
    S′::TileArray4,
    Dk::Int,
    Dv::Int,
    NormEps::Float32,
    TILE_DK::Int,
    V_PER_K::Int
)
    padding_mode = ct.PaddingMode.Zero
    h_v, b = ct.bid(1), ct.bid(2)   # value head — owns state, gates, output
    h_k = cld(h_v, Int32(V_PER_K))  # its shared qk head
    num = cld(Int32(Dk), Int32(TILE_DK))

    # Per-value-head gate scalars
    α = Alpha[h_v, b] → Float32
    β_raw = BetaRaw[h_v, b] → Float32
    a_log = A_log[h_v] → Float32
    dt_bias = DtBias[h_v] → Float32
    sp_input = α + dt_bias
    sp = sp_input > 20.0f0 ? sp_input : log(1 + exp(sp_input))  # softplus
    decay = exp(-exp(a_log) * sp)
    beta = sigmoid(β_raw)

    # L2 norms over the shared qk head
    q_ss = zeros(Float32, TILE_DK)
    k_ss = zeros(Float32, TILE_DK)
    for i in 1i32:num
        qt = ct.load(Q_raw, (i, h_k, b), (TILE_DK,); padding_mode) → Float32
        kt = ct.load(K_raw, (i, h_k, b), (TILE_DK,); padding_mode) → Float32
        q_ss = q_ss .+ qt .* qt
        k_ss = k_ss .+ kt .* kt
    end
    q_scale = 1 / √(sum(q_ss) + 1f-6) / √Float32(Dk)
    k_scale = 1 / √(sum(k_ss) + 1f-6)

    v = ct.load(V, (1, h_v, b), (Dv,)) → Float32

    # Pass 1: decay + S^T @ k_norm — reads S, writes S′ (safe aliased: each
    # tile is loaded before it is stored, and the block owns its state slice)
    acc = zeros(Float32, Dv)
    for i in 1i32:num
        s = ct.load(S, (i, 1, h_v, b), (TILE_DK, Dv); padding_mode) → Float32
        kt = ct.load(K_raw, (i, h_k, b), (TILE_DK,); padding_mode) → Float32
        s = s .* decay
        ct.store(S′, (i, 1, h_v, b), s → eltype(S′))
        acc = muladd((s)ᵀ, kt .* k_scale, acc)
    end
    delta = beta .* (v .- acc)

    # Pass 2: rank-1 update + output — entirely within S′
    output = zeros(Float32, Dv)
    for i in 1i32:num
        s = ct.load(S′, (i, 1, h_v, b), (TILE_DK, Dv); padding_mode) → Float32
        kt = ct.load(K_raw, (i, h_k, b), (TILE_DK,); padding_mode) → Float32
        qt = ct.load(Q_raw, (i, h_k, b), (TILE_DK,); padding_mode) → Float32
        s = s .+ (kt .* k_scale) .* (delta)ᵀ
        ct.store(S′, (i, 1, h_v, b), s → eltype(S′))
        output = muladd((s)ᵀ, qt .* q_scale, output)
    end

    # Gated RMSNorm + z-gate epilogue
    rms_sq = sum(output .* output) / Dv
    rstd = 1 / √(rms_sq + NormEps)
    norm_w = ct.load(NormW, (1,), (Dv,)) → Float32
    z = ct.load(Z, (1, h_v, b), (Dv,)) → Float32
    final = output .* rstd .* norm_w .* silu.(z)

    ct.store(O, (1, h_v, b), final → eltype(O))
    return
end

function fused_deltanet_decode!(
    (; O, S′), Q_raw, K_raw, V, Alpha, BetaRaw, Z,
    A_log, DtBias, NormWeight, S;
    eps=1f-6
)
    @sizes begin
        O          => (Dv, Hv, B)
        Q_raw      => (Dk, Hk, B)
        K_raw      => (Dk, Hk, B)
        V          => (Dv, Hv, B)
        Alpha      => (Hv, B)
        BetaRaw    => (Hv, B)
        Z          => (Dv, Hv, B)
        A_log      => (Hv,)
        DtBias     => (Hv,)
        NormWeight => (Dv,)
        S          => (Dk, Dv, Hv, B)
        S′         => (Dk, Dv, Hv, B)
    end
    iszero(Hv % Hk) ||
        throw(DimensionMismatch("Value heads ($Hv) is not a multiple of qk heads ($Hk)"))
    v_per_k = Hv ÷ Hk

    TILE_DK = 16
    @cutile(blocks=(Hv, B),
        fused_deltanet_decode_fwd(
            O, Q_raw, K_raw, V, Alpha, BetaRaw, Z,
            A_log, DtBias, NormWeight, S, S′,
            Constant(Dk), Constant(Dv), Constant(Float32(eps)),
            Constant(TILE_DK), Constant(v_per_k)
        )
    )
end

Pol.outputs(::typeof(fused_deltanet_decode!),
            Q, K, V, Alpha, BetaRaw, Z, A_log, DtBias, NormWeight, S; kws...) =
    (; O = Undef(V), S′ = Undef(S))

const fused_deltanet_decode = Allocating(fused_deltanet_decode!)
