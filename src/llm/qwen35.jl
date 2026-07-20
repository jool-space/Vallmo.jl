# qwen35.jl — the model is a file. Weights are nouns (the checkpoint's own
# tree, packed once at load), forwards are verbs (kernels + Vall).
# Decode temporaries live in a model-owned Arena: step! opens a step frame
# (residual x/xn/h), each layer a nested frame (everything else), so all
# layers share one region and every step replays the same addresses — which
# is exactly what graph capture needs, and eager gets bump-allocation and
# automatic retraction from the same code.
#
# Two layer kinds, two structs; the per-layer verbs are methods:
#   decode!(h, l, m, gen, xn; space)       — one token, arena-framed, capturable
#   forward(l, m, xn, T, B)                — prefill, eager, allocates freely
# The engine skeletons (step!/prefill!) hold what the kinds share: norm,
# residual, MLP.
#
# Checkpoint facts (Qwen3.5-4B, config.json + HF names):
#   model.language_model.{embed_tokens, norm, layers.N.*}; visual branch unread.
#   attention: Hq=16, Hkv=4, Dh=256, q_proj packs [q|gate] per head
#   (attn_output_gate), qk-norm per head, partial rope 64 dims.
#   deltanet: Dk=Dv=128, Hk=16, Hv=32, conv kernel 4, state kept Float32
#   (mamba_ssm_dtype). in_proj packs [qkv|a|b|z].
#   mrope note: mrope_interleaved sections only reassign which *position
#   stream* feeds each frequency; text-only runs have t=h=w, so the table
#   degenerates to standard rope with theta=1e7. Nothing to implement.
#   tie_word_embeddings: lm head IS the embedding — (D, vocab) K-major raw,
#   exactly lm_head_argmax!'s W. Zero-copy tie.
#   mtp_* weights (multi-token prediction) sit in the tree, unread.

using CUDACore: CUDACore, cu, CuArray, CuMatrix, CuVector
using Vall: linear!, linear
import cuBLASLt    # loading it is the opt-in: Vall's :lt provider activates

softplus(x) = log1p(exp(x))

# A layer is still a NamedTuple; the struct is a dispatch tag around it.
# `l.q` etc. forward through, so the verbs read the same as before.
struct AttnLayer{NT<:NamedTuple}
    w::NT
end
struct DeltaNetLayer{NT<:NamedTuple}
    w::NT
end
const Layer = Union{AttnLayer, DeltaNetLayer}
Base.getproperty(l::Layer, s::Symbol) =
    s === :w ? getfield(l, :w) : getproperty(getfield(l, :w), s)
Base.propertynames(l::Layer) = propertynames(getfield(l, :w))

struct Qwen35{CFG,E,NW,R,L,SP}
    cfg    :: CFG       # parsed text_config as a NamedTuple
    E      :: E         # (D, vocab) BF16 — embedding AND lm head (tied)
    norm_w :: NW        # (D,) final norm
    rope   :: R         # (; cos, sin) :: (HALF_ROT, Ctx) Float32
    layers :: L         # Vector{Union{AttnLayer, DeltaNetLayer}} (union-split)
    space  :: SP        # Arena for decode temporaries; watermark(m.space)
                        #   measures the true per-step byte requirement
    eos    :: Int32     # 1-based
end

# ── load ─────────────────────────────────────────────────────────────────────

function qwen35(dir::AbstractString; Ctx::Int, B::Int, arena_bytes::Int = 64 << 20)
    tc = JSON.parsefile(joinpath(expanduser(dir), "config.json"))["text_config"]
    cfg = (;
        D    = Int(tc["hidden_size"]),
        eps  = Float32(tc["rms_norm_eps"]),
        Dh   = Int(tc["head_dim"]),
        Hq   = Int(tc["num_attention_heads"]),
        Hkv  = Int(tc["num_key_value_heads"]),
        rot_half = Int(tc["head_dim"] * tc["rope_parameters"]["partial_rotary_factor"]) ÷ 2,
        theta = Float64(tc["rope_parameters"]["rope_theta"]),
        DkL = Int(tc["linear_key_head_dim"]),   HkL = Int(tc["linear_num_key_heads"]),
        DvL = Int(tc["linear_value_head_dim"]), HvL = Int(tc["linear_num_value_heads"]),
        convK = Int(tc["linear_conv_kernel_dim"]),
        inter = Int(tc["intermediate_size"]),
        layer_types = String.(tc["layer_types"]),
        Ctx, B,
    )

    st = SafeTensors(checkpoint_shards(expanduser(dir)))
    GC.@preserve st begin   # tree's leaves alias st's mmaps; every cu() reads them
        w = tree(st).model.language_model
        E = cu(w.embed_tokens.weight)                    # (D, vocab), tied head

        layers = map(collect(enumerate(w.layers))) do (i, lw)
            cfg.layer_types[i] == "full_attention" ?
                attn_layer(lw, cfg) : deltanet_layer(lw, cfg)
        end

        return Qwen35(cfg, E, cu(w.norm.weight), rope_tables(cfg), layers,
                      Arena(CuVector{UInt8}(undef, arena_bytes)),
                      Int32(tc["eos_token_id"] + 1))
    end
end

function rope_tables(cfg)
    inv_freq = Float32.(cfg.theta .^ .-((0:cfg.rot_half-1) ./ cfg.rot_half))
    angles = inv_freq * Float32.(0:cfg.Ctx-1)'           # (HALF_ROT, Ctx); col p = pos p-1
    return (; cos = cu(cos.(angles)), sin = cu(sin.(angles)))
end

# HF Conv1d weight is (dim, 1, K) row-major → raw (K, 1, dim); we want (dim, K).
conv_weight_matrix(w) = permutedims(dropdims(w; dims=2), (2, 1))

mlp_parts(mw) = (;
    Wgu = cu(hcat(mw.gate_proj.weight, mw.up_proj.weight)),  # (D, 2I)
    Wd  = cu(mw.down_proj.weight),                           # (I, D)
)

function attn_layer(lw, cfg)
    (; Dh, Hkv, B, Ctx) = cfg
    sa = lw.self_attn
    return AttnLayer((;
        norm1 = cu(lw.input_layernorm.weight),
        norm2 = cu(lw.post_attention_layernorm.weight),
        Wqkv = cu(hcat(sa.q_proj.weight, sa.k_proj.weight, sa.v_proj.weight)),
        Wo = cu(sa.o_proj.weight),
        qw = cu(sa.q_norm.weight), kw = cu(sa.k_norm.weight),
        K_cache = CUDACore.zeros(BFloat16, Dh, Ctx, Hkv, B),
        V_cache = CUDACore.zeros(BFloat16, Dh, Ctx, Hkv, B),
        mlp = mlp_parts(lw.mlp),
    ))
end

function deltanet_layer(lw, cfg)
    (; DkL, DvL, HkL, HvL, convK, B) = cfg
    dn = lw.linear_attn
    @assert DkL == DvL                                   # slot math in decode! assumes it
    n = 2DkL * HkL + DvL * HvL                           # conv_dim: q|k|v channels
    return DeltaNetLayer((;
        norm1 = cu(lw.input_layernorm.weight),
        norm2 = cu(lw.post_attention_layernorm.weight),
        # [qkv|z] in one GEMV — 12352 total rows don't divide by DkL, so the
        # a|b tail is its own tiny GEMV.
        Wqz = cu(hcat(dn.in_proj_qkv.weight, dn.in_proj_z.weight)),
        Wab = cu(hcat(dn.in_proj_a.weight, dn.in_proj_b.weight)),
        Wout = cu(dn.out_proj.weight),
        conv_w = cu(conv_weight_matrix(dn.conv1d.weight)),
        conv_b = :bias in propertynames(dn.conv1d) ? cu(dn.conv1d.bias) : nothing,
        conv_state = CUDACore.zeros(BFloat16, n, convK, B),
        A_log = cu(Float32.(dn.A_log)), dt_bias = cu(Float32.(dn.dt_bias)),
        gnorm_w = cu(dn.norm.weight),
        S = CUDACore.zeros(Float32, DkL, DvL, HvL, B),   # mamba_ssm_dtype: float32
        mlp = mlp_parts(lw.mlp),
    ))
end

# ── the decode step (the capture target) ─────────────────────────────────────

function decode!(h, l::AttnLayer, m::Qwen35, gen, xn)
    (; eps, Dh, Hq, Hkv, B) = m.cfg
    # Slot layout of the packed GEMV output, reshaped (Dh, slots, B):
    # [q₁ g₁ q₂ g₂ … q_Hq g_Hq | k₁…k_Hkv | v₁…v_Hkv]. Kernel args must be at
    # most one wrapper over a CuArray (cuTile's device_pointer limit), so:
    # reshape the whole buffer (still a CuArray), then single strided views.
    qkv = alloc(BFloat16, Dh * (2Hq + 2Hkv), B)
    linear!((; y = qkv), xn, l.Wqkv)
    qkv3 = reshape(qkv, (Dh, 2Hq + 2Hkv, B))
    qg, k, v = splitaxis(qkv3, (2Hq, Hkv, Hkv); dims = 2)
    q      = view(qkv3, :, 1:2:2Hq, :)     # [q|gate] interleaves per head:
    q_gate = view(qkv3, :, 2:2:2Hq, :)     #   stride-2 slots, beyond splitaxis

    qnorm_rope!(q, l.qw, m.rope.cos, m.rope.sin;
        positions = gen.positions, eps, offset = 1f0)
    knorm_rope_append!(l.K_cache, l.V_cache, k, v, l.kw,
        m.rope.cos, m.rope.sin; positions = gen.positions, eps, offset = 1f0)
    (; O) = decode_attention(q, l.K_cache, l.V_cache; lengths = gen.positions)
    O .*= sigmoid.(q_gate)
    linear!((; y = h), reshape(O, :, B), l.Wo)
    return
end

function decode!(h, l::DeltaNetLayer, m::Qwen35, gen, xn)
    (; eps, DkL, DvL, HkL, HvL, B) = m.cfg
    n = 2DkL * HkL + DvL * HvL
    qz = alloc(BFloat16, n + DvL * HvL, B)
    ab = alloc(BFloat16, 2HvL, B)
    linear!((; y = qz), xn, l.Wqz)
    linear!((; y = ab), xn, l.Wab)
    qz3 = reshape(qz, (DkL, 2HkL + 2HvL, B))
    conv_in = view(qz, 1:n, :)                       # the q|k|v slots, flat
    _, _, _, z = splitaxis(qz3, (HkL, HkL, HvL, HvL); dims = 2)
    alpha, beta = splitaxis(ab, 2)

    (; Y) = causal_conv1d(conv_in, l.conv_state, l.conv_w, l.conv_b;
        σ = silu, output_override = (; S′ = l.conv_state))       # S′ = S: in place
    conv3 = reshape(Y, (DkL, 2HkL + HvL, B))
    q, k, v = splitaxis(conv3, (HkL, HkL, HvL); dims = 2)

    (; O) = fused_deltanet_decode(q, k, v, alpha, beta, z,
        l.A_log, l.dt_bias, l.gnorm_w, l.S;
        eps, output_override = (; S′ = l.S))                     # S′ = S: in place
    linear!((; y = h), reshape(O, :, B), l.Wout)
    return
end

function mlp!(h, mlp, cfg, xn)
    (; inter, B) = cfg
    gu = alloc(BFloat16, 2inter, B)
    linear!((; y = gu), xn, mlp.Wgu)
    gate, up = splitaxis(gu, 2)
    mid = alloc(BFloat16, inter, B)
    mid .= silu.(gate) .* up
    linear!((; y = h), mid, mlp.Wd)
    return
end

function step!(m::Qwen35, gen)
    (; D, B, eps) = m.cfg
    scratchspace(m.space) do step          # the step frame: residual stream
        x  = alloc(BFloat16, D, B)
        xn = alloc(BFloat16, D, B)
        h  = alloc(BFloat16, D, B)
        x .= uview(m.E, :, gen.result)     # in bounds: result is a vocab id

        # Every norm rides fused with the residual add before it; layer 1's
        # norm1 has no residual before it (raw embedding), so it's the loop
        # prologue, and the final norm is the last iteration's `next_w`.
        rms_norm!((; Y = xn), x, first(m.layers).norm1; eps, offset = 1f0)

        for (i, l) in enumerate(m.layers)
            scratchspace() do              # the layer frame: all temporaries
                decode!(h, l, m, gen, xn)
                fused_add_rms_norm!((; Y = xn, X′ = x), x, h, l.norm2; eps, offset = 1f0)
                mlp!(h, l.mlp, m.cfg, xn)
            end
            next_w = i == length(m.layers) ? m.norm_w : m.layers[i+1].norm1
            fused_add_rms_norm!((; Y = xn, X′ = x), x, h, next_w; eps, offset = 1f0)
        end

        lm_head_argmax!((; Result = gen.result), xn, m.E)   # scratch: step frame
    end
    return
end

# ── prefill (eager: allocations, permutes, and host offsets are all fine) ────

# Reference-style helpers, Float32 math. `x` is (Dh, H, T, B).
head_rmsnorm(x, w, eps) =
    Float32.(x) .* (Float32.(w) .+ 1f0) ./
        .√(sum(abs2, Float32.(x); dims=1) ./ size(x, 1) .+ eps)

function apply_rope(x, cos_t, sin_t, half)
    r = 2half
    c = reshape(cos_t, (half, 1, size(cos_t, 2), 1))
    s = reshape(sin_t, (half, 1, size(sin_t, 2), 1))
    x1 = x[1:half, :, :, :]
    x2 = x[half+1:r, :, :, :]
    return vcat(x1 .* c .- x2 .* s, x2 .* c .+ x1 .* s, x[r+1:end, :, :, :])
end

# (Dh, H, T, B) → (Dh, T, H, B), the kernels' time-second convention
tsecond(x) = permutedims(x, (1, 3, 2, 4))

function forward(l::AttnLayer, m::Qwen35, xn, T, B)
    cos_t = @view m.rope.cos[:, 1:T]
    sin_t = @view m.rope.sin[:, 1:T]
    (; eps, Dh, Hq, Hkv, rot_half) = m.cfg
    qkv = linear(Similar(xn), xn, l.Wqkv).y
    Nq = 2Dh * Hq
    qg = reshape(view(qkv, 1:Nq, :), (Dh, 2, Hq, T, B))
    q  = qg[:, 1, :, :, :]                       # (Dh, Hq, T, B)
    g  = qg[:, 2, :, :, :]
    k  = reshape(qkv[Nq+1:Nq+Dh*Hkv, :], (Dh, Hkv, T, B))
    v  = reshape(qkv[Nq+Dh*Hkv+1:end, :], (Dh, Hkv, T, B))

    qr = tsecond(apply_rope(head_rmsnorm(q, l.qw, eps), cos_t, sin_t, rot_half))
    kr = tsecond(apply_rope(head_rmsnorm(k, l.kw, eps), cos_t, sin_t, rot_half))
    copyto!(view(l.K_cache, :, 1:T, :, :), BFloat16.(kr))
    copyto!(view(l.V_cache, :, 1:T, :, :), BFloat16.(tsecond(v)))

    o = similar(xn, Dh, T, Hq, B)
    attention!((; O = o), BFloat16.(qr),
        view(l.K_cache, :, 1:T, :, :), view(l.V_cache, :, 1:T, :, :);
        causal = true)
    o .*= sigmoid.(tsecond(Float32.(g)))
    return linear(Similar(xn),
                  reshape(permutedims(o, (1, 3, 2, 4)), (Dh * Hq, T * B)), l.Wo).y
end

function forward(l::DeltaNetLayer, m::Qwen35, xn, T, B)
    (; eps, DkL, DvL, HkL, HvL) = m.cfg
    qz = linear(Similar(xn), xn, l.Wqz).y
    ab = linear(Similar(xn), xn, l.Wab).y
    n = 2DkL * HkL + DvL * HvL
    conv = similar(xn, n, T, B)
    causal_conv1d_sequence!((; Y = conv), reshape(qz[1:n, :], (n, T, B)),
        l.conv_w, l.conv_b; σ = silu, S′ = l.conv_state)
    cf = reshape(conv, (n, T * B))

    l2n(y) = Float32.(y) ./ .√(sum(abs2, Float32.(y); dims=1) .+ 1f-6)
    q = l2n(reshape(cf[1:DkL*HkL, :], (DkL, HkL, T, B))) .* (1f0 / √Float32(DkL))
    k = l2n(reshape(cf[DkL*HkL+1:2DkL*HkL, :], (DkL, HkL, T, B)))
    qe = tsecond(q)                              # (Dk, T, Hk, B); kernel maps heads
    ke = tsecond(k)
    ve = tsecond(reshape(cf[2DkL*HkL+1:n, :], (DvL, HvL, T, B)))

    a = reshape(ab[1:HvL, :], (HvL, T, B))
    b = reshape(ab[HvL+1:2HvL, :], (HvL, T, B))
    gate = .-exp.(Float32.(l.A_log)) .* softplus.(Float32.(a) .+ l.dt_bias)
    beta = sigmoid.(Float32.(b))

    o = CuArray{Float32}(undef, DvL, T, HvL, B)
    deltanet_sequence!((; O = o, S′ = l.S),                 # S′ = S: in place
        Float32.(qe), Float32.(ke), Float32.(ve), beta, gate, l.S)

    z = reshape(qz[n+1:end, :], (DvL, HvL, T, B))
    rstd = 1f0 ./ .√(sum(abs2, o; dims=1) ./ DvL .+ eps)
    o .*= rstd .* reshape(Float32.(l.gnorm_w), (DvL, 1, 1, 1)) .*
          silu.(tsecond(Float32.(z)))
    return linear(Similar(xn),
        BFloat16.(reshape(permutedims(o, (1, 3, 2, 4)), (DvL * HvL, T * B))), l.Wout).y
end

function prefill!(m::Qwen35, gen, ids)
    (; D, eps) = m.cfg
    T, B = size(ids)

    x = m.E[:, vec(ids)]                                 # (D, T·B) gather
    xn = similar(x)

    rms_norm!((; Y = xn), x, first(m.layers).norm1; eps, offset = 1f0)   # prologue (see step!)

    for (i, l) in enumerate(m.layers)
        h = forward(l, m, xn, T, B)
        fused_add_rms_norm!((; Y = xn, X′ = x), x, h, l.norm2; eps, offset = 1f0)

        gu = linear(Similar(xn), xn, l.mlp.Wgu).y
        mid = silu.(view(gu, 1:m.cfg.inter, :)) .* view(gu, m.cfg.inter+1:2m.cfg.inter, :)
        h = linear(Similar(x), mid, l.mlp.Wd).y
        next_w = i == length(m.layers) ? m.norm_w : m.layers[i+1].norm1
        fused_add_rms_norm!((; Y = xn, X′ = x), x, h, next_w; eps, offset = 1f0)
    end

    xlast = view(reshape(xn, (D, T, B)), :, T, :)        # uniform T; ragged is M2+
    lm_head_argmax!((; Result = gen.result), xlast, m.E)
    gen.positions .= Int32(T + 1)                        # the final act: seed the clock
    return
end
