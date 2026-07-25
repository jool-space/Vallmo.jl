# gguf_llamacpp.jl — llama.cpp-convention GGUF (unsloth et al.) → the HF
# checkpoint's own naming and layout, so qwen35's loader stays source-blind.
#
# llama.cpp's qwen35 converter is not a rename; it rewrites tensors.
# Everything below was established empirically by byte-comparing unsloth's
# BF16 GGUF against the HF Qwen3.5-9B checkpoint:
#   - deltanet value heads are deinterleaved, vperm = [1:2:Hv; 2:2:Hv]:
#     the v segment of qkv's columns, z's columns, out_proj's rows, the
#     conv's v channels, a/b's columns, and the A/dt entries. q/k segments
#     are untouched.
#   - ssm_a = -exp(A_log[vperm]) — the training parameterization folded
#     for inference; log.(-a) recovers it (float-exact to well under the
#     BF16 grid the original lived on).
#   - every RMSNorm weight except linear_attn.norm is stored as w + 1
#     (llama.cpp folds Qwen3.5's (1+w) norm; +1 is exact in F32, so
#     subtracting recovers the original bits).
#   - norms and conv are upcast to F32 in the file; converted back to the
#     HF dtypes here.
# Untouched tensors — attention, mlp, embeddings, most of the bytes —
# pass through as the mmap-aliasing arrays themselves.

# The rewritten tensors are fresh copies, but the pass-throughs alias the
# GGUF's mmap: the wrapper keeps it rooted through GC.@preserve at the
# load site, exactly like SafeTensors.
struct AdaptedTensors <: AbstractDict{String,Array}
    root::GGUF
    tensors::Dict{String,Array}
end
Base.getindex(a::AdaptedTensors, k::AbstractString) = a.tensors[k]
Base.haskey(a::AdaptedTensors, k::AbstractString) = haskey(a.tensors, k)
Base.keys(a::AdaptedTensors) = keys(a.tensors)
Base.length(a::AdaptedTensors) = length(a.tensors)
Base.iterate(a::AdaptedTensors, state...) = iterate(a.tensors, state...)

_headcols(perm, dh) = reduce(vcat, [(h - 1) * dh .+ (1:dh) for h in perm])

function llamacpp_qwen35(g::GGUF)
    md = g.metadata
    arch = get(md, "general.architecture", "?")
    arch == "qwen35" || error("expected general.architecture qwen35, got $arch")
    nb   = Int(md["qwen35.block_count"])
    ival = Int(md["qwen35.full_attention_interval"])
    Dh   = Int(md["qwen35.attention.key_length"])
    Dk   = Int(md["qwen35.ssm.state_size"])
    Hk   = Int(md["qwen35.ssm.group_count"])
    Hv   = Int(md["qwen35.ssm.time_step_rank"])
    layer_types = [i % ival == 0 ? "full_attention" : "linear_attention"
                   for i in 1:nb]
    tied = !haskey(g, "output.weight")
    cj = Dict{String,Any}(
        "tie_word_embeddings" => tied,
        "text_config" => Dict{String,Any}(
            "hidden_size" => Int(md["qwen35.embedding_length"]),
            "rms_norm_eps" => Float64(md["qwen35.attention.layer_norm_rms_epsilon"]),
            "head_dim" => Dh,
            "num_attention_heads" => Int(md["qwen35.attention.head_count"]),
            "num_key_value_heads" => Int(md["qwen35.attention.head_count_kv"]),
            "rope_parameters" => Dict{String,Any}(
                "partial_rotary_factor" => Int(md["qwen35.rope.dimension_count"]) / Dh,
                "rope_theta" => Float64(md["qwen35.rope.freq_base"])),
            "linear_key_head_dim" => Dk, "linear_value_head_dim" => Dk,
            "linear_num_key_heads" => Hk, "linear_num_value_heads" => Hv,
            "linear_conv_kernel_dim" => Int(md["qwen35.ssm.conv_kernel"]),
            "intermediate_size" => Int(md["qwen35.feed_forward_length"]),
            "layer_types" => layer_types,
            "eos_token_id" => Int(md["tokenizer.ggml.eos_token_id"]),
        ))

    ivp = invperm([1:2:Hv; 2:2:Hv])            # undo the v-head deinterleave
    vcols = _headcols(ivp, Dk)
    unnorm(w) = BFloat16.(w .- 1)              # unfold the (1+w) offset

    d = Dict{String,Array}()
    P = "model.language_model."
    d[P*"embed_tokens.weight"] = g["token_embd.weight"]
    d[P*"norm.weight"] = unnorm(g["output_norm.weight"])
    tied || (d["lm_head.weight"] = g["output.weight"])
    for i in 0:nb-1
        b = "blk.$i."
        L = P * "layers.$i."
        d[L*"input_layernorm.weight"] = unnorm(g[b*"attn_norm.weight"])
        d[L*"post_attention_layernorm.weight"] = unnorm(g[b*"post_attention_norm.weight"])
        d[L*"mlp.gate_proj.weight"] = g[b*"ffn_gate.weight"]
        d[L*"mlp.up_proj.weight"]   = g[b*"ffn_up.weight"]
        d[L*"mlp.down_proj.weight"] = g[b*"ffn_down.weight"]
        if layer_types[i+1] == "full_attention"
            S = L * "self_attn."
            d[S*"q_proj.weight"] = g[b*"attn_q.weight"]     # [q|gate], as HF packs it
            d[S*"k_proj.weight"] = g[b*"attn_k.weight"]
            d[S*"v_proj.weight"] = g[b*"attn_v.weight"]
            d[S*"o_proj.weight"] = g[b*"attn_output.weight"]
            d[S*"q_norm.weight"] = unnorm(g[b*"attn_q_norm.weight"])
            d[S*"k_norm.weight"] = unnorm(g[b*"attn_k_norm.weight"])
        else
            A = L * "linear_attn."
            qk = 2 * Dk * Hk                                # q|k channels before v
            qkv = g[b*"attn_qkv.weight"]
            d[A*"in_proj_qkv.weight"] = hcat(qkv[:, 1:qk], qkv[:, qk .+ vcols])
            d[A*"in_proj_z.weight"] = g[b*"attn_gate.weight"][:, vcols]
            d[A*"in_proj_a.weight"] = g[b*"ssm_alpha.weight"][:, ivp]
            d[A*"in_proj_b.weight"] = g[b*"ssm_beta.weight"][:, ivp]
            d[A*"out_proj.weight"]  = g[b*"ssm_out.weight"][vcols, :]
            conv = g[b*"ssm_conv1d.weight"]                 # (K, qk + Hv*Dk) F32
            conv = hcat(conv[:, 1:qk], conv[:, qk .+ vcols])
            d[A*"conv1d.weight"] = BFloat16.(reshape(conv, size(conv, 1), 1, :))
            d[A*"A_log"]   = log.(-g[b*"ssm_a"])[ivp]       # F32; loader wants F32
            d[A*"dt_bias"] = g[b*"ssm_dt.bias"][ivp]
            d[A*"norm.weight"] = BFloat16.(g[b*"ssm_norm.weight"])
        end
    end
    return cj, AdaptedTensors(g, d)
end
