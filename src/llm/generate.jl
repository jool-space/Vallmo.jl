# generate.jl — the decode engine: closed-loop generation with optional
# graph capture. Model-agnostic; a model is a file that extends the two
# stubs below.
#
# ── The model contract ──────────────────────────────────────────────────────
#
#   prefill!(model, gen, ids)   ids :: (T, B) device Int32, 1-based.
#     Eager. Fills the model's caches (bulk writes, host offsets fine here),
#     computes the first generated token into `gen.result`, and — as its
#     FINAL act — seeds the clock: `gen.positions .= prompt_lengths .+ 1`.
#     (Attention inside prefill uses the pre-seed prompt lengths; seeding
#     last keeps "positions = decode's clock" exception-free.)
#
#   step!(model, gen)
#     One decode step, and the capture target, so it must be:
#       - device-driven: reads `gen.result` (last token) and `gen.positions`
#         (the clock); no host-computed offsets, sizes, or copies;
#       - pool-allocation-free: temporaries carved from a model-owned
#         Arena, whose frames retract each step so every step replays the
#         same addresses — never the CUDA pool, whose allocation under
#         capture records a memory node that replays against garbage
#         (see Pol/capture.jl).
#     It writes the next token into `gen.result` and touches nothing else
#     of the engine's.
#
# The engine owns the clock. `advance!` — the only writer — harvests
# `gen.result` into the output buffer and bumps `positions` and `step`,
# on device, so a captured (step!; advance!) is self-advancing: the decode
# loop is `launch(exec)` and nothing else. Host work per token: zero.
# EOS detection and streaming happen in an amortized poll every
# `poll_every` tokens (one small device→host copy), off the hot path.
#
# Four phases (the qwen3.5-triton ritual, minus its per-token host writes):
# prefill (eager) → warmup (eager step!s; plan caches and autotune spend
# themselves) → capture one (step!; advance!) → replay. Stream capture
# records without executing, so the capture itself produces no token —
# state is untouched and the first replay continues exactly where warmup
# left off. Token ids are 1-based end to end; subtract 1 at the tokenizer.

using CUDACore: CUDACore, CuVector, CuMatrix, capture, instantiate, launch

function prefill! end
function step! end

# view minus the bounds check. GPUArrays' `view` checks device indices with a
# mapreduce + scalar read — a host sync, capture-illegal — and `@inbounds`
# can't elide it across a non-inlined call. Same SubArray its NonContiguous
# path builds; indices must be in bounds by construction.
uview(A, I...) = Base.unsafe_view(A, Base.to_indices(A, I)...)

struct Generation
    positions   :: CuVector{Int32}   # (B,) the device clock: current token's column
    step        :: CuVector{Int32}   # (1,) next row of tokens_out to write
    result      :: CuVector{Int32}   # (B,) last generated token, 1-based
    tokens_out  :: CuMatrix{Int32}   # (max_new, B) harvested tokens
    lin         :: CuVector{Int32}   # (B,) scratch: linear harvest indices
    col_offsets :: CuVector{Int32}   # (B,) 0-based column strides into tokens_out
end

function Generation(B::Int, max_new::Int)
    Generation(
        CuVector{Int32}(undef, B),
        CuVector{Int32}(undef, 1),
        CuVector{Int32}(undef, B),
        CuMatrix{Int32}(undef, max_new, B),
        CuVector{Int32}(undef, B),
        CuVector{Int32}((0:B-1) .* max_new),
    )
end

function reset!(gen::Generation)
    gen.step .= Int32(1)
    gen.tokens_out .= Int32(0)
    return gen
end

# Harvest the step's token and advance the clock — the only writer of
# `positions`/`step`, once per step, captured along with step!.
function advance!(gen::Generation)
    gen.lin .= gen.step .+ gen.col_offsets
    uview(vec(gen.tokens_out), gen.lin) .= gen.result   # in bounds: step ≤ max_new
    gen.step .+= 1
    gen.positions .+= 1
    return gen
end

# One device→host copy per poll: rows lo:hi of tokens_out. Returns the
# host chunk so the caller can stream it and check for EOS.
poll(gen::Generation, lo::Int, hi::Int) = Array(@view gen.tokens_out[lo:hi, :])

function _run!(model, gen::Generation, ids;
    max_new_tokens, launch_step!::F, eos, poll_every, on_tokens,
) where {F}
    reset!(gen)
    prefill_s = CUDACore.@elapsed begin
        prefill!(model, gen, ids)
        advance!(gen)                        # harvest prefill's token as row 1
    end

    n = 1
    polled = 0
    stop = false
    decode_s = CUDACore.@elapsed while n < max_new_tokens && !stop
        launch_step!()
        n += 1
        if n - polled >= poll_every || n == max_new_tokens
            chunk = poll(gen, polled + 1, n)
            isnothing(on_tokens) || on_tokens(chunk)
            isnothing(eos) || (stop = any(==(Int32(eos)), chunk))
            polled = n
        end
    end

    tokens = Array(@view gen.tokens_out[1:n, :])
    timing = (;
        prefill_s,
        decode_s,
        decode_tokens = n - 1,
        tok_s = (n - 1) / decode_s,
        itl_ms = 1000 * decode_s / (n - 1),
    )
    return (; tokens, timing)
end

"""
    generate!(model, gen, ids; max_new_tokens, eos = nothing, poll_every = 8,
              on_tokens = nothing)

Eager generation (M1): every step launched from the host. The correctness
baseline that `generate_captured!` must match token-for-token.
"""
function generate!(model, gen::Generation, ids; max_new_tokens,
    eos = nothing, poll_every = 8, on_tokens = nothing,
)
    _run!(model, gen, ids; max_new_tokens, eos, poll_every, on_tokens,
        launch_step! = () -> (step!(model, gen); advance!(gen)))
end

"""
    generate_captured!(model, gen, ids; max_new_tokens, warmup = 3, kwargs...)

Captured generation (M2): `warmup` eager steps (autotune, plan caches, and
pool allocations spend themselves), then one (step!; advance!) recorded as a
CUDA graph and replayed for the remaining tokens. Capture records without
executing, so warmup's last state flows straight into the first replay.
"""
function generate_captured!(model, gen::Generation, ids; max_new_tokens,
    warmup = 3, eos = nothing, poll_every = 8, on_tokens = nothing,
)
    exec = nothing
    n_warm = 0
    function launch_step!()
        if n_warm < warmup
            step!(model, gen); advance!(gen)
            n_warm += 1
        else
            if isnothing(exec)
                graph = capture() do
                    step!(model, gen); advance!(gen)
                end
                exec = instantiate(graph)
            end
            launch(exec)
        end
    end
    _run!(model, gen, ids; max_new_tokens, eos, poll_every, on_tokens, launch_step!)
end
