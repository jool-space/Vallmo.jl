# Fused lm-head GEMV + streaming argmax — never materializes the logits.
#
# The lm head is the fattest single read at decode (W is (K, N) with N = vocab);
# writing N logits to HBM and scanning them back is pure waste when greedy
# decode only wants one index. Each block owns TILE_N vocab columns: it streams
# the GEMV over K and reduces its columns to a (max, index) pair in registers.
# `lm_head_argmax_combine` then folds the per-tile pairs to one index per
# sequence. Argmax only (no top-k): the output shape is deterministic, which is
# what graph capture needs; sampling wants the same split with a per-tile top-k
# — a later variant, same scratch.
#
# The batch dimension rides in the tile (the decode_attention! GROUP trick):
# each K/W tile is read once for all B sequences, so there is no batched
# variant — B=1 is just the degenerate tile.
#
# Layout (column-major):
#   X (K, B)                      — hidden state per sequence
#   W (K, N)                      — K-major; the raw-checkpoint orientation,
#                                   so a tied embedding loads with no transpose
#   Result (B,) Int32             — 1-based token index per sequence
#
# Partials (allocated by the host wrapper):
#   LocalMax (num_tiles, B) Float32
#   LocalIdx (num_tiles, B) Int32
#
# Ties resolve to the smallest index, within tiles and across them.

function lm_head_argmax_split_fwd(
    X::TileMatrix,
    W::TileMatrix,
    LocalMax::TileMatrix{Float32},
    LocalIdx::TileMatrix{Int32},
    Tc::Type,
    B::Int,
    TILE_N::Int,
    TILE_K::Int,
)
    padding_mode = ct.PaddingMode.Zero
    pid = ct.bid(1)
    N = size(W, 2)
    num_k = ct.num_tiles(W, 1, (TILE_K, TILE_N))

    acc = zeros(Float32, (TILE_N, B))
    for i in 1i32:num_k
        x = ct.load(X, (i, 1), (TILE_K, B); padding_mode)
        w = ct.load(W, (i, pid), (TILE_K, TILE_N); padding_mode, latency=2)
        acc = muladd((w)ᵀ → Tc, x → Tc, acc)
    end

    # 0-based vocab ids, converted to 1-based only at the store — the same
    # `scalar .+ (arange .- 1)` idiom as the attention kernels. (A cuTile dev
    # snapshot once miscompiled the bare `scalar .+ arange` fold here;
    # registered releases fold it correctly, so this is style, not workaround.)
    ids0 = (pid - 1i32) * Int32(TILE_N) .+ (ct.arange(TILE_N) .- 1i32)
    valid = ids0 .< Int32(N)
    acc = ifelse.(valid, acc, Float32(-Inf32))

    local_max = maximum(acc, dims=1)
    is_max = acc .== local_max
    local_idx = minimum(ifelse.(is_max, ids0, Int32(N)), dims=1) .+ 1i32

    ct.store(LocalMax, (pid, 1), reshape(local_max, (1, B)))
    ct.store(LocalIdx, (pid, 1), reshape(local_idx, (1, B)))

    return
end

function lm_head_argmax_combine(
    Result::TileVector{Int32},
    LocalMax::TileMatrix{Float32},
    LocalIdx::TileMatrix{Int32},
    B::Int,
    TILE_T::Int,
)
    padding_mode = ct.PaddingMode.Zero
    num_tiles = size(LocalMax, 1)
    num_chunks = ct.num_tiles(LocalMax, 1, (TILE_T, 1))

    offs_t = ct.arange(TILE_T) .- 1i32

    best_val = fill(-Inf32, (1, B))
    best_idx = zeros(Int32, (1, B))

    for t in 1i32:num_chunks
        vals = ct.load(LocalMax, (t, 1), (TILE_T, B); padding_mode) → Float32
        idxs = ct.load(LocalIdx, (t, 1), (TILE_T, B); padding_mode) → Int32

        # Zero-padded rows past num_tiles would beat negative logits; mask them.
        rows = (t - 1i32) * Int32(TILE_T) .+ offs_t
        vals = ifelse.(rows .< Int32(num_tiles), vals, Float32(-Inf32))

        chunk_max = maximum(vals, dims=1)
        is_max = vals .== chunk_max
        chunk_idx = minimum(ifelse.(is_max, idxs, typemax(Int32)), dims=1)

        # Strict > keeps the earlier (smaller-index) chunk on cross-chunk ties.
        take = chunk_max .> best_val
        best_idx = ifelse.(take, chunk_idx, best_idx)
        best_val = max.(best_val, chunk_max)
    end

    ct.store(Result, (1,), reshape(best_idx, B))

    return
end

# Fused lm-head GEMV + argmax: `Result[b] = argmax(Wᵀ X[:, b])`, 1-based,
# without materializing the `(N, B)` logits.
function lm_head_argmax!(
    (; Result),
    X, W;
    tensorcore = tensorcore_type(eltype(X)),
    TILE_N = 256,
    TILE_K = 128,
    space = ambientspace(Similar(X)),
    scratch = alloc(space, Pol.scratch(lm_head_argmax!, X, W; TILE_N)),
)
    (; LocalMax, LocalIdx) = scratch

    @sizes begin
        X => (K, B)
        W => (K, N)
        Result => (B,)
        LocalMax => (num_tiles, B)
        LocalIdx => (num_tiles, B)
    end

    @assert num_tiles == cld(N, TILE_N)
    @cutile(blocks=num_tiles,
        lm_head_argmax_split_fwd(
            X, W, LocalMax, LocalIdx,
            tensorcore,
            Constant(B),
            Constant(TILE_N),
            Constant(TILE_K),
        )
    )

    TILE_T = min(nextpow(2, num_tiles), 1024)
    @cutile(blocks=1,
        lm_head_argmax_combine(
            Result, LocalMax, LocalIdx,
            Constant(B),
            Constant(TILE_T),
        )
    )
end

Pol.outputs(::typeof(lm_head_argmax!), X, W; kwargs...) =
    (; Result = Undef(Int32, (size(X, 2),)))

function Pol.scratch(::typeof(lm_head_argmax!), X, W; TILE_N)
    @sizes begin
        X => (K, B)
        W => (K, N)
    end
    num_tiles = cld(N, TILE_N)
    return (;
        LocalMax = Undef(Float32, (num_tiles, B)),
        LocalIdx = Undef(Int32, (num_tiles, B)),
    )
end

Pol.@takes_space lm_head_argmax!
const lm_head_argmax = Allocating(lm_head_argmax!)
