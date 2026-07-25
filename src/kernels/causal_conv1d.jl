# Depthwise causal conv1d over the channel dim, with optional bias and
# activation. Decode form shifts a (D, K, B) ring-buffer state; sequence
# form reads left context directly from X.

#=== single step ===#

function causal_conv1d_update_fwd(
    Y::TileMatrix,       # (D, B) — output
    X::TileMatrix,       # (D, B) — new input
    S::TileArray3,   # (D, K, B) — ring buffer in, read only
    S′::TileArray3,  # (D, K, B) — shifted ring buffer out (alias in place:
                         #   col i+1 is read before iteration i+1 stores it)
    Weight::TileMatrix,  # (D, K) — conv weights
    Bias::Optional{TileVector},    # (D,)
    σ::Optional{Function},
    TILE_D::Int,
    KERNEL_SIZE::Int,
)
    padding_mode = ct.PaddingMode.Zero
    pid, b = ct.bid(1), ct.bid(2)

    x = ct.load(X, (pid, b), (TILE_D,); padding_mode) → Float32
    acc = zeros(Float32, TILE_D)

    for i in 1i32:KERNEL_SIZE
        s = i == KERNEL_SIZE ? x :
            ct.load(S, (pid, i + 1i32, b), (TILE_D,); padding_mode) → Float32

        ct.store(S′, (pid, i, b), s → eltype(S′))

        w = ct.load(Weight, (pid, i), (TILE_D,); padding_mode) → Float32
        acc = acc .+ s .* w
    end

    isnothing(Bias) || (acc = acc .+ ct.load(Bias, (pid,), (TILE_D,); padding_mode) → Float32)
    isnothing(σ) || (acc = σ.(acc))

    ct.store(Y, (pid, b), acc → eltype(Y))

    return
end

function causal_conv1d!((; Y, S′), X, S, Weight, Bias; σ=nothing)
    @sizes begin
        Y      => (D, B)
        X      => (D, B)
        S      => (D, K, B)
        S′     => (D, K, B)
        Weight => (D, K)
        Bias   => (D,)
    end

    TILE_D = 64
    @cutile(blocks=(cld(D, TILE_D), B),
        causal_conv1d_update_fwd(
            Y, X, S, S′, Weight, Bias, σ,
            Constant(TILE_D),
            Constant(K),
        )
    )
end

Pol.outputs(::typeof(causal_conv1d!), X, S, args...; kws...) =
    (; Y = Undef(X), S′ = Undef(S))

const causal_conv1d = Allocating(causal_conv1d!)


#=== full sequence ===#

# Grid: (ceil(D/TILE_D), B), each block handles TILE_D channels for all T positions.

function causal_conv1d_sequence_fwd(
    Y::TileArray3,       # (D, T, B) — output
    X::TileArray3,       # (D, T, B)
    Weight::TileMatrix,  # (D, K)
    Bias::Optional{TileVector},    # (D,)
    σ::Optional{Function},
    S′::Optional{TileArray3},  # (D, K, B) — seed for decode handoff (write-only)
    T_len::Int,
    TILE_D::Int,
    KERNEL_SIZE::Int,
)
    padding_mode = ct.PaddingMode.Zero
    pid = ct.bid(1)  # tile along D
    b   = ct.bid(2)  # batch

    bias = isnothing(Bias) ? nothing :
        ct.load(Bias, (pid,), (TILE_D,); padding_mode) → Float32

    for t in 1i32:T_len
        acc = zeros(Float32, TILE_D)

        for k in 1i32:KERNEL_SIZE
            src_t = t - Int32(KERNEL_SIZE) + k
            if src_t >= 1i32
                x_val = ct.load(X, (pid, src_t, b), (TILE_D,); padding_mode) → Float32
            else
                x_val = zeros(Float32, TILE_D)
            end
            w_val = ct.load(Weight, (pid, k), (TILE_D,); padding_mode) → Float32
            acc = acc .+ x_val .* w_val
        end

        isnothing(bias) || (acc = acc .+ bias)
        isnothing(σ) || (acc = σ.(acc))

        ct.store(Y, (pid, t, b), acc → eltype(Y))
    end

    # Seed the decode ring buffer: S′[:, i] = X[:, T - K + i] — raw inputs,
    # matching the update kernel's convention (column i holds x_{t-(K-i)},
    # newest in column K) — zeros where the sequence is shorter than the
    # window, so causal_conv1d! continues seamlessly at position T + 1.
    if !isnothing(S′)
        for i in 1i32:KERNEL_SIZE
            src_t = Int32(T_len) - Int32(KERNEL_SIZE) + i
            if src_t >= 1i32
                s = ct.load(X, (pid, src_t, b), (TILE_D,); padding_mode) → Float32
            else
                s = zeros(Float32, TILE_D)
            end
            ct.store(S′, (pid, i, b), s → eltype(S′))
        end
    end

    return
end

function causal_conv1d_sequence!((; Y), X, Weight, Bias; σ=nothing, S′=nothing)
    @sizes begin
        Y      => (D, T, B)
        X      => (D, T, B)
        Weight => (D, K)
        Bias   => (D,)
        S′ => (D, K, B)
    end

    TILE_D = 64
    @cutile(blocks=(cld(D, TILE_D), B),
        causal_conv1d_sequence_fwd(
            Y, X, Weight, Bias, σ, S′,
            T,               # runtime: only a loop bound — a Constant here
                             # would recompile the kernel per sequence length
            Constant(TILE_D),
            Constant(K),
        )
    )
end

Pol.outputs(::typeof(causal_conv1d_sequence!), X, W, B; kws...) =
    (; Y = Undef(X))

const causal_conv1d_sequence = Allocating(causal_conv1d_sequence!)
