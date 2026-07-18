# Fused: x′ = x + Δx; y = (x′ / rms) * (w + offset)
# Single kernel, two outputs.

function fused_add_rms_norm_fwd(
    Y::TileMatrix,
    X′::TileMatrix,
    X::TileMatrix,
    ΔX::TileMatrix,
    W::TileVector,
    offset::Float32,
    eps::Float32,
    TILE_M::Int,
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    ss = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode) → Float32
        Δx = ct.load(ΔX, (i, bid_n), (TILE_M,); padding_mode) → Float32
        x′ = x + Δx
        ct.store(X′, (i, bid_n), x′ → eltype(X′))
        ss = ss .+ x′ .^ 2
    end
    rstd = 1 / √(sum(ss) / M + eps)

    for i in 1i32:num_tiles
        x′ = ct.load(X′, (i, bid_n), (TILE_M,); padding_mode) → Float32
        w = ct.load(W, i, (TILE_M,); padding_mode) → Float32
        y = x′ .* rstd .* (w .+ offset)
        ct.store(Y, (i, bid_n), y → eltype(Y))
    end

    return
end

function fused_add_rms_norm!(
    (; Y, X′), X, ΔX, W;
    eps, offset
)
    @sizes begin
        Y  => (M, N)
        X′ => (M, N)
        X  => (M, N)
        ΔX => (M, N)
        W  => (M,)
    end

    TILE_M = 1024
    @cutile(blocks=size(X, 2),
        fused_add_rms_norm_fwd(
            Y, X′, X, ΔX, W,
            Float32(offset),
            Float32(eps),
            Constant(TILE_M)
        )
    )
end

Pol.outputs(::typeof(fused_add_rms_norm!), X, args...; kws...) =
    (; Y = Undef(X), X′ = Undef(X))

const fused_add_rms_norm = Allocating(fused_add_rms_norm!)
