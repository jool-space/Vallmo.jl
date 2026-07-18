# RMSNorm along dim 1, column-major (M, N). Grid: one block per column.

function rms_norm_fwd(
    X::TileMatrix,
    W::TileVector,
    Y::TileMatrix,
    offset::Float32,
    eps::Float32,
    TILE_M::Int
)
    padding_mode = ct.PaddingMode.Zero
    bid_n = ct.bid(1)
    num_tiles = ct.num_tiles(X, 1, (TILE_M, 1))
    M = size(X, 1)

    ss = zeros(Float32, TILE_M)
    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode) → Float32
        ss = ss .+ x .^ 2
    end
    rstd = 1 / √(sum(ss) / M .+ eps)

    for i in 1i32:num_tiles
        x = ct.load(X, (i, bid_n), (TILE_M,); padding_mode) → Float32
        w = ct.load(W, i, (TILE_M,); padding_mode) → Float32
        y = x .* rstd .* (w .+ offset)
        ct.store(Y, (i, bid_n), y → eltype(Y))
    end

    return
end

function rms_norm!(
    (; Y), X::AbstractMatrix, W::AbstractVector;
    eps, offset = 0f0
)
    @sizes begin
        Y => (M, N)
        X => (M, N)
        W => (M,)
    end

    TILE_M = 1024
    @cutile(blocks=N,
        rms_norm_fwd(
            X, W, Y,
            Float32(offset),
            Float32(eps),
            Constant(TILE_M)
        )
    )
end

Pol.outputs(::typeof(rms_norm!), X, W; kws...) =
    (; Y = Undef(X))

const rms_norm = Allocating(rms_norm!)
