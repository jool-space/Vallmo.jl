# Shared kernel utilities. One file per kernel below, mirroring the
# qwen3.5-triton layout (triton_<name>.py ↔ <name>.jl).

import cuTile
import cuTile as ct
using cuTile: Constant, TFloat32
using CUDACore: @cuda, i32
using Pol: Pol, alloc, Similar, Undef, Allocating, Arena, scratchspace,
    withspace, ambientspace

using Vall: @sizes

macro cutile(args...)
    esc(:(@cuda backend=$cuTile $(args...)))
end

const TileVector{T} = cuTile.TileArray{T,1}
const TileMatrix{T} = cuTile.TileArray{T,2}
const TileArray3{T} = cuTile.TileArray{T,3}
const TileArray4{T} = cuTile.TileArray{T,4}

const Optional{T} = Union{T,Nothing}

x → T::Type = T.(x)

struct TransposePostfix end
const ᵀ = TransposePostfix()
Base.:(*)(x, ::TransposePostfix) = transpose(x)

arithmetic_type(T::Type) = T
arithmetic_type(::Type{TFloat32}) = Float32

tensorcore_type(T::Type) = T
tensorcore_type(::Type{Float32}) = TFloat32

sigmoid(x) = 1 / (1 + exp(-x))
silu(x) = x / (1 + exp(-x))

include("rms_norm.jl")
include("residual_norm.jl")
include("attention.jl")
include("attention_decode.jl")
include("deltanet_recurrent.jl")
include("deltanet_fused.jl")
include("causal_conv1d.jl")
include("qknorm_rope_append.jl")
include("lm_head_topk.jl")
