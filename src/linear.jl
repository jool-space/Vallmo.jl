# linear.jl — `y .= Wᵀx` on cuBLASLt, directly. Activations `x :: (K, N)`,
# weights stored `W :: (K, M)` (K-major, the orientation checkpoint bytes and
# narrow types want), giving `y :: (M, N)`.
#
# One plan per problem (element types and M, N, K), built on first use and
# cached for the process; a warm call is a single Lt launch. The plan's
# workspace is carved from the ambient Pol space — a decode step's arena
# frame, or a pool round-trip outside one — never cuBLASLt's own default,
# which would cudaMalloc under graph capture. Shapes are checked once here;
# a problem cuBLASLt has no algorithm for falls back to `mul!` (cuBLAS).

using LinearAlgebra: mul!, transpose
using CUDACore: CuMatrix
using Pol: Pol, Allocating
using cuBLASLt: cuBLASLt, MatmulPlan

const LT_PLANS = Dict{Tuple{DataType,DataType,DataType,Int,Int,Int},Union{MatmulPlan,Nothing}}()
const LT_PLANS_LOCK = ReentrantLock()

# `nothing` means cuBLASLt has no algorithm for this problem: remembered, so
# the fallback is taken without re-asking the heuristic every call.
function lt_plan(y::CuMatrix, x::CuMatrix, W::CuMatrix)
    (K, N), M = size(x), size(W, 2)
    key = (eltype(y), eltype(x), eltype(W), M, N, K)
    lock(LT_PLANS_LOCK) do
        get!(LT_PLANS, key) do
            try
                MatmulPlan(; M, N, K, typeA = eltype(W), typeB = eltype(x),
                           typeD = eltype(y), transA = 'T', compute = :f32)
            catch err
                err isa ArgumentError && return nothing       # no algorithm
                err isa cuBLASLt.CUBLASError &&               # unsupported dtype/layout
                    err.code == cuBLASLt.CUBLAS_STATUS_NOT_SUPPORTED && return nothing
                rethrow()
            end
        end
    end
end

"""
    linear!((; y), x, W)

`y .= Wᵀx` — `x :: (K, N)`, `W :: (K, M)` (K-major), `y :: (M, N)`.
cuBLASLt with F32 accumulation; workspace from the ambient Pol space, so a
warm call inside an arena frame allocates nothing (capture-safe).

The functional form is [`linear`](@ref)` = Pol.Allocating(linear!)`.
"""
function linear!((; y), x, W; kwargs...)
    @sizes begin
        x => (K, N)
        W => (K, M)
        y => (M, N)
    end
    lt_linear!(y, x, W)
    return y
end

function lt_linear!(y::CuMatrix, x::CuMatrix, W::CuMatrix)
    plan = lt_plan(y, x, W)
    plan === nothing && return mul!(y, transpose(W), x)
    space = Pol.ambientspace(Pol.Similar(y))
    (; workspace) = Pol.alloc(space, (; workspace = Pol.Undef(UInt8, max(plan.workspace_size, 1))))
    plan(y, W, x; workspace)
    return y
end
lt_linear!(y, x, W) = mul!(y, transpose(W), x)     # anything not a plain CuMatrix

Pol.outputs(::typeof(linear!), x, W; kwargs...) =
    (; y = Pol.Undef(eltype(x), (size(W, 2), size(x, 2))))

"""
    linear(space, x, W) -> (; y)
    linear(x, W) -> (; y)

The allocating form of [`linear!`](@ref): `y` is materialized from `space`,
or from the ambient space when called without one.
"""
const linear = Allocating(linear!)
