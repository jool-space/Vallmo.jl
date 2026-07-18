# safetensors.jl — zero-copy safetensors reading.
#
# A safetensors file is: 8 bytes of little-endian header length, a JSON
# header (padded so the data section starts 8-byte aligned), then raw
# little-endian tensor bytes at header-given offsets. The header's shapes
# are row-major labels; read column-major, the same bytes are the reversed
# shape — there is nothing physical to permute. A weight PyTorch labels
# (out, in) is (in, out) here: K-major, the orientation the TN paths want.
# Only tensors whose *label* is already (in, out) — embeddings — need a
# transpose, and it belongs at the call site.
#
# Scope: read-only, mmap-backed, little-endian hosts. Sharded checkpoints
# are just several files merged: each shard's own header names its tensors,
# so the HF index.json's weight_map adds nothing a merge doesn't — its one
# real job is naming the shard list when given a directory (mmap makes
# "avoid opening files" moot; opening a shard touches zero pages).

using Mmap: mmap
using JSON: JSON
using BFloat16s: BFloat16
using Microfloats: Float8_E4M3FN, Float8_E5M2

const DTYPES = Dict(
    "F64" => Float64, "F32" => Float32, "F16" => Float16, "BF16" => BFloat16,
    "I64" => Int64, "I32" => Int32, "I16" => Int16, "I8" => Int8,
    "U64" => UInt64, "U32" => UInt32, "U16" => UInt16, "U8" => UInt8,
    "BOOL" => Bool,
    "F8_E4M3" => Float8_E4M3FN, "F8_E5M2" => Float8_E5M2,
)

"""
    SafeTensors(path)
    SafeTensors(paths::AbstractVector)

Open one safetensors file — or several shards, merged — and expose every
tensor as an `Array` aliasing the files' mmap'd pages: zero copy, contiguous,
dimensions reversed from the header's row-major labels. The pages are
read-only — copy before mutating. The struct roots the mappings; keep it
alive as long as any tensor from it is in use.
"""
struct SafeTensors <: AbstractDict{String,Array}
    roots::Vector{Vector{UInt8}}
    tensors::Dict{String,Array}
    metadata::Dict{String,String}
end

SafeTensors(path::AbstractString) = SafeTensors([path])

function SafeTensors(paths::AbstractVector{<:AbstractString})
    ENDIAN_BOM == 0x04030201 ||
        error("safetensors data is little-endian; this host is not")
    roots = Vector{UInt8}[]
    tensors = Dict{String,Array}()
    metadata = Dict{String,String}()
    for path in paths
        bytes = mmap(path, Vector{UInt8})
        push!(roots, bytes)
        headerlen = Int(only(reinterpret(UInt64, @view bytes[1:8])))
        header = JSON.parse(String(bytes[9:8+headerlen]))
        datastart = 8 + headerlen
        for (name, entry) in header
            if name == "__metadata__"
                merge!(metadata, entry)
                continue
            end
            haskey(tensors, name) &&
                error("tensor $name appears in more than one shard")
            T = DTYPES[entry["dtype"]]
            dims = reverse(Tuple(Int.(entry["shape"]))) # reverse is key distinction between FluxML/SafeTensors.jl
            start, stop = Int.(entry["data_offsets"])
            stop - start == prod(dims) * sizeof(T) ||
                error("tensor $name: data_offsets disagree with dtype × shape")
            ptr = Ptr{T}(pointer(bytes) + datastart + start)
            UInt(ptr) % sizeof(T) == 0 || error("tensor $name is misaligned for $T")
            tensors[name] = unsafe_wrap(Array, ptr, dims; own=false)
        end
    end
    return SafeTensors(roots, tensors, metadata)
end

# Resolve a checkpoint: a .safetensors file, or a directory holding either
# model.safetensors or an HF index (whose weight_map names the shard list —
# possibly "00001-of-00001"; small models ship indexes too).
function checkpoint_shards(path::AbstractString)
    isdir(path) || return [path]
    index = joinpath(path, "model.safetensors.index.json")
    single = joinpath(path, "model.safetensors")
    if isfile(index)
        weight_map = JSON.parsefile(index)["weight_map"]
        return [joinpath(path, f) for f in sort!(unique(String.(values(weight_map))))]
    elseif isfile(single)
        return [single]
    end
    files = sort!(filter(endswith(".safetensors"), readdir(path; join=true)))
    isempty(files) && error("no .safetensors files found in $path")
    return files
end

Base.getindex(rt::SafeTensors, name::AbstractString) = rt.tensors[name]
Base.haskey(rt::SafeTensors, name::AbstractString) = haskey(rt.tensors, name)
Base.keys(rt::SafeTensors) = keys(rt.tensors)
Base.length(rt::SafeTensors) = length(rt.tensors)
Base.iterate(rt::SafeTensors, state...) = iterate(rt.tensors, state...)

include("tree.jl")
safetensors(path) = tree(SafeTensors(checkpoint_shards(expanduser(path))))
