# gguf.jl — zero-copy GGUF reading.
#
# A GGUF file is one self-contained checkpoint: magic "GGUF", u32 version,
# u64 tensor and kv counts, metadata key-values (config + tokenizer — the
# single-file property is most of GGUF's point), tensor infos (name, dims,
# ggml dtype, offset), then — aligned to general.alignment (default 32) —
# the raw little-endian tensor bytes.
#
# ggml stores dims fastest-first: ne[0] is the contiguous axis. That is
# column-major — Julia's own order — so the dims as stored ARE the Julia
# shape: no transpose, no PermutedDimsArray, nothing physical to permute.
# It lands in the same orientation as the safetensors reader's reversed
# row-major labels: a weight PyTorch calls (out, in) is (in, out) from
# either reader.
#
# Scope: read-only, mmap-backed, little-endian hosts, plain dtypes only
# (floats, ints, bfloat16) — ggml quantization blocks are not decoded.

using Mmap: mmap
using BFloat16s: BFloat16

const GGML_DTYPES = Dict{UInt32,DataType}(
    0 => Float32, 1 => Float16, 24 => Int8, 25 => Int16,
    26 => Int32, 27 => Int64, 28 => Float64, 30 => BFloat16)

# ids we recognize but do not decode, for error messages
const GGML_QUANTS = Dict{UInt32,String}(
    2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0",
    9 => "Q8_1", 10 => "Q2_K", 11 => "Q3_K", 12 => "Q4_K", 13 => "Q5_K",
    14 => "Q6_K", 15 => "Q8_K", 16 => "IQ2_XXS", 17 => "IQ2_XS",
    18 => "IQ3_XXS", 19 => "IQ1_S", 20 => "IQ4_NL", 21 => "IQ3_S",
    22 => "IQ2_S", 23 => "IQ4_XS", 29 => "IQ1_M")

# metadata value types; BOOL (7), STRING (8) and ARRAY (9) read specially
const GGUF_VTYPES = Dict{UInt32,DataType}(
    0 => UInt8, 1 => Int8, 2 => UInt16, 3 => Int16, 4 => UInt32, 5 => Int32,
    6 => Float32, 10 => UInt64, 11 => Int64, 12 => Float64)

gguf_string(io::IO) = String(read(io, Int(read(io, UInt64))))

function gguf_value(io::IO, t::UInt32)
    t == 7 && return read(io, UInt8) != 0          # bool: one byte
    t == 8 && return gguf_string(io)
    if t == 9                                      # array: elem type, count, values
        et = read(io, UInt32)
        n = Int(read(io, UInt64))
        T = get(GGUF_VTYPES, et, nothing)
        T === nothing && return [gguf_value(io, et) for _ in 1:n]  # strings/bools/nested
        return read!(io, Vector{T}(undef, n))
    end
    T = get(GGUF_VTYPES, t, nothing)
    T === nothing && error("GGUF metadata value of unknown type $t")
    return read(io, T)
end

align_up(n::Integer, a::Integer) = n + (a - n % a) % a

"""
    GGUF(path)

Open a GGUF file: `metadata` holds the key-values (config, tokenizer), and
the object is an `AbstractDict{String,Array}` of the tensors, each a plain
`Array` aliasing the file's mmap'd pages — zero copy, contiguous, dims as
stored (ggml's fastest-first order is column-major already). The pages are
read-only — copy before mutating. The struct roots the mapping; keep it
alive as long as any tensor from it is in use. `tree(GGUF(path))` gives the
same weight tree `safetensors` checkpoints get.
"""
struct GGUF <: AbstractDict{String,Array}
    root::Vector{UInt8}
    metadata::Dict{String,Any}
    tensors::Dict{String,Array}
end

function GGUF(path::AbstractString)
    ENDIAN_BOM == 0x04030201 ||
        error("GGUF data is little-endian; this host is not")
    bytes = mmap(expanduser(path), Vector{UInt8})
    io = IOBuffer(bytes)
    String(read(io, 4)) == "GGUF" || error("$path: not a GGUF file (bad magic)")
    version = Int(read(io, UInt32))
    # The magic is the same four chars in both byte orders; a big-endian
    # file betrays itself by its version reading as a byte-swapped int.
    Int(bswap(UInt32(version))) in (1, 2, 3) &&
        error("$path: big-endian GGUF is not supported")
    version in (2, 3) ||
        error("unsupported GGUF version $version (v1's 32-bit counts are not handled)")
    n_tensors = Int(read(io, UInt64))
    n_kv = Int(read(io, UInt64))

    metadata = Dict{String,Any}()
    for _ in 1:n_kv
        k = gguf_string(io)
        metadata[k] = gguf_value(io, read(io, UInt32))
    end

    infos = [(gguf_string(io),                                   # name
              ntuple(_ -> Int(read(io, UInt64)),                 # dims, Julia order
                     Int(read(io, UInt32))),
              read(io, UInt32), Int(read(io, UInt64)))           # dtype id, offset
             for _ in 1:n_tensors]

    alignment = Int(get(metadata, "general.alignment", 32))
    alignment > 0 || error("bad general.alignment $alignment")
    datastart = align_up(position(io), alignment)

    tensors = Dict{String,Array}()
    for (name, dims, tid, off) in infos
        haskey(tensors, name) && error("duplicate tensor $name")
        T = get(GGML_DTYPES, tid, nothing)
        T === nothing && error("tensor $name: " * (haskey(GGML_QUANTS, tid) ?
            "$(GGML_QUANTS[tid])-quantized tensors are not supported" :
            "unknown ggml dtype $tid"))
        datastart + off + prod(dims) * sizeof(T) <= length(bytes) ||
            error("tensor $name: data runs past the end of the file")
        ptr = Ptr{T}(pointer(bytes) + datastart + off)
        UInt(ptr) % sizeof(T) == 0 || error("tensor $name is misaligned for $T")
        tensors[name] = unsafe_wrap(Array, ptr, dims; own=false)
    end
    return GGUF(bytes, metadata, tensors)
end

Base.getindex(g::GGUF, name::AbstractString) = g.tensors[name]
Base.haskey(g::GGUF, name::AbstractString) = haskey(g.tensors, name)
Base.keys(g::GGUF) = keys(g.tensors)
Base.length(g::GGUF) = length(g.tensors)
Base.iterate(g::GGUF, state...) = iterate(g.tensors, state...)
