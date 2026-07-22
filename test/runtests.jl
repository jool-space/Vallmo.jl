using Vallmo
using Vallmo: SafeTensors, checkpoint_shards, safetensors, tree, GGUF
using Test
using BFloat16s: BFloat16

# Build a GGUF v3 file by hand: magic, version, tensor/kv counts, metadata
# key-values, tensor infos, then data aligned to general.alignment (32).
gguf_str(io, s) = (write(io, UInt64(ncodeunits(s))); write(io, s))
function write_gguf(path; kvs, tensors)
    open(path, "w") do io
        write(io, "GGUF", UInt32(3), UInt64(length(tensors)), UInt64(length(kvs)))
        for (k, vt, writer) in kvs
            gguf_str(io, k)
            write(io, UInt32(vt))
            writer(io)
        end
        off = 0
        for (name, dims, tid, data) in tensors
            gguf_str(io, name)
            write(io, UInt32(length(dims)), UInt64.(collect(dims)), UInt32(tid),
                  UInt64(off))
            off += 32 * cld(sizeof(data), 32)     # next tensor starts aligned
        end
        pad = 32 * cld(position(io), 32) - position(io)
        write(io, zeros(UInt8, pad))
        for (_, _, _, data) in tensors
            write(io, data)
            pad = 32 * cld(sizeof(data), 32) - sizeof(data)
            write(io, zeros(UInt8, pad))
        end
    end
end

# Build a safetensors file by hand: 8-byte little-endian header length,
# JSON header padded to 8-byte alignment, then raw little-endian data.
# Byte-exact: sizeof, not length — a stray non-ASCII char in a header
# would silently shift the data section otherwise.
function write_safetensors(path, header_json::String, data::Vector{UInt8})
    @assert isascii(header_json)
    n = 8 * cld(sizeof(header_json), 8)
    open(path, "w") do io
        write(io, htol(UInt64(n)))
        write(io, header_json, " "^(n - sizeof(header_json)))
        write(io, data)
    end
end

@testset "Vallmo.jl" begin
    @testset "SafeTensors" begin
        # "float32_35": Python shape (3, 5), values arange(15) — row-major on
        # disk, so the column-major truth is the 5×3 with columns 0:4, 5:9, 10:14.
        # "bf16_4" follows at offset 60.
        header = """{
            "float32_35": {"dtype": "F32", "shape": [3, 5], "data_offsets": [0, 60]},
            "bf16_4": {"dtype": "BF16", "shape": [4], "data_offsets": [60, 68]},
            "__metadata__": {"format": "pt"}
        }"""
        data = vcat(
            reinterpret(UInt8, Float32.(0:14)),
            reinterpret(UInt8, BFloat16[1, 2, 3, 4]),
        )
        path = joinpath(mktempdir(), "model.safetensors")
        write_safetensors(path, header, Vector(data))

        rt = SafeTensors(path)
        @test length(rt) == 2
        @test haskey(rt, "float32_35")
        @test rt.metadata["format"] == "pt"

        W = rt["float32_35"]
        @test W isa Matrix{Float32}
        @test size(W) == (5, 3)                          # header's [3, 5], reversed
        @test W == reshape(Float32.(0:14), 5, 3)
        @test strides(W) == (1, 5)                       # honest contiguous Array

        v = rt["bf16_4"]
        @test v isa Vector{BFloat16}
        @test v == BFloat16[1, 2, 3, 4]

        # Zero copy: the tensor aliases the mmap'd pages, no materialization.
        root = only(rt.roots)
        @test UInt(pointer(W)) - UInt(pointer(root)) < length(root)

        # A lying header fails loudly, not silently.
        badheader = """{"w": {"dtype": "F32", "shape": [3, 5], "data_offsets": [0, 59]}}"""
        badpath = joinpath(mktempdir(), "bad.safetensors")
        write_safetensors(badpath, badheader, Vector(data))
        @test_throws ErrorException SafeTensors(badpath)
    end

    @testset "GGUF" begin
        path = joinpath(mktempdir(), "model.gguf")
        write_gguf(path;
            kvs = [
                ("general.architecture", 8, io -> gguf_str(io, "qwen")),
                ("qwen.block_count", 4, io -> write(io, UInt32(2))),
                ("qwen.rope.freq_base", 6, io -> write(io, Float32(10000))),
                ("tokenizer.ggml.add_bos", 7, io -> write(io, UInt8(1))),
                # string array and numeric array (numeric takes the read! fast path)
                ("tokenizer.ggml.tokens", 9, io -> begin
                    write(io, UInt32(8), UInt64(3))
                    foreach(s -> gguf_str(io, s), ["<s>", "a", "b"])
                end),
                ("tokenizer.ggml.token_type", 9, io -> begin
                    write(io, UInt32(5), UInt64(3))
                    write(io, Int32[3, 1, 1])
                end),
            ],
            tensors = [
                # ne = [3, 2] fastest-first: contiguous 0:5 is the 3×2 whose
                # columns are 0:2 and 3:5 — dims used as-is, nothing flipped
                ("blk.0.w", (3, 2), 0, collect(reinterpret(UInt8, Float32.(0:5)))),
                ("blk.1.v", (4,), 30, collect(reinterpret(UInt8, BFloat16[1, 2, 3, 4]))),
            ])

        g = GGUF(path)
        @test length(g) == 2
        @test g.metadata["general.architecture"] == "qwen"
        @test g.metadata["qwen.block_count"] === UInt32(2)
        @test g.metadata["qwen.rope.freq_base"] === Float32(10000)
        @test g.metadata["tokenizer.ggml.add_bos"] === true
        @test g.metadata["tokenizer.ggml.tokens"] == ["<s>", "a", "b"]
        @test g.metadata["tokenizer.ggml.token_type"] == Int32[3, 1, 1]

        W = g["blk.0.w"]
        @test W isa Matrix{Float32}                      # honest Array, no wrapper
        @test size(W) == (3, 2)                          # ggml dims are Julia dims
        @test W == reshape(Float32.(0:5), 3, 2)
        @test strides(W) == (1, 3)

        v = g["blk.1.v"]
        @test v isa Vector{BFloat16}
        @test v == BFloat16[1, 2, 3, 4]

        # Zero copy: tensors alias the mmap'd pages.
        @test UInt(pointer(W)) - UInt(pointer(g.root)) < length(g.root)

        # The same tree the safetensors path gets.
        w = tree(g)
        @test w.blk[1].w === W

        # Quantized tensors refuse loudly; truncated data fails the bounds check.
        qpath = joinpath(mktempdir(), "quant.gguf")
        write_gguf(qpath; kvs = [],
            tensors = [("q", (256,), 12, zeros(UInt8, 144))])
        @test_throws ErrorException GGUF(qpath)
        tpath = joinpath(mktempdir(), "trunc.gguf")
        write_gguf(tpath; kvs = [],
            tensors = [("t", (64, 64), 0, zeros(UInt8, 16))])
        @test_throws ErrorException GGUF(tpath)

        # A big-endian GGUF has the same magic; its version reads swapped.
        bpath = joinpath(mktempdir(), "be.gguf")
        open(bpath, "w") do io
            write(io, "GGUF", bswap(UInt32(3)), bswap(UInt64(0)), bswap(UInt64(0)))
        end
        @test_throws "big-endian" GGUF(bpath)
    end

    @testset "sharded checkpoints" begin
        entry(name, shape, offs) =
            "\"$name\": {\"dtype\": \"F32\", \"shape\": [$(join(shape, ","))], \"data_offsets\": [$(join(offs, ","))]}"
        dir = mktempdir()
        write_safetensors(joinpath(dir, "model-00001-of-00002.safetensors"),
            "{" * entry("model.a.weight", (2, 3), (0, 24)) * "}",
            Vector(reinterpret(UInt8, Float32.(1:6))))
        write_safetensors(joinpath(dir, "model-00002-of-00002.safetensors"),
            "{" * entry("model.layers.0.w", (4,), (0, 16)) * ", " *
                  entry("model.layers.1.w", (4,), (16, 32)) * "}",
            Vector(reinterpret(UInt8, Float32.(1:8))))
        open(joinpath(dir, "model.safetensors.index.json"), "w") do io
            write(io, "{\"metadata\": {\"total_size\": 56}, \"weight_map\": {" *
                "\"model.a.weight\": \"model-00001-of-00002.safetensors\", " *
                "\"model.layers.0.w\": \"model-00002-of-00002.safetensors\", " *
                "\"model.layers.1.w\": \"model-00002-of-00002.safetensors\"}}")
        end

        @test length(checkpoint_shards(dir)) == 2            # via the index
        w = safetensors(dir)
        @test size(w.model.a.weight) == (3, 2)               # reversed, cross-shard
        @test w.model.layers[2].w == Float32.(5:8)

        # a tensor in two shards fails loudly
        @test_throws ErrorException SafeTensors(
            fill(joinpath(dir, "model-00002-of-00002.safetensors"), 2))

        # directory without an index falls back to globbing shards
        rm(joinpath(dir, "model.safetensors.index.json"))
        @test length(checkpoint_shards(dir)) == 2
    end

    @testset "tree" begin
        t = Vallmo.tree(Dict(
            "model.layers.0.w" => [1.0], "model.layers.1.w" => [2.0],
            "model.norm" => [3.0], "lm_head" => [4.0],
        ))
        @test t.model.layers[2].w == [2.0]                    # 0-indexed keys, 1-indexed access
        @test t.model.norm == [3.0]
        @test t.lm_head == [4.0]
        @test t.model.layers isa Vector
        @test isconcretetype(eltype(t.model.layers))          # homogeneous blocks stay concrete

        # non-contiguous layer indices fail loudly
        @test_throws ErrorException Vallmo.tree(Dict("a.0.w" => [1], "a.2.w" => [2]))
        # a key that is both leaf and prefix fails loudly
        @test_throws ErrorException Vallmo.tree(Dict("a" => [1], "a.b" => [2]))
    end

end
