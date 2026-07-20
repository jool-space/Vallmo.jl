using Vallmo
using Vallmo: SafeTensors, checkpoint_shards, safetensors, tree
using Test
using BFloat16s: BFloat16

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
