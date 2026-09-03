# Vallmo

A from-scratch Qwen3.5 inference engine in Julia, running on the jool GPU stack
(cuTile kernels, CUDA graph capture), with an OpenAI-compatible server and a
terminal chat client. *Vallmo* is Swedish for poppy.

## What's here

- **`src/`** — the engine.
  - Zero-copy, mmap-backed loaders for safetensors checkpoints (sharded or not) and GGUF files, plus a llama.cpp-convention GGUF adapter that maps unsloth-style BF16 GGUFs back onto the HF layout.
  - Matmuls go straight to cuBLASLt plans (F32 accumulation, workspace carved from the arena). cuTile kernels for Qwen3.5's hybrid architecture: gated DeltaNet (fused and recurrent paths), causal conv1d, gated attention with per-head QK-norm and partial RoPE, RMSNorm, and a fused LM head.
  - A model-agnostic decode engine. Prefill runs eagerly; the decode step is captured once as a CUDA graph and replayed with zero host work per token. Recurrent state can be snapshotted and restored, which gives the server prefix caching across turns.
  - `Vallmo.serve`: a single-GPU `/v1/chat/completions` server with streaming.
- **`chat/`** — Poppy, a server-agnostic chat TUI for any `/v1/chat/completions` endpoint. A poppy blooms in the terminal while the model streams.
- **`cli/`** — the `vallmo` app: `vallmo serve` runs the engine, `vallmo chat` opens Poppy with the Vallmo persona.

## Requirements

- Julia 1.12 and an NVIDIA GPU with Compute Capability >8.0.
- The [jool registry](https://github.com/jool-space/registry)
- Weights: the HF `Qwen/Qwen3.5-4B` (or 9B) checkpoint directory with its `tokenizer.json`, or a BF16 GGUF.

## Running

```julia
julia> using Pkg; Pkg.Registry.add(url = "https://registry.jool.space")
julia> Pkg.instantiate()            # from the repo root
julia> Pkg.Apps.develop(path = "cli")
```

```sh
vallmo serve --model models/Qwen3.5-4B      # or a .gguf; --port, --ctx, --max-new
vallmo chat                                 # in another terminal
```

`serve` warms up at boot (kernel compiles, plan sieve, graph capture) so the
first request only pays its own prefill. Decoding is greedy; sampling
parameters are accepted and ignored.

From Julia, the same pieces are available directly:

```julia
using Vallmo, Bop, CUDACore
tok   = Bop.from_file("models/Qwen3.5-4B/tokenizer.json")
model = Vallmo.qwen35("models/Qwen3.5-4B"; Ctx = 4096, B = 1)
gen   = Vallmo.Generation(1, 256)
ids   = CuMatrix{Int32}(reshape(Bop.encode(tok, "Hello").ids .+ 1, :, 1))
out   = Vallmo.generate_captured!(model, gen, ids; max_new_tokens = 256)
Bop.decode(tok, vec(out.tokens) .- 1)
```
