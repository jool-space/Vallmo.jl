"""
    VallmoCLI

The `vallmo` command — Julia app (1.12) wrapping the jool stack's Qwen3.5
serving loop and its poppy-fronted chat client.

    vallmo chat  [--url …] [--fade …] [--theme …]
    vallmo serve [--dir …] [--host …] [--port …] [--ctx …] [--max-new …]

`chat` is the default command and loads only the TUI stack; `serve`
lazily imports Vallmo and the CUDA packages, so the client stays snappy
and runs on GPU-less machines.
"""
module VallmoCLI

using Tachikoma
@tachikoma_app          # extend view/update!/init!/should_quit for our Model
using HTTP, JSON
using Base64: base64encode

include("poppy_render.jl")
include("commands.jl")
include("chat.jl")

const USAGE = """
vallmo — Qwen3.5 on the jool stack: server and chat client

  vallmo [chat] [--url http://127.0.0.1:8080]
      The poppy chat client. The flower blooms while the model streams.
      Settings live in-chat: type `/` for fuzzy-matched commands
      (/fade, /theme, /url, /model, /maxtokens, /clear) with live preview.

  vallmo serve [--dir MODELS_DIR] [--host 127.0.0.1] [--port 8080]
               [--ctx 4096] [--max-new 2048]
      /v1/chat/completions over captured decode with prefix caching.

  vallmo help
"""

function chat_cmd(args)
    opts = Dict("--url" => "http://127.0.0.1:8080")
    for i in 1:2:length(args)
        haskey(opts, args[i]) || (println(stderr, "vallmo chat: unknown flag $(args[i])"); return 1)
        i + 1 <= length(args) || (println(stderr, "vallmo chat: $(args[i]) needs a value"); return 1)
        opts[args[i]] = args[i+1]
    end
    chat(; base_url = opts["--url"])
    return 0
end

function serve_cmd(args)
    @eval begin
        import Vallmo
        import CUDACore
        import cuBLASLt
        import HuggingFaceTokenizers
    end
    if !isdefined(@__MODULE__, :_serve_main)
        Base.include(@__MODULE__, joinpath(@__DIR__, "serve_impl.jl"))
    end
    Base.invokelatest(_serve_main, args)
    return 0
end

function (@main)(argv)
    argv = collect(String, argv)
    isempty(argv) && return chat_cmd(argv)
    cmd, rest = argv[1], argv[2:end]
    cmd == "chat" && return chat_cmd(rest)
    cmd == "serve" && return serve_cmd(rest)
    startswith(cmd, "--") && return chat_cmd(argv)   # bare flags → chat
    cmd in ("help", "-h") && (print(USAGE); return 0)
    println(stderr, "vallmo: unknown command '$cmd'\n")
    print(stderr, USAGE)
    return 1
end

public chat, main

end
