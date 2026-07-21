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
using CommonMark        # activates TachikomaMarkdownExt at load (and precompile)
using HTTP, JSON
using Base64: base64encode
using PrecompileTools

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

# Bake the chat's hot paths into the package image: the render loop,
# markdown → spans, the poppy point cloud, slash-command mode. Without
# this the first frame pays ~2.6 s of JIT; with it, milliseconds. The
# serve path is left alone — its startup is model load + GPU warmup,
# and CUDA can't run at precompile time.
@compile_workload begin
    redirect_stdout(devnull) do
        m = VallmoChatModel()
        push!(m.messages, (:user, "hello"))
        push!(m.messages,
              (:assistant, "<think>think</think>\n# H\n**b** `c` and\n- item\n"))
        m.bloom = 0.6
        area = Rect(1, 1, 100, 30)
        f = Frame(Buffer(area), area, GraphicsRegion[], PixelSnapshot[])
        view(m, f)
        for c in collect("hi\\")
            update!(m, KeyEvent(:char, c))
        end
        update!(m, KeyEvent(:enter))            # backslash continuation
        update!(m, KeyEvent(:backspace))
        view(m, f)
        for c in collect("/the")                # command mode + fuzzy + preview
            update!(m, KeyEvent(:char, c))
        end
        update!(m, KeyEvent(:tab))
        update!(m, KeyEvent(:down))
        view(m, f)
        update!(m, KeyEvent(:escape))
        update!(m, MouseEvent(5, 5, mouse_scroll_down, mouse_press,
                              false, false, false))
        view(m, f)
        main(["help"])
    end
end

end
