"""
    VallmoCLI

The `vallmo` command — Julia app (1.12) tying the twins together:
Poppy (the server-agnostic chat TUI) opened with the Vallmo persona,
and Vallmo's server (Qwen3.5 on the jool stack).

    vallmo chat  [--url …]
    vallmo serve [--model …] [--tokenizer …] [--host …] [--port …] [--ctx …] [--max-new …]

`chat` is the default command and loads only Poppy's TUI stack; `serve`
lazily imports Vallmo (and the CUDA stack with it), so the client stays
snappy and runs on GPU-less machines.
"""
module VallmoCLI

import Poppy
using PrecompileTools

const USAGE = """
vallmo — Qwen3.5 on the jool stack: server and chat client

  vallmo [chat] [--url http://127.0.0.1:8080]
      The poppy chat client. The flower blooms while the model streams.
      Settings live in-chat: type `/` for fuzzy-matched commands
      (/fade, /theme, /url, /model, /maxtokens, /clear) with live preview.

  vallmo serve [--model HF_MODEL_DIR|model.gguf] [--tokenizer tokenizer.json]
               [--host 127.0.0.1] [--port 8080] [--ctx 4096] [--max-new 2048]
      /v1/chat/completions over captured decode with prefix caching.

  vallmo help
"""

# The Vallmo persona over Poppy's neutral defaults. A constant prefix
# on every request, so the server's prefix cache keeps paying off
# across turns.
const SYSTEM_PROMPT = """
    You are Vallmo (Swedish word meaning 'Poppy'), a \
    helpful poppy. A detailed poppy rendering of you \
    is displayed in the background of the terminal chat interface, \
    blooming as the conversation begins. \
    Since the visual representation is already present, \
    please avoid the use of emojis. \
    Be yourself: helpful first, flower second."""

const SETTINGS = joinpath(homedir(), ".vallmo", "settings.json")

function chat_cmd(args)
    # nothing = flag not given → saved settings (then defaults) apply
    opts = Dict{String,Union{Nothing,String}}("--url" => nothing)
    for i in 1:2:length(args)
        haskey(opts, args[i]) || (println(stderr, "vallmo chat: unknown flag $(args[i])"); return 1)
        i + 1 <= length(args) || (println(stderr, "vallmo chat: $(args[i]) needs a value"); return 1)
        opts[args[i]] = args[i+1]
    end
    Poppy.chat(; base_url = opts["--url"], assistant_name = "Vallmo",
                 system_prompt = SYSTEM_PROMPT, settings_path = SETTINGS)
    return 0
end

function serve_cmd(args)
    @eval import Vallmo         # the CUDA stack comes with it
    # Resolve the binding via invokelatest: naming `Vallmo` directly
    # reads the global in this function's (pre-import) world, which
    # 1.12 warns on and later versions will error on.
    V = Base.invokelatest(getglobal, @__MODULE__, :Vallmo)
    Base.invokelatest(V.serve, args)
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

public main

# The chat's hot paths (render loop, markdown, the poppy point cloud)
# are baked into Poppy's package image by its own workload; here only
# the dispatcher needs covering. The serve path is left alone — its
# startup is model load + GPU warmup, and CUDA can't run at precompile
# time.
@compile_workload begin
    redirect_stdout(devnull) do
        main(["help"])
    end
end

end
