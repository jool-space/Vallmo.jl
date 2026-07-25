"""
    Poppy

A server-agnostic chat TUI for `/v1/chat/completions` endpoints — a
poppy flower blooms in the terminal while the model streams. Points at
anything OpenAI-compatible; knows nothing about what serves it.

    poppy [--url http://127.0.0.1:8080]

Sibling of Vallmo (Swedish for poppy), whose `vallmo` command opens
this client with Vallmo-flavored defaults. Branding is parameterized:
`chat(; assistant_name, system_prompt, settings_path, …)`.
"""
module Poppy

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
poppy — chat TUI for any /v1/chat/completions server

  poppy [--url http://127.0.0.1:8080]
      The flower blooms while the model streams. Settings live
      in-chat: type `/` for fuzzy-matched commands (/fade, /theme,
      /url, /model, /maxtokens, /clear) with live preview.

  poppy --help
"""

function (@main)(argv)
    argv = collect(String, argv)
    any(a -> a in ("--help", "-h", "help"), argv) && (print(USAGE); return 0)
    # nothing = flag not given → saved settings (then defaults) apply
    opts = Dict{String,Union{Nothing,String}}("--url" => nothing)
    for i in 1:2:length(argv)
        haskey(opts, argv[i]) || (println(stderr, "poppy: unknown flag $(argv[i])\n");
                                  print(stderr, USAGE); return 1)
        i + 1 <= length(argv) || (println(stderr, "poppy: $(argv[i]) needs a value"); return 1)
        opts[argv[i]] = argv[i+1]
    end
    chat(; base_url = opts["--url"])
    return 0
end

public chat, main

# Bake the chat's hot paths into the package image: the render loop,
# markdown → spans, the poppy point cloud, slash-command mode. Without
# this the first frame pays ~2.6 s of JIT; with it, milliseconds.
@compile_workload begin
    redirect_stdout(devnull) do
        m = PoppyChatModel()
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
        main(["--help"])
    end
end

end
