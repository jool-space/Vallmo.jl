# server.jl — `/v1/chat/completions` over the captured decode; what
# `vallmo serve` runs. The engine's canonical interface, vLLM-style:
# HTTP + the Bop tokenizer over generate_captured!, nothing else.
#
# One model, one GPU worker, one queue: requests serialize through a
# `Channel{Job}` into the worker task that owns the model, so the HTTP
# layer never touches the GPU. The worker holds a single `CaptureSession` —
# warmup and graph capture are paid on the first request, every later
# request replays the same graph from token 1 (same buffers, same
# addresses). Decoding is greedy (lm_head_argmax!); sampling params are
# accepted and ignored. The tokenizer (Bop) is pure Julia and safe to
# share across handler threads; the GPU stays serialized via the queue.

import CUDACore
import HTTP
import JSON
import Bop

# ── chat template (Qwen3.5) ──────────────────────────────────────────────────

content_text(c::AbstractString) = String(c)
content_text(parts::AbstractVector) =
    join(p["text"] for p in parts if get(p, "type", "text") == "text")

# History never carries thought: the template strips assistant think blocks.
strip_think(s) = replace(s, r"<think>.*?</think>\s*"s => "")

# enable_think=false prefills an empty think block after the assistant
# header — the model sees thinking as already over and answers directly
# (the same trick the official template's enable_thinking=false uses).
function render_chat(messages; enable_think::Bool=true)
    io = IOBuffer()
    for m in messages
        role = m["role"]
        text = content_text(m["content"])
        role == "assistant" && (text = strip_think(text))
        print(io, "<|im_start|>", role, "\n", text, "<|im_end|>\n")
    end
    print(io, "<|im_start|>assistant\n")
    enable_think || print(io, "<think>\n\n</think>\n\n")
    return String(take!(io))
end

# ── the GPU worker ───────────────────────────────────────────────────────────

struct Job
    ids     :: Matrix{Int32}   # (T, 1) host, 1-based; the worker moves it over
    max_new :: Int
    out     :: Channel{Any}    # (:delta, str)… then (:done, meta) | (:error, msg)
end

# Streaming decodes the whole accumulated sequence each poll and emits the
# byte-suffix beyond what was already sent — append-only for byte-level BPE,
# except a trailing replacement char (a token split mid-UTF-8), which is
# withheld until its partner token lands.
function serve_worker(model, gen, tok, session, jobs)
    # Prefix cache, one conversation slot: `snap` is the recurrent state
    # right after prefilling `cached` (think-blocks are stripped from
    # history by the template, so the *generated* sequence is never a
    # prefix of the next prompt — the post-prefill state is). A request
    # whose ids strictly extend `cached` restores and prefills only the
    # suffix; anything else resets and prefills in full.
    cached = Int32[]
    snap = nothing
    for job in jobs
        try
            ids_h = vec(job.ids)
            Tc = length(cached)
            hit = snap !== nothing && 0 < Tc < length(ids_h) &&
                  Base.view(ids_h, 1:Tc) == cached   # Base.: Tachikoma claims `view`
            if hit
                Vallmo.restore!(model, snap)
            else
                Vallmo.reset!(model)
                Tc = 0
            end
            cached = Int32[]        # invalid until the new snapshot lands
            on_prefilled = () -> begin
                snap = snap === nothing ? Vallmo.snapshot(model) :
                                          Vallmo.snapshot!(snap, model)
                cached = ids_h
            end
            ids = CUDACore.CuMatrix{Int32}(reshape(ids_h[Tc+1:end], :, 1))
            acc = Int32[]
            hit_eos = false
            sent = ""
            emit! = function ()
                full = Bop.decode(tok, acc .- 1)
                # Byte-level BPE token boundaries can split a codepoint:
                # Bop.decode then passes the raw incomplete bytes through
                # (no '�' substitution), so trim by validity, not by char.
                safe = full
                while !isempty(safe)
                    c = safe[lastindex(safe)]
                    (c == '�' || !isvalid(c)) || break
                    safe = safe[1:prevind(safe, lastindex(safe))]
                end
                if startswith(safe, sent) && ncodeunits(safe) > ncodeunits(sent)
                    put!(job.out, (:delta, safe[ncodeunits(sent)+1:end]))
                    sent = safe
                end
            end
            on_tokens = function (chunk)
                hit_eos && return
                for t in vec(chunk)
                    t == model.eos && (hit_eos = true; break)
                    push!(acc, t)
                end
                emit!()
            end
            out = generate_captured!(model, gen, ids; session, on_tokens,
                max_new_tokens = job.max_new, eos = model.eos,
                prefill_offset = Tc, on_prefilled)
            Tc > 0 && @info "prefix cache: reused $Tc tokens, prefilled $(size(ids, 1))"
            emit!()                                    # flush a withheld tail
            put!(job.out, (:done, (;
                finish = hit_eos ? "stop" : "length",
                completion_tokens = length(acc),
                reused = Tc,
                timing = out.timing)))
        catch err
            if err isa InvalidStateException      # handler closed `out`: client gone
                @info "generation aborted (client disconnected)"
            else
                @error "generation failed" exception = (err, catch_backtrace())
                try
                    put!(job.out, (:error, sprint(showerror, err)))
                catch
                end
            end
        finally
            close(job.out)
        end
    end
end

# ── HTTP ─────────────────────────────────────────────────────────────────────

function respond_json(http, status, obj)
    HTTP.setstatus(http, status)
    HTTP.setheader(http, "Content-Type" => "application/json")
    HTTP.startwrite(http)
    write(http, JSON.json(obj))
    return
end

function handle_chat(http, st)
    body = JSON.parse(String(read(http)))
    prompt = render_chat(body["messages"];
                         enable_think = get(body, "enable_thinking", true) === true)
    ids = Bop.encode(st.tok, prompt).ids .+ 1            # 0-based → 1-based
    T = length(ids)
    room = st.ctx - T - 1
    room >= 1 || return respond_json(http, 400,
        (; error = (; message = "prompt is $T tokens; context is $(st.ctx)")))
    req_max = get(body, "max_tokens", nothing)
    max_new = clamp(isnothing(req_max) ? st.max_new : Int(req_max),
                    1, min(st.max_new, room))

    out = Channel{Any}(64)
    put!(st.jobs, Job(reshape(Int32.(ids), :, 1), max_new, out))
    id = "chatcmpl-" * string(st.counter[] += 1)
    created = round(Int, time())

    # If the client disconnects, the SSE write below throws and this
    # handler dies — closing `out` turns the worker's next put! into an
    # exception (its abort path) instead of a forever-block on a full
    # channel nobody drains.
    try
    if get(body, "stream", false) === true
        HTTP.setstatus(http, 200)
        HTTP.setheader(http, "Content-Type" => "text/event-stream")
        HTTP.setheader(http, "Cache-Control" => "no-cache")
        HTTP.startwrite(http)
        sse(obj) = write(http, "data: ", JSON.json(obj), "\n\n")
        chunk(delta, finish) = (; id, object = "chat.completion.chunk", created,
            model = st.model_id,
            choices = [(; index = 0, delta, finish_reason = finish)])
        sse(chunk((; role = "assistant", content = ""), nothing))
        finish = "stop"
        for (kind, payload) in out
            kind === :delta && sse(chunk((; content = payload), nothing))
            kind === :done && (finish = payload.finish)
            kind === :error && sse((; error = (; message = payload)))
        end
        sse(chunk((;), finish))
        write(http, "data: [DONE]\n\n")
    else
        parts = String[]
        finish, completion_tokens = "stop", 0
        vallmo = (;)
        for (kind, payload) in out
            kind === :delta && push!(parts, payload)
            kind === :done && begin
                finish, completion_tokens = payload.finish, payload.completion_tokens
                vallmo = (; payload.reused, prefill_s = payload.timing.prefill_s,
                    tok_s = payload.timing.tok_s)
            end
            kind === :error && return respond_json(http, 500,
                (; error = (; message = payload)))
        end
        respond_json(http, 200, (;
            id, object = "chat.completion", created, model = st.model_id,
            choices = [(; index = 0, finish_reason = finish,
                message = (; role = "assistant", content = join(parts)))],
            usage = (; prompt_tokens = T, completion_tokens,
                total_tokens = T + completion_tokens),
            vallmo))
    end
    finally
        close(out)
    end
    return
end

function handle(http, st)
    req = http.message
    target = String(HTTP.URI(req.target).path)
    if req.method == "POST" && target == "/v1/chat/completions"
        handle_chat(http, st)
    elseif req.method == "GET" && target in ("/v1/models", "/health")
        respond_json(http, 200, target == "/health" ? (; status = "ok") :
            (; object = "list", data = [(; id = st.model_id, object = "model",
                created = st.created, owned_by = "vallmo")]))
    else
        respond_json(http, 404,
            (; error = (; message = "no route: $(req.method) $target")))
    end
end

# ── main ─────────────────────────────────────────────────────────────────────

function parse_args(args)
    cfg = Dict("--model" => joinpath(pkgdir(Vallmo), "models", "Qwen3.5-4B"),
               "--host" => "127.0.0.1", "--port" => "8080",
               "--ctx" => "4096", "--max-new" => "2048",
               "--tokenizer" => "")
    for i in 1:2:length(args)
        haskey(cfg, args[i]) || error("unknown flag $(args[i]); knows $(keys(cfg))")
        cfg[args[i]] = args[i+1]
    end
    return (; model_path = abspath(expanduser(cfg["--model"])), host = cfg["--host"],
        port = parse(Int, cfg["--port"]), ctx = parse(Int, cfg["--ctx"]),
        max_new = parse(Int, cfg["--max-new"]),
        tokenizer = isempty(cfg["--tokenizer"]) ? nothing :
                    abspath(expanduser(cfg["--tokenizer"])))
end

# Resolution order: --tokenizer flag, then tokenizer.json in the
# checkpoint directory / beside the .gguf, then the GGUF's own metadata
# (Bop reads tokens/merges/pre-tokenizer straight from the header, so a
# bare .gguf serves self-contained).
function load_tokenizer(model_path, tokenizer)
    tokenizer === nothing || return Bop.from_file(tokenizer)
    path = joinpath(isfile(model_path) ? dirname(model_path) : model_path, "tokenizer.json")
    isfile(path) && return Bop.from_file(path)
    isfile(model_path) && endswith(model_path, ".gguf") && return Bop.from_gguf(model_path)
    error("no tokenizer at $path — pass --tokenizer path/to/tokenizer.json")
end

"""
    serve(args)

Run the server; `args` are the CLI flags: `--model`, `--tokenizer`,
`--host`, `--port`, `--ctx`, `--max-new`.
"""
function serve(args::AbstractVector{<:AbstractString})
    cfg = parse_args(args)
    @info "loading tokenizer…"
    tok = load_tokenizer(cfg.model_path, cfg.tokenizer)
    @info "loading model…" cfg.model_path cfg.ctx
    t = @elapsed model = qwen35(cfg.model_path; Ctx = cfg.ctx, B = 1)
    @info "model loaded" seconds = round(t; digits = 1)

    gen = Generation(1, cfg.max_new)
    session = CaptureSession()

    # Warm before listening: the first generation ever pays the cuTile
    # prefill-kernel compiles (one per shape/divisibility class), the
    # cuBLASLt plan sieve, eager warmup decode steps, and graph capture
    # — ~25 s that has nothing to do with any real prompt. Spend it at
    # boot on synthetic prompts covering the shape classes (odd, /2, /4,
    # /8, /16 — full and suffix prefill both), so the first request pays
    # only its own prefill and replays the graph.
    @info "warming up (kernel compiles, plan sieve, graph capture)…"
    wids(T) = CUDACore.CuMatrix{Int32}(fill(Int32(5), T, 1))
    t = @elapsed begin
        for T in (17, 18, 20, 24, 32)
            Vallmo.reset!(model)
            generate_captured!(model, gen, wids(T); session,
                max_new_tokens = 4, warmup = 3)
        end
        snap = Ref{Any}(nothing)
        Vallmo.reset!(model)
        generate_captured!(model, gen, wids(17); session, max_new_tokens = 4,
            on_prefilled = () -> (snap[] = Vallmo.snapshot(model)))
        for Ts in (19, 20, 22, 26, 31)          # suffix classes, offset 17
            Vallmo.restore!(model, snap[])
            generate_captured!(model, gen, wids(Ts); session,
                max_new_tokens = 4, prefill_offset = 17)
        end
        Vallmo.reset!(model)
        CUDACore.synchronize()
    end
    @info "warm" seconds = round(t; digits = 1)

    jobs = Channel{Job}(16)
    st = (; tok, jobs, cfg.ctx, cfg.max_new, counter = Ref(0),
        model_id = replace(basename(cfg.model_path), r"\.gguf$" => ""),
        created = round(Int, time()))
    errormonitor(@async serve_worker(model, gen, tok, session, jobs))

    @info "serving" cfg.host cfg.port
    # Ctrl+C: threaded SIGINT delivery is unsafe in 1.12 — the
    # InterruptException can land in a thread that's inside the scheduler
    # (task_done_hook of a finished connection task), where no handler
    # exists: "fatal: error thrown and no exception handler available".
    # So on a TTY we take SIGINT out of the picture: cbreak mode reads
    # the ^C byte ourselves for a quiet, graceful shutdown. Off-TTY (or
    # `kill -INT`), exit_on_sigint(true) exits directly from the signal
    # thread — Julia's standard interrupt printout, but never the crash.
    Base.exit_on_sigint(true)
    server = HTTP.serve!(cfg.host, cfg.port; stream = true) do http
        try
            handle(http, st)
        catch err
            @error "handler failed" exception = (err, catch_backtrace())
            try   # best-effort: headers may already be on the wire
                respond_json(http, 500, (; error = (; message = sprint(showerror, err))))
            catch
            end
        end
    end
    stop = Ref(false)
    old = stdin isa Base.TTY ? _cbreak!() : nothing
    old === nothing || errormonitor(@async try
        while !stop[]
            read(stdin, UInt8) == 0x03 || continue
            stop[] = true
        end
    catch                       # stdin closed — SIGINT fallback still works
    end)
    try
        while isopen(server) && !stop[]
            sleep(0.2)
        end
        stop[] && @info "^C — shutting down"
    catch err
        err isa InterruptException || rethrow()
    finally
        old === nothing || _restore_tty!(old)
        close(server)
    end
end

# ── stdin cbreak mode ───────────────────────────────────────────────
# Clear ISIG|ICANON|ECHO on the local-mode flags only: ^C arrives as a
# readable 0x03 instead of SIGINT, bytes come unbuffered, and — unlike
# full raw mode — OPOST stays on so log lines keep their newlines.

mutable struct Termios      # struct termios, linux
    c_iflag::UInt32; c_oflag::UInt32; c_cflag::UInt32; c_lflag::UInt32
    c_line::UInt8
    c_cc::NTuple{32,UInt8}
    c_ispeed::UInt32; c_ospeed::UInt32
    Termios() = new(0, 0, 0, 0, 0, ntuple(_ -> 0x00, 32), 0, 0)
end

function _cbreak!()
    t = Termios()
    ccall(:tcgetattr, Cint, (Cint, Ref{Termios}), 0, t) == 0 || return nothing
    old = deepcopy(t)
    t.c_lflag &= ~UInt32(0x1 | 0x2 | 0x8)       # ISIG | ICANON | ECHO
    cc = collect(t.c_cc)
    cc[6+1], cc[5+1] = 0x01, 0x00               # VMIN = 1, VTIME = 0
    t.c_cc = Tuple(cc)
    ccall(:tcsetattr, Cint, (Cint, Cint, Ref{Termios}), 0, 0, t) == 0 ||
        return nothing
    return old
end

_restore_tty!(old::Termios) =
    ccall(:tcsetattr, Cint, (Cint, Cint, Ref{Termios}), 0, 0, old)
