# server.jl — /v1/chat/completions over the captured decode.
#
#     vallmo serve [--dir MODELS] [--port 8080] [--ctx 4096] [--max-new 2048]
#
# Loaded lazily by VallmoCLI.serve_cmd — `vallmo chat` never pays for
# the CUDA stack.
#
# One model, one GPU worker, one queue: requests serialize through a
# `Channel{Job}` into the worker task that owns the model, so the HTTP
# layer never touches the GPU. The worker holds a single `CaptureSession` —
# warmup and graph capture are paid on the first request, every later
# request replays the same graph from token 1 (same buffers, same
# addresses). Decoding is greedy (lm_head_argmax!); sampling params are
# accepted and ignored. Run single-threaded: the tokenizer is PythonCall-
# backed, and everything here is designed to interleave on one thread.

using Vallmo: Vallmo, qwen35, Generation, generate_captured!, CaptureSession
import HuggingFaceTokenizers as Tokenizers
using CUDACore: CuMatrix
using HTTP, JSON

# ── chat template (Qwen3.5) ──────────────────────────────────────────────────

content_text(c::AbstractString) = String(c)
content_text(parts::AbstractVector) =
    join(p["text"] for p in parts if get(p, "type", "text") == "text")

# History never carries thought: the template strips assistant think blocks.
strip_think(s) = replace(s, r"<think>.*?</think>\s*"s => "")

function render_chat(messages)
    io = IOBuffer()
    for m in messages
        role = m["role"]
        text = content_text(m["content"])
        role == "assistant" && (text = strip_think(text))
        print(io, "<|im_start|>", role, "\n", text, "<|im_end|>\n")
    end
    print(io, "<|im_start|>assistant\n")
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
                  view(ids_h, 1:Tc) == cached
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
            ids = CuMatrix{Int32}(reshape(ids_h[Tc+1:end], :, 1))
            acc = Int32[]
            hit_eos = false
            sent = ""
            emit! = function ()
                full = Tokenizers.decode(tok, acc .- 1)
                safe = endswith(full, '�') ?
                    full[1:prevind(full, lastindex(full))] : full
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
    prompt = render_chat(body["messages"])
    ids = Tokenizers.encode(st.tok, prompt).ids .+ 1     # 0-based → 1-based
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
    cfg = Dict("--dir" => joinpath(pkgdir(Vallmo), "models", "Qwen3.5-4B"),
               "--host" => "127.0.0.1", "--port" => "8080",
               "--ctx" => "4096", "--max-new" => "2048")
    for i in 1:2:length(args)
        haskey(cfg, args[i]) || error("unknown flag $(args[i]); knows $(keys(cfg))")
        cfg[args[i]] = args[i+1]
    end
    return (; dir = abspath(expanduser(cfg["--dir"])), host = cfg["--host"],
        port = parse(Int, cfg["--port"]), ctx = parse(Int, cfg["--ctx"]),
        max_new = parse(Int, cfg["--max-new"]))
end

function _serve_main(args)
    cfg = parse_args(args)
    @info "loading tokenizer…"
    tok = Tokenizers.from_file(Tokenizers.Tokenizer, joinpath(cfg.dir, "tokenizer.json"))
    @info "loading model…" cfg.dir cfg.ctx
    t = @elapsed model = qwen35(cfg.dir; Ctx = cfg.ctx, B = 1)
    @info "model loaded" seconds = round(t; digits = 1)

    gen = Generation(1, cfg.max_new)
    jobs = Channel{Job}(16)
    st = (; tok, jobs, cfg.ctx, cfg.max_new, counter = Ref(0),
        model_id = basename(cfg.dir), created = round(Int, time()))
    errormonitor(@async serve_worker(model, gen, tok, CaptureSession(), jobs))

    @info "serving" cfg.host cfg.port
    HTTP.serve(cfg.host, cfg.port; stream = true) do http
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
end

