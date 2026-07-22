# ═══════════════════════════════════════════════════════════════════════
# Vallmo Chat ── the poppy as the generation indicator for a real model
#
# A chat client for Vallmo's /v1/chat/completions server (Qwen3.5 on
# the jool stack; vallmo is Swedish for poppy).  The flower IS the
# state: a closed green bud while idle, blooming open and spinning
# while the model streams tokens, folding shut when the answer ends.
# The transcript covers the screen with the flower behind it (faded to
# taste — the `fade` knob); glyphs over flower cells take the cell's
# tint as their background, so the bloom glows through the words.
#
#     vallmo chat [--url http://127.0.0.1:8080]
#
# Streaming runs on a spawned task that feeds SSE deltas through a
# Channel; the view drains it each frame, so the UI thread never
# blocks on the network.  <think> blocks render dim.
# ═══════════════════════════════════════════════════════════════════════

@kwdef mutable struct VallmoChatModel <: Model
    quit::Bool = false
    tick::Int = 0
    truecolor::Bool = true
    base_url::String = "http://127.0.0.1:8080"
    model_id::String = "vallmo"
    messages::Vector{Tuple{Symbol,String}} = Tuple{Symbol,String}[]
    live::String = ""                    # assistant text mid-stream
    streaming::Bool = false
    status::String = ""
    deltas::Channel{Any} = Channel{Any}(1024)
    cancel::Union{CancelToken,Nothing} = nothing
    input::TextArea = TextArea(text="", label="", focused=true)
    pane::ScrollPane = ScrollPane(Vector{Span}[]; following=true)
    open_until::Int = 0                  # "actively streaming" until this tick
    opened::Bool = false                 # bud has opened; only clear closes it
    spin::Float64 = 1/4                  # eased speed factor: 1 streaming, 1/4 at rest
    fade::Float64 = 0.0                  # flower presence: 1.0 = full, 0.0 = hidden
    built::Vector{Vector{Span}} = Vector{Span}[]   # finished messages, memoized
    built_n::Int = 0                     # messages count the memo covers
    built_w::Int = 0                     # width the memo was wrapped for
    bloom::Float64 = 0.0
    phase::Float64 = 0.0
    # line-granular mouse selection over the transcript
    selecting::Bool = false
    sel_anchor::Int = 0                  # buffer row where the drag started (0 = none)
    sel_head::Int = 0
    shown::Vector{Vector{Span}} = Vector{Span}[]   # content as last rendered
    shown_off::Int = 0                   # pane offset at last render
    pane_rect::Rect = Rect()
    smooth::Matrix{ColorRGBA} = Matrix{ColorRGBA}(undef, 0, 0)  # per-cell color EMA
    max_tokens::Int = 0                  # 0 = let the server default
    show_think::Bool = false             # false: replace think text with a spinner line
    think_enabled::Bool = true           # false: server prefills an empty think block
    show_help::Bool = false              # /help panel; any key dismisses
    # prompt history (this session): ↑↓ prefix-search over past inputs
    history::Vector{String} = String[]
    hist_idx::Int = 0                    # index being shown (0 = not walking)
    hist_prefix::String = ""             # prefix captured when the walk began
    hist_draft::String = ""              # in-progress draft, restored on ↓ past newest
    # slash-command mode (see commands.jl)
    cmd_sel::Int = 1
    cmd_last::String = ""
    cmd_revert::Union{Nothing,NamedTuple} = nothing
end

should_quit(m::VallmoChatModel) = m.quit

function init!(m::VallmoChatModel, ::Terminal)
    ct = lowercase(get(ENV, "COLORTERM", ""))
    m.truecolor = occursin("truecolor", ct) || occursin("24bit", ct)
    Tachikoma.enable_markdown()
end

function update!(m::VallmoChatModel, evt::KeyEvent)
    if m.show_help                       # any key closes the help panel
        m.show_help = false
        return
    end
    # Any key but ↑↓ ends a history walk; the next ↑ re-captures the prefix
    evt.key in (:up, :down) || (m.hist_idx = 0)
    # Command mode owns navigation/confirm/cancel while the input is "/…"
    if evt.key != :ctrl_c
        cmdline = _vc_cmdline(m)
        if cmdline !== nothing && evt.key in (:up, :down, :tab, :enter, :escape)
            _vc_cmd_key!(m, evt, cmdline) && return
        end
    end
    if evt.key == :ctrl_c
        # Mid-stream: abort the exchange and reclaim the prompt as a
        # draft (the partial reply is dropped). Idle: quit as usual.
        if m.streaming
            _vc_cancel!(m)
            if !isempty(m.messages) && m.messages[end][1] === :user
                set_text!(m.input, m.messages[end][2])
                pop!(m.messages)
            end
        else
            m.quit = true
        end
    elseif evt.key == :escape
        if m.selecting || m.sel_anchor != 0
            m.selecting = false
            m.sel_anchor = m.sel_head = 0
        elseif m.streaming
            m.cancel !== nothing && cancel!(m.cancel)   # keep the partial
        else
            m.quit = true
        end
    elseif evt.key == :enter
        # A line ending in \ continues on the next line; otherwise send.
        li = m.input.lines[m.input.cursor_row]
        if !isempty(li) && li[end] == '\\'
            pop!(li)
            insert!(m.input.lines, m.input.cursor_row + 1, Char[])
            m.input.cursor_row += 1
            m.input.cursor_col = 0
        else
            m.streaming || _vc_submit!(m)
        end
    elseif evt.key == :ctrl && evt.char == 'd'
        # Shell convention: EOF quits on an empty draft, deletes forward
        # in a non-empty one.
        isempty(strip(text(m.input))) ? (m.quit = true) :
                                        handle_key!(m.input, KeyEvent(:delete))
    elseif evt.key == :ctrl && evt.char == 'w'
        _vc_delete_word!(m.input)                # readline: also what most
    elseif evt.key == :ctrl && evt.char == 'u'   # macOS terminals send for
        _vc_delete_bol!(m.input)                 # opt-/cmd-backspace
    elseif evt.key == :alt_backspace
        _vc_delete_word!(m.input)
    elseif evt.key in (:alt_left, :ctrl_left) || (evt.key == :alt && evt.char == 'b')
        _vc_word_left!(m.input)                  # opt+← arrives as CSI-mod,
    elseif evt.key in (:alt_right, :ctrl_right) || (evt.key == :alt && evt.char == 'f')
        _vc_word_right!(m.input)                 # kitty mod, or readline M-b/M-f
    elseif evt.key == :super_left                # cmd+← (kitty terminals; VS Code
        handle_key!(m.input, KeyEvent(:home))    # sends Home/End directly, which
    elseif evt.key == :super_right               # falls through to the input)
        handle_key!(m.input, KeyEvent(:end_key))
    elseif evt.key in (:pageup, :pagedown)
        handle_key!(m.pane, evt)
    elseif evt.key in (:up, :down)
        # a multi-line draft owns ↑↓ for cursor movement; on a single line
        # they walk prompt history (prefix search), scrolling as fallback
        if length(m.input.lines) > 1
            handle_key!(m.input, evt)
        else
            _vc_history!(m, evt.key == :up ? -1 : 1) || handle_key!(m.pane, evt)
        end
    elseif evt.key == :ctrl && evt.char == 'l'
        _vc_clear!(m)
    else
        handle_key!(m.input, evt)
    end
end

# Left-drag over the transcript selects whole lines; release copies them
# (OSC 52 — reaches the *local* clipboard through ssh and vscode-remote).
# The scrollbar column and the wheel still belong to the pane. Native
# terminal selection also works: hold Shift to bypass mouse reporting.
function update!(m::VallmoChatModel, evt::MouseEvent)
    r = m.pane_rect
    if evt.button == mouse_left && r.width > 0 &&
       Base.contains(r, evt.x, evt.y) && evt.x < right(r)
        if evt.action == mouse_press
            m.selecting = true
            m.sel_anchor = m.sel_head = evt.y
            return
        elseif evt.action == mouse_drag && m.selecting
            m.sel_head = clamp(evt.y, r.y, bottom(r))
            return
        elseif evt.action == mouse_release && m.selecting
            m.selecting = false
            _vc_copy_selection!(m)
            return
        end
    end
    handle_mouse!(m.pane, evt)
end

function _vc_copy_selection!(m::VallmoChatModel)
    m.sel_anchor == 0 && return
    lo, hi = minmax(m.sel_anchor, m.sel_head)
    idxs = filter(i -> 1 <= i <= length(m.shown),
                  [m.shown_off + (y - m.pane_rect.y) + 1 for y in lo:hi])
    m.sel_anchor = m.sel_head = 0
    isempty(idxs) && return
    txt = join((rstrip(join(sp.content for sp in m.shown[i])) for i in idxs), '\n')
    _vc_copy_text(txt)
    m.status = "copied $(length(idxs)) line$(length(idxs) == 1 ? "" : "s")"
    return
end

# Every clipboard road at once: OSC 52 (ST-terminated; wrapped for tmux
# passthrough) reaches the terminal's clipboard through ssh and
# vscode-remote; clipboard_copy! (xclip/pbcopy) covers a local session
# where the terminal ignores OSC 52.
function _vc_copy_text(txt::AbstractString)
    seq = "\e]52;c;" * base64encode(txt) * "\e\\"
    haskey(ENV, "TMUX") &&
        (seq = "\ePtmux;" * replace(seq, "\e" => "\e\e") * "\e\\")
    print(stdout, seq)
    flush(stdout)
    clipboard_copy!(String(txt))
    return
end

# Readline-style motions TextArea doesn't have natively.
function _vc_word_left!(ta::TextArea)
    c = ta.cursor_col
    c == 0 && return handle_key!(ta, KeyEvent(:left))   # wrap to prev line end
    line = ta.lines[ta.cursor_row]
    i = c
    while i > 0 && line[i] == ' '
        i -= 1
    end
    while i > 0 && line[i] != ' '
        i -= 1
    end
    ta.cursor_col = i
    return true
end

function _vc_word_right!(ta::TextArea)
    line = ta.lines[ta.cursor_row]
    n = length(line)
    c = ta.cursor_col
    c >= n && return handle_key!(ta, KeyEvent(:right))  # wrap to next line
    i = c + 1
    while i <= n && line[i] == ' '
        i += 1
    end
    while i <= n && line[i] != ' '
        i += 1
    end
    ta.cursor_col = i - 1
    return true
end

# Readline-style edits TextArea doesn't have natively.
function _vc_delete_word!(ta::TextArea)
    c = ta.cursor_col
    c == 0 && return handle_key!(ta, KeyEvent(:backspace))
    line = ta.lines[ta.cursor_row]
    i = c
    while i > 0 && line[i] == ' '
        i -= 1
    end
    while i > 0 && line[i] != ' '
        i -= 1
    end
    deleteat!(line, i+1:c)
    ta.cursor_col = i
    return true
end

function _vc_delete_bol!(ta::TextArea)
    c = ta.cursor_col
    c == 0 && return handle_key!(ta, KeyEvent(:backspace))
    deleteat!(ta.lines[ta.cursor_row], 1:c)
    ta.cursor_col = 0
    return true
end

# Shell-style prompt history over this session's submitted messages.
# ↑ recalls the most recent entry starting with what's typed ("he" ↑ →
# "hey"); repeated ↑ walks older matches, ↓ walks back and past the
# newest match restores the stashed draft. Any other key ends the walk
# (update! resets hist_idx), so the next ↑ captures a fresh prefix.
# Returns false when there is nothing to recall — ↑↓ then fall through
# to transcript scrolling.
function _vc_history!(m::VallmoChatModel, dir::Int)
    if m.hist_idx == 0
        dir > 0 && return false                  # ↓ outside a walk
        m.hist_draft = text(m.input)
        m.hist_prefix = m.hist_draft
        m.hist_idx = length(m.history) + 1
    end
    cur = text(m.input)
    i = m.hist_idx + dir
    while 1 <= i <= length(m.history)
        h = m.history[i]
        startswith(h, m.hist_prefix) && h != cur && break
        i += dir
    end
    if i < 1                                     # already at the oldest match
        m.hist_idx == length(m.history) + 1 && (m.hist_idx = 0)
        return m.hist_idx != 0
    elseif i > length(m.history)                 # ↓ past the newest → draft
        set_text!(m.input, m.hist_draft)
        m.hist_idx = 0
    else
        set_text!(m.input, m.history[i])
        m.hist_idx = i
    end
    return true
end

# ── Streaming ───────────────────────────────────────────────────────

# Sent as the first message of every request; never shown in the
# transcript. A constant prefix, so the server's prefix cache keeps
# paying off across turns.
const SYSTEM_PROMPT = """
    You are Vallmo (Swedish word meaning 'Poppy'), a \
    helpful poppy. A detailed poppy rendering of you \
    is displayed in the background of the terminal chat interface, \
    blooming as the conversation begins. \
    Since the visual representation is already present, \
    please avoid the use of emojis. \
    Be yourself: helpful first, flower second."""

function _vc_submit!(m::VallmoChatModel)
    txt = strip(text(m.input))
    (isempty(txt) || startswith(txt, "/")) && return
    (isempty(m.history) || m.history[end] != txt) && push!(m.history, String(txt))
    m.hist_idx = 0
    push!(m.messages, (:user, String(txt)))
    set_text!(m.input, "")
    m.live = ""
    m.status = ""
    m.streaming = true
    m.pane.following = true
    m.cancel = CancelToken()
    history = vcat(Dict("role" => "system", "content" => SYSTEM_PROMPT),
                   [Dict("role" => String(r), "content" => c) for (r, c) in m.messages])
    req = Dict{String,Any}("model" => m.model_id, "messages" => history,
                           "stream" => true,
                           "enable_thinking" => m.think_enabled)
    m.max_tokens > 0 && (req["max_tokens"] = m.max_tokens)
    body = JSON.json(req)
    url, ch, tok = m.base_url * "/v1/chat/completions", m.deltas, m.cancel
    Threads.@spawn _vc_stream(url, body, ch, tok)
end

function _vc_stream(url::String, body::String, ch::Channel, tok::CancelToken)
    try
        # retry=false: a retried POST would double-submit the generation
        HTTP.open("POST", url, ["Content-Type" => "application/json"];
                  retry=false) do io
            write(io, body)
            HTTP.closewrite(io)
            HTTP.startread(io)
            # Cancellation = socket death, by design. The read loop blocks
            # in eof/readavailable, so it can't poll the token; and any
            # break/throw out of this block has HTTP.jl drain the rest of
            # the stream to recycle the connection — the server never sees
            # a disconnect and generates to max_new. Instead a watcher
            # closes the connection under the read: the loop dies on the
            # read error (caught below as a clean cancel) and the server's
            # next SSE write fails, aborting generation within a token.
            alive = Ref(true)
            watcher = @async begin
                while alive[] && !is_cancelled(tok)
                    sleep(0.05)
                end
                is_cancelled(tok) && alive[] && close(io.stream)
            end
            # Chunked line assembly: readline-per-byte on an HTTP.Stream
            # is slow, and SSE events may split across chunks.
            pending = ""
            done = false
            try
            while !done && !eof(io)
                pending *= String(readavailable(io))
                while (nl = findfirst('\n', pending)) !== nothing
                    line = rstrip(pending[1:nl-1])
                    pending = pending[nl+1:end]
                    startswith(line, "data: ") || continue
                    payload = SubString(line, 7)
                    payload == "[DONE]" && (done = true; break)
                    obj = JSON.parse(String(payload))
                    if haskey(obj, "error")
                        put!(ch, (:error, string(get(obj["error"], "message", obj["error"]))))
                        done = true
                        break
                    end
                    for c in obj["choices"]
                        d = get(c, "delta", nothing)
                        d === nothing && continue
                        s = get(d, "content", nothing)
                        s isa AbstractString && !isempty(s) && put!(ch, (:delta, String(s)))
                    end
                end
            end
            finally
                alive[] = false           # normal end: watcher stands down
            end
        end
        put!(ch, (:done, nothing))
    catch err
        # The watcher's socket close surfaces here as a read/stream error;
        # with the token cancelled that's a clean end, not a failure.
        is_cancelled(tok) ? put!(ch, (:done, nothing)) :
                            put!(ch, (:error, sprint(showerror, err)))
    end
end

# Stop a live stream, dropping the partial. Late events from the dying
# task are ignored by the drain guards below (streaming is already off).
function _vc_cancel!(m::VallmoChatModel)
    m.streaming || return
    m.cancel === nothing || cancel!(m.cancel)
    m.streaming = false
    m.live = ""
end

# Clear the whole conversation — including one still streaming.
function _vc_clear!(m::VallmoChatModel)
    _vc_cancel!(m)
    empty!(m.messages)
    m.status = ""
    m.built_n = -1
    m.opened = false
end

function _vc_drain!(m::VallmoChatModel)
    while isready(m.deltas)
        kind, payload = take!(m.deltas)
        m.streaming || continue        # stray events from a cancelled stream
        if kind === :delta
            m.live *= payload
            m.open_until = m.tick + 90   # tokens flowing: open, linger 3 s
        elseif kind === :done
            isempty(m.live) || push!(m.messages, (:assistant, m.live))
            m.live = ""
            m.streaming = false
        elseif kind === :error
            m.status = "⚠ " * payload
            isempty(m.live) || push!(m.messages, (:assistant, m.live))
            m.live = ""
            m.streaming = false
        end
    end
end

# ── Transcript shaping ──────────────────────────────────────────────

# Assistant text → (text, kind) segments; think-blocks render dim.
# An unclosed <think> (mid-stream) dims everything after it.
function _vc_segments(s::AbstractString)
    segs = Tuple{String,Symbol}[]
    rest = s
    while true
        i = findfirst("<think>", rest)
        if i === nothing
            isempty(rest) || push!(segs, (String(rest), :plain))
            return segs
        end
        pre = rest[1:prevind(rest, first(i))]
        isempty(strip(pre)) || push!(segs, (String(pre), :plain))
        rest = rest[nextind(rest, last(i)):end]
        j = findfirst("</think>", rest)
        if j === nothing
            push!(segs, (String(rest), :think_open))   # still streaming
            return segs
        end
        push!(segs, (String(rest[1:prevind(rest, first(j))]), :think))
        rest = lstrip(rest[nextind(rest, last(j)):end])
    end
end

# Greedy word wrap of one paragraph (no newlines) to `w` columns.
function _vc_wrap(par::AbstractString, w::Int)
    w = max(w, 8)
    out = String[]
    line = ""
    for word in split(par, ' '; keepempty=false)
        while length(word) > w                     # pathological long token
            room = w - length(line) - (isempty(line) ? 0 : 1)
            if room >= 8
                cut = min(room, length(word))
                line *= (isempty(line) ? "" : " ") * first(word, cut)
                word = word[nextind(word, 0, cut + 1):end]
            end
            push!(out, line)
            line = ""
        end
        if isempty(line)
            line = String(word)
        elseif length(line) + 1 + length(word) <= w
            line *= " " * word
        else
            push!(out, line)
            line = String(word)
        end
    end
    (isempty(out) || !isempty(line)) && push!(out, line)
    return out
end

# Role labels: the poppy's petal red (poppy_render.jl), desaturated to
# sit comfortably next to prose, and its purple counterpart (same value
# and saturation, hue swung to violet). Quantized per-terminal through
# _poppy_term_color so 256-color mode gets cube colors, not raw RGB.
const VALLMO_RED    = ColorRGBA(0xf2, 0x6d, 0x6f, 0xff)
const VALLMO_PURPLE = ColorRGBA(0xb0, 0x6d, 0xf2, 0xff)

# Prefill "Working…" spinner: grows and folds back, ping-pong with the
# ends held one extra frame:  · ✢ ✶ ✻ ✽ ✽ ✻ ✶ ✢ ·
const WORKING_FRAMES = ('·', '✢', '✶', '✻', '✽')

# One message → span lines. User text and think-blocks are wrapped plain
# (bold / dim-italic); assistant prose renders as markdown. With
# show_think off, think text collapses — a mid-stream (unclosed) block
# shows a "Thinking..." spinner line instead (tick drives the spinner).
function _vc_message_lines!(lines::Vector{Vector{Span}}, role::Symbol,
                            s::AbstractString, w::Int;
                            show_think::Bool=true, tick::Int=0,
                            truecolor::Bool=true)
    isempty(lines) || push!(lines, Span[])
    label = role === :user ? VALLMO_PURPLE : VALLMO_RED
    push!(lines, [Span(role === :user ? "── You" : "── Vallmo",
                       Style(fg=_poppy_term_color(label, truecolor), bold=true))])
    plain(seg, style) = for par in split(seg, '\n')
        if isempty(strip(par))
            push!(lines, Span[])
        else
            for l in _vc_wrap(par, w)
                push!(lines, [Span(l, style)])
            end
        end
    end
    if role === :user
        plain(s, tstyle(:text, bold=true))
    else
        for (seg, kind) in _vc_segments(s)
            if kind === :think || kind === :think_open
                if show_think
                    seg = strip(seg)               # empty think → no lines
                    isempty(seg) || plain(seg, tstyle(:text_dim, italic=true))
                elseif kind === :think_open
                    si = mod1(tick ÷ 3, length(SPINNER_BRAILLE))
                    push!(lines, [Span("$(SPINNER_BRAILLE[si]) Thinking...",
                                       tstyle(:text_dim, italic=true))])
                end
            else
                append!(lines, markdown_to_spans(seg, w; text_style=tstyle(:text),
                                                 emph_style=tstyle(:text, italic=true),
                                                 code_style=Style(fg=Tachikoma.GREEN.c400)))
            end
        end
    end
    return lines
end

# ── View ────────────────────────────────────────────────────────────

function view(m::VallmoChatModel, f::Frame)
    m.tick += 1
    _vc_drain!(m)
    buf = f.buffer
    area = f.area
    (area.width < 24 || area.height < 8) && return

    # Slash-command mode: preview live settings; restore if "/" was erased
    cmdline = _vc_cmdline(m)
    cmd_st = nothing
    if cmdline === nothing
        m.cmd_revert === nothing || _vc_cmd_restore!(m)
        m.cmd_last = ""
    else
        cmd_st = _vc_cmd_frame!(m, cmdline)
    end

    # The generation indicator: the bud opens as tokens actually arrive
    # (not on submit — prefill wait keeps it closed) and then stays open —
    # the chat is still open to talk; only a cleared conversation closes
    # it. Streaming shows in the spin instead: full speed while tokens
    # flow, easing to a quarter of it at rest. `open_until` (refreshed by
    # every delta) covers gaps between token bursts mid-stream; the
    # `streaming` flag ends `active` the moment the stream finishes,
    # without waiting out that grace window.
    active = m.streaming && m.tick < m.open_until
    active && (m.opened = true)
    m.bloom += ((m.opened ? 1.0 : 0.0) - m.bloom) * 0.015
    m.spin += ((active ? 1.0 : 1/4) - m.spin) * 0.08
    m.phase += 0.018 * (0.2 + 0.8 * m.bloom) * m.spin

    input_h = clamp(length(m.input.lines), 1, 5)
    rows = split_layout(Layout(Vertical, [Fill(), Fixed(1),
                                          Fixed(input_h)]), area)
    length(rows) < 3 && return
    main, seprow, inputrow = rows

    # The transcript covers the whole main area — text flows from the
    # left like poppy_chat's poem, only the scrollbar rides the far right
    # edge — and the flower sits behind everything, center-right where
    # ragged line ends leave it room. Text glyphs over flower cells take
    # the cell's tint as their background (the glass pass below), so the
    # bloom glows through the words instead of being punched out by them.
    transcript = main
    tints = nothing
    if area.width >= 64 && m.fade > 0
        pixels = fill(POPPY_BG, main.height * 4, main.width * 2)
        zbuf = _draw_poppy!(pixels, m.tick, m.phase, 0.35;
                            cx_frac=0.60, cy_frac=0.50, roll=0.12,
                            ppu_frac=0.62, bloom=m.bloom)
        # :cover, both polarities: :full paints bg only on all-8-dot cells,
        # and spin/flutter makes edge cells oscillate across that threshold
        # — visible as flicker, worse when faded. :cover backs every braille
        # cell with a coverage-faded tint; no boundary to cross.
        if size(m.smooth) != (main.height, main.width)
            m.smooth = fill(POPPY_BG, main.height, main.width)
        end
        tints = _render_poppy_braille!(pixels, zbuf, main, buf, m.truecolor;
                                       bg=:cover, fade=m.fade, smooth=m.smooth)
    end

    # Transcript: a ScrollPane of styled span lines. Finished messages are
    # memoized per (count, width); the live message rebuilds every frame
    # (its markdown re-parses as it grows — a few KB, cheap at 30 fps).
    pane_area = Rect(transcript.x + 2, transcript.y,
                     transcript.width - 2, transcript.height)
    wrapw = max(pane_area.width - 2, 12)    # room for the scrollbar column
    if m.built_n != length(m.messages) || m.built_w != wrapw
        m.built = Vector{Span}[]
        for (role, s) in m.messages
            _vc_message_lines!(m.built, role, s, wrapw; show_think=m.show_think,
                               truecolor=m.truecolor)
        end
        m.built_n, m.built_w = length(m.messages), wrapw
    end
    content = m.built
    # Nothing shows until real tokens arrive — the reply (red label and
    # all) appears the moment the flower opens; the prefill wait is noted
    # on the separator instead. The blinking tail cursor is gray while
    # inside a think block, plain text color once prose is streaming.
    if m.streaming && !isempty(m.live)
        content = copy(m.built)
        _vc_message_lines!(content, :assistant, m.live, wrapw;
                           show_think=m.show_think, tick=m.tick,
                           truecolor=m.truecolor)
        if (m.tick ÷ 8) % 2 == 0            # blinking cursor on the tail line
            thinking = last(_vc_segments(m.live))[2] === :think_open
            content[end] = vcat(content[end],
                                [Span(isempty(content[end]) ? "▌" : " ▌",
                                      thinking ? tstyle(:text_dim) : tstyle(:text))])
        end
    end
    set_content!(m.pane, content)
    m.pane.tick = m.tick
    render(m.pane, pane_area, buf)
    m.pane_rect = pane_area           # for mouse selection hit testing
    m.shown = content
    m.shown_off = m.pane.offset       # finalized by render (auto-follow)

    # Separator (status rides its right end), then the input
    set_string!(buf, seprow.x, seprow.y, "─"^seprow.width,
                tstyle(:text_dim, dim=true))
    prefill = m.streaming && isempty(m.live) && isempty(m.status)
    stat = !isempty(m.status) ? m.status : prefill ? "Working…" : ""
    if !isempty(stat)
        # each element padded a space on both sides for room from the ──
        s = " " * first(stat, max(seprow.width - 12, 8)) * " "
        if prefill
            n = length(WORKING_FRAMES)
            k = mod(m.tick ÷ 3, 2n)
            s = " $(WORKING_FRAMES[k < n ? k + 1 : 2n - k]) " * s
        end
        set_string!(buf, right(seprow) - length(s) - 2, seprow.y, s,
                    startswith(stat, "⚠") ? tstyle(:error) : tstyle(:text_dim))
    end
    set_string!(buf, inputrow.x + 1, inputrow.y, "❯", tstyle(:accent, bold=true))
    m.input.tick = m.tick
    render(m.input, Rect(inputrow.x + 3, inputrow.y,
                         inputrow.width - 3, inputrow.height), buf)

    # Command popup, riding just above the separator
    cmd_st === nothing ||
        _vc_cmd_popup!(m, buf, Rect(main.x + 1, main.y, main.width - 2,
                                    main.height), cmd_st)

    m.show_help && _vc_help_panel!(buf, area)

    # Selection highlight: whole lines between anchor and head.
    if m.sel_anchor != 0
        selbg = light_mode() ? SLATE.c300 : SLATE.c600
        lo, hi = minmax(m.sel_anchor, m.sel_head)
        for y in max(lo, pane_area.y):min(hi, bottom(pane_area))
            for x in pane_area.x:right(pane_area)-1
                i = Tachikoma.buf_index(buf, x, y)
                c = buf.content[i]
                s = c.style
                buf.content[i] = Tachikoma.Cell(c.char,
                    Style(fg=s.fg, bg=selbg, bold=s.bold, dim=s.dim,
                          italic=s.italic), c.suffix)
            end
        end
    end

    # Canvas + glass, one final pass. The app owns its background — on a
    # terminal theme whose canvas is lighter than the tints (gray VS Code),
    # the flower otherwise reads as darker-than-background holes. Cells
    # with an explicit bg (braille :cover tints, input cursor, markdown
    # code, selection) keep it; glyphs over flower cells take the cell's
    # tint (glass); everything else gets the canvas color.
    canvas = m.truecolor ?
        (light_mode() ? ColorRGB(0xff, 0xff, 0xff) : ColorRGB(0x00, 0x00, 0x00)) :
        (light_mode() ? Color256(231) : Color256(16))
    for y in area.y:bottom(area), x in area.x:right(area)
        i = Tachikoma.buf_index(buf, x, y)
        cell = buf.content[i]
        cell.style.bg isa Tachikoma.NoColor || continue
        bgc = canvas
        if tints !== nothing && Base.contains(main, x, y) &&
           !(0x2800 <= UInt32(cell.char) <= 0x28ff)
            t = tints[y - main.y + 1, x - main.x + 1]
            t == POPPY_BG || (bgc = _poppy_term_color(t, m.truecolor))
        end
        s = cell.style
        buf.content[i] = Tachikoma.Cell(cell.char,
            Style(fg=s.fg, bg=bgc, bold=s.bold, dim=s.dim, italic=s.italic,
                  underline=s.underline, strikethrough=s.strikethrough,
                  hyperlink=s.hyperlink), cell.suffix)
    end
end

# ── Persistent settings ─────────────────────────────────────────────
# The file holds only options the user has explicitly set — a committed
# slash command merges just its own key(s) in. Anything absent keeps
# following the code default, so defaults can change under users who
# never overrode them. chat() loads at startup (kwargs/CLI flags win).

const VALLMO_SETTINGS = joinpath(homedir(), ".vallmo", "settings.json")

_vc_setting(m::VallmoChatModel, k::String) =
    k == "fade"             ? m.fade :
    k == "theme"            ? theme().name :
    k == "url"              ? m.base_url :
    k == "model"            ? m.model_id :
    k == "maxtokens"        ? m.max_tokens :
    k == "thinking_show"    ? m.show_think :
    k == "thinking_enabled" ? m.think_enabled : nothing

function _vc_save_settings(m::VallmoChatModel, keys)
    isempty(keys) && return
    d = try
        isfile(VALLMO_SETTINGS) ? JSON.parse(read(VALLMO_SETTINGS, String)) : Dict()
    catch
        Dict()
    end
    d isa AbstractDict || (d = Dict())
    for k in keys
        d[k] = _vc_setting(m, k)
    end
    try
        mkpath(dirname(VALLMO_SETTINGS))
        write(VALLMO_SETTINGS, JSON.json(d, 2))
    catch
    end
    return
end

function _vc_load_settings!(m::VallmoChatModel)
    d = try
        isfile(VALLMO_SETTINGS) ? JSON.parse(read(VALLMO_SETTINGS, String)) : nothing
    catch
        nothing                            # unreadable/corrupt: fresh defaults
    end
    d isa AbstractDict || return
    haskey(d, "fade") && (m.fade = clamp(Float64(d["fade"]), 0.0, 1.0))
    haskey(d, "url") && (m.base_url = String(d["url"]))
    haskey(d, "model") && (m.model_id = String(d["model"]))
    haskey(d, "maxtokens") && (m.max_tokens = max(Int(d["maxtokens"]), 0))
    haskey(d, "thinking_show") && (m.show_think = d["thinking_show"] === true)
    haskey(d, "thinking_enabled") && (m.think_enabled = d["thinking_enabled"] === true)
    haskey(d, "theme") && try _vc_apply_theme!(String(d["theme"])) catch end
    return
end

"""
    chat(; base_url = nothing, fade = nothing, theme_name = nothing)

Chat with a Vallmo `/v1/chat/completions` server; the poppy blooms while
the model generates and closes when idle. `fade` sets the flower's
presence (1.0 = full color, lower recedes it into the canvas, 0.0 hides
it; /fade adjusts at runtime). Options persist in ~/.vallmo/settings.json;
kwargs left as `nothing` fall back to the saved settings, then defaults.
"""
function chat(; base_url=nothing, fade=nothing, theme_name=nothing)
    m = VallmoChatModel()
    _vc_load_settings!(m)
    base_url === nothing || (m.base_url = String(rstrip(base_url, '/')))
    fade === nothing || (m.fade = clamp(Float64(fade), 0.0, 1.0))
    theme_name !== nothing && set_theme!(theme_name)
    app(m; fps=30)
end
