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
    open_until::Int = 0                  # bloom stays open until this tick
    fade::Float64 = 0.75                 # flower presence: 1.0 = full, lower = ambience
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
    # Command mode owns navigation/confirm/cancel while the input is "/…"
    if evt.key != :ctrl_c
        cmdline = _vc_cmdline(m)
        if cmdline !== nothing && evt.key in (:up, :down, :tab, :enter, :escape)
            _vc_cmd_key!(m, evt, cmdline) && return
        end
    end
    if evt.key == :ctrl_c
        m.quit = true
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
        m.streaming || _vc_submit!(m)
    elseif evt.key == :ctrl && evt.char == 'j'
        handle_key!(m.input, KeyEvent(:enter))   # newline inside the input
    elseif evt.key == :ctrl && evt.char == 'w'
        _vc_delete_word!(m.input)                # readline: also what most
    elseif evt.key == :ctrl && evt.char == 'u'   # macOS terminals send for
        _vc_delete_bol!(m.input)                 # opt-/cmd-backspace
    elseif evt.key in (:pageup, :pagedown)
        handle_key!(m.pane, evt)
    elseif evt.key in (:up, :down)
        # a multi-line draft owns ↑↓ for cursor movement; else they scroll
        length(m.input.lines) > 1 ? handle_key!(m.input, evt) :
                                    handle_key!(m.pane, evt)
    elseif evt.key == :ctrl && evt.char == 'l'
        m.streaming || (empty!(m.messages); m.status = ""; m.built_n = -1)
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
    print(stdout, "\e]52;c;", base64encode(txt), "\a")
    flush(stdout)
    m.status = "copied $(length(idxs)) line$(length(idxs) == 1 ? "" : "s")"
    return
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

# ── Streaming ───────────────────────────────────────────────────────

function _vc_submit!(m::VallmoChatModel)
    txt = strip(text(m.input))
    (isempty(txt) || startswith(txt, "/")) && return
    push!(m.messages, (:user, String(txt)))
    set_text!(m.input, "")
    m.live = ""
    m.status = ""
    m.streaming = true
    m.pane.following = true
    m.cancel = CancelToken()
    history = [Dict("role" => String(r), "content" => c) for (r, c) in m.messages]
    req = Dict{String,Any}("model" => m.model_id, "messages" => history,
                           "stream" => true)
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
            # Chunked line assembly: readline-per-byte on an HTTP.Stream
            # is slow, and SSE events may split across chunks.
            pending = ""
            done = false
            while !done && !eof(io)
                is_cancelled(tok) && break
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
        end
        put!(ch, (:done, nothing))
    catch err
        put!(ch, (:error, sprint(showerror, err)))
    end
end

function _vc_drain!(m::VallmoChatModel)
    while isready(m.deltas)
        kind, payload = take!(m.deltas)
        if kind === :delta
            m.live *= payload
            m.open_until = m.tick + 90   # tokens flowing: open, linger 3 s
        elseif kind === :done
            isempty(m.live) || push!(m.messages, (:assistant, m.live))
            m.live = ""
            m.streaming = false
        elseif kind === :error
            m.status = payload
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
            push!(segs, (String(rest), :think))
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

# One message → span lines. User text and think-blocks are wrapped plain
# (bold / dim-italic); assistant prose renders as markdown.
function _vc_message_lines!(lines::Vector{Vector{Span}}, role::Symbol,
                            s::AbstractString, w::Int)
    isempty(lines) || push!(lines, Span[])
    push!(lines, [Span(role === :user ? "── you" : "── vallmo",
                       role === :user ? tstyle(:primary, bold=true) :
                                        tstyle(:accent, bold=true))])
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
            if kind === :think
                plain(seg, tstyle(:text_dim, italic=true))
            else
                append!(lines, markdown_to_spans(seg, w; text_style=tstyle(:text)))
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
    # (not on submit — prefill wait keeps it closed) and folds shut 3 s
    # after the last one. `open_until` is refreshed by every delta.
    m.bloom += ((m.tick < m.open_until ? 1.0 : 0.0) - m.bloom) * 0.015
    m.phase += 0.018 * (0.2 + 0.8 * m.bloom)

    input_h = clamp(length(m.input.lines), 1, 5)
    rows = split_layout(Layout(Vertical, [Fixed(1), Fill(), Fixed(1),
                                          Fixed(input_h), Fixed(1)]), area)
    length(rows) < 5 && return
    header, main, seprow, inputrow, footer = rows

    # The transcript covers the whole main area — text flows from the
    # left like poppy_chat's poem, only the scrollbar rides the far right
    # edge — and the flower sits behind everything, center-right where
    # ragged line ends leave it room. Text glyphs over flower cells take
    # the cell's tint as their background (the glass pass below), so the
    # bloom glows through the words instead of being punched out by them.
    transcript = main
    tints = nothing
    if area.width >= 64
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

    # Header: title + endpoint; spinner while streaming
    si = mod1(m.tick ÷ 3, length(SPINNER_BRAILLE))
    m.streaming && set_char!(buf, header.x, header.y, SPINNER_BRAILLE[si],
                             tstyle(:accent))
    set_string!(buf, header.x + 2, header.y, "vallmo", tstyle(:accent, bold=true))
    hdr = "$(DOT) $(m.base_url)" * (m.truecolor ? "" : " $(DOT) 256color")
    set_string!(buf, header.x + 9, header.y, hdr, tstyle(:text_dim))

    # Transcript: a ScrollPane of styled span lines. Finished messages are
    # memoized per (count, width); the live message rebuilds every frame
    # (its markdown re-parses as it grows — a few KB, cheap at 30 fps).
    pane_area = Rect(transcript.x + 2, transcript.y,
                     transcript.width - 2, transcript.height)
    wrapw = max(pane_area.width - 2, 12)    # room for the scrollbar column
    if m.built_n != length(m.messages) || m.built_w != wrapw
        m.built = Vector{Span}[]
        for (role, s) in m.messages
            _vc_message_lines!(m.built, role, s, wrapw)
        end
        m.built_n, m.built_w = length(m.messages), wrapw
    end
    content = m.built
    if m.streaming
        content = copy(m.built)
        if isempty(m.live)
            push!(content, Span[])
            push!(content, [Span("── vallmo", tstyle(:accent, bold=true))])
        else
            _vc_message_lines!(content, :assistant, m.live, wrapw)
        end
        if (m.tick ÷ 8) % 2 == 0            # blinking cursor on the tail line
            content[end] = vcat(content[end],
                                [Span(isempty(content[end]) ? "▌" : " ▌",
                                      tstyle(:accent))])
        end
    end
    set_content!(m.pane, content)
    m.pane.tick = m.tick
    render(m.pane, pane_area, buf)
    m.pane_rect = pane_area           # for mouse selection hit testing
    m.shown = content
    m.shown_off = m.pane.offset       # finalized by render (auto-follow)

    # Separator, input (❯ prompt + multi-line TextArea), status
    set_string!(buf, seprow.x, seprow.y, "─"^seprow.width,
                tstyle(:text_dim, dim=true))
    set_string!(buf, inputrow.x + 1, inputrow.y, "❯", tstyle(:accent, bold=true))
    m.input.tick = m.tick
    render(m.input, Rect(inputrow.x + 3, inputrow.y,
                         inputrow.width - 3, inputrow.height), buf)
    hints = m.streaming ? " [Esc]stop [PgUp/wheel]scroll [drag]copy [/]settings" :
        " [Enter]send [^J]newline [/]settings [Esc]quit [^L]clear [drag]copy"
    left = isempty(m.status) ? hints : " ⚠ $(m.status)"
    set_string!(buf, footer.x + 1, footer.y, first(left, area.width - 2),
                isempty(m.status) ? tstyle(:text_dim, dim=true) : tstyle(:error))

    # Command popup, riding just above the separator
    cmd_st === nothing ||
        _vc_cmd_popup!(m, buf, Rect(main.x + 1, main.y, main.width - 2,
                                    main.height), cmd_st)

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

"""
    chat(; base_url = "http://127.0.0.1:8080", fade = 0.75,
         theme_name = nothing)

Chat with a Vallmo `/v1/chat/completions` server; the poppy blooms while
the model generates and closes when idle. `fade` sets the flower's
presence (1.0 = full color, lower recedes it into the canvas; ^F cycles
at runtime).
"""
function chat(; base_url::String="http://127.0.0.1:8080",
                     fade::Float64=0.75, theme_name=nothing)
    theme_name !== nothing && set_theme!(theme_name)
    app(VallmoChatModel(; base_url, fade); fps=30)
end
