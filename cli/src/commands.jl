# ═══════════════════════════════════════════════════════════════════════
# Slash commands ── in-chat settings with fuzzy suggestions
#
# Typing `/` in the (single-line) input opens command mode: a popup over
# the transcript lists matching commands with descriptions and current
# values. ↑↓ moves, Tab completes, Enter applies, Esc cancels. Commands
# marked `live` preview as you narrow (theme swaps under your cursor,
# fade dims in place); the pre-`/` state is snapshotted and Esc restores
# it, so browsing is free.
# ═══════════════════════════════════════════════════════════════════════

struct ChatCommand
    name::String
    argspec::String            # "" = takes no argument
    desc::String
    current::Function          # m -> String, shown in the popup
    options::Function          # m -> Vector{String}; empty = free-form arg
    apply!::Function           # (m, arg) -> status String (may throw)
    live::Bool                 # preview the candidate while browsing
end

_vc_theme_names() = String[t.name for t in Tachikoma.ALL_THEMES]

function _vc_apply_theme!(arg::AbstractString)
    for t in Tachikoma.ALL_THEMES
        if t.name == arg
            set_light_mode!(t in Tachikoma.LIGHT_THEMES)
            set_theme!(t)
            return
        end
    end
    error("unknown theme '$arg'")
end

const CHAT_COMMANDS = ChatCommand[
    ChatCommand("fade", "<0.2–1.0>",
        "flower presence — 1.0 full color, lower recedes into the canvas",
        m -> string(m.fade),
        m -> ["1.0", "0.85", "0.75", "0.6", "0.45"],
        (m, arg) -> (m.fade = clamp(parse(Float64, arg), 0.05, 1.0);
                     "fade $(m.fade)"),
        true),
    ChatCommand("theme", "<name>",
        "color theme (light themes switch the canvas polarity too)",
        m -> theme().name,
        m -> _vc_theme_names(),
        (m, arg) -> (_vc_apply_theme!(arg); m.built_n = -1; "theme $(arg)"),
        true),
    ChatCommand("url", "<base_url>",
        "the /v1/chat/completions server to talk to",
        m -> m.base_url,
        m -> String[],
        (m, arg) -> (m.base_url = String(rstrip(arg, '/')); "url $(m.base_url)"),
        false),
    ChatCommand("model", "<id>",
        "model id sent with each request (informational for vallmo serve)",
        m -> m.model_id,
        m -> String[],
        (m, arg) -> (m.model_id = String(arg); "model $(arg)"),
        false),
    ChatCommand("maxtokens", "<n>",
        "completion token cap per request (0 = server default)",
        m -> m.max_tokens == 0 ? "server default" : string(m.max_tokens),
        m -> String[],
        (m, arg) -> (m.max_tokens = max(parse(Int, arg), 0);
                     "maxtokens $(m.max_tokens == 0 ? "server default" : m.max_tokens)"),
        false),
    ChatCommand("clear", "",
        "clear the conversation",
        m -> "$(length(m.messages)) messages",
        m -> String[],
        (m, _) -> (empty!(m.messages); m.built_n = -1; "cleared"),
        false),
]

# Subsequence fuzzy match: nothing on miss, higher score = better.
# Consecutive hits and early positions win; shorter candidates break ties.
function _vc_fuzzy(pat::AbstractString, s::AbstractString)
    isempty(pat) && return -0.05 * length(s)
    sl = lowercase(s)
    score = 0.0
    prev = 0
    pos = 0
    for c in lowercase(pat)
        i = findnext(c, sl, pos + 1)
        i === nothing && return nothing
        score += (i == prev + 1 ? 2.0 : 1.0) - 0.02 * i
        prev = i
        pos = i
    end
    return score - 0.05 * length(sl)
end

_vc_rank(pat, xs) =
    isempty(strip(pat)) ? [(0.0, x) for x in xs] :   # keep authored order
    sort!([(s, x) for x in xs
           for s in (_vc_fuzzy(pat, x),) if s !== nothing];
          by = first, rev = true)

# The input is a command line when it is a single line starting with "/".
_vc_cmdline(m) =
    length(m.input.lines) == 1 && !isempty(m.input.lines[1]) &&
    m.input.lines[1][1] == '/' ? String(m.input.lines[1][2:end]) : nothing

# Command-mode state for one frame: `cmd === nothing` means still picking
# a command (suggestions = commands); otherwise picking/typing its arg
# (suggestions = matching option strings, empty for free-form).
function _vc_cmd_state(m, cmdline::String)
    sp = findfirst(' ', cmdline)
    if sp === nothing
        ranked = _vc_rank(cmdline, [c.name for c in CHAT_COMMANDS])
        return (; cmd = nothing, arg = "",
                suggestions = String[x for (_, x) in ranked])
    end
    name, arg = cmdline[1:sp-1], lstrip(cmdline[sp+1:end])
    i = findfirst(c -> c.name == name, CHAT_COMMANDS)
    i === nothing && return (; cmd = nothing, arg = "", suggestions = String[])
    cmd = CHAT_COMMANDS[i]
    opts = cmd.options(m)
    ranked = _vc_rank(arg, opts)
    return (; cmd, arg = String(arg),
            suggestions = String[x for (_, x) in ranked])
end

# What Enter would apply: the selected option if a list is showing,
# else whatever was typed.
function _vc_candidate(st, sel::Int)
    isempty(st.suggestions) && return st.arg
    return st.suggestions[clamp(sel, 1, length(st.suggestions))]
end

function _vc_cmd_snapshot!(m)
    m.cmd_revert === nothing || return
    m.cmd_revert = (; fade = m.fade, theme = theme(),
                    light = light_mode())
end

function _vc_cmd_restore!(m)
    r = m.cmd_revert
    r === nothing && return
    m.fade = r.fade
    set_light_mode!(r.light)
    set_theme!(r.theme)
    m.built_n = -1
    m.cmd_revert = nothing
end

# Keys consumed in command mode. Returns true when handled.
function _vc_cmd_key!(m, evt::KeyEvent, cmdline::String)
    st = _vc_cmd_state(m, cmdline)
    n = length(st.suggestions)
    if evt.key == :up
        n > 0 && (m.cmd_sel = mod1(m.cmd_sel - 1, n))
        return true
    elseif evt.key == :down
        n > 0 && (m.cmd_sel = mod1(m.cmd_sel + 1, n))
        return true
    elseif evt.key == :tab
        if st.cmd === nothing && n > 0
            pick = st.suggestions[clamp(m.cmd_sel, 1, n)]
            c = CHAT_COMMANDS[findfirst(c -> c.name == pick, CHAT_COMMANDS)]
            set_text!(m.input, "/" * pick * (isempty(c.argspec) ? "" : " "))
        elseif st.cmd !== nothing && n > 0
            set_text!(m.input, "/" * st.cmd.name * " " *
                               st.suggestions[clamp(m.cmd_sel, 1, n)])
        end
        return true
    elseif evt.key == :enter
        if st.cmd === nothing
            # Enter on a command suggestion behaves like Tab (or runs an
            # argument-less command outright).
            n == 0 && return true
            pick = st.suggestions[clamp(m.cmd_sel, 1, n)]
            c = CHAT_COMMANDS[findfirst(c -> c.name == pick, CHAT_COMMANDS)]
            if isempty(c.argspec)
                _vc_cmd_run!(m, c, "")
            else
                set_text!(m.input, "/" * pick * " ")
            end
        else
            _vc_cmd_run!(m, st.cmd, _vc_candidate(st, m.cmd_sel))
        end
        return true
    elseif evt.key == :escape
        _vc_cmd_restore!(m)
        set_text!(m.input, "")
        return true
    end
    return false
end

function _vc_cmd_run!(m, c::ChatCommand, arg::String)
    m.cmd_revert = nothing            # committing: nothing to restore
    m.status = try
        c.apply!(m, arg)
    catch err
        "/$(c.name): $(sprint(showerror, err))"
    end
    set_text!(m.input, "")
    m.cmd_sel = 1
    return
end

# Per-frame preview + popup. Called from `view` when command mode is on.
function _vc_cmd_frame!(m, cmdline::String)
    _vc_cmd_snapshot!(m)
    st = _vc_cmd_state(m, cmdline)
    # Reset the selection only when the *list* changes, not on navigation
    key = join(st.suggestions, '\1')
    key == m.cmd_last || (m.cmd_sel = 1; m.cmd_last = key)
    if st.cmd !== nothing && st.cmd.live
        cand = _vc_candidate(st, m.cmd_sel)
        isempty(cand) || try
            st.cmd.apply!(m, cand)
        catch
        end
    end
    return st
end

function _vc_cmd_popup!(m, buf, above::Rect, st)
    rows = if st.cmd === nothing
        [(c.name * (isempty(c.argspec) ? "" : " " * c.argspec),
          c.desc * "  ·  " * c.current(m), c.name)
         for c in (CHAT_COMMANDS[findfirst(x -> x.name == name, CHAT_COMMANDS)]
                   for name in st.suggestions)]
    elseif !isempty(st.suggestions)
        [(o, "", o) for o in st.suggestions]
    else
        [(st.cmd.name * " " * st.cmd.argspec,
          st.cmd.desc * "  ·  " * st.cmd.current(m), nothing)]
    end
    isempty(rows) && (rows = [("no matching command", "", nothing)])
    h = min(length(rows), 6, above.height)
    top = bottom(above) - h + 1
    base = light_mode() ? SLATE.c200 : SLATE.c800
    selbg = light_mode() ? SLATE.c300 : SLATE.c700
    for (k, y) in zip(1:h, top:bottom(above))
        label, desc, _ = rows[k]
        selected = k == clamp(m.cmd_sel, 1, length(rows))
        bg = selected ? selbg : base
        set_string!(buf, above.x, y, " "^above.width, Style(bg = bg))
        set_string!(buf, above.x + 1, y, "/" * label,
                    Style(fg = tstyle(:accent).fg, bg = bg,
                          bold = selected))
        isempty(desc) ||
            set_string!(buf, above.x + 3 + length(label) + 2, y,
                        first(desc, max(above.width - length(label) - 7, 0)),
                        Style(fg = tstyle(:text_dim).fg, bg = bg))
    end
    return
end
