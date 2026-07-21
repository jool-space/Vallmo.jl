# ═══════════════════════════════════════════════════════════════════════
# The poppy renderer ── vallmo's flower (Swedish for poppy), extracted
# from TachikomaDemos' poppy.jl (which keeps its own copy for the pure
# demos). Parametric petal surfaces as a shaded braille point cloud with
# a per-dot z-buffer; `bloom` morphs bud ↔ open, `fade` recedes the
# flower into the canvas, `smooth` EMAs per-cell colors across frames.
# ═══════════════════════════════════════════════════════════════════════

# ── Point-cloud renderer with z-buffer ──────────────────────────────

struct PoppyCamera
    cs::Float64   # cos/sin of spin (about y)
    ss::Float64
    ct::Float64   # cos/sin of tilt (about x)
    st::Float64
    cro::Float64  # cos/sin of roll (about view axis; +ve leans left)
    sro::Float64
    cx::Float64   # screen center px
    cy::Float64
    ppu::Float64  # pixels per world unit
    dist::Float64 # camera distance
end

# World point → (px, py, z_cam) after spin, tilt, roll, perspective
@inline function _poppy_project(cam::PoppyCamera, x::Float64, y::Float64, z::Float64)
    # Spin about vertical axis
    rx = x * cam.cs + z * cam.ss
    rz = -x * cam.ss + z * cam.cs
    # Tilt toward camera
    ry = y * cam.ct - rz * cam.st
    rz2 = y * cam.st + rz * cam.ct
    # Roll in the view plane
    rx2 = rx * cam.cro - ry * cam.sro
    ry2 = rx * cam.sro + ry * cam.cro
    zc = cam.dist - rz2
    zc < 0.3 && return (-1.0, -1.0, zc)
    persp = cam.dist / zc
    (cam.cx + rx2 * persp * cam.ppu, cam.cy - ry2 * persp * cam.ppu, zc)
end

# Rotate a direction by the same spin (for world-space normals)
@inline function _poppy_spin_dir(cam::PoppyCamera, x::Float64, y::Float64, z::Float64)
    (x * cam.cs + z * cam.ss, y, -x * cam.ss + z * cam.cs)
end

# Transparent sentinel for unlit dots
const POPPY_BG = ColorRGBA(0x00, 0x00, 0x00, 0x00)

@inline function _poppy_splat!(pixels::Matrix{ColorRGBA}, zbuf::Matrix{Float64},
                               px::Float64, py::Float64, zc::Float64,
                               color::ColorRGBA)
    x = round(Int, px)
    y = round(Int, py)
    (x >= 1 && y >= 1 && x <= size(pixels, 2) && y <= size(pixels, 1)) || return
    if zc < zbuf[y, x]
        zbuf[y, x] = zc
        pixels[y, x] = color
    end
    nothing
end

# Saturation-preserving shade: the dominant channel dims linearly,
# the minor channels dim quadratically, so shadows deepen the hue
# instead of drifting toward gray.  (Uniformly-dimmed muted colors
# read as gray — and on non-truecolor terminals they literally
# quantize onto the 256-palette grayscale ramp.)
@inline function _poppy_shade(r::Integer, g::Integer, b::Integer, light::Float64)
    l2 = light * light
    m = max(r, max(g, b))
    ColorRGBA(UInt8(clamp(round(Int, r * (r == m ? light : l2)), 0, 255)),
              UInt8(clamp(round(Int, g * (g == m ? light : l2)), 0, 255)),
              UInt8(clamp(round(Int, b * (b == m ? light : l2)), 0, 255)))
end

# Draws into a braille-dot-resolution grid (2×4 dots per cell), one
# point per dot — the intended look is a colored point cloud.
# `bloom` ∈ [0,1] morphs the flower: 0 = closed pre-bloom bud wrapped
# in green sepals, 1 = fully open.  Petals sweep up/inward and pinch
# closed at the tip as bloom → 0.
function _draw_poppy!(pixels::Matrix{ColorRGBA}, tick::Int, spin::Float64, tilt::Float64;
                      cx_frac::Float64=0.5, cy_frac::Float64=0.42,
                      roll::Float64=0.0, ppu_frac::Float64=0.40,
                      bloom::Float64=1.0)
    ph, pw = size(pixels)
    zbuf = fill(Inf, ph, pw)
    bl = clamp(bloom, 0.0, 1.0)
    be = bl * bl * (3.0 - 2.0 * bl)     # smoothstep ease

    cam = PoppyCamera(cos(spin), sin(spin), cos(tilt), sin(tilt),
                      cos(roll), sin(roll),
                      pw * cx_frac, ph * cy_frac,
                      min(pw, ph) * ppu_frac, 3.2)

    # Fixed light: mostly top-down with a gentle lateral offset.
    # Deliberately no toward-viewer (z) component — that would dim
    # every surface facing the camera and make the near side look
    # desaturated regardless of spin.  The lateral part is kept small:
    # cupped petals' INNER faces point at a side-light from the far
    # side, so a strong lateral light lights one half of the cup.
    lx, ly, lz = -0.22, 0.96, 0.10
    flutter = tick * 0.05

    # Sampling density: ≲1 dot spacing across the widest petal arc
    nu = clamp(max(pw, ph) ÷ 2, 64, 200)
    nv = clamp(max(pw, ph), 96, 280)

    # ── Stem: slightly bowed green cylinder below the flower ──
    n_stem = clamp(ph ÷ 2, 60, 220)
    for i in 0:n_stem
        t = i / n_stem
        sy = 0.04 - 1.77 * t
        bow = 0.10 * sin(t * 2.2)          # gentle S-curve
        for a in 0:9
            ang = a / 10 * 2π
            sxo = bow + 0.028 * cos(ang)
            szo = 0.028 * sin(ang)
            px, py, zc = _poppy_project(cam, sxo, sy, szo)
            px < 0 && continue
            dnx, dny, dnz = _poppy_spin_dir(cam, cos(ang), 0.0, sin(ang))
            lam = max(0.0, dnx * lx + dny * ly + dnz * lz)
            light = (0.60 + 0.40 * lam) * (1.0 - 0.15 * t)
            _poppy_splat!(pixels, zbuf, px, py, zc,
                          _poppy_shade(0x2a, 0x92, 0x30, light))
        end
    end

    # ── Petals: two whorls of 4 broad overlapping petals each ──
    # u ∈ [0,1] root→tip, v ∈ [-1,1] across.  Each petal is evaluated
    # into a position grid first (all pow-heavy terms hoisted to the
    # u/v axes), then splatted with normals from grid differences.
    # Layer 1: 4 wide base petals, surface ~80° from the stem axis.
    # Layer 2: 3 smaller ear-shaped petals at ~50°, offset rotation.
    #   n: petals in the whorl.  ca/cb: cup profile
    #   y = scale·(ca·u + cb·u²) — slope sets the inclination.
    #   halfw: max petal half-angle (rad).  pu/pw: width-profile
    #   exponents — lower pu widens the petal near the root, lower pw
    #   squares off the sides (ear shape).
    # The sepal layer is the bud's wrapper: two wide green husk
    # halves rendered with the same surface machinery.  They exist at
    # every bloom stage — wrapped around the closed petals at bloom=0,
    # hinging open and folding down beneath the flower as it opens.
    layers = (
        (n = 2, off = π / 2,  scale = 0.0,  lift = 0.00, ca = 0.0,  cb = 0.0,
         halfw = 1.72, pu = 0.80, pw = 0.38, sepal = true),
        (n = 4, off = 0.0,    scale = 0.88, lift = 0.00, ca = 0.08, cb = 0.08,
         halfw = 1.05, pu = 0.90, pw = 0.50, sepal = false),
        (n = 3, off = π / 6,  scale = 0.60, lift = 0.03, ca = 0.28, cb = 0.43,
         halfw = 1.25, pu = 0.75, pw = 0.42, sepal = false),
    )
    X = Matrix{Float64}(undef, nu + 1, nv + 1)
    Y = Matrix{Float64}(undef, nu + 1, nv + 1)
    Z = Matrix{Float64}(undef, nu + 1, nv + 1)
    edge_v = [0.03 * abs(2.0 * iv / nv - 1.0)^1.5 for iv in 0:nv]
    # Closed-bud morph targets: steep near-vertical cup, narrower
    # petals, tip pinched toward the axis
    ca_closed, cb_closed = 0.92, 0.26
    for L in layers
        # Fully-enclosed petals are invisible inside the husk; skip
        # them so sparse sepal dot coverage can't leak red at grazing
        # angles.  They start drawing right as the crown splits.
        L.sepal || be > 0.20 || continue
        # Sepals and petals move together across the whole bloom;
        # only the petal delay (below) keeps the bud sealed at the
        # closed end.
        be_s = be
        if L.sepal
            # Wrapper opens in two overlapping stages so each point
            # arcs OUT and then DOWN — not a straight-line morph:
            #   1. unwrap: tip releases from the crown, blade uncurls
            #      radially outward while still steep (making room)
            #   2. droop: the whole blade hinges down at the base
            unwrap = clamp(be_s / 0.50, 0.0, 1.0)
            unwrap = unwrap * unwrap * (3.0 - 2.0 * unwrap)
            droop = clamp((be_s - 0.35) / 0.65, 0.0, 1.0)
            droop = droop * droop * (3.0 - 2.0 * droop)
            fold = 1.25 - 1.88 * droop
            sfold, cfold = sincos(fold)
            # Once folded, the sepals DROP (like real poppies): the
            # blade shrinks away over the last stretch of the bloom,
            # leaving no green skirt on the open flower.
            drop = clamp((be - 0.60) / 0.35, 0.0, 1.0)
            drop = drop * drop * (3.0 - 2.0 * drop)
            drop >= 0.999 && continue
            # Blade length is coupled to the fold angle: full length
            # only while wrapping (steep); contracted while swung out
            # flat, where a full blade would dominate the view.
            eff_scale = 0.60 * (1.0 - drop) * (0.46 + 0.54 * clamp(sfold, 0.0, 1.0))
            pinchL = 0.93 - 0.60 * unwrap  # tip sealed shut when closed
            hw_f = 1.0 - 0.22 * unwrap     # relaxes as it unwraps
        else
            sfold = cfold = 0.0           # unused for petals
            # Petals run on a DELAYED clock relative to the sepals'
            # accelerated one: they finish closing before the husk
            # seals, and only emerge once it has unwrapped — so red
            # is never outside the green in either direction.
            be_p = clamp((be - 0.26) / 0.74, 0.0, 1.0)
            # Ease-in growth: petals stay near bud-size while first
            # emerging and accelerate later, so there's no jump from
            # the diminished state.  Angle swings on a smoothstep.
            size = be_p^1.7
            grow = be_p * be_p * (3.0 - 2.0 * be_p)
            eff_scale = L.scale * (0.34 + 0.66 * size)
            eff_ca = L.ca + (ca_closed - L.ca) * (1.0 - grow)
            eff_cb = L.cb + (cb_closed - L.cb) * (1.0 - grow)
            pinchL = 0.78 * (1.0 - grow)
            hw_f = 0.62 + 0.38 * grow
        end
        for k in 0:(L.n - 1)
            base = k * (2π / L.n) + L.off
            scale = eff_scale
            @inbounds for iu in 0:nu
                u = iu / nu
                # Broad rounded petal: neighbors overlap, closing to a
                # rounded tip
                halfw = L.halfw * hw_f * sin(u^L.pu * π)^L.pw
                if L.sepal
                    # Blade point at arc length u along fold angle,
                    # bowed outward while closed (plump egg silhouette),
                    # straightening as it unwraps
                    bow = 0.20 * sin(u * π) * (1.0 - unwrap)
                    r_u = (0.06 + scale * (u * cfold + bow)) * (1.0 - pinchL * u^1.5)
                    cup = scale * (u * sfold - 0.06 * u * u)
                    ruff = 0.0
                else
                    r_u = scale * (0.10 + 0.92 * u) * (1.0 - pinchL * u^1.5)
                    cup = L.lift + scale * (eff_ca * u + eff_cb * u * u)
                    ruff = 0.05 * scale * u * u * be
                end
                for iv in 0:nv
                    v = 2.0 * iv / nv - 1.0
                    theta = base + v * halfw
                    sth, cth = sincos(theta)
                    # Sepals are boat-shaped shells: the blade's
                    # midline bulges outward across its width
                    r_pt = L.sepal ?
                        r_u + 0.07 * scale * (1.0 - v * v) * sin(clamp(u, 0.0, 1.0) * π) :
                        r_u
                    y = cup + ruff * sin(5.0 * theta + flutter) +
                        scale * edge_v[iv + 1] * (1.0 + 0.4 * sin(3.0 * theta - flutter))
                    X[iu + 1, iv + 1] = r_pt * cth
                    Y[iu + 1, iv + 1] = y
                    Z[iu + 1, iv + 1] = r_pt * sth
                end
            end
            @inbounds for iu in 1:(nu + 1)
                u = (iu - 1) / nu
                iu0 = max(iu - 1, 1)
                iu1 = min(iu + 1, nu + 1)
                for iv in 1:(nv + 1)
                    v = 2.0 * (iv - 1) / nv - 1.0
                    px, py, zc = _poppy_project(cam, X[iu, iv], Y[iu, iv], Z[iu, iv])
                    px < 0 && continue
                    # Normal from grid differences
                    iv0 = max(iv - 1, 1)
                    iv1 = min(iv + 1, nv + 1)
                    dux = X[iu1, iv] - X[iu0, iv]
                    duy = Y[iu1, iv] - Y[iu0, iv]
                    duz = Z[iu1, iv] - Z[iu0, iv]
                    dvx = X[iu, iv1] - X[iu, iv0]
                    dvy = Y[iu, iv1] - Y[iu, iv0]
                    dvz = Z[iu, iv1] - Z[iu, iv0]
                    nx0 = duy * dvz - duz * dvy
                    ny0 = duz * dvx - dux * dvz
                    nz0 = dux * dvy - duy * dvx
                    nlen = sqrt(nx0 * nx0 + ny0 * ny0 + nz0 * nz0)
                    nlen < 1e-12 && (nlen = 1.0; ny0 = 1.0)
                    dnx, dny, dnz = _poppy_spin_dir(cam, nx0 / nlen, ny0 / nlen, nz0 / nlen)
                    lam = abs(dnx * lx + dny * ly + dnz * lz)  # two-sided petals
                    if L.sepal
                        light = min(1.0, 0.62 + 0.42 * lam)
                        # Green husk, faintly streaked, with darkened
                        # rims so the shell reads thick, not papery
                        streak = 0.92 + 0.08 * sin(9.0 * v)
                        rim = 0.72 + 0.28 * (1.0 - abs(v)^3)
                        r = round(Int, 0x3c * rim)
                        g = round(Int, 0x8a * streak * rim)
                        b = round(Int, 0x34 * rim)
                    else
                        # Like real vallmo: darkest at the flower center,
                        # vivid toward the rim.  The gradient is driven
                        # through the light factor (with a flat Lambert
                        # weight) so surface tilt can never invert it.
                        rad = u < 0.16 ? 0.0 : min(1.0, (u - 0.16) / 0.70)
                        light = min(1.0, 0.80 + 0.24 * lam) * (0.52 + 0.48 * rad)
                        if u < 0.16
                            blk = u / 0.16
                            r = round(Int, 0x30 + blk * (0xea - 0x30))
                            g = round(Int, 0x08 + blk * (0x14 - 0x08))
                            b = round(Int, 0x10 + blk * (0x1a - 0x10))
                        else
                            r, g, b = 0xf2, 0x1a, 0x1e
                        end
                    end
                    # Sepals get a slight depth bias so coincident
                    # surfaces resolve green instead of z-fighting
                    _poppy_splat!(pixels, zbuf, px, py, zc - 0.03 * L.sepal,
                                  _poppy_shade(r, g, b, light))
                end
            end
        end
    end

    # ── Seed pod + stamens: only once the crown has split; while
    # enclosed they'd glint gray through gaps in the husk dot cloud ──
    if be > 0.20
    pod_r = 0.08
    n_ring = 28
    for it in 0:n_ring
        phi = it / n_ring * π
        sphi, cphi = sincos(phi)
        for ip in 0:(2 * n_ring)
            ang = ip / (2 * n_ring) * 2π
            sx = pod_r * sphi * cos(ang)
            sz = pod_r * sphi * sin(ang)
            sy = 0.16 + pod_r * cphi * 0.9
            px, py, zc = _poppy_project(cam, sx, sy, sz)
            px < 0 && continue
            dnx, dny, dnz = _poppy_spin_dir(cam, sphi * cos(ang), cphi, sphi * sin(ang))
            lam = max(0.0, dnx * lx + dny * ly + dnz * lz)
            light = 0.55 + 0.45 * lam
            # Ridged green pod — green must clearly dominate r/b or the
            # small dark shape reads (and quantizes) as gray
            ridge = 0.88 + 0.12 * sin(8.0 * ang)
            _poppy_splat!(pixels, zbuf, px, py, zc,
                          _poppy_shade(0x30, 0x84, 0x26, light * ridge))
        end
    end

    # ── Stamens: blackish-brown ring around the green pod center ──
    n_stamen = 36
    for si in 0:(n_stamen - 1)
        ang = si / n_stamen * 2π + 0.07 * sin(flutter + si)
        for j in 0:8
            t = j / 8
            sr = 0.08 + t * 0.10
            sy = 0.15 + 0.08 * t
            sx = sr * cos(ang)
            sz = sr * sin(ang)
            px, py, zc = _poppy_project(cam, sx, sy, sz)
            px < 0 && continue
            color = j >= 7 ? ColorRGBA(0x1c, 0x10, 0x06) :     # anther tip
                             ColorRGBA(0x30, 0x1e, 0x0e)       # filament
            _poppy_splat!(pixels, zbuf, px, py, zc, color)
        end
    end
    end  # be > 0.20
    zbuf
end

# ── Colored braille render ──────────────────────────────────────────
# The pixel grid is exactly 2×4 dots per cell.  Each cell's braille
# char is colored by its FRONT-MOST lit dot (via the z-buffer), not an
# average — averaging mixes bright front faces with dark shadowed dots
# and desaturates the whole cell toward gray.

# Nearest xterm-256 palette entry (6×6×6 cube or grayscale ramp)
const POPPY_CUBE = (0, 95, 135, 175, 215, 255)
@inline _poppy_cube_idx(c::Int) = c < 48 ? 0 : c < 115 ? 1 : min(5, (c - 35) ÷ 40)

function _poppy_rgb_to_256(r::Int, g::Int, b::Int)
    ri, gi, bi = _poppy_cube_idx(r), _poppy_cube_idx(g), _poppy_cube_idx(b)
    cr, cg, cb = POPPY_CUBE[ri + 1], POPPY_CUBE[gi + 1], POPPY_CUBE[bi + 1]
    d_cube = (r - cr)^2 + (g - cg)^2 + (b - cb)^2
    avg = (r + g + b) ÷ 3
    gi2 = clamp((avg - 3) ÷ 10, 0, 23)
    gv = 8 + gi2 * 10
    d_gray = (r - gv)^2 + (g - gv)^2 + (b - gv)^2
    d_gray < d_cube ? Color256(232 + gi2) : Color256(16 + 36 * ri + 6 * gi + bi)
end

# Cell-background tint for either canvas polarity: on dark terminals a
# shade toward black (saturation-preserving), on light terminals a pale
# wash toward white.  `s` is "how much of the dot color survives" in
# both cases, so callers tune one knob.
@inline function _poppy_tint(best::ColorRGBA, s::Float64)
    if Tachikoma.light_mode()
        ColorRGBA(UInt8(255 - round(Int, (255 - Int(best.r)) * s)),
                  UInt8(255 - round(Int, (255 - Int(best.g)) * s)),
                  UInt8(255 - round(Int, (255 - Int(best.b)) * s)))
    else
        _poppy_shade(Int(best.r), Int(best.g), Int(best.b), s)
    end
end

@inline function _poppy_term_color(c::ColorRGBA, truecolor::Bool)
    truecolor ? ColorRGB(c.r, c.g, c.b) :
        _poppy_rgb_to_256(Int(c.r), Int(c.g), Int(c.b))
end

# Renders the dot grid as colored braille and returns a per-cell tint
# matrix: the front dot's color dimmed hard (saturation-preserving) and
# scaled by dot coverage, POPPY_BG where the cell is empty.  `bg`
# controls whether the tint is also painted as the cell background:
# :off paints none, :cover paints every braille cell (coverage-faded at
# the edges), :full paints only fully-set 2×4 cells (hard edges).
# `fade` scales the whole flower toward the canvas (1.0 = full presence,
# lower = ambience): dot colors and cell tints fade together, saturation-
# preserving on dark canvases, washing toward white on light ones.
#
# `smooth`: a caller-persisted (rows, cols) ColorRGBA matrix. A cell's
# color is its FRONT-MOST dot's — and as the flower spins, which dot wins
# switches abruptly (different normal, z-fighting petals), so cell colors
# pop rather than glide. With `smooth`, each cell blends toward its new
# color (EMA), turning winner-switches into short fades. Alpha 0.35 ≈
# settles in ~5 frames; empty cells reset so a returning petal snaps in.
@inline _poppy_lerp8(a::UInt8, b::UInt8, t::Float64) =
    UInt8(clamp(round(Int, a + (Int(b) - Int(a)) * t), 0, 255))
@inline _poppy_lerp(a::ColorRGBA, b::ColorRGBA, t::Float64) =
    ColorRGBA(_poppy_lerp8(a.r, b.r, t), _poppy_lerp8(a.g, b.g, t),
              _poppy_lerp8(a.b, b.b, t), b.a)

function _render_poppy_braille!(pixels::Matrix{ColorRGBA}, zbuf::Matrix{Float64},
                                rect::Rect, buf::Buffer, truecolor::Bool;
                                bg::Symbol=:off, fade::Float64=1.0,
                                smooth::Union{Nothing,Matrix{ColorRGBA}}=nothing)
    cw, ch = rect.width, rect.height
    ph, pw = size(pixels)
    tints = fill(POPPY_BG, ch, cw)
    for cy in 1:ch
        for cx in 1:cw
            bits = 0x00
            ndots = 0
            best_z = Inf
            best = POPPY_BG
            for sy in 0:3, sx in 0:1
                px = (cx - 1) * 2 + sx + 1
                py = (cy - 1) * 4 + sy + 1
                (px <= pw && py <= ph) || continue
                p = pixels[py, px]
                p == POPPY_BG && continue
                bits |= Tachikoma.BRAILLE_MAP[sy + 1][sx + 1]
                ndots += 1
                if zbuf[py, px] < best_z
                    best_z = zbuf[py, px]
                    best = p
                end
            end
            if bits == 0x00
                smooth === nothing || (smooth[cy, cx] = POPPY_BG)
                continue
            end
            if smooth !== nothing
                prev = smooth[cy, cx]
                prev == POPPY_BG || (best = _poppy_lerp(prev, best, 0.35))
                smooth[cy, cx] = best
            end
            color = _poppy_term_color(fade < 1.0 ? _poppy_tint(best, fade) : best,
                                      truecolor)
            # Coverage falloff: c^1.25 -- near-linear with the sparse end
            # eased down. Anything steeper makes dense cells jump visibly
            # when their dot count changes frame to frame (flicker at the
            # petal body); anything shallower (sqrt) paints blocky squares
            # under lone rim dots. Steps stay a near-even ~7 RGB apart.
            cov = (ndots / 8)^1.25
            tint = _poppy_tint(best, 0.38 * fade * cov)
            tints[cy, cx] = tint
            style = if bg == :cover
                Style(fg=color, bg=_poppy_term_color(tint, truecolor))
            elseif bg == :full && bits == 0xff
                # Slightly fainter than the under-text tint, so overlaid
                # text sits on a pane of glass that catches the light.
                Style(fg=color, bg=_poppy_term_color(_poppy_tint(best, 0.30 * fade),
                                                     truecolor))
            else
                Style(fg=color)
            end
            set_char!(buf, rect.x + cx - 1, rect.y + cy - 1,
                      Char(Tachikoma.BRAILLE_OFFSET + bits), style)
        end
    end
    tints
end

