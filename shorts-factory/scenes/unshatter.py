"""UNSHATTER: a key, a bulb or a vase is struck and breaks. What was it before it broke?

The video is one recorded shatter played BACKWARD (and says so on screen), then forward again.
Rules, all on screen:
  * we draw one of three generic silhouettes (KEY, BULB, VASE), fracture it into 40 to 70 shards (Voronoi cells around the
    hammer's impact point, denser near it) and simulate the shards as rigid convex pieces: gravity, spin, friction, damped
    bounces off the floor and walls, and collisions between shards;
  * the video plays that recording backward: shards lift, fly back and lock together, outer pieces first (outer shards
    detach first in the recording). The rebuild speeds up as it goes;
  * the three possible answers are on screen from frame 0; the whole object is shown only in the last 1.5 s;
  * then the hammer strikes and the same recording plays forward at fast speed and ends in exactly the opening heap, so the
    last frame cuts cleanly to frame 0.
The simulation is deterministic per seed. The object is KEY, BULB or VASE according to seed % 3.
"""
import math
import os

import cairo
import numpy as np
from PIL import Image, ImageFilter

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, darken, ease_out, ease_out_back, glow, hex_rgb, lerp, lighten, mix,
                        polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30
T_TOTAL = 24.0                 # seconds, 720 frames; frame 0 is the heap, the forward shatter ends in it
T_REB = 22.5                   # the rebuild ends: the whole object exists only from here (last 1.5 s)
T_HIT = 23.35                  # hammer strike
HX = 475.0
OBJ_CX, OBJ_CY = 475.0, 975.0  # object origin on screen
FLOOR_Y = 1250.0
WALL_L, WALL_R = 45.0, 905.0   # shards stay out of the right 12 percent
BX0, BY0, BW, BH = -260.0, -320.0, 520, 620      # art box in object units
S_FULL = 2                     # art and sprites are drawn at 2x
G_ACC = 560.0                  # slow-motion gravity (px/s^2): this is a toy recording, not real physics
DT = 1.0 / 120.0
MAXT = 16.0
DELAY_SPAN = 2.6               # outer shards detach first; inner ones up to this much later
KINDS = ("KEY", "BULB", "VASE")

# ====================================================================== silhouettes (our own generic drawings)


def pieces(kind):
    """Path builders of the filled parts and the holes of a silhouette, in object units."""
    def rr(x, y, w, h, r=8):
        return lambda c: rounded_rect(c, x, y, w, h, r)

    def circ(x, y, r):
        def f(c):
            c.new_sub_path()
            c.arc(x, y, r, 0, TAU)
        return f

    def poly(pts):
        return lambda c: polygon(c, pts)

    if kind == 0:      # key
        parts = [circ(0, -172, 112), rr(-42, -70, 84, 34, 6), rr(-24, -66, 48, 336, 4), rr(24, 98, 46, 40, 4),
                 rr(24, 158, 80, 44, 4), rr(24, 224, 58, 44, 4)]
        holes = [circ(0, -172, 46)]
    elif kind == 1:    # bulb
        parts = [circ(0, -82, 190), poly([(-110, 36), (110, 36), (68, 150), (-68, 150)]), rr(-72, 148, 144, 30, 6),
                 rr(-60, 176, 120, 12, 2), rr(-72, 186, 144, 30, 6), rr(-48, 214, 96, 30, 10), circ(0, 232, 40)]
        holes = []
    else:              # vase

        def body(c):
            c.move_to(-96, -268)
            c.line_to(96, -268)
            c.line_to(62, -208)
            c.curve_to(62, -140, 218, -120, 218, -6)
            c.curve_to(218, 110, 112, 160, 106, 268)
            c.line_to(-106, 268)
            c.curve_to(-112, 160, -218, 110, -218, -6)
            c.curve_to(-218, -120, -62, -140, -62, -208)
            c.close_path()
        parts = [body, rr(-112, -300, 224, 36, 10), circ(128, -150, 62), circ(-128, -150, 62)]
        holes = [circ(128, -150, 28), circ(-128, -150, 28)]
    return parts, holes


def new_surface(scale):
    w, h = int(BW * scale), int(BH * scale)
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    ctx = cairo.Context(surf)
    ctx.scale(scale, scale)
    ctx.translate(-BX0, -BY0)
    return surf, ctx


def mask_of(kind, scale):
    surf, ctx = new_surface(scale)
    parts, holes = pieces(kind)
    ctx.set_source_rgba(1, 1, 1, 1)
    for p in parts:
        ctx.new_path()
        p(ctx)
        ctx.fill()
    ctx.set_operator(cairo.OPERATOR_CLEAR)
    for p in holes:
        ctx.new_path()
        p(ctx)
        ctx.fill()
    surf.flush()
    h, w = int(BH * scale), int(BW * scale)
    return np.frombuffer(surf.get_data(), np.uint8).reshape(h, surf.get_stride() // 4, 4)[:, :w, 3].copy()


KCOL = [hex_rgb("#FFC93C"), hex_rgb("#5CD6FF"), hex_rgb("#FF6B52")]       # key gold, bulb cyan, vase coral


def art_of(kind, scale):
    """The picture that is cut into shards (drawn once)."""
    surf, ctx = new_surface(scale)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    if kind == 0:
        g = cairo.LinearGradient(-140, -320, 160, 280)
        g.add_color_stop_rgb(0, 1.0, 0.89, 0.50)
        g.add_color_stop_rgb(0.5, 0.95, 0.70, 0.18)
        g.add_color_stop_rgb(1, 0.66, 0.40, 0.08)
        ctx.set_source(g)
        ctx.paint()
        for r, a, wd in ((92, 0.40, 7), (62, 0.30, 4)):
            ctx.new_path()
            ctx.arc(0, -172, r, 0, TAU)
            ctx.set_source_rgba(0.35, 0.18, 0.02, a)
            ctx.set_line_width(wd)
            ctx.stroke()
        for i in range(10):
            ctx.new_path()
            ctx.arc(80 * math.cos(i * TAU / 10), -172 + 80 * math.sin(i * TAU / 10), 5, 0, TAU)
            ctx.set_source_rgba(1, 0.97, 0.80, 0.9)
            ctx.fill()
        ctx.new_path()
        ctx.move_to(0, -30)
        ctx.line_to(0, 255)
        ctx.set_source_rgba(0.35, 0.18, 0.02, 0.35)
        ctx.set_line_width(5)
        ctx.stroke()
        for y in (108, 168, 234):
            ctx.new_path()
            ctx.move_to(30, y)
            ctx.line_to(96, y)
            ctx.set_source_rgba(1, 0.95, 0.7, 0.45)
            ctx.set_line_width(4)
            ctx.stroke()
    elif kind == 1:
        g = cairo.RadialGradient(-70, -150, 10, 0, -82, 215)
        g.add_color_stop_rgb(0, 0.90, 0.99, 1.0)
        g.add_color_stop_rgb(0.55, 0.42, 0.82, 0.96)
        g.add_color_stop_rgb(1, 0.14, 0.45, 0.74)
        ctx.set_source(g)
        ctx.paint()
        g2 = cairo.RadialGradient(0, -60, 5, 0, -60, 150)
        g2.add_color_stop_rgba(0, 1.0, 0.78, 0.30, 0.75)
        g2.add_color_stop_rgba(1, 1.0, 0.78, 0.30, 0.0)
        ctx.set_source(g2)
        ctx.paint()
        ctx.set_line_width(6)
        ctx.new_path()
        ctx.move_to(-16, 60)
        ctx.line_to(-38, -84)
        ctx.move_to(16, 60)
        ctx.line_to(38, -84)
        ctx.set_source_rgba(0.35, 0.28, 0.22, 0.9)
        ctx.stroke()
        ctx.new_path()
        ctx.move_to(-38, -84)
        for i in range(1, 8):
            ctx.line_to(-38 + i * 76 / 7.0, -84 + (-22 if i % 2 else 12))
        ctx.set_source_rgb(1.0, 0.86, 0.45)
        ctx.set_line_width(6)
        ctx.stroke()
        ctx.new_path()
        ctx.arc(0, -82, 152, math.radians(196), math.radians(250))
        ctx.set_source_rgba(1, 1, 1, 0.65)
        ctx.set_line_width(14)
        ctx.stroke()
        g3 = cairo.LinearGradient(-80, 0, 80, 0)
        g3.add_color_stop_rgb(0, 0.40, 0.44, 0.50)
        g3.add_color_stop_rgb(0.4, 0.86, 0.90, 0.94)
        g3.add_color_stop_rgb(1, 0.34, 0.38, 0.46)
        ctx.set_source(g3)
        ctx.rectangle(-90, 146, 180, 120)
        ctx.fill()
        for y in (170, 200, 226):
            ctx.new_path()
            ctx.move_to(-76, y)
            ctx.line_to(76, y + 4)
            ctx.set_source_rgba(0.1, 0.12, 0.16, 0.55)
            ctx.set_line_width(4)
            ctx.stroke()
    else:
        g = cairo.LinearGradient(-230, 0, 230, 0)
        g.add_color_stop_rgb(0, 0.62, 0.20, 0.14)
        g.add_color_stop_rgb(0.35, 1.0, 0.45, 0.32)
        g.add_color_stop_rgb(1, 0.66, 0.22, 0.16)
        ctx.set_source(g)
        ctx.paint()
        for y, h_ in ((-92, 24), (128, 28)):
            ctx.rectangle(-260, y, 520, h_)
            ctx.set_source_rgba(1.0, 0.93, 0.78, 0.95)
            ctx.fill()
        for i in range(-5, 6):
            ctx.new_path()
            ctx.arc(i * 36, -80, 6, 0, TAU)
            ctx.set_source_rgba(0.55, 0.15, 0.10, 0.9)
            ctx.fill()
        for i in range(-4, 5):
            x = i * 44
            polygon(ctx, [(x, 26), (x + 17, 48), (x, 70), (x - 17, 48)])
            ctx.set_source_rgba(1.0, 0.93, 0.78, 0.9 if i % 2 == 0 else 0.5)
            ctx.fill()
        ctx.new_path()
        ctx.move_to(-150, -210)
        ctx.curve_to(-170, -80, -190, 0, -150, 150)
        ctx.set_source_rgba(1, 1, 1, 0.22)
        ctx.set_line_width(16)
        ctx.stroke()
    surf.flush()
    h, w = int(BH * scale), int(BW * scale)
    return np.frombuffer(surf.get_data(), np.uint8).reshape(h, surf.get_stride() // 4, 4)[:, :w, :3].copy()   # BGR


# ====================================================================== fracture
def erode(mask_bool, px):
    im = Image.fromarray((mask_bool * 255).astype(np.uint8))
    im = im.filter(ImageFilter.MinFilter(2 * px + 1))
    return np.array(im) > 127


def components(lab, inside):
    """Connected components (4-neighbor) of equal label inside the mask; tiny ones join their biggest neighbor."""
    h, w = lab.shape
    comp = np.full((h, w), -1, np.int32)
    flat_l, flat_c, flat_in = lab.ravel(), comp.ravel(), inside.ravel()
    sizes = []
    n = 0
    for start in np.flatnonzero(flat_in):
        if flat_c[start] >= 0:
            continue
        flat_c[start] = n
        stack, cnt, L = [start], 0, flat_l[start]
        while stack:
            p = stack.pop()
            cnt += 1
            y, x = divmod(p, w)
            for q, ok in ((p - w, y > 0), (p + w, y < h - 1), (p - 1, x > 0), (p + 1, x < w - 1)):
                if ok and flat_c[q] < 0 and flat_in[q] and flat_l[q] == L:
                    flat_c[q] = n
                    stack.append(q)
        sizes.append(cnt)
        n += 1
    sizes = np.array(sizes)
    return comp, sizes


def fracture(kind, seed, scale):
    """Shard label image (at `scale` px per unit), impact point, per-shard comp ids."""
    rng = np.random.default_rng(seed * 104729 + 7)
    m1 = mask_of(kind, 1) > 127
    inner = erode(m1, 11)
    ys, xs = np.nonzero(inner)
    mid = (ys > 120) & (ys < 500)
    k = int(rng.integers(int(mid.sum())))
    iy, ix = int(ys[mid][k]), int(xs[mid][k])
    seeds = [(ix, iy)]
    target = int(rng.integers(50, 66))
    base = 0.74 * math.sqrt(float(m1.sum()) / target)
    for _ in range(8000):
        if len(seeds) >= target:
            break
        j = int(rng.integers(len(ys)))
        p = (int(xs[j]), int(ys[j]))
        r = math.hypot(p[0] - ix, p[1] - iy)
        smin = base * (0.62 + 0.75 * min(1.0, r / 280.0))
        if all(math.hypot(p[0] - q[0], p[1] - q[1]) >= smin for q in seeds):
            seeds.append(p)
    sd = np.array(seeds, np.float64) * scale
    mk = mask_of(kind, scale)
    inside = mk > 127
    yy, xx = np.nonzero(inside)
    best = np.full(len(yy), 1e18)
    lab_px = np.zeros(len(yy), np.int32)
    for i, (sx, sy) in enumerate(sd):
        d = (xx - sx) ** 2 + (yy - sy) ** 2
        up = d < best
        best[up] = d[up]
        lab_px[up] = i
    lab = np.full(mk.shape, -1, np.int32)
    lab[yy, xx] = lab_px
    f = max(1, int(round(scale)))
    l1, in1 = lab[::f, ::f], inside[::f, ::f]
    comp1, sizes = components(l1, in1)
    # merge tiny pieces (under 160 px^2) into the neighbor they touch most
    ncomp = len(sizes)
    min_px = 160.0 * (1.0 / f) ** 2 if f == 1 else 160.0 / 1.0
    min_px = 160.0 * (scale / f) ** 2
    remap = np.arange(ncomp)
    h1, w1 = comp1.shape
    for c in np.argsort(sizes):
        if sizes[c] >= min_px:
            break
        cy, cx = np.nonzero(comp1 == c)
        cnt = {}
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny_, nx_ = np.clip(cy + dy, 0, h1 - 1), np.clip(cx + dx, 0, w1 - 1)
            for v in comp1[ny_, nx_]:
                if v >= 0 and remap[v] != c:
                    cnt[remap[v]] = cnt.get(remap[v], 0) + 1
        if cnt:
            tgt = max(cnt, key=cnt.get)
            remap[remap == c] = tgt
            comp1[comp1 == c] = tgt
            sizes[tgt] += sizes[c]
            sizes[c] = 0
    ids = np.unique(comp1[comp1 >= 0])
    lut = np.full(ncomp, -1, np.int32)
    lut[ids] = np.arange(len(ids))
    comp1 = np.where(comp1 >= 0, lut[np.maximum(comp1, 0)], -1)
    comp = np.repeat(np.repeat(comp1, f, axis=0), f, axis=1)[:mk.shape[0], :mk.shape[1]]
    if comp.shape != mk.shape:
        comp = np.pad(comp, ((0, mk.shape[0] - comp.shape[0]), (0, mk.shape[1] - comp.shape[1])), constant_values=-1)
    # pixels whose coarse parent is outside the mask: take the label's biggest piece
    miss = inside & (comp < 0)
    if miss.any():
        main = {}
        for L in np.unique(lab[miss]):
            cs = comp[(lab == L) & (comp >= 0)]
            main[L] = int(np.bincount(cs).argmax()) if len(cs) else -1
        for L, c in main.items():
            comp[miss & (lab == L)] = c
    return dict(comp=comp, mask=mk, impact=(ix, iy), n=int(comp.max()) + 1, seeds=len(seeds))


def convex_hull(pts):
    pts = sorted(set(map(tuple, np.round(pts, 2))))
    if len(pts) < 3:
        return np.array(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, up = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0:
            lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0:
            up.pop()
        up.append(p)
    return np.array(lo[:-1] + up[:-1])


def simplify_hull(h, nmax=12):
    h = [tuple(p) for p in h]
    while len(h) > nmax:
        best, bi = 1e18, 0
        for i in range(len(h)):
            a, b, c = h[i - 1], h[i], h[(i + 1) % len(h)]
            area = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) / 2
            if area < best:
                best, bi = area, i
        h.pop(bi)
    return np.array(h)


def build_shards(kind, seed, scale=S_FULL, with_sprites=True):
    fr = fracture(kind, seed, scale)
    comp, mk = fr["comp"], fr["mask"]
    n = fr["n"]
    art = art_of(kind, scale) if with_sprites else None
    shards = []
    ix, iy = fr["impact"]
    for k in range(n):
        m = comp == k
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            continue
        area = len(ys) / scale ** 2
        cx2, cy2 = xs.mean(), ys.mean()
        com = (cx2 / scale + BX0, cy2 / scale + BY0)                  # object units
        er = erode(m, max(1, int(scale)))
        edge = m & ~er
        ey, ex = np.nonzero(edge)
        hull = convex_hull(np.stack([(ex - cx2) / scale, (ey - cy2) / scale], 1)[::2])
        hull = simplify_hull(hull, 12)
        c = hull.mean(0)
        hull = c + (hull - c) * (1.0 + 0.6 / max(8.0, np.hypot(*(hull - c).T).mean()))      # 0.6 px padding
        dx, dy = (xs - cx2) / scale, (ys - cy2) / scale
        inn = er[ys, xs]
        inner_pts = np.stack([dx[inn], dy[inn]], 1)[:: max(1, int(scale) ** 2)]
        sh = dict(k=k, area=float(area), com=com, hull=hull, inner=inner_pts,
                  inertia=float(0.25 * np.sum(dx * dx + dy * dy) / (scale ** 2) * 4.0 / 4.0) if False else float(np.sum(dx * dx + dy * dy) / scale ** 2),
                  dist=math.hypot(com[0] - (ix / 1.0 + BX0), com[1] - (iy / 1.0 + BY0)))
        if with_sprites:
            y0, y1, x0, x1 = ys.min() - 2, ys.max() + 3, xs.min() - 2, xs.max() + 3
            y0, x0 = max(0, y0), max(0, x0)
            sub = m[y0:y1, x0:x1]
            alpha = (mk[y0:y1, x0:x1].astype(np.float32) / 255.0) * sub
            rgb = art[y0:y1, x0:x1].astype(np.float32)
            subs = sub.copy()
            er2 = erode(subs, 2)
            edge2 = subs & ~er2
            rgb *= np.where(edge2, 0.58, 1.0)[..., None]
            er1 = erode(subs, 1)
            rgb *= np.where(subs & ~er1, 0.8, 1.0)[..., None]
            tint = 1.0 + 0.07 * math.sin(k * 12.9898)
            rgb = np.clip(rgb * tint, 0, 255)
            spr = np.zeros(sub.shape + (4,), np.uint8)
            spr[..., :3] = (rgb * alpha[..., None]).astype(np.uint8)
            spr[..., 3] = (alpha * 255).astype(np.uint8)
            sh["sprite"] = spr
            sh["anchor"] = (cx2 - x0, cy2 - y0)
        shards.append(sh)
    return shards, fr


# ====================================================================== physics
def support(v, n):
    return v[int(np.argmax(v @ n))]


def sat2(wi, ni, wj, nj):
    """Penetration of two convex polygons: (depth, unit normal from i to j) or None (separating-axis test)."""
    p = wi @ ni.T
    q = wj @ ni.T
    a1 = p.max(0) - q.min(0)
    p = wj @ nj.T
    q = wi @ nj.T
    a2 = p.max(0) - q.min(0)
    k1, k2 = int(np.argmin(a1)), int(np.argmin(a2))
    if a1[k1] <= 0 or a2[k2] <= 0:
        return None
    if a1[k1] < a2[k2]:
        return float(a1[k1]), ni[k1]
    return float(a2[k2]), -nj[k2]


def simulate(shards, impact, seed):
    rng = np.random.default_rng(seed * 31 + 5)
    N = len(shards)
    V = 12
    HL = np.zeros((N, V, 2))
    for i, s in enumerate(shards):
        h = s["hull"]
        HL[i, :len(h)] = h
        HL[i, len(h):] = h[-1]
    pos0 = np.array([[OBJ_CX + s["com"][0], OBJ_CY + s["com"][1]] for s in shards])
    mass = np.array([s["area"] for s in shards])
    inertia = np.array([s["inertia"] for s in shards])
    rad = np.hypot(HL[..., 0], HL[..., 1]).max(1)
    ipx, ipy = OBJ_CX + impact[0] + BX0, OBJ_CY + impact[1] + BY0
    d = np.hypot(pos0[:, 0] - ipx, pos0[:, 1] - ipy)
    rel = DELAY_SPAN * (1.0 - d / d.max()) ** 1.2             # outer shards detach first
    pos, ang = pos0.copy(), np.zeros(N)
    vel, om = np.zeros((N, 2)), np.zeros(N)
    held = np.ones(N, bool)
    states = [np.concatenate([pos, ang[:, None]], 1).astype(np.float32)]
    events = []
    nsteps = int(MAXT / DT)
    moving = []
    mu = 0.45

    NL = np.zeros((N, V, 2))
    for i, s_h in enumerate(shards):
        h = s_h["hull"]
        cen = h.mean(0)
        nn = []
        for q in range(len(h)):
            e = h[(q + 1) % len(h)] - h[q]
            nv = np.array([e[1], -e[0]])
            nv = nv / max(1e-9, math.hypot(*nv))
            if (h[q] + e / 2 - cen) @ nv < 0:
                nv = -nv
            nn.append(nv)
        NL[i, :len(h)] = nn
        NL[i, len(h):] = nn[-1]

    def worlds():
        c, s_ = np.cos(ang)[:, None], np.sin(ang)[:, None]
        wv = np.stack([pos[:, 0:1] + c * HL[..., 0] - s_ * HL[..., 1], pos[:, 1:2] + s_ * HL[..., 0] + c * HL[..., 1]], 2)
        nw = np.stack([c * NL[..., 0] - s_ * NL[..., 1], s_ * NL[..., 0] + c * NL[..., 1]], 2)
        return wv, nw

    def impulse(i, j, n, pt, depth, t, e0=0.30):
        """Contact between i and j (j None = static world); n points from i to j (or into the world)."""
        ri = pt - pos[i]
        vi = vel[i] + om[i] * np.array([-ri[1], ri[0]])
        if j is None:
            vr = -vi
            rj = None
            inv = 1.0 / mass[i] + (ri[0] * n[1] - ri[1] * n[0]) ** 2 / inertia[i]
        else:
            rj = pt - pos[j]
            vj = vel[j] + om[j] * np.array([-rj[1], rj[0]])
            vr = vj - vi
            inv = (1.0 / mass[i] + 1.0 / mass[j] + (ri[0] * n[1] - ri[1] * n[0]) ** 2 / inertia[i]
                   + (rj[0] * n[1] - rj[1] * n[0]) ** 2 / inertia[j])
        vn = vr @ n
        if vn >= 0:
            return
        e = e0 if -vn > 150 else 0.0
        jn = -(1 + e) * vn / inv
        tg = vr - vn * n
        tl = math.hypot(*tg)
        jt = np.zeros(2)
        if tl > 1e-6:
            tdir = tg / tl
            invt = inv
            jt = tdir * min(mu * jn, tl / invt)
        pi = -jn * n + jt
        vel[i] += pi / mass[i]
        om[i] += (ri[0] * pi[1] - ri[1] * pi[0]) / inertia[i]
        if j is not None:
            pj = jn * n - jt
            vel[j] += pj / mass[j]
            om[j] += (rj[0] * pj[1] - rj[1] * pj[0]) / inertia[j]
        if -vn > 70:
            events.append((t, i, min(1.0, -vn / 600.0)))
            if j is not None:
                events.append((t, j, min(1.0, -vn / 600.0)))

    for step in range(1, nsteps):
        t = step * DT
        new = held & (t >= rel)
        for i in np.flatnonzero(new):
            dv = np.array([pos0[i, 0] - ipx, pos0[i, 1] - ipy])
            dv = dv / max(1e-6, np.hypot(*dv))
            a = rng.normal(0, 0.28)
            dv = np.array([dv[0] * math.cos(a) - dv[1] * math.sin(a), dv[0] * math.sin(a) + dv[1] * math.cos(a)])
            sp = 110 + 300 * math.exp(-d[i] / 140.0) + rng.uniform(0, 60)
            vel[i] = dv * sp
            vel[i, 1] = max(-400.0, vel[i, 1] - 140.0 - 60 * rng.random())
            om[i] = rng.normal(0, 4.0) * min(1.0, 2500.0 / mass[i]) ** 0.5
            held[i] = False
            events.append((t, i, 0.9))
        free = ~held
        vel[free, 1] += G_ACC * DT
        pos[free] += vel[free] * DT
        ang[free] += om[free] * DT
        for _pass in range(2):
            fl = np.flatnonzero(free)
            WV, NW = worlds()
            for i in fl:
                v = WV[i]
                # floor
                low = np.flatnonzero(v[:, 1] > FLOOR_Y)
                if len(low):
                    for q in low[np.argsort(-v[low, 1])][:2]:
                        dep = v[q, 1] - FLOOR_Y
                        impulse(i, None, np.array([0.0, 1.0]), v[q], dep, t, 0.35)
                    pos[i, 1] -= (v[low, 1].max() - FLOOR_Y) * 0.9
                for lim, sgn in ((WALL_L, 1.0), (WALL_R, -1.0)):
                    out = np.flatnonzero((v[:, 0] - lim) * sgn < 0)
                    if len(out):
                        q = out[np.argmin((v[out, 0] - lim) * sgn)]
                        impulse(i, None, np.array([-sgn, 0.0]), v[q], 0, t, 0.3)
                        pos[i, 0] += sgn * abs(v[q, 0] - lim) * 0.9
            if _pass == 0 and t > 0.12:
                P = pos[fl]
                dd = np.hypot(P[:, None, 0] - P[None, :, 0], P[:, None, 1] - P[None, :, 1])
                cand = np.argwhere(np.triu(dd < (rad[fl][:, None] + rad[fl][None, :]), 1))
                for a_, b_ in cand:
                    i, j = fl[a_], fl[b_]
                    r = sat2(WV[i], NW[i], WV[j], NW[j])
                    if r is None:
                        continue
                    depth, n = r
                    if depth < 0.4:
                        continue
                    pa, pb = support(WV[i], n), support(WV[j], -n)
                    pt = (pa + pb) / 2
                    impulse(i, j, n, pt, depth, t, 0.25)
                    wi, wj = 1.0 / mass[i], 1.0 / mass[j]
                    corr = 0.6 * max(0.0, depth - 0.3) / (wi + wj)
                    pos[i] -= n * corr * wi
                    pos[j] += n * corr * wj
        slow = free & (np.hypot(vel[:, 0], vel[:, 1]) < 40.0) & (np.abs(om) * rad < 40.0)
        vel[slow] *= 0.93
        om[slow] *= 0.93
        vel[free] *= 1.0 - 0.10 * DT
        sp_ = np.hypot(vel[:, 0], vel[:, 1]) + np.abs(om) * rad * 0.5
        moving.append(int(((sp_ > 70) & free).sum() + (held.sum() > 0) * 100))
        states.append(np.concatenate([pos, ang[:, None]], 1).astype(np.float32))
        if step * DT > DELAY_SPAN + 0.5 and moving[-1] == 0 and step > 60:
            if all(m_ == 0 for m_ in moving[-120:]):
                break
    St = np.array(states)
    vm = np.hypot(*np.diff(St[:, :, :2], axis=0).transpose(2, 0, 1)).mean(1) / DT
    vm = np.convolve(vm, np.ones(30) / 30.0, mode="same")
    act = np.flatnonzero(vm >= 30.0)
    cut = int(act[-1]) + 20 if len(act) else len(states) - 1       # the recording ends while the heap is still settling
    cut = min(cut, len(states) - 1, int(10.0 / DT))
    S = np.array(states[:cut + 1])
    ev = [(t, i, s_) for t, i, s_ in events if t <= cut * DT]
    return dict(S=S, T=cut * DT, rel=rel, events=ev, pos0=pos0, d=d, ipx=ipx, ipy=ipy, rad=rad)


def pose(sim, tau):
    K = len(sim["S"]) - 1
    x = clamp(tau / sim["T"]) * K
    i0 = int(min(K - 1, math.floor(x)))
    a = x - i0
    return sim["S"][i0] * (1 - a) + sim["S"][i0 + 1] * a


def overlap_stats(shards, sim, tau):
    P = pose(sim, tau)
    canvas = np.zeros((H, W), np.int16)
    tot = 0.0
    for i, s in enumerate(shards):
        c, s_ = math.cos(P[i, 2]), math.sin(P[i, 2])
        q = s["inner"]
        x = np.round(P[i, 0] + c * q[:, 0] - s_ * q[:, 1]).astype(int)
        y = np.round(P[i, 1] + s_ * q[:, 0] + c * q[:, 1]).astype(int)
        ok = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        idx = np.unique(y[ok] * W + x[ok])
        canvas.ravel()[idx] += 1
        tot += len(idx)
    return float((canvas >= 2).sum()) / max(1.0, tot)


# ====================================================================== drama / gates
def drama_light(seed):
    kind = seed % 3
    try:
        sh, fr = build_shards(kind, seed, scale=0.5, with_sprites=False)
    except Exception as e:                                   # pragma: no cover
        return dict(seed=seed, score=-100.0, valid=False, note=str(e))
    areas = np.array([s["area"] for s in sh])
    n = len(sh)
    cv = float(areas.std() / areas.mean())
    gates = dict(count=40 <= n <= 70, min_area=bool(areas.min() >= 140), max_area=bool(areas.max() <= 9000))
    ok = all(gates.values())
    score = (100.0 if ok else 0.0) - 20.0 * cv - 0.02 * abs(n - 56)
    return dict(seed=seed, score=round(float(score), 2), valid=ok, gates=gates, kind=KINDS[kind], shards=n,
                area_cv=round(cv, 2))


def score_seed(seed):
    return drama_light(seed)


# ====================================================================== look
INK = hex_rgb("#0B0716")
CREAM = hex_rgb("#F7F1E3")
MUTED = hex_rgb("#A79FC4")
AMBER = hex_rgb("#FFC23D")
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
Q1, Q2 = "WHAT WAS IT", "BEFORE IT BROKE?"
RULE = "SHARDS FLY BACK TOGETHER: PICK THE OBJECT"
SUBTXT = "THE REBUILD IS A REVERSED RECORDING OF THE SHATTER"
END_TEXT = "Got it right? +1 streak. Subscribe to keep it."
CARD_X = (HX - 290.0, HX, HX + 290.0)
CARD_Y, CARD_W, CARD_H = 1300.0, 262.0, 150.0


def hash1(a, b=0.0):
    v = math.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return v - math.floor(v)


def symbol(ctx, k, x, y, s):
    if k == 0:
        ctx.new_sub_path()
        ctx.arc(x, y, s, 0, TAU)
    elif k == 1:
        ctx.rectangle(x - s * 0.86, y - s * 0.86, s * 1.72, s * 1.72)
    else:
        polygon(ctx, [(x, y - s * 1.15), (x + s * 1.1, y + s * 0.8), (x - s * 1.1, y + s * 0.8)])


def chip(ctx, font, txt, cx, y, size, pad=20, h=None, bg=CREAM, fg=INK):
    tw = font.width(txt, size, 1.2)
    h = h or size * 1.42
    rounded_rect(ctx, cx - tw / 2 - pad, y - h * 0.76, tw + 2 * pad, h, 12)
    ctx.set_source_rgba(*bg, 1)
    ctx.fill()
    font.draw(ctx, txt, cx, y, size, fg, anchor="c", tracking=1.2)


def make_background():
    rng = np.random.default_rng(3)
    yy = np.arange(H, dtype=np.float32)[:, None]
    top, mid, bot = (np.array(x, np.float32) for x in ((0.045, 0.030, 0.090), (0.105, 0.060, 0.190), (0.030, 0.022, 0.060)))
    u1 = np.clip(yy / 1250.0, 0, 1)
    u2 = np.clip((yy - 1250.0) / 400.0, 0, 1)
    img = (top[None, :] + (mid - top)[None, :] * u1) + (bot - mid)[None, :] * u2
    img = np.repeat(img[:, None, :], W, axis=1)
    img += rng.normal(0, 0.004, (H, W, 1)).astype(np.float32)
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    ctx = cairo.Context(surf)
    g = cairo.RadialGradient(OBJ_CX, 1000, 20, OBJ_CX, 1000, 620)          # warm pool of light under the object
    g.add_color_stop_rgba(0, 0.95, 0.55, 0.30, 0.20)
    g.add_color_stop_rgba(1, 0.95, 0.55, 0.30, 0.0)
    ctx.set_source(g)
    ctx.rectangle(0, 500, W, 1000)
    ctx.fill()
    for i in range(22):                                                      # soft vertical light shafts
        x = 30 + i * 48
        ctx.set_source_rgba(0.7, 0.6, 1.0, 0.018)
        ctx.rectangle(x, 560, 14, 700)
        ctx.fill()
    g = cairo.LinearGradient(0, FLOOR_Y, 0, FLOOR_Y + 70)
    g.add_color_stop_rgba(0, 0.20, 0.14, 0.34, 1)
    g.add_color_stop_rgba(1, 0.05, 0.03, 0.10, 1)
    ctx.set_source(g)
    ctx.rectangle(0, FLOOR_Y, W, H - FLOOR_Y)
    ctx.fill()
    ctx.set_source_rgba(1.0, 0.80, 0.60, 0.55)
    ctx.set_line_width(3)
    ctx.move_to(0, FLOOR_Y + 1)
    ctx.line_to(W, FLOOR_Y + 1)
    ctx.stroke()
    return surf, buf


def make_header(font):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, 460)
    ctx = cairo.Context(surf)
    s1 = min(124.0, 850.0 / font.width(Q2, 1.0))
    font.draw(ctx, Q1, HX, 124, s1, CREAM, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    font.draw(ctx, Q2, HX, 124 + s1 * 1.1, s1, CREAM, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    size = min(40.0, 880.0 / font.width(RULE, 1.0))
    chip(ctx, font, RULE, HX, 124 + s1 * 1.1 + 90, size, pad=22, bg=AMBER)
    font.draw(ctx, SUBTXT, HX, 124 + s1 * 1.1 + 146, 25, CREAM, anchor="c", stroke=INK, stroke_w=6, tracking=0.6)
    surf.flush()
    return surf


# ====================================================================== the scene
class Scene:
    W, H, FPS = W, H, FPS
    EQ = ((50, -7.0), (120, -3.0), (300, -1.0), (900, 0.0), (3000, 1.5), (8000, 0.5), (16000, -4.0))
    DRIVE = 2.0
    CEILING_DB = -6.0
    HEAD_Y = 150
    WARP_K, WARP_P = 1.4, 1.5

    def __init__(self, seed):
        self.seed = seed
        self.kind = seed % 3
        self.shards, fr = build_shards(self.kind, seed)
        self.impact = fr["impact"]
        self.n = len(self.shards)
        self.sim = simulate(self.shards, self.impact, seed)
        self.T = self.sim["T"]
        self.nframes = int(round(T_TOTAL * FPS))
        self.dur = self.nframes / FPS
        rng = np.random.default_rng(seed + 99)
        self.order = [int(x) for x in rng.permutation(3)]                  # which silhouette is option A, B, C
        # time warp: remaining recording time tau as a function of video time during the rebuild. The recording is played
        # at a rate that keeps the visible motion steady (the slow settling at the start of the reverse is sped up, the
        # flights are slowed down) and the target speed grows over the rebuild, so the reverse accelerates.
        S = self.sim["S"]
        v = np.hypot(*np.diff(S[:, :, :2], axis=0).transpose(2, 0, 1)).mean(1) / DT
        v = np.convolve(v, np.ones(30) / 30.0, mode="same")
        self.act = np.concatenate([v, v[-1:]])
        tg = np.arange(0, T_REB + 1e-9, 0.005)
        u = tg / T_REB

        def run(v0):
            tau, out = self.T, np.empty(len(tg))
            for k in range(len(tg)):
                out[k] = tau
                a_ = self.act[min(len(self.act) - 1, int(tau / DT))]
                rate = min(4.0, max(0.10, v0 * (1.0 + self.WARP_K * u[k] ** self.WARP_P) / (a_ + 10.0)))
                tau = max(0.0, tau - rate * 0.005)
            return out

        lo, hi = 1.0, 20000.0
        for _ in range(40):
            mid = math.sqrt(lo * hi)
            if run(mid)[-1] > 0.0:
                lo = mid
            else:
                hi = mid
        self.tg = tg
        self.tau_g = run(hi)
        # whole-object sprite and its alpha
        mk = mask_of(self.kind, S_FULL)
        art = art_of(self.kind, S_FULL)
        al = mk.astype(np.float32) / 255.0
        whole = np.zeros(mk.shape + (4,), np.uint8)
        whole[..., :3] = (art.astype(np.float32) * al[..., None]).astype(np.uint8)
        whole[..., 3] = mk
        self.whole = whole
        self.mask2 = mk
        self.stats = self._drama()
        if not self.stats["valid"]:
            raise ValueError(f"seed {seed}: failed gates {[k for k, v in self.stats['gates'].items() if not v]}")
        self._res = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_res"] = None
        return d

    # ------------------------------------------------------------ time mapping
    def tau_at(self, t):
        if t <= T_REB:
            return float(np.interp(t, self.tg, self.tau_g))
        if t < T_HIT:
            return 0.0
        return self.T * (1.0 - (1.0 - clamp((t - T_HIT) / (T_TOTAL - T_HIT))) ** 2)     # fast burst, then it settles into the heap

    def t_reverse(self, tau):
        return float(np.interp(tau, self.tau_g[::-1], self.tg[::-1]))

    def _drama(self):
        sim = self.sim
        heap = pose(sim, self.T)
        t25 = self.tau_at(2.5)
        p25 = pose(sim, t25)
        moved = int((np.hypot(*(p25[:, :2] - heap[:, :2]).T) >= 10.0).sum())
        tops = [float(pose(sim, tt)[:, 1].min()) for tt in np.linspace(0, self.T, 60)]
        ovl = [overlap_stats(self.shards, sim, tt) for tt in (self.T, self.T * 0.97, self.T * 0.9, 0.0)]
        t12 = self.tau_at(12.0)
        p12 = pose(sim, t12)
        near_home = int((np.hypot(*(p12[:, :2] - sim["pos0"]).T) < 40.0).sum())
        areas = np.array([s["area"] for s in self.shards])
        gates = dict(shards=40 <= self.n <= 70, event_by_2_5s=moved >= 4, rest_time=self.T <= 11.0,
                     heap_overlap=max(ovl[0], ovl[1], ovl[2]) <= 0.035, assembled_overlap=ovl[3] <= 0.03,
                     min_area=bool(areas.min() >= 140))
        return dict(seed=self.seed, valid=bool(all(gates.values())), gates={k: bool(v) for k, v in gates.items()},
                    answer=KINDS[self.kind], shards=self.n, recording_seconds=round(self.T, 2),
                    shards_moved_10px_by_2_5s=moved, shards_home_at_12s=near_home,
                    heap_overlap_fraction=round(ovl[0], 4), overlap_at_0_97=round(ovl[1], 4),
                    overlap_at_0_9=round(ovl[2], 4), assembled_overlap_fraction=round(ovl[3], 4),
                    highest_shard_y=round(min(tops), 1), score=100.0 if all(gates.values()) else 0.0,
                    option_order=[KINDS[k] for k in self.order])

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            font = Font(FONT_PATH)
            bg, keep = make_background()
            spr = []
            for s in self.shards:
                a = np.ascontiguousarray(s["sprite"])
                h, w = a.shape[:2]
                spr.append(cairo.ImageSurface.create_for_data(bytearray(a.tobytes()), cairo.FORMAT_ARGB32, w, h, w * 4))
            wh = np.ascontiguousarray(self.whole)
            whole = cairo.ImageSurface.create_for_data(bytearray(wh.tobytes()), cairo.FORMAT_ARGB32, wh.shape[1],
                                                       wh.shape[0], wh.shape[1] * 4)
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep, font=font,
                             head=make_header(font), spr=spr, whole=whole)
        return self._res

    def frame(self, f):
        res = self._resources()
        ctx = res["ctx"]
        ctx.save()
        self.draw(ctx, f, res)
        ctx.restore()
        res["surf"].flush()
        return bytes(res["surf"].get_data())

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, f, res):
        t = f / FPS
        font = res["font"]
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(res["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        tau = self.tau_at(t)
        P = pose(self.sim, tau)
        self._dust(ctx, t)
        self._shards(ctx, res, P, t)
        self._seal(ctx, res, t)
        self._glints(ctx, P, t)
        self._hammer(ctx, t)
        fl = self._flash(t)
        if fl > 0.004:
            ctx.set_source_rgba(0.95, 0.93, 1.0, fl)
            ctx.paint()
        ctx.set_source_surface(res["head"], 0, self.HEAD_Y)
        ctx.paint()
        self._cards(ctx, font, t)
        self._countdown(ctx, font, t)
        self._result(ctx, font, t)
        self._endcard(ctx, font, t)

    def _dust(self, ctx, t):
        for q, ph in ((3, 0.0), (2, 0.5)):                                   # two slow lamp glows sweep over the heap
            glow(ctx, 475 + 440 * math.sin(TAU * q * t / T_TOTAL + ph * TAU), 1150, 280, (1.0, 0.82, 0.62), 0.14)
        for i in range(36):                                                  # slow drifting motes, whole cycles over the video
            k = 1 + i % 3
            ph = (hash1(i, 3.0) + k * t / T_TOTAL) % 1.0
            x = 30 + hash1(i, 9.0) * 900 + 18 * math.sin(TAU * (ph * 2 + hash1(i, 1.0)))
            y = 1240 - ph * 760
            a = 0.30 * math.sin(math.pi * ph) ** 2
            ctx.new_path()
            ctx.arc(x, y, 2.4 + (i % 3), 0, TAU)
            ctx.set_source_rgba(1.0, 0.86, 0.70, a)
            ctx.fill()

    def _shards(self, ctx, res, P, t):
        whole_a = ramp(t, T_REB + 0.05, T_REB + 0.45) * (1.0 if t < T_HIT else 0.0)
        if whole_a >= 0.999:
            return
        ctx.new_path()
        for i, s in enumerate(self.shards):
            x, y, a = float(P[i, 0]), float(P[i, 1]), float(P[i, 2])
            ax, ay = s["anchor"]
            ctx.save()
            ctx.translate(x, y)
            ctx.rotate(a)
            ctx.scale(1.0 / S_FULL, 1.0 / S_FULL)
            spr = res["spr"][i]
            ctx.set_source_surface(spr, -ax, -ay)
            ctx.get_source().set_filter(cairo.FILTER_BILINEAR)
            ctx.rectangle(-ax, -ay, spr.get_width(), spr.get_height())
            ctx.fill()
            ctx.restore()

    def _seal(self, ctx, res, t):
        a = ramp(t, T_REB + 0.05, T_REB + 0.45) * (1.0 if t < T_HIT else 0.0)
        if a <= 0.003:
            return
        ox, oy = OBJ_CX + BX0, OBJ_CY + BY0
        glow(ctx, OBJ_CX, OBJ_CY, 420, KCOL[self.kind], 0.20 * a)
        ctx.save()
        ctx.translate(ox, oy)
        ctx.scale(1.0 / S_FULL, 1.0 / S_FULL)
        ctx.set_source_surface(res["whole"], 0, 0)
        ctx.paint_with_alpha(a)
        # one soft sheen sweeps across the whole object
        u = (t - (T_REB + 0.25)) / 0.75
        if 0 < u < 1:
            x0 = -400 + 1700 * u
            g = cairo.LinearGradient(x0, 0, x0 + 300, 600)
            g.add_color_stop_rgba(0, 1, 1, 1, 0)
            g.add_color_stop_rgba(0.5, 1, 1, 1, 0.6)
            g.add_color_stop_rgba(1, 1, 1, 1, 0)
            ctx.set_source(g)
            ctx.mask_surface(res["whole"], 0, 0)
        ctx.restore()

    def _glints(self, ctx, P, t):
        if T_REB < t < T_HIT:
            return
        for i in range(self.n):
            m = 1 + i % 3
            ph = (t / T_TOTAL * m + hash1(i, 5.0)) % 1.0
            a = max(0.0, 1.0 - abs(ph - 0.5) / 0.035) if False else max(0.0, 1.0 - abs(ph - 0.5) * 12.0)
            if a <= 0.02:
                continue
            x, y = float(P[i, 0]), float(P[i, 1])
            r = 16 + 14 * a
            ctx.new_path()
            ctx.move_to(x - r, y)
            ctx.line_to(x + r, y)
            ctx.move_to(x, y - r)
            ctx.line_to(x, y + r)
            ctx.set_source_rgba(1, 1, 0.95, 0.85 * a)
            ctx.set_line_width(3)
            ctx.stroke()
            glow(ctx, x, y, 22, (1, 0.95, 0.8), 0.45 * a)

    def _hammer(self, ctx, t):
        u = (t - (T_HIT - 0.30)) / 0.30
        if not (0.0 <= u <= 1.0) and not (T_HIT < t < T_HIT + 0.25):
            return
        ix, iy = self.sim["ipx"], self.sim["ipy"]
        if u <= 1.0:
            th, al = 0.78 + 1.3 * (1.0 - u * u), 1.0
        else:
            th, al = 0.78, 1.0 - ramp(t, T_HIT + 0.02, T_HIT + 0.22)
        ctx.save()
        ctx.translate(ix + 150, iy - 150)                                   # grip pivot; the head lands on the impact point
        ctx.rotate(th)
        L = 199.0
        ctx.new_path()
        rounded_rect(ctx, -10, 0, 20, L, 6)
        ctx.set_source_rgba(0.62, 0.40, 0.22, al)
        ctx.fill()
        rounded_rect(ctx, -46, L - 16, 92, 58, 8)
        ctx.set_source_rgba(0.72, 0.76, 0.84, al)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.12, 0.10, 0.20, al)
        ctx.set_line_width(5)
        ctx.stroke()
        ctx.restore()

    def _flash(self, t):
        d = t - T_HIT
        if d < 0:
            return 0.0
        return 0.30 * ramp(d, 0.0, 0.05) * math.exp(-max(0.0, d - 0.05) / 0.10)

    def _cards(self, ctx, font, t):
        dim_on = ramp(t, T_REB + 0.05, T_REB + 0.35) * (1.0 - ramp(t, T_HIT + 0.15, T_HIT + 0.45))
        for slot in range(3):
            kk = self.order[slot]
            win = kk == self.kind
            cx = CARD_X[slot]
            a = 1.0 - 0.58 * dim_on * (0.0 if win else 1.0)
            x0, y0 = cx - CARD_W / 2, CARD_Y
            rounded_rect(ctx, x0, y0, CARD_W, CARD_H, 20)
            ctx.set_source_rgba(0.07, 0.05, 0.14, 0.88 * a)
            ctx.fill_preserve()
            ctx.set_source_rgba(*KCOL[kk], a * (0.9 + 0.1 * dim_on))
            ctx.set_line_width(4 + 5 * dim_on * (1.0 if win else 0.0))
            ctx.stroke()
            ctx.new_path()                                                    # symbol plate
            ctx.arc(x0 + 38, y0 + 40, 25, 0, TAU)
            ctx.set_source_rgba(*KCOL[kk], a)
            ctx.fill()
            ctx.new_path()
            symbol(ctx, slot, x0 + 38, y0 + 40, 13)
            ctx.set_source_rgba(*INK, a)
            ctx.fill()
            font.draw(ctx, KINDS[kk], x0 + 82, y0 + 56, 46, CREAM, anchor="l", alpha=a, stroke=INK, stroke_w=7, tracking=1.0)
            ctx.save()                                                        # small icon of the silhouette
            ctx.rectangle(x0, y0, CARD_W, CARD_H)
            ctx.clip()
            ctx.translate(cx, y0 + 108)
            ctx.scale(0.082, 0.082)
            ctx.translate(0, 0 if kk == 2 else -30.0)
            parts, holes = pieces(kk)
            ctx.push_group()
            ctx.set_source_rgba(*KCOL[kk], 1.0)
            for p in parts:
                ctx.new_path()
                p(ctx)
                ctx.fill()
            ctx.set_operator(cairo.OPERATOR_CLEAR)
            for h_ in holes:
                ctx.new_path()
                h_(ctx)
                ctx.fill()
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(0.95 * a)
            ctx.restore()
            if win and dim_on > 0.02:
                glow(ctx, cx, y0 + CARD_H / 2, 190, KCOL[kk], 0.22 * dim_on)
                ctx.new_path()
                ctx.move_to(x0 + CARD_W - 52, y0 + 30)
                ctx.line_to(x0 + CARD_W - 38, y0 + 46)
                ctx.line_to(x0 + CARD_W - 14, y0 + 16)
                ctx.set_source_rgba(0.55, 1.0, 0.65, dim_on)
                ctx.set_line_width(9)
                ctx.stroke()

    def _countdown(self, ctx, font, t):
        if t < 3.0:
            a = 1.0 - ramp(t, 2.75, 3.0)
            cur = 3 - int(t)
            u = t - int(t)
        elif t > T_HIT + 0.2:
            a = ramp(t, T_HIT + 0.2, T_TOTAL - 0.1)               # back to the opening look for the loop
            cur, u = 3, 0.0
        else:
            return
        if a <= 0.01:
            return
        y = 705
        rounded_rect(ctx, HX - 400, y - 126, 800, 150, 24)
        ctx.set_source_rgba(0.04, 0.02, 0.09, 0.78 * a)
        ctx.fill()
        ctx.set_source_rgba(*AMBER, a)
        ctx.set_line_width(4)
        ctx.stroke()
        font.draw(ctx, "PICK NOW", HX - 190, y - 40, 74, CREAM, anchor="c", alpha=a, stroke=INK, stroke_w=8)
        for j, v in enumerate((3, 2, 1)):
            on = v == cur
            sc = 1.0 + (0.28 * (1 - u) ** 2 if on else 0.0)
            ctx.save()
            ctx.translate(HX + 120 + j * 82, y - 40)
            ctx.scale(sc, sc)
            font.draw(ctx, str(v), 0, 0, 84, AMBER if on else MUTED, anchor="c", alpha=a * (1.0 if on else 0.55),
                      stroke=INK, stroke_w=8)
            ctx.restore()
        font.draw(ctx, "COMMENT YOUR PICK", HX, y + 8, 36, AMBER, anchor="c", alpha=a, stroke=INK, stroke_w=7,
                  tracking=1.2)

    def _result(self, ctx, font, t):
        a = ramp(t, T_REB + 0.10, T_REB + 0.35) * (1.0 - ramp(t, T_HIT - 0.05, T_HIT + 0.10))
        if a <= 0.01:
            return
        k = self.kind
        s = 0.85 + 0.15 * ease_out_back(ramp(t, T_REB + 0.1, T_REB + 0.4))
        ctx.save()
        ctx.translate(HX, 650)
        ctx.scale(s, s)
        txt = "IT WAS A " + KINDS[k]
        tw = font.width(txt, 82, 1.0)
        rounded_rect(ctx, -tw / 2 - 34, -72, tw + 68, 104, 22)
        ctx.set_source_rgba(0.04, 0.02, 0.09, 0.80 * a)
        ctx.fill()
        font.draw(ctx, txt, 0, 0, 82, KCOL[k], anchor="c", alpha=a, stroke=INK, stroke_w=10, shadow=4, tracking=1.0)
        ctx.restore()

    def _endcard(self, ctx, font, t):
        a = ramp(t, T_REB + 0.12, T_REB + 0.40) * (1.0 - ramp(t, T_HIT + 0.30, T_HIT + 0.55))
        if a <= 0.01:
            return
        size = min(38.0, 880.0 / font.width(END_TEXT, 1.0))
        chip(ctx, font, END_TEXT, HX, 1522, size, pad=22, h=size * 1.5)

    # ------------------------------------------------------------ sound
    def audio(self):
        rng = np.random.default_rng(self.seed + 4242)
        sr = au.SR
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        bed = np.zeros((n + tail, 2))
        ts_ = np.arange(n) / sr

        def tt(dd):
            return np.arange(int(dd * sr)) / sr

        sim = self.sim
        # a soft air bed and a slow two-note drone that tremolos (no silence under -50 dB)
        air = au.noise(rng, self.dur, 400, 3000)[:n] * (0.5 + 0.5 * np.sin(TAU * ts_ / 7.0 + 0.7)) * 0.05
        hiss = au.noise(rng, self.dur, 4000, 9000)[:n] * 0.025
        drone = (np.sin(TAU * 110.0 * ts_) + 0.6 * np.sin(TAU * 164.8 * ts_)) * (0.6 + 0.4 * np.sin(TAU * ts_ / 3.3)) * 0.05
        w = air + hiss + drone
        bed[:n, 0] += w
        bed[:n, 1] += np.roll(w, 300)
        # countdown 3-2-1
        for j, m_ in enumerate((69, 72, 76)):
            t_ = tt(0.45)
            x = np.sin(TAU * au.hz(m_) * t_) * np.minimum(1.0, t_ / 0.004) * np.exp(-t_ / 0.12)
            au.place(fx, x, 0.0 + j, 0.24)

        # pings: one voice per shard (2 to 6 kHz, fixed per shard); in the rebuild they swell into the contact, reversed
        pf = 2000.0 + 4000.0 * np.array([hash1(i, 2.0) for i in range(self.n)])

        def ping(f, rev, dur=0.28):
            t_ = tt(dur)
            x = np.sin(TAU * f * t_) + 0.3 * np.sin(TAU * f * 2.01 * t_)
            if rev:
                x = x * np.exp(-(dur - t_) / 0.07) * np.minimum(1.0, (dur - t_) / 0.004)
            else:
                x = x * np.minimum(1.0, t_ / 0.002) * np.exp(-t_ / 0.06)
            return x

        last = {}
        for (te, i, st) in sorted(sim["events"], key=lambda e: e[0]):
            tv = self.t_reverse(te)
            if tv >= T_REB - 0.05:
                continue
            if tv - last.get(i, -9.0) < 0.07:
                continue
            last[i] = tv
            pan = clamp((float(pose(sim, te)[i, 0]) - 475.0) / 450.0, -0.8, 0.8)
            au.place(fx, ping(pf[i], True), max(0.0, tv - 0.28), 0.05 + 0.17 * st, pan)
        # a soft knock that speeds up with the rebuild, so there is always a readable beat
        tk = 3.2
        while tk < T_REB - 0.3:
            u = clamp((tk - 3.0) / (T_REB - 3.0))
            au.place(fx, au.marimba(au.hz(57 + 5 * u), 0.35), tk, 0.18 + 0.10 * u)
            tk += lerp(1.25, 0.40, u ** 1.2)
        # reversed swell into the assembly: noise sweep rising, then the reversed burst of the strike
        dur_r = 2.4
        sw = au.swept_noise(rng, dur_r, 600, 7000)
        au.place(fx, sw * (np.arange(len(sw)) / len(sw)) ** 2.2, T_REB - dur_r, 0.30)
        rise = au.glide(au.hz(48), au.hz(84), dur_r, decay=3.0, harm=0.25)
        au.place(fx, rise * (np.arange(len(rise)) / len(rise)) ** 1.5, T_REB - dur_r, 0.14)
        bt = tt(0.7)
        burst = au.noise(rng, 0.7, 1500, 11000) * np.exp(-bt * 6.0)
        au.place(fx, burst[::-1], T_REB - 0.7, 0.38)
        # assembly tone: one clean sine with a slow tremolo
        ta = tt(1.0)
        tone = (np.sin(TAU * 659.25 * ta) + 0.5 * np.sin(TAU * 987.77 * ta)) * (1 + 0.28 * np.sin(TAU * 5.0 * ta))
        tone *= np.minimum(1.0, ta / 0.01) * np.exp(-ta / 0.7)
        au.place(fx, tone, T_REB, 0.20)
        # hammer strike and the forward shatter (high-passed noise burst plus every ping forward)
        au.place(fx, au.marimba(au.hz(43), 0.3), T_HIT - 0.01, 0.55)
        au.place(fx, au.tick(1900.0, 0.05), T_HIT, 0.5)
        bt = tt(0.6)
        au.place(fx, au.noise(rng, 0.6, 2500, 13000) * np.exp(-bt * 7.0), T_HIT, 0.55)
        au.place(fx, au.boom(rng, 0.9, 80.0, 38.0), T_HIT, 0.50)
        last = {}
        for (te, i, st) in sim["events"]:
            tv = T_HIT + (T_TOTAL - T_HIT) * (1.0 - math.sqrt(max(0.0, 1.0 - te / self.T)))
            if tv - last.get(i, -9.0) < 0.05:
                continue
            last[i] = tv
            pan = clamp((float(pose(sim, te)[i, 0]) - 475.0) / 450.0, -0.8, 0.8)
            au.place(fx, ping(pf[i], False), tv, 0.04 + 0.14 * st, pan)
        mix_ = au.reverb(fx, rng, decay=1.1, wet=0.10) + bed
        out_ = mix_[:n].copy()
        out_[:tail] += mix_[n:]
        out_ = au.eq(out_, self.EQ)
        out_[:int(0.05 * sr)] *= np.linspace(0, 1, int(0.05 * sr))[:, None]
        out_[-int(0.30 * sr):] *= np.linspace(1, 0, int(0.30 * sr))[:, None]
        pk = np.abs(out_).max()
        return np.tanh(out_ * (self.DRIVE / max(1e-9, pk)))

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        st = self.stats
        title = "What Was It Before It Broke? Pick One 🔨"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "What was it before it broke: a key, a bulb or a vase? Pick one before the shards fly back together, "
                "then check the last seconds.\n\n"
                "We drew one object, cracked it into about "
                f"{self.n} shards around the point where the hammer lands, and simulated every shard as a rigid piece: "
                "gravity, spin, friction, bounces off the floor and walls, and collisions between shards. The rebuild is "
                "that recording played backward, so the shards lift, fly back and lock together, outer pieces first, and it "
                "speeds up as it goes. The whole object appears only in the last 1.5 seconds. Then the hammer strikes and "
                "the same recording plays forward, so the video loops exactly. Slow-motion gravity: a toy recording, not "
                "real glass physics.\n"
                f"Shapes, picture and sound are generated by code (seed {self.seed}), with no footage, samples or stock media.\n"
                "Got it right? +1 streak. Which one did you pick?\n"
                "#simulation #satisfying #shatter"),
            tags=["simulation", "shatter reverse", "what was it", "guess the object", "reverse shatter", "prediction",
                  "satisfying", "made with code", "generative animation", "key bulb vase"],
            pinned_comment=("Key, bulb or vase: what was it before it broke? Pick before the shards lock together. "
                            "The rebuild is the shatter recording played backward."),
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of an abstract object breaking; a toy "
                                              "physics recording that could not be mistaken for real footage, people, "
                                              "places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / FPS, 2),
            drama=st,
        )
