"""MERGE FORM: how many merges until it reaches the final form? (lane 6, merge to final form.)

A patch of space. Rules, all on screen from frame 0:
  * six forms, smallest to largest: DUST, ROCK, MOON, PLANET, STAR, BLACK HOLE;
  * the same form pulls the same form (each body is drawn toward the nearest body of its own form);
  * two of a kind that touch merge into one body a tier up; different forms bounce off each other;
  * new dust drifts in from the rim, faster and faster, and the view pulls back as things grow;
  * the merge counter counts every merge; the run ends the moment the first BLACK HOLE forms;
  * pick the band (A, B, C, D) that holds the final merge count; a band is crossed out when the counter passes its top edge.
Toy model, not astrophysics. Deterministic per seed, so make.py can score many seeds without rendering.
"""
import math
import os

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, ease_in, ease_out, ease_out_back, glow, hex_rgb, lighten, mix,
                        polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30
T_TOTAL = 25.0                  # seconds; frame 0 is the clean loop point
T_COL = 23.5                    # the final form: the answer exists only in the last 1.5 s
RW0, RW1, TREF = 9.5, 26.0, 23.0     # world radius of the view at the start, at the end, and the time scale
DT = 1.0 / 120.0
NT = 6
TIERS = ("DUST", "ROCK", "MOON", "PLANET", "STAR", "BLACK HOLE")
RAD = [1.0 * 1.45 ** k for k in range(NT)]
MASS = [2.0 ** k for k in range(NT)]
ATTR, DRAG = 2.0, 0.9           # pull toward the nearest body of the same form; drag
T_MAX = 38.0
BANDS = ("A", "B", "C", "D")


# ====================================================================== simulation
def rw_at(tau):
    return RW0 + (RW1 - RW0) * min(1.15, max(0.0, tau / TREF)) ** 1.4


def edges_of(seed):
    r = np.random.default_rng(seed * 104729 + 5)
    e0 = int(r.integers(26, 50))
    return (e0, e0 + 8, e0 + 16)


def band_of(M, ed):
    return sum(1 for e in ed if M >= e)


def simulate(seed, record=False):
    rng = np.random.default_rng(seed * 6007 + 31)
    sp_t, t = [], 0.6
    for i in range(140):                                   # dust arrives faster and faster
        sp_t.append(t)
        t += max(0.12, (0.60 - 0.30 * min(1.0, t / 20.0)) * rng.uniform(0.7, 1.3))
    ang0 = rng.uniform(0, 2 * math.pi, 400)
    spd0 = rng.uniform(1.0, 3.0, 400)
    x = np.zeros((0, 2))
    v = np.zeros((0, 2))
    tier = np.zeros(0, int)
    ids = np.zeros(0, int)
    births, deaths, tier_of = [], [], []
    merges, spawns = [], []

    def add(px, py, vx, vy, k, step):
        nonlocal x, v, tier, ids
        x = np.vstack([x, [px, py]])
        v = np.vstack([v, [vx, vy]])
        tier = np.append(tier, k)
        ids = np.append(ids, len(births))
        births.append(step)
        deaths.append(-1)
        tier_of.append(k)
        return len(births) - 1

    placed = []
    for k in (0, 0, 0, 0, 0, 0, 0, 0, 1):                  # first nine bodies, no overlaps
        for _ in range(200):
            rr = 0.70 * RW0 * math.sqrt(rng.random())
            a = rng.uniform(0, 2 * math.pi)
            p = (rr * math.cos(a), rr * math.sin(a))
            if all(math.hypot(p[0] - q[0], p[1] - q[1]) > RAD[k] + RAD[kk] + 0.6 for q, kk in placed):
                break
        placed.append((p, k))
        a = rng.uniform(0, 2 * math.pi)
        s = rng.uniform(1.0, 3.0)
        add(p[0], p[1], s * math.cos(a), s * math.sin(a), k, 0)
    nsp, step = 0, 0
    traj = []
    t_bh = None
    radv = np.array(RAD)
    massv = np.array(MASS)
    S = int(T_MAX / DT)
    while step < S:
        tt = step * DT
        rw = rw_at(tt)
        while nsp < len(sp_t) and sp_t[nsp] <= tt:
            a = ang0[nsp]
            px, py = 0.93 * rw * math.cos(a), 0.93 * rw * math.sin(a)
            if len(tier) == 0 or np.min(np.hypot(x[:, 0] - px, x[:, 1] - py) - (1.0 + radv[tier])) > 0.1:
                i = add(px, py, -spd0[nsp] * math.cos(a) * 1.5, -spd0[nsp] * math.sin(a) * 1.5, 0, step)
                spawns.append((tt, i))
                nsp += 1
            else:
                sp_t[nsp] += 0.05                          # wait a moment for room
                break
        n = len(tier)
        r = radv[tier]
        m = massv[tier]
        if record:
            traj.append((ids.copy(), x.copy()))
        d = x[None, :, :] - x[:, None, :]
        dist = np.hypot(d[..., 0], d[..., 1]) + np.eye(n) * 1e9
        same = tier[:, None] == tier[None, :]
        dm = np.where(same, dist, 1e18)
        j = dm.argmin(axis=1)
        dj = dm[np.arange(n), j]
        has = dj < 1e17
        acc = np.zeros((n, 2))
        vec = x[j] - x
        acc[has] = ATTR * vec[has] / np.maximum(dj[has], 1e-6)[:, None]
        v = v * (1 - DRAG * DT) + acc * DT
        sp = np.hypot(v[:, 0], v[:, 1])
        v = v * np.minimum(1.0, 14.0 / np.maximum(sp, 1e-9))[:, None]
        x = x + v * DT
        rho = np.hypot(x[:, 0], x[:, 1])                   # the rim
        lim = 0.97 * rw - r
        out = rho > lim
        if out.any():
            nx = x[out] / rho[out][:, None]
            x[out] -= nx * (rho[out] - lim[out])[:, None]
            vn = (v[out] * nx).sum(1)
            bad = vn > 0
            v[np.nonzero(out)[0][bad]] -= 1.5 * vn[bad][:, None] * nx[bad]
        d = x[None, :, :] - x[:, None, :]                  # contacts: same form merges, others bounce
        dist = np.hypot(d[..., 0], d[..., 1]) + np.eye(n) * 1e9
        ov = (r[:, None] + r[None, :]) - dist
        iu = np.argwhere(np.triu(ov > 0, 1))
        used = set()
        newb = []
        for a_, b_ in sorted(iu.tolist(), key=lambda p: -ov[p[0], p[1]]):
            if a_ in used or b_ in used:
                continue
            if tier[a_] == tier[b_]:
                used.add(a_)
                used.add(b_)
                mm = m[a_] + m[b_]
                newb.append((a_, b_, (x[a_] * m[a_] + x[b_] * m[b_]) / mm, (v[a_] * m[a_] + v[b_] * m[b_]) / mm,
                             int(tier[a_]) + 1))
            else:
                nvec = (x[b_] - x[a_]) / max(dist[a_, b_], 1e-9)
                o = ov[a_, b_]
                x[a_] -= nvec * o * m[b_] / (m[a_] + m[b_])
                x[b_] += nvec * o * m[a_] / (m[a_] + m[b_])
                vn = float(((v[b_] - v[a_]) * nvec).sum())
                if vn < 0:
                    J = -(1 + 0.4) * vn / (1 / m[a_] + 1 / m[b_])
                    v[a_] -= J / m[a_] * nvec
                    v[b_] += J / m[b_] * nvec
        if newb:
            kill = np.zeros(n, bool)
            for a_, b_, c, vv, kn in newb:
                kill[a_] = kill[b_] = True
                deaths[int(ids[a_])] = step + 1
                deaths[int(ids[b_])] = step + 1
            keep = ~kill
            x, v, tier, ids = x[keep], v[keep], tier[keep], ids[keep]
            for a_, b_, c, vv, kn in newb:
                i = add(c[0], c[1], vv[0], vv[1], kn, step + 1)
                merges.append(dict(t=(step + 1) * DT, x=float(c[0]), y=float(c[1]), tier=kn, id=i))
                if kn == NT - 1 and t_bh is None:
                    t_bh = (step + 1) * DT
        step += 1
        if t_bh is not None:
            if record:
                traj.append((ids.copy(), x.copy()))
            break
    res = dict(merges=merges, spawns=spawns, t_bh=t_bh, M=len(merges) if t_bh else None, edges=edges_of(seed))
    if record:
        res.update(traj=traj, births=births, deaths=deaths, tier_of=tier_of)
    return res


# ====================================================================== drama / seed gates
def drama(seed, res=None):
    res = res or simulate(seed)
    st = dict(seed=seed, valid=False, score=-100.0)
    if res["t_bh"] is None:
        st["note"] = "no black hole"
        return st
    ed = res["edges"]
    M = res["M"]
    w = T_COL / res["t_bh"]
    tm = [m_["t"] * w for m_ in res["merges"]]
    gaps = [b - a for a, b in zip([0.0] + tm, tm)]
    gmax = max(gaps[:-1]) if len(gaps) > 1 else 99.0
    n1 = sum(1 for t in tm if t < T_COL / 3)
    n3 = sum(1 for t in tm if t > 2 * T_COL / 3)
    edge_d = min(abs(M - e) for e in ed)
    near = min(range(3), key=lambda i: abs(M - ed[i]))
    band = band_of(M, ed)
    gates = dict(
        early_merge=bool(tm[0] <= 2.2),
        warp_ok=bool(0.85 <= w <= 1.2),
        merges_range=bool(34 <= M <= 72),
        steady=bool(gmax <= 1.7),
        escalates=bool(n3 >= n1 + 2),
        near_edge=bool(edge_d <= 1),
        not_first_band=bool(band >= 1),
    )
    ok = all(gates.values())
    score = (100.0 if ok else 0.0) + 8.0 * sum(gates.values())
    score += -5.0 * edge_d + 0.5 * (n3 - n1) - 3.0 * max(0.0, gmax - 1.2) - 20.0 * abs(w - 1.0)
    runner = BANDS[near] if band == near + 1 else BANDS[near + 1]
    return dict(seed=seed, score=round(float(score), 2), valid=bool(ok), gates={k: bool(v) for k, v in gates.items()},
                winner=BANDS[band], answer_merges=int(M), edges=[int(e) for e in ed], runner_up=runner,
                nearest_edge_merges=int(edge_d), first_merge_s=round(float(tm[0]), 2), max_gap_s=round(float(gmax), 2),
                merges_first_third=int(n1), merges_last_third=int(n3), warp=round(float(w), 3),
                black_hole_sim_s=round(float(res["t_bh"]), 2))


def score_seed(seed):
    return drama(seed)


def beat_times():
    """Kick times of audio.tension_bed (same tempo ramp), so the picture can pulse on the beat."""
    out, t, beat = [], 0.0, 0
    while t < T_COL - 0.35:
        k = t / T_COL
        step = 60.0 / (96.0 + 36.0 * k ** 1.5) / 2
        if beat % 2 == 0:
            out.append(t)
        t += step
        beat += 1
    return np.array(out)


# ====================================================================== look
BAND_COL = [hex_rgb("#FF7A9C"), hex_rgb("#7CE38B"), hex_rgb("#4CC9FF"), hex_rgb("#FFD23F")]    # circle, square, triangle, diamond
TIER_COL = [hex_rgb("#E6C27E"), hex_rgb("#D08C62"), hex_rgb("#C9D6F2"), hex_rgb("#4FD0E0"), hex_rgb("#FFC93C"),
            hex_rgb("#C07CFF")]
INK = hex_rgb("#070614")
CREAM = hex_rgb("#F6F0E4")
MUTED = hex_rgb("#8F8AB0")
AMBER = hex_rgb("#FFC23D")
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
HX, AY, AR = 475.0, 950.0, 316.0
Q1, Q2 = "HOW MANY MERGES", "UNTIL THE FINAL FORM?"
RULE = "PICK THE BAND OF THE FINAL MERGE COUNT"
SUBTXT = "SAME FORM PULLS SAME FORM   •   2 OF A KIND = 1 TIER UP"
WHY = "WHY: SAME FORMS PULL TOGETHER, AND TWO OF A KIND MAKE ONE TIER UP"
END_TEXT = "Got it right? +1 streak. Subscribe to keep it."
YB = 1366.0


def hash1(a, b=0.0):
    v = math.sin(a * 12.9898 + b * 78.233) * 43758.5453
    return v - math.floor(v)


def symbol(ctx, k, x, y, s):
    if k == 0:
        ctx.new_sub_path()
        ctx.arc(x, y, s, 0, TAU)
    elif k == 1:
        ctx.rectangle(x - s * 0.86, y - s * 0.86, s * 1.72, s * 1.72)
    elif k == 2:
        polygon(ctx, [(x, y - s * 1.15), (x + s * 1.1, y + s * 0.8), (x - s * 1.1, y + s * 0.8)])
    else:
        polygon(ctx, [(x, y - s * 1.2), (x + s * 0.95, y), (x, y + s * 1.2), (x - s * 0.95, y)])


def chip(ctx, font, txt, cx, y, size, pad=20, h=None, bg=CREAM, fg=INK):
    tw = font.width(txt, size, 1.2)
    h = h or size * 1.42
    rounded_rect(ctx, cx - tw / 2 - pad, y - h * 0.76, tw + 2 * pad, h, 12)
    ctx.set_source_rgba(*bg, 1)
    ctx.fill()
    font.draw(ctx, txt, cx, y, size, fg, anchor="c", tracking=1.2)


def ellipse_path(ctx, x, y, rx, ry, tilt, a0, a1):
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(tilt)
    ctx.scale(1.0, ry / rx)
    ctx.new_sub_path()
    ctx.arc(0, 0, rx, a0, a1)
    ctx.restore()


def add_glow(ctx, x, y, r, col, a):
    ctx.set_operator(cairo.OPERATOR_ADD)
    glow(ctx, x, y, r, col, a)
    ctx.set_operator(cairo.OPERATOR_OVER)


def draw_body(ctx, k, x, y, r, rot, uid, al=1.0):
    """One form, drawn from a radius in pixels. Every tier has its own silhouette, not only its own color."""
    if r < 1.0 or al <= 0.01:
        return
    if k == 0:                                           # dust: a bright grain with specks around it
        g = cairo.RadialGradient(x - r * 0.3, y - r * 0.3, r * 0.1, x, y, r)
        g.add_color_stop_rgba(0, 1.0, 0.94, 0.74, al)
        g.add_color_stop_rgba(1, 0.74, 0.52, 0.30, al)
        ctx.new_path()
        ctx.arc(x, y, r, 0, TAU)
        ctx.set_source(g)
        ctx.fill()
        for j in range(5):
            a = rot * 1.4 + j * TAU / 5 + hash1(uid, j)
            d = r * (1.5 + 0.45 * hash1(uid, j + 7))
            ctx.new_path()
            ctx.arc(x + math.cos(a) * d, y + math.sin(a) * d, max(1.2, r * 0.13), 0, TAU)
            ctx.set_source_rgba(0.95, 0.80, 0.55, 0.85 * al)
            ctx.fill()
    elif k == 1:                                         # rock: a rough lump
        pts = []
        for j in range(9):
            a = rot + j * TAU / 9
            rr = r * (0.80 + 0.32 * hash1(uid, j + 3))
            pts.append((x + math.cos(a) * rr, y + math.sin(a) * rr))
        ctx.new_path()
        polygon(ctx, pts)
        g = cairo.RadialGradient(x - r * 0.35, y - r * 0.35, r * 0.1, x, y, r * 1.1)
        g.add_color_stop_rgba(0, 0.88, 0.64, 0.48, al)
        g.add_color_stop_rgba(1, 0.38, 0.25, 0.22, al)
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_source_rgba(0.16, 0.09, 0.10, 0.9 * al)
        ctx.set_line_width(max(1.5, r * 0.07))
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.stroke()
        for j in (0, 1):
            ctx.new_path()
            ctx.move_to(pts[j * 3][0] * 0.5 + x * 0.5, pts[j * 3][1] * 0.5 + y * 0.5)
            ctx.line_to(pts[j * 3 + 4][0] * 0.4 + x * 0.6, pts[j * 3 + 4][1] * 0.4 + y * 0.6)
            ctx.set_source_rgba(0.16, 0.09, 0.10, 0.6 * al)
            ctx.set_line_width(max(1.0, r * 0.05))
            ctx.stroke()
    elif k == 2:                                         # moon: pale sphere with craters
        g = cairo.RadialGradient(x - r * 0.35, y - r * 0.35, r * 0.1, x, y, r * 1.05)
        g.add_color_stop_rgba(0, 0.97, 0.98, 1.0, al)
        g.add_color_stop_rgba(1, 0.50, 0.54, 0.68, al)
        ctx.new_path()
        ctx.arc(x, y, r, 0, TAU)
        ctx.set_source(g)
        ctx.fill()
        for ox, oy, rr in ((-0.35, -0.2, 0.25), (0.3, 0.1, 0.20), (-0.05, 0.45, 0.16), (0.35, -0.45, 0.12)):
            a = rot * 0.25
            cx_ = x + (ox * math.cos(a) - oy * math.sin(a)) * r
            cy_ = y + (ox * math.sin(a) + oy * math.cos(a)) * r
            ctx.new_path()
            ctx.arc(cx_, cy_, r * rr, 0, TAU)
            ctx.set_source_rgba(0.30, 0.33, 0.48, 0.38 * al)
            ctx.fill()
    elif k == 3:                                         # planet: banded sphere with a ring
        tilt = -0.38
        ellipse_path(ctx, x, y, r * 1.85, r * 0.55, tilt, math.pi, TAU)
        ctx.set_source_rgba(0.96, 0.84, 0.58, 0.85 * al)
        ctx.set_line_width(max(2.0, r * 0.17))
        ctx.stroke()
        g = cairo.RadialGradient(x - r * 0.35, y - r * 0.4, r * 0.1, x, y, r * 1.05)
        g.add_color_stop_rgba(0, 0.62, 0.97, 1.0, al)
        g.add_color_stop_rgba(1, 0.08, 0.34, 0.60, al)
        ctx.new_path()
        ctx.arc(x, y, r, 0, TAU)
        ctx.set_source(g)
        ctx.fill()
        ctx.save()
        ctx.new_path()
        ctx.arc(x, y, r, 0, TAU)
        ctx.clip()
        for off in (-0.35, 0.12, 0.5):
            ctx.new_path()
            ctx.rectangle(x - r, y + off * r - r * 0.08, 2 * r, r * 0.16)
            ctx.set_source_rgba(0.05, 0.25, 0.45, 0.28 * al)
            ctx.fill()
        ctx.restore()
        ellipse_path(ctx, x, y, r * 1.85, r * 0.55, tilt, 0.0, math.pi)
        ctx.set_source_rgba(0.99, 0.90, 0.66, 0.95 * al)
        ctx.set_line_width(max(2.0, r * 0.17))
        ctx.stroke()
    elif k == 4:                                         # star: hot core, corona and spikes
        add_glow(ctx, x, y, r * 3.0, (1.0, 0.62, 0.15), 0.55 * al)
        for j in range(8):
            a = rot * 0.6 + j * TAU / 8
            ln = r * (2.1 if j % 2 == 0 else 1.55)
            ctx.new_path()
            polygon(ctx, [(x + math.cos(a - 0.11) * r * 0.9, y + math.sin(a - 0.11) * r * 0.9),
                          (x + math.cos(a) * ln, y + math.sin(a) * ln),
                          (x + math.cos(a + 0.11) * r * 0.9, y + math.sin(a + 0.11) * r * 0.9)])
            ctx.set_source_rgba(1.0, 0.82, 0.30, 0.9 * al)
            ctx.fill()
        g = cairo.RadialGradient(x, y, 0, x, y, r)
        g.add_color_stop_rgba(0, 1.0, 1.0, 0.92, al)
        g.add_color_stop_rgba(0.6, 1.0, 0.86, 0.35, al)
        g.add_color_stop_rgba(1, 1.0, 0.60, 0.12, al)
        ctx.new_path()
        ctx.arc(x, y, r, 0, TAU)
        ctx.set_source(g)
        ctx.fill()
    else:                                                # black hole: dark disc, photon ring, tilted accretion disc
        add_glow(ctx, x, y, r * 3.6, (0.55, 0.28, 1.0), 0.40 * al)
        tilt = 0.30
        ellipse_path(ctx, x, y, r * 2.35, r * 0.62, tilt, math.pi, TAU)
        ctx.set_source_rgba(1.0, 0.58, 0.18, 0.80 * al)
        ctx.set_line_width(max(3.0, r * 0.30))
        ctx.stroke()
        ctx.new_path()
        ctx.arc(x, y, r, 0, TAU)
        ctx.set_source_rgba(0.0, 0.0, 0.02, al)
        ctx.fill_preserve()
        ctx.set_source_rgba(1.0, 0.92, 0.75, 0.95 * al)
        ctx.set_line_width(max(2.0, r * 0.08))
        ctx.stroke()
        ellipse_path(ctx, x, y, r * 2.35, r * 0.62, tilt, 0.0, math.pi)
        ctx.set_source_rgba(1.0, 0.80, 0.42, 0.95 * al)
        ctx.set_line_width(max(3.0, r * 0.30))
        ctx.stroke()
        ellipse_path(ctx, x, y, r * 1.55, r * 0.40, tilt, 0.0, math.pi)
        ctx.set_source_rgba(1.0, 0.97, 0.88, 0.9 * al)
        ctx.set_line_width(max(2.0, r * 0.10))
        ctx.stroke()


def make_background():
    yy = np.arange(H, dtype=np.float32)
    stops = ((0, (0.020, 0.016, 0.060)), (700, (0.050, 0.030, 0.120)), (1000, (0.075, 0.040, 0.150)),
             (1500, (0.040, 0.025, 0.100)), (1920, (0.015, 0.012, 0.045)))
    img = np.zeros((H, 1, 3), np.float32)
    for c in range(3):
        img[:, 0, c] = np.interp(yy, [a for a, _ in stops], [b[c] for _, b in stops])
    img = np.repeat(img, W, axis=1)
    img += np.random.default_rng(3).normal(0, 0.003, (H, W, 1)).astype(np.float32)
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    del img
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    ctx = cairo.Context(surf)
    for cx, cy, rr, col in ((220, 760, 380, (0.35, 0.15, 0.55)), (800, 1130, 420, (0.15, 0.25, 0.55))):
        g = cairo.RadialGradient(cx, cy, 0, cx, cy, rr)
        g.add_color_stop_rgba(0, *col, 0.20)
        g.add_color_stop_rgba(1, *col, 0.0)
        ctx.set_source(g)
        ctx.rectangle(0, 0, W, H)
        ctx.fill()
    return surf, buf


def make_header(font):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, 400)
    ctx = cairo.Context(surf)
    s1 = min(118.0, 860.0 / font.width(Q2, 1.0))
    font.draw(ctx, Q1, HX, 124, s1, CREAM, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    font.draw(ctx, Q2, HX, 124 + s1 * 1.1, s1, CREAM, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    chip(ctx, font, RULE, HX, 124 + s1 * 1.1 + 90, 38, pad=22, bg=AMBER)
    font.draw(ctx, SUBTXT, HX, 124 + s1 * 1.1 + 142, 25, CREAM, anchor="c", stroke=INK, stroke_w=6, tracking=0.6)
    surf.flush()
    return surf


# ====================================================================== the scene
class Scene:
    W, H, FPS = W, H, FPS
    EQ = ((50, -6.0), (120, -3.0), (300, -0.5), (900, 0.0), (3000, 1.5), (8000, 0.5), (16000, -4.0))
    DRIVE = 3.0
    CEILING_DB = -3.4
    HEAD_Y = 150

    def __init__(self, seed):
        self.seed = seed
        res = simulate(seed, record=True)
        if res["t_bh"] is None:
            raise ValueError(f"seed {seed}: no black hole forms")
        self.stats = drama(seed, res)
        self.ed = res["edges"]
        self.M = res["M"]
        self.band = band_of(self.M, self.ed)
        self.w = T_COL / res["t_bh"]
        self.nframes = int(round(T_TOTAL * FPS))
        self.dur = self.nframes / FPS
        traj = res["traj"]
        nid = len(res["births"])
        S = len(traj)
        X = np.full((S, nid, 2), np.nan, np.float32)
        for s, (ids, xy) in enumerate(traj):
            X[s, ids] = xy
        self.X = X
        self.S = S
        self.tier_of = res["tier_of"]
        self.birth_t = np.array(res["births"]) * DT * self.w
        self.mg = res["merges"]
        self.mt = np.array([m_["t"] * self.w for m_ in self.mg])
        self.bh_id = self.mg[-1]["id"]
        first = {}
        for i, k in enumerate(self.tier_of):
            first[k] = min(first.get(k, 1e9), self.birth_t[i])
        self.first_t = [first.get(k, 1e9) for k in range(NT)]
        self.edge_t = [float(self.mt[e - 1]) if e <= self.M else None for e in self.ed]
        self.spawn_t = [(s_[0] * self.w, s_[1]) for s_ in res["spawns"]]
        self.beats = beat_times()
        self._res = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_res"] = None
        return d

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            font = Font(FONT_PATH)
            bg, keep = make_background()
            rng = np.random.default_rng(11)
            stars = [(rng.uniform(0, W), rng.uniform(0, H), rng.uniform(0.5, 2.6), rng.uniform(0, 6.28)) for _ in range(240)]
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep, font=font,
                             head=make_header(font), stars=stars)
        return self._res

    def frame(self, f):
        res = self._resources()
        ctx = res["ctx"]
        ctx.save()
        t = f / FPS
        a = ramp(t - T_COL, 1.0, 1.45)
        if a <= 0.0:
            self.draw(ctx, f, res)
        else:
            ctx.push_group()
            self.draw(ctx, f, res)
            ctx.pop_group_to_source()
            ctx.paint()
            ctx.push_group()
            self.draw(ctx, 0, res)
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(a)
        ctx.restore()
        res["surf"].flush()
        return bytes(res["surf"].get_data())

    # ------------------------------------------------------------ helpers
    def tau(self, t):
        return min(t, T_COL) / self.w

    def nmerge(self, t):
        return int(np.searchsorted(self.mt, t, side="right"))

    def scale(self, t):
        return AR / rw_at(self.tau(t))

    def shake(self, t):
        sx = sy = 0.0
        k0 = int(np.searchsorted(self.mt, t - 0.4, side="left"))
        for i in range(k0, self.nmerge(t)):
            a = t - self.mt[i]
            tier = self.mg[i]["tier"]
            big = 1.0 if tier < 5 else 3.2
            amp = big * (0.5 + 1.1 * tier ** 1.3) * math.exp(-a / 0.11) * (1.0 + 0.6 * self.mt[i] / T_COL)
            sx += amp * math.sin(i * 7.1 + a * 90)
            sy += amp * math.cos(i * 3.3 + a * 83)
        j = int(np.searchsorted(self.beats, t, side="right")) - 1
        if j >= 0 and t < T_COL:
            a = t - self.beats[j]
            amp = 5.0 * (t / T_COL) ** 2.5 * math.exp(-a / 0.09)
            sx += amp * math.sin(j * 5.1)
            sy += amp * math.cos(j * 2.9 + 1.0)
        return sx, sy

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, f, res):
        t = f / FPS
        d = t - T_COL
        font = res["font"]
        prog = clamp(self.nmerge(t) / self.M)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(res["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        sc = self.scale(t)
        shx, shy = self.shake(t)
        self._space(ctx, res, t, sc, prog, shx, shy)
        self._bodies(ctx, t, d, sc, shx, shy)
        self._fx(ctx, font, t, sc, shx, shy)
        ctx.set_source_surface(res["head"], 0, self.HEAD_Y)
        ctx.paint()
        self._counter(ctx, font, t)
        self._ladder(ctx, font, t)
        self._bands(ctx, font, t)
        self._countdown(ctx, font, t)
        if d >= 0:
            self._result(ctx, d, font)
        self._endcard(ctx, font, d)

    def _space(self, ctx, res, t, sc, prog, shx, shy):
        u = min(1.0, t / T_COL)
        travel = 14.0 * t + 330.0 * T_COL * u ** 3 / 3.0                # the starfield streams inward, faster and faster
        speed = 14.0 + 330.0 * u ** 2
        for (x, y, r, ph) in res["stars"]:
            dx, dy = x - HX, y - AY
            rho = math.hypot(dx, dy) + 1e-6
            rho2 = (rho - travel * (0.5 + 0.5 * hash1(x, y))) % 1250.0
            ux, uy = dx / rho, dy / rho
            px, py = HX + ux * rho2 + shx * 0.4, AY + uy * rho2 + shy * 0.4
            tw = 0.55 + 0.45 * math.sin(ph + t * (1.5 + prog * 3.0))
            ln = 2.0 + 0.045 * speed * (0.5 + hash1(x, y))
            ctx.new_path()
            ctx.move_to(px, py)
            ctx.line_to(px + ux * ln, py + uy * ln)
            ctx.set_source_rgba(0.85, 0.88, 1.0, (0.25 + 0.5 * prog) * tw)
            ctx.set_line_width(r * (0.8 + 0.5 * prog))
            ctx.stroke()
        pulse = 0.0
        j = int(np.searchsorted(self.beats, t, side="right")) - 1
        if j >= 0 and t < T_COL:
            pulse = math.exp(-(t - self.beats[j]) / 0.13) * (0.06 + 0.34 * u ** 1.2)
        col = mix((0.35, 0.20, 0.75), (1.0, 0.45, 0.20), prog)
        add_glow(ctx, HX + shx, AY + shy, AR * (1.3 + 0.9 * prog), col, 0.05 + 0.40 * prog ** 1.5 + pulse)
        ctx.new_path()
        ctx.arc(HX + shx, AY + shy, AR * 0.97, 0, TAU)
        ctx.set_source_rgba(*mix((0.55, 0.60, 0.90), (1.0, 0.70, 0.35), prog), 0.30 + 0.35 * prog + 1.2 * pulse)
        ctx.set_line_width(3.0 + 3.0 * prog + 14.0 * pulse)
        ctx.set_dash([14, 12])
        ctx.stroke()
        ctx.set_dash([])

    def _bodies(self, ctx, t, d, sc, shx, shy):
        tau = self.tau(t)
        step = int(min(self.S - 1, round(tau / DT)))
        row = self.X[step]
        alive = np.nonzero(~np.isnan(row[:, 0]))[0]
        e = ease_in(d / 0.95) if d > 0 else 0.0
        bx, by = self.X[self.S - 1][self.bh_id]
        for i in sorted(alive.tolist(), key=lambda q: self.tier_of[q]):
            k = self.tier_of[i]
            x, y = float(row[i, 0]), float(row[i, 1])
            age = t - self.birth_t[i]
            pop = 1.0
            if i == self.bh_id:
                pop = (0.5 + 0.75 * ease_out_back(clamp(age / 0.45))) if age < 0.45 else 1.0
                if d > 0:
                    pop *= 1.0 + 0.30 * e
            elif self.birth_t[i] > 0.02 and 0 <= age < 0.30:
                pop = (0.3 + 0.7 * ease_out_back(clamp(age / 0.22))) if k == 0 else 1.0 + 0.28 * math.exp(-age / 0.09)
            al = 1.0
            if d > 0 and i != self.bh_id:                  # after the reveal everything is drawn into the black hole
                dx, dy = x - bx, y - by
                ca, sa = math.cos(e * 2.2), math.sin(e * 2.2)
                x, y = bx + (dx * ca - dy * sa) * (1 - e), by + (dx * sa + dy * ca) * (1 - e)
                pop *= max(0.05, 1 - 0.9 * e)
                al = 1 - 0.6 * e
            draw_body(ctx, k, HX + x * sc + shx, AY + y * sc + shy, RAD[k] * sc * pop,
                      0.6 * t * (1 + 0.3 * k) + i, i, al)

    def _fx(self, ctx, font, t, sc, shx, shy):
        k0 = int(np.searchsorted(self.mt, t - 0.9, side="left"))
        for i in range(k0, self.nmerge(t)):
            m_ = self.mg[i]
            a = t - self.mt[i]
            tier = m_["tier"]
            step = int(min(self.S - 1, round(self.mt[i] / self.w / DT)))
            pos = self.X[step, m_["id"]]
            if np.isnan(pos[0]):
                pos = (m_["x"], m_["y"])
            if tier == NT - 1:
                pos = self.X[self.S - 1][self.bh_id]
            x, y = HX + float(pos[0]) * sc + shx, AY + float(pos[1]) * sc + shy
            r = RAD[tier] * sc
            col = TIER_COL[tier]
            u = a / 0.55
            if u < 1:
                for q in range(2 if tier < 5 else 4):
                    uu = u - 0.14 * q
                    if uu <= 0:
                        continue
                    ctx.new_path()
                    ctx.arc(x, y, r + (46 + 24 * tier) * ease_out(uu) * (1.0 if tier < 5 else 1.6), 0, TAU)
                    ctx.set_source_rgba(*lighten(col, 0.35), (0.9 if tier < 5 else 0.5) * (1 - uu))
                    ctx.set_line_width((4 + tier) * (1 - 0.5 * uu))
                    ctx.stroke()
                ns = 6 + 3 * tier
                for j in range(ns):
                    ang = TAU * (j + hash1(i, j)) / ns
                    sp = (120 + 220 * hash1(i, j + 40)) * (1.0 + 0.55 * tier) ** 0.5
                    ctx.new_path()
                    ctx.move_to(x + math.cos(ang) * (r + sp * a), y + math.sin(ang) * (r + sp * a))
                    ctx.line_to(x + math.cos(ang) * (r + sp * a + 14 + 4 * tier),
                                y + math.sin(ang) * (r + sp * a + 14 + 4 * tier))
                    ctx.set_source_rgba(*lighten(col, 0.4), 1 - u)
                    ctx.set_line_width(3.5)
                    ctx.stroke()
            if a < 0.25:
                add_glow(ctx, x, y, r * (2.4 + 0.6 * tier), lighten(col, 0.3),
                         0.8 * (1 - a / 0.25) * (1.0 if tier < 5 else 1.4))
            if tier >= 2 and a < 0.9 and abs(self.mt[i] - self.first_t[tier]) < 1e-6:
                al = (1 - (ramp(a, 0.55, 0.9) if tier < 5 else ramp(a, 0.10, 0.20))) * ramp(a, 0.0, 0.08)
                ty = clamp(y - r * 1.4 - 36 - 30 * ease_out(min(1.0, a / 0.4)), 700.0, 1230.0)
                font.draw(ctx, TIERS[tier] + "!", clamp(x, 200.0, 780.0), ty, 52 + 8 * tier, lighten(col, 0.25),
                          anchor="c", alpha=al, stroke=INK, stroke_w=10, shadow=4)
        a = t - T_COL
        if 0 <= a < 0.5:                                   # white flash on the final form
            ctx.rectangle(0, 0, W, H)
            ctx.set_source_rgba(1.0, 0.95, 0.85, 0.55 * (1 - a / 0.5) ** 2)
            ctx.fill()

    def _counter(self, ctx, font, t):
        n = self.nmerge(t)
        txt = str(n)
        sz = 104.0
        w1 = font.width(txt, sz, 1.0)
        w2 = font.width("MERGES", 50.0, 1.0)
        x0 = HX - (w1 + 22 + w2) / 2
        y = 612.0
        pop, col = 1.0, AMBER
        if n:
            age = t - float(self.mt[n - 1])
            if 0 <= age < 0.30:
                pop = 1.0 + 0.16 * math.exp(-age / 0.09)
                col = mix(AMBER, (1, 1, 1), 0.7 * math.exp(-age / 0.1))
        ctx.save()
        ctx.translate(HX, y - 38)
        ctx.scale(pop, pop)
        ctx.translate(-HX, -(y - 38))
        font.draw(ctx, txt, x0 + w1, y, sz, col, anchor="r", stroke=INK, stroke_w=12, shadow=5)
        font.draw(ctx, "MERGES", x0 + w1 + 22, y, 50, CREAM, anchor="l", stroke=INK, stroke_w=8)
        ctx.restore()

    def _ladder(self, ctx, font, t):
        y = 1296.0
        for k in range(NT):
            x = HX + (k - 2.5) * 148.0
            lit = t >= self.first_t[k]
            age = t - self.first_t[k]
            if k < NT - 1:
                ctx.new_path()
                ctx.move_to(x + 56, y - 8)
                ctx.line_to(x + 70, y)
                ctx.line_to(x + 56, y + 8)
                ctx.set_source_rgba(*MUTED, 0.8)
                ctx.set_line_width(3)
                ctx.stroke()
            pulse = 1.0 + (0.28 * math.exp(-age / 0.2) if 0 <= age < 0.8 and k > 0 else 0.0)
            ctx.push_group()
            draw_body(ctx, k, x, y, 19.0 * pulse * (1.0 if k < 4 else 0.85), 0.5 * k, 100 + k)
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(1.0 if lit else 0.30)
            if k == NT - 1:
                ctx.new_path()
                ctx.arc(x, y, 38, 0, TAU)
                ctx.set_source_rgba(*AMBER, 1.0 if lit else 0.9)
                ctx.set_line_width(3.0)
                ctx.set_dash([7, 6])
                ctx.stroke()
                ctx.set_dash([])
            font.draw(ctx, TIERS[k], x, y + 62, 22, CREAM if lit else MUTED, anchor="c", stroke=INK, stroke_w=5,
                      tracking=0.5)

    def _bands(self, ctx, font, t):
        texts = (f"UNDER {self.ed[0]}", f"{self.ed[0]} TO {self.ed[1] - 1}", f"{self.ed[1]} TO {self.ed[2] - 1}",
                 f"{self.ed[2]} AND UP")
        cw, gap, x0 = 207.0, 12.0, 38.0
        for i in range(4):
            x = x0 + i * (cw + gap)
            te = self.edge_t[i] if i < 3 else None
            out = te is not None and t >= te
            win = t >= T_COL + 0.05 and self.band == i
            lose = t >= T_COL + 0.05 and self.band != i
            col = BAND_COL[i]
            a = 0.38 if (out or lose) else 1.0
            s = 1.0
            fl = 0.0
            if te is not None and 0 <= t - te < 0.5:
                fl = 1 - (t - te) / 0.5
            if win:
                s = 1.0 + 0.05 * ease_out_back(clamp((t - T_COL - 0.05) / 0.3))
            ctx.save()
            ctx.translate(x + cw / 2, YB + 48)
            ctx.scale(s, s)
            rounded_rect(ctx, -cw / 2, -44, cw, 90, 16)
            ctx.set_source_rgba(0.04, 0.03, 0.11, 0.9)
            ctx.fill_preserve()
            ctx.set_source_rgba(*mix(col, (1, 1, 1), 0.8 * fl + (0.35 if win else 0.0)), a)
            ctx.set_line_width(4.5 + (4 if win else 0) + 3 * fl)
            ctx.stroke()
            ctx.new_path()
            symbol(ctx, i, -48, -8, 17)
            ctx.set_source_rgba(*col, a)
            ctx.fill()
            font.draw(ctx, BANDS[i], 10, 6, 58, col, anchor="c", alpha=a, stroke=INK, stroke_w=7)
            font.draw(ctx, texts[i], 0, 34, 28, CREAM, anchor="c", alpha=a, tracking=0.8)
            if out and not win:
                ctx.new_path()
                ctx.move_to(-cw / 2 + 12, 36)
                ctx.line_to(cw / 2 - 12, -36)
                ctx.set_source_rgba(1.0, 0.25, 0.2, 0.95)
                ctx.set_line_width(7)
                ctx.stroke()
            ctx.restore()

    def _countdown(self, ctx, font, t):
        if t >= 3.0:
            return
        a = 1.0 - ramp(t, 2.75, 3.0)
        cur = 3 - int(t)
        u = t - int(t)
        if a <= 0.01:
            return
        y = 780
        rounded_rect(ctx, HX - 400, y - 126, 800, 150, 24)
        ctx.set_source_rgba(0.04, 0.02, 0.10, 0.80 * a)
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

    def _result(self, ctx, d, font):
        k1 = ramp(d, 0.20, 0.42)
        a = k1 * (1 - ramp(d, 0.72, 0.98))
        if a <= 0.01:
            return
        k = self.band
        rounded_rect(ctx, HX - 410, 640, 820, 330, 26)
        ctx.set_source_rgba(0.04, 0.02, 0.10, 0.95 * a)
        ctx.fill()
        s = 0.85 + 0.15 * ease_out_back(k1)
        ctx.save()
        ctx.translate(HX, 0)
        ctx.scale(s, s)
        font.draw(ctx, "BLACK HOLE AFTER", 0, 725 / s, 52, CREAM, anchor="c", alpha=a, stroke=INK, stroke_w=8, tracking=1.5)
        font.draw(ctx, f"{self.M} MERGES", 0, 850 / s, 138, AMBER, anchor="c", alpha=a, stroke=INK, stroke_w=13, shadow=5)
        ctx.restore()
        tx = f"BAND {BANDS[k]} WINS"
        tw = font.width(tx, 68)
        ctx.new_path()
        symbol(ctx, k, HX - tw / 2 - 40, 920 - 22, 24)
        ctx.set_source_rgba(*BAND_COL[k], a)
        ctx.fill()
        font.draw(ctx, tx, HX + 20, 940, 68, BAND_COL[k], anchor="c", alpha=a, stroke=INK, stroke_w=9)
        size = min(30.0, 800.0 / font.width(WHY, 1.0))
        ctx.push_group()
        chip(ctx, font, WHY, HX, 1040, size, pad=20, h=size * 1.5)
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(a)

    def _endcard(self, ctx, font, d):
        a = ramp(d, 0.30, 0.55) * (1 - ramp(d, 0.78, 1.02))
        if a <= 0.01:
            return
        size = min(38.0, 880.0 / font.width(END_TEXT, 1.0))
        ctx.push_group()
        chip(ctx, font, END_TEXT, HX, 1522, size, pad=22, h=size * 1.5)
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(a)

    # ------------------------------------------------------------ sound
    def audio(self):
        rng = np.random.default_rng(self.seed + 991)
        sr = au.SR
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        fx[:n] += au.tension_bed(self.dur, T_COL, root=45, seed=self.seed % 97 + 3, gain=0.5)
        for j, m_ in enumerate((60, 64, 72)):                      # countdown 3-2-1
            au.place(fx, au.marimba(au.hz(m_), 0.5), float(j), 0.30)
        pent = (0, 3, 5, 7, 10)
        for i, m_ in enumerate(self.mg[:-1]):
            tm = float(self.mt[i])
            tier = m_["tier"]
            frac = (i + 1) / self.M
            note = 57 + pent[i % 5] + 12 * (tier // 2) + (5 if tier >= 4 else 0)
            g = 0.16 + 0.22 * frac + 0.04 * tier
            pan = clamp((m_["x"] * 12.0) / 300.0, -0.7, 0.7)
            if tier == 1:
                au.place(fx, au.marimba(au.hz(note + 12), 0.4), tm, g, pan)
            elif tier == 2:
                au.place(fx, au.marimba(au.hz(note + 5), 0.5), tm, g, pan)
                au.place(fx, au.bell(au.hz(note + 12), 0.8), tm, g * 0.5, pan)
            elif tier == 3:
                au.place(fx, au.bell(au.hz(note), 1.1), tm, g * 0.9, pan)
                au.place(fx, au.pluck(au.hz(note - 12), 0.6), tm, g * 0.7, pan)
            else:
                for dd in (0, 7, 12):
                    au.place(fx, au.bell(au.hz(note - 12 + dd), 1.4), tm, g * 0.7, pan)
                au.place(fx, au.boom(rng, 0.9, 80.0, 40.0), tm, 0.35)
        for ts_, _ in self.spawn_t:                                 # a soft tick for every new grain of dust
            if ts_ < T_COL - 0.4:
                au.place(fx, au.tick(2200 + 900 * rng.random(), 0.04), ts_, 0.05 + 0.04 * ts_ / T_COL)
        tl = np.arange(int(2.2 * sr)) / sr                          # the black hole: a falling low swell under the boom
        sub = np.sin(TAU * np.cumsum(46.0 - 14.0 * np.minimum(1.0, tl / 1.5)) / sr) * np.exp(-tl / 1.0)
        au.place(fx, sub * np.minimum(1.0, tl / 0.01), T_COL, 0.55)
        for j, m_ in enumerate((69, 72, 76, 81)):
            au.place(fx, au.marimba(au.hz(m_), 0.4), T_TOTAL - 0.62 + 0.13 * j, 0.12)
        mix_ = au.reverb(fx, rng, decay=1.1, wet=0.10)
        out_ = mix_[:n].copy()
        out_[:tail] += mix_[n:]
        out_ = au.eq(out_, self.EQ)
        out_[:int(0.04 * sr)] *= np.linspace(0, 1, int(0.04 * sr))[:, None]
        out_[-int(0.25 * sr):] *= np.linspace(1, 0, int(0.25 * sr))[:, None]
        pk = np.abs(out_).max()
        return np.tanh(out_ * (self.DRIVE / max(1e-9, pk)))

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        st = self.stats
        e = self.ed
        title = "How Many Merges Until the Final Form? \U0001F30C"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "Comment A, B, C or D \U0001F447 How many merges until dust becomes a black hole?\n\n"
                f"Band A is under {e[0]} merges, B is {e[0]} to {e[1] - 1}, C is {e[1]} to {e[2] - 1}, D is {e[2]} and up. "
                "Pick one before the first merge! "
                "Six forms: dust, rock, moon, planet, star, black hole. The same form pulls the same form, and two of a kind "
                "that touch merge into one body a tier up. Different forms bounce off each other. New dust drifts in from the "
                "rim faster and faster, the view pulls back as things grow, and the counter counts every merge. The run "
                "ends the moment the first black hole forms, and the band that holds the final count wins. A band is "
                "crossed out when the counter passes its top edge. It is a toy model, not astrophysics.\n"
                f"Everything here is generated by code (seed {self.seed}), with no footage, samples or stock media.\n"
                "Which band did you pick?\n"
                "#simulation #satisfying #merge"),
            tags=["simulation", "merge game", "final form", "merge simulation", "dust to black hole", "which one wins",
                  "prediction", "satisfying", "made with code", "generative animation", "black hole", "space"],
            pinned_comment=("Comment A, B, C or D \U0001F447 How many merges until it reaches the final form? "
                            f"A is under {e[0]}, B {e[0]} to {e[1] - 1}, C {e[1]} to {e[2] - 1}, D {e[2]} and up. "
                            "Pick before the first merge: same forms pull together, and two of a kind make one tier up."),
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of abstract space objects; a toy merge "
                                              "model that could not be mistaken for real footage, people, places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / FPS, 2),
            drama=st,
        )
