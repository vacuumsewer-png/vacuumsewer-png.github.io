"""STAR FORGE: three colored cores sit in a rotating disc of 600 dust grains. The first core to reach mass 250 ignites.

A toy model (real stars do not form by pairwise merging and the mass numbers are arbitrary). Rules, all on screen:
  * gravity is softened inverse-square between every pair of bodies;
  * bodies that touch merge, conserving mass and momentum;
  * dust has no color: a merge with dust keeps the core's color; when two cores merge the heavier keeps its color
    and the other core is swallowed;
  * bodies change form at mass tiers: rock (under 6), moon (6), planet (30), giant (100);
  * the first core to reach mass 250 ignites and blows the rest away.
The simulation runs in real time (30 frames per second, two integration steps per frame) and is deterministic per seed,
so make.py can score many seeds without rendering and render only the best one.
"""
import math
import os

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, darken, ease_in_out, ease_out, ease_out_back, glow, hex_rgb, lerp,
                        lighten, mix, polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30
SUB = 2                         # integration steps per frame
TV = 1.0 / (FPS * SUB)         # video seconds per step
SLOW = 2.1                      # the model runs this many times slower than the physics clock, so the build-up fills the video
DT = TV / SLOW
N_DUST = 600
CORE_M = 12.0                   # starting mass of every core
DUST_M = 1.6                    # mass of one dust grain
IGNITE = 250.0
TIERS = (6.0, 30.0, 100.0)      # moon, planet, giant
G = 3000.0
EPS = 12.0                      # softening length, px
RAD_K, RAD_P = 5.0, 0.42        # drawn radius = RAD_K * mass ** RAD_P
TOUCH = 0.8                     # bodies merge when closer than TOUCH * (r1 + r2)
VLO, VHI, VNOISE = 0.80, 1.0, 7.0   # circular-speed fraction range and random speed spread
DY = 75.0                       # the disc is drawn this far below the simulation center
CX, CY = 475.0, 905.0           # disc center (text and bars are centered on 475, clear of the right 12%)
R_IN, R_OUT = 70.0, 385.0
T_MAX = 27.0
PAYOFF = 4.6                    # seconds from ignition to the loop point
HYS = 2.0                       # mass a challenger must be ahead by to count as the new leader
NAMES = ("BLUE", "RED", "GOLD")  # symbols: circle, square, triangle


def radius(m):
    return RAD_K * np.power(m, RAD_P)


def tier(m):
    return int(m >= TIERS[0]) + int(m >= TIERS[1]) + int(m >= TIERS[2])


# ====================================================================== simulation
def initial(seed):
    rng = np.random.default_rng(seed * 104729 + 17)
    n = 3 + N_DUST
    pos = np.zeros((n, 2))
    mass = np.full(n, DUST_M)
    mass[:3] = CORE_M
    col = np.full(n, -1, np.int64)
    col[:3] = (0, 1, 2)
    a0 = rng.uniform(0, TAU)
    for k in range(3):
        a = a0 + k * TAU / 3 + rng.uniform(-0.25, 0.25)
        r = rng.uniform(150.0, 235.0)
        pos[k] = (r * math.cos(a), r * math.sin(a))
    rr = R_IN + (R_OUT - R_IN) * np.sqrt(rng.uniform(0, 1, N_DUST))
    th = rng.uniform(0, TAU, N_DUST)
    pos[3:] = np.stack([rr * np.cos(th), rr * np.sin(th)], axis=1)
    # circular speed from the mass enclosed (softened), a little random spread, all spinning the same way
    r_all = np.hypot(pos[:, 0], pos[:, 1])
    order = np.argsort(r_all)
    enc = np.cumsum(mass[order])
    menc = np.empty(n)
    menc[order] = enc
    v = np.sqrt(G * menc * r_all ** 2 / (r_all ** 2 + EPS ** 2) ** 1.5) * rng.uniform(VLO, VHI, n)
    vel = np.stack([-pos[:, 1] / np.maximum(r_all, 1e-9), pos[:, 0] / np.maximum(r_all, 1e-9)], axis=1) * v[:, None]
    vel += rng.normal(0, VNOISE, (n, 2))
    pos += (CX, CY)
    return pos, vel, mass, col


def accel(pos, mass):
    dx = pos[None, :, 0] - pos[:, None, 0]
    dy = pos[None, :, 1] - pos[:, None, 1]
    d2 = dx * dx + dy * dy
    s = d2 + EPS * EPS
    w = mass[None, :] / (s * np.sqrt(s))
    return G * np.stack([(dx * w).sum(1), (dy * w).sum(1)], axis=1), d2


def merge_pass(pos, vel, mass, col, d2, t, events):
    """Merge touching pairs (closest first); returns the new arrays and whether anything merged."""
    n = len(mass)
    r = radius(mass)
    thr = (TOUCH * (r[:, None] + r[None, :])) ** 2
    hit = np.triu(d2 < thr, 1)
    ii, jj = np.nonzero(hit)
    if len(ii) == 0:
        return pos, vel, mass, col, False
    order = np.argsort(d2[ii, jj])
    dead = np.zeros(n, bool)
    for k in order:
        i, j = int(ii[k]), int(jj[k])
        if dead[i] or dead[j]:
            continue
        if mass[j] > mass[i]:
            i, j = j, i                               # i is the heavier body
        mi, mj = mass[i], mass[j]
        m = mi + mj
        ci, cj = col[i], col[j]
        if ci >= 0 and cj >= 0:
            c = ci                                     # core meets core: the heavier keeps its color
        else:
            c = ci if ci >= 0 else cj                  # dust has no color: the core keeps its own
        p = (pos[i] * mi + pos[j] * mj) / m
        v = (vel[i] * mi + vel[j] * mj) / m
        t0, t1 = tier(mi) if ci >= 0 else -1, tier(m)
        events.append(dict(t=t, x=float(p[0]), y=float(p[1]), m=float(m), c=int(c), small=float(mj),
                           both=bool(ci >= 0 and cj >= 0), lost=int(cj) if (ci >= 0 and cj >= 0) else -1,
                           tier_up=bool(c >= 0 and t1 > t0 and t0 >= 0 and c == ci) or bool(c >= 0 and t1 > tier(mi))))
        pos[i], vel[i], mass[i], col[i] = p, v, m, c
        dead[j] = True
    keep = ~dead
    return pos[keep], vel[keep], mass[keep], col[keep], True


def simulate(seed, record=False):
    pos, vel, mass, col = initial(seed)
    init = dict(pos=pos.copy(), mass=mass.copy(), col=col.copy())
    events = []
    frames = []                                        # per frame: pos float32 (n, 2), mass, col, color masses
    acc, d2 = accel(pos, mass)
    ign = None
    nsteps = int(T_MAX * FPS * SUB)
    for s in range(nsteps):
        t = s * TV
        if s % SUB == 0:
            cm = [float(mass[col == k].sum()) for k in range(3)]
            if record:
                frames.append((pos.astype(np.float32), mass.astype(np.float32), col.astype(np.int8)))
            else:
                frames.append((None, None, None))
            frames[-1] = frames[-1] + (cm,)
            if max(cm) >= IGNITE:
                ign = s // SUB
                break
        vel += acc * (DT * 0.5)
        pos += vel * DT
        acc, d2 = accel(pos, mass)
        pos, vel, mass, col, merged = merge_pass(pos, vel, mass, col, d2, t + TV, events)
        if merged:
            acc, d2 = accel(pos, mass)
        vel += acc * (DT * 0.5)
    return dict(seed=seed, init=init, frames=frames, events=events, ign=ign, n_final=len(mass))


def lead_series(frames):
    out, lead = [], -1
    for fr in frames:
        cm = fr[3]
        j = max(range(3), key=lambda i: cm[i])
        if lead < 0:
            lead = j
        elif j != lead and cm[j] > cm[lead] + HYS:
            lead = j
        out.append(lead)
    return out


def drama(sim):
    frames = sim["frames"]
    ign = sim["ign"]
    valid = ign is not None
    stats = dict(seed=sim["seed"], valid=False, score=-100.0)
    if not valid:
        stats["note"] = "nobody ignited in time"
        return stats
    lead = lead_series(frames)
    fin = frames[ign][3]
    win = int(np.argmax(fin))
    order = sorted(range(3), key=lambda i: -fin[i])
    run = order[1]
    t_ign = ign / FPS
    dur = t_ign + PAYOFF
    changes = [(f / FPS, lead[f]) for f in range(1, len(frames)) if lead[f] != lead[f - 1] and f / FPS > 3.0]
    f75 = int(0.75 * ign)
    alive = [fin[i] > 0 for i in range(3)]
    core_merges = [e for e in sim["events"] if e["both"]]
    big_last = max((e for e in sim["events"] if e["c"] == win), key=lambda e: e["t"], default=None)
    et = sorted(e["t"] for e in sim["events"] if e["c"] >= 0 and 3.0 < e["t"] < t_ign)
    gap = max(np.diff([3.0] + et + [t_ign]))
    gates = dict(
        length=24.3 <= dur <= 31.0,
        no_lull=bool(gap <= 4.5),
        all_cores_alive=all(alive),
        lead_changes=len(changes) >= 2,
        winner_not_leading_at_75=lead[f75] != win,
        margin=fin[run] >= 0.72 * IGNITE,
        no_overshoot=fin[win] < IGNITE + 40.0,
    )
    ok = all(gates.values())
    score = (100.0 if ok else 0.0) + 8.0 * sum(gates.values()) - 40.0 * (not gates["length"])
    score += 30.0 * (fin[run] / IGNITE) + min(len(changes), 6) * 1.5 - 2.0 * min(gap, 8.0)
    if changes:
        score += 6.0 * ramp(changes[-1][0], 0.55 * t_ign, 0.95 * t_ign)
    stats = dict(
        seed=sim["seed"], score=round(float(score), 2), valid=bool(ok), gates={k: bool(v) for k, v in gates.items()},
        winner=NAMES[win], winner_index=win, runner_up=NAMES[run],
        ignition_s=round(t_ign, 2), duration_s=round(dur, 2),
        final_masses={NAMES[i]: round(float(fin[i]), 1) for i in range(3)},
        runner_up_gap=round(float(IGNITE - fin[run]), 1),
        longest_core_pause_s=round(float(gap), 2),
        lead_changes=len(changes), lead_change_times_s=[round(c[0], 2) for c in changes],
        lead_change_to=[NAMES[c[1]] for c in changes],
        leader_at_75pct=NAMES[lead[f75]],
        core_merges=len(core_merges),
        merges=len(sim["events"]), bodies_left=int(sim["n_final"]),
    )
    return stats


def score_seed(seed):
    return drama(simulate(seed))


# ====================================================================== look
COL = [hex_rgb("#4CC9FF"), hex_rgb("#FF5A6E"), hex_rgb("#FFC93C")]
INK = hex_rgb("#070B1A")
CREAM = hex_rgb("#F2EEE4")
MUTED = hex_rgb("#8E9AC0")
AMBER = hex_rgb("#FFC93C")
DUSTC = hex_rgb("#AEB8DC")
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
HX = 475.0
Q1, Q2 = "WHICH CORE", "BECOMES A STAR?"
RULE = "FIRST CORE TO MASS 250 IGNITES"
SUBTXT = "TOUCHING BODIES MERGE   •   A HEAVIER CORE KEEPS ITS COLOR"
TIER_NAMES = ("ROCK", "MOON", "PLANET", "GIANT")
BAR_CX = (175.0, 475.0, 775.0)
BAR_W = 250.0
SHOCK_V = 1050.0
PUSH = 230.0
RESET0, RESET1 = 2.1, 4.35      # seconds after ignition: the first frame rebuilds itself


def symbol(ctx, k, x, y, s):
    if k == 0:
        ctx.new_sub_path()
        ctx.arc(x, y, s, 0, TAU)
    elif k == 1:
        ctx.rectangle(x - s * 0.86, y - s * 0.86, s * 1.72, s * 1.72)
    else:
        polygon(ctx, [(x, y - s * 1.15), (x + s * 1.1, y + s * 0.8), (x - s * 1.1, y + s * 0.8)])


def badge(ctx, k, x, y, r, alpha=1.0):
    ctx.new_path()
    ctx.arc(x, y, r, 0, TAU)
    ctx.set_source_rgba(*COL[k], alpha)
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK, alpha)
    ctx.set_line_width(3)
    ctx.stroke()
    ctx.new_path()
    symbol(ctx, k, x, y, r * 0.45)
    ctx.set_source_rgba(*INK, alpha)
    ctx.fill()


def chip(ctx, font, txt, cx, y, size, pad=20, h=None):
    tw = font.width(txt, size, 1.2)
    h = h or size * 1.42
    rounded_rect(ctx, cx - tw / 2 - pad, y - h * 0.76, tw + 2 * pad, h, 12)
    ctx.set_source_rgba(*CREAM, 1)
    ctx.fill()
    font.draw(ctx, txt, cx, y, size, INK, anchor="c", tracking=1.2)


def tier_icon(ctx, tr, x, y, s, a=1.0, col=None):
    """Our own drawings: rock = dot, moon = cratered disc, planet = disc with a ring, giant = banded disc with halo."""
    base = [(0.62, 0.65, 0.80), (0.80, 0.84, 0.92), (0.45, 0.78, 0.76), (0.98, 0.62, 0.30)][tr]
    if col is not None:
        base = mix(base, col, 0.5)
    if tr == 3:
        glow(ctx, x, y, s * 2.3, (1.0, 0.62, 0.3), 0.45 * a)
    if tr == 2:
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(-0.35)
        ctx.scale(1.0, 0.32)
        ctx.new_path()
        ctx.arc(0, 0, s * 1.75, math.pi, TAU)          # back half of the ring
        ctx.restore()
        ctx.set_source_rgba(*lighten(base, 0.5), 0.8 * a)
        ctx.set_line_width(max(1.5, s * 0.2))
        ctx.stroke()
    g = cairo.RadialGradient(x - s * 0.35, y - s * 0.4, s * 0.1, x, y, s * 1.05)
    g.add_color_stop_rgba(0, *lighten(base, 0.55), a)
    g.add_color_stop_rgba(0.6, *base, a)
    g.add_color_stop_rgba(1, *darken(base, 0.55), a)
    ctx.set_source(g)
    ctx.new_path()
    ctx.arc(x, y, s, 0, TAU)
    ctx.fill()
    if tr == 1:
        ctx.set_source_rgba(*darken(base, 0.45), 0.8 * a)
        for cx_, cy_, cr in ((-0.35, -0.2, 0.26), (0.30, 0.25, 0.2), (0.15, -0.45, 0.14)):
            ctx.new_path()
            ctx.arc(x + cx_ * s, y + cy_ * s, cr * s, 0, TAU)
            ctx.fill()
    if tr == 3:
        ctx.save()
        ctx.new_path()
        ctx.arc(x, y, s, 0, TAU)
        ctx.clip()
        ctx.set_source_rgba(*darken(base, 0.45), 0.5 * a)
        for dy_, hh in ((-0.45, 0.2), (0.05, 0.28), (0.55, 0.18)):
            ctx.rectangle(x - s, y + dy_ * s - hh * s / 2, 2 * s, hh * s)
            ctx.fill()
        ctx.restore()
    if tr == 2:
        ctx.save()
        ctx.translate(x, y)
        ctx.rotate(-0.35)
        ctx.scale(1.0, 0.32)
        ctx.new_path()
        ctx.arc(0, 0, s * 1.75, 0, math.pi)             # front half of the ring
        ctx.restore()
        ctx.set_source_rgba(*lighten(base, 0.6), 0.95 * a)
        ctx.set_line_width(max(1.5, s * 0.2))
        ctx.stroke()


def make_background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xx - CX, (yy - CY - DY) * 0.92) / 560
    img = np.array([0.018, 0.020, 0.060], np.float32) + (np.clip(1 - d, 0, 1) ** 1.6)[..., None] * np.array(
        [0.060, 0.050, 0.130], np.float32)
    img += (np.clip(1 - np.hypot(xx - 140, yy - 520) / 420, 0, 1) ** 2)[..., None] * np.array(
        [0.015, 0.045, 0.060], np.float32)
    rng = np.random.default_rng(5)
    img += rng.normal(0, 0.004, (H, W, 1)).astype(np.float32)
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    del img, yy, xx, d
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    ctx = cairo.Context(surf)
    for _ in range(170):                                       # far stars
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        ctx.new_path()
        ctx.arc(x, y, rng.uniform(0.7, 1.7), 0, TAU)
        ctx.set_source_rgba(0.8, 0.85, 1.0, rng.uniform(0.15, 0.5))
        ctx.fill()
    ctx.set_source_rgba(0.55, 0.62, 0.95, 0.10)                # faint orbit guides
    ctx.set_line_width(2)
    for r in (110, 220, 330):
        ctx.new_path()
        ctx.arc(CX, CY + DY, r, 0, TAU)
        ctx.set_dash([8, 14])
        ctx.stroke()
    ctx.set_dash([])
    return surf, buf


def make_header(font):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, 470)
    ctx = cairo.Context(surf)
    s1 = min(124.0, 850.0 / font.width(Q2, 1.0))
    font.draw(ctx, Q1, HX, 124, s1, CREAM, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    font.draw(ctx, Q2, HX, 124 + s1 * 1.1, s1, CREAM, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    chip(ctx, font, RULE, HX, 124 + s1 * 1.1 + 92, 46, pad=24)
    font.draw(ctx, SUBTXT, HX, 124 + s1 * 1.1 + 148, 27, CREAM, anchor="c", stroke=INK, stroke_w=6, tracking=0.8)
    surf.flush()
    return surf


# ====================================================================== the scene
class Scene:
    W, H, FPS = W, H, FPS
    EQ = ((50, -9.0), (120, -5.0), (300, -1.0), (900, 0.0), (3000, 2.5), (8000, 1.0), (16000, -3.0))
    DRIVE = 4.0
    CEILING_DB = -3.4                                # AAC adds about 1.5 dB of true peak on top of the limiter ceiling
    HEAD_Y = 150                                     # header surface is drawn at this y

    def __init__(self, seed):
        self.seed = seed
        self.sim = simulate(seed, record=True)
        if self.sim["ign"] is None:
            raise ValueError(f"seed {seed}: no core ignites within {T_MAX} s")
        self.stats = drama(self.sim)
        self.ign = self.sim["ign"]
        self.t_ign = self.ign / FPS
        self.nframes = int(round((self.t_ign + PAYOFF) * FPS))
        self.dur = self.nframes / FPS
        self._res = None
        self._prep()

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_res"] = None
        return d

    def _prep(self):
        sim = self.sim
        frames = sim["frames"]
        ign = self.ign
        self.fin = frames[ign][3]
        self.win = int(np.argmax(self.fin))
        self.order = sorted(range(3), key=lambda i: -self.fin[i])
        self.run = self.order[1]
        self.lead = lead_series(frames)
        self.changes = [(f / FPS, self.lead[f]) for f in range(1, len(frames))
                        if self.lead[f] != self.lead[f - 1] and f / FPS > 3.0]
        # path of every core (for trails), by frame
        self.core_xy = []
        for k in range(3):
            row = []
            for pos, mass, col, cm in frames:
                idx = np.nonzero(col == k)[0]
                row.append((float(pos[idx[0], 0]), float(pos[idx[0], 1])) if len(idx) else None)
            self.core_xy.append(row)
        # events by frame
        self.ev_by_f = {}
        for e in sim["events"]:
            self.ev_by_f.setdefault(int(e["t"] * FPS), []).append(e)
        self.tier_events = [e for e in sim["events"] if e["tier_up"] and e["t"] * FPS <= ign]
        self.core_events = [e for e in sim["events"] if e["both"] and e["t"] * FPS <= ign]
        # star position and the frozen final state
        pos, mass, col, cm = frames[ign]
        w = np.nonzero(col == self.win)[0][0]
        self.star = (float(pos[w, 0]), float(pos[w, 1]))
        self.fpos, self.fmass, self.fcol = pos.astype(np.float64), mass.astype(np.float64), col
        self.win_i = int(w)
        # phantom start state: the opening disc returns from the outside
        ini = sim["init"]
        self.ipos, self.imass, self.icol = ini["pos"], ini["mass"], ini["col"]
        rng = np.random.default_rng(self.seed + 77)
        vec = self.ipos - (CX, CY)
        ang = np.arctan2(vec[:, 1], vec[:, 0]) + rng.uniform(-0.9, 0.9, len(vec))
        far = rng.uniform(430.0, 760.0, len(vec))
        self.off = np.stack([CX + far * np.cos(ang), CY + far * np.sin(ang)], axis=1)

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            font = Font(FONT_PATH)
            bg, keep = make_background()
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep, font=font,
                             head=make_header(font))
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
        d = t - self.t_ign
        font = res["font"]
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(res["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        fi = min(f, self.ign)
        pos, mass, col, cm = self.sim["frames"][fi]
        ctx.save()
        ctx.translate(0, DY)
        if d < 0:
            self._bodies(ctx, pos.astype(np.float64), mass, col, np.ones(len(mass)), font)
            self._trails(ctx, fi)
            self._pulses(ctx, f, font)
            bars = cm
            lead = self.lead[fi]
        else:
            self._after(ctx, d, font)
            k = ease_in_out(ramp(d, RESET0 + 0.5, RESET1 - 0.2))
            bars = [lerp(self.fin[i], CORE_M, k) for i in range(3)]
            lead = self.win if k < 0.5 else -1
        ctx.restore()
        ctx.set_source_surface(res["head"], 0, self.HEAD_Y)
        ctx.paint()
        self._bars(ctx, font, bars, lead, d)
        if d >= 0:
            self._result(ctx, d, font)
            w = math.exp(-d / 0.16) * 0.7
            if w > 0.01:
                ctx.set_source_rgba(1, 0.97, 0.9, w)
                ctx.paint()

    def _bodies(self, ctx, pos, mass, col, alpha, font, skip=None):
        """Dust and clumps first, heavier last; then the colored bodies get their symbol and mass number."""
        rad = radius(mass)
        dist = np.hypot(pos[:, 0] - CX, pos[:, 1] - CY)
        vis = alpha * (1 - np.clip((dist - 430.0) / 90.0, 0, 1))
        tr = (mass >= TIERS[0]).astype(int) + (mass >= TIERS[1]) + (mass >= TIERS[2])
        order = np.argsort(mass)
        for j in order:
            if vis[j] <= 0.02 or (skip is not None and j == skip):
                continue
            x, y, r, a = pos[j, 0], pos[j, 1], rad[j], vis[j]
            c = int(col[j])
            if c >= 0:
                glow(ctx, x, y, r * 2.6 + 14, COL[c], 0.30 * a)
            if tr[j] == 0:
                ctx.new_path()
                ctx.arc(x, y, r * 0.8, 0, TAU)
                ctx.set_source_rgba(*(DUSTC if c < 0 else mix(DUSTC, COL[c], 0.7)), 0.92 * a)
                ctx.fill()
            else:
                tier_icon(ctx, tr[j], x, y, r, a, COL[c] if c >= 0 else None)
            if c >= 0:
                ctx.new_path()
                ctx.arc(x, y, r + 3.5, 0, TAU)
                ctx.set_source_rgba(*COL[c], 0.95 * a)
                ctx.set_line_width(4)
                ctx.stroke()
        for j in order:
            c = int(col[j])
            if c < 0 or vis[j] <= 0.02 or (skip is not None and j == skip):
                continue
            x, y, r, a = pos[j, 0], pos[j, 1], rad[j], vis[j]
            ctx.new_path()
            symbol(ctx, c, x, y - r - 24, 13)
            ctx.set_source_rgba(*COL[c], a)
            ctx.fill_preserve()
            ctx.set_source_rgba(*INK, a)
            ctx.set_line_width(4)
            ctx.stroke()
            font.draw(ctx, str(int(round(mass[j]))), min(x, 905.0), y + r + 34, 34, COL[c], anchor="c", alpha=a,
                      stroke=INK, stroke_w=7)

    def _trails(self, ctx, fi):
        for k in range(3):
            pts = [self.core_xy[k][g] for g in range(max(0, fi - 30), fi + 1)]
            pts = [p for p in pts if p is not None]
            if len(pts) < 3:
                continue
            for q in range(1, len(pts)):
                a = (q / len(pts)) ** 1.6 * 0.55
                ctx.move_to(*pts[q - 1])
                ctx.line_to(*pts[q])
                ctx.set_source_rgba(*COL[k], a)
                ctx.set_line_width(2 + 4.0 * q / len(pts))
                ctx.stroke()

    def _pulses(self, ctx, f, font):
        for g in range(max(0, f - 9), f + 1):
            for e in self.ev_by_f.get(g, ()):
                u = (f - g) / 9.0
                big = e["c"] >= 0 and (e["tier_up"] or e["both"])
                if e["m"] < 8.0 and not big:
                    continue
                r0 = float(radius(e["m"]))
                col = COL[e["c"]] if e["c"] >= 0 else DUSTC
                ctx.new_path()
                ctx.arc(e["x"], e["y"], r0 + (46 if big else 16) * ease_out(u), 0, TAU)
                ctx.set_source_rgba(*lighten(col, 0.3), (0.85 if big else 0.35) * (1 - u))
                ctx.set_line_width(5 if big else 2.5)
                ctx.stroke()
        # tier name tags: the new form, for a moment
        for e in self.tier_events:
            g = e["t"] * FPS
            if g <= f <= g + 24:
                c = e["c"]
                m_ = e["m"]
                u = (f - g) / 24.0
                a = ramp(u, 0, 0.1) * (1 - ramp(u, 0.7, 1.0))
                label = TIER_NAMES[tier(m_)]
                x = min(max(e["x"], 130.0), 860.0)
                y = e["y"] - float(radius(m_)) - 62 - 12 * ease_out(u)
                tw = font.width(label, 30, 1.0)
                rounded_rect(ctx, x - tw / 2 - 12, y - 26, tw + 24, 40, 10)
                ctx.set_source_rgba(*INK, 0.78 * a)
                ctx.fill_preserve()
                ctx.set_source_rgba(*COL[c], a)
                ctx.set_line_width(3)
                ctx.stroke()
                font.draw(ctx, label, x, y + 3, 30, COL[c], anchor="c", alpha=a, tracking=1.0)

    # ------------------------------------------------------------ ignition and the reset
    def _after(self, ctx, d, font):
        sx, sy = self.star
        pos = self.fpos
        vec = pos - (sx, sy)
        dist = np.hypot(vec[:, 0], vec[:, 1])
        unit = vec / np.maximum(dist, 1e-6)[:, None]
        hit = dist / SHOCK_V
        k = np.maximum(0.0, d - hit)
        newpos = pos + unit * (PUSH * (1 - np.exp(-k / 0.45)))[:, None] * (0.6 + dist / 500.0)[:, None]
        alpha = 1 - np.clip((k - 0.2) / 1.2, 0, 1)
        alpha[k <= 0] = 1.0
        self._bodies(ctx, newpos, self.fmass, self.fcol, alpha, font, skip=self.win_i)
        # the reset: the opening disc flies back in
        e = ease_out(ramp(d, RESET0, RESET1), 3)
        ra = ramp(d, RESET0, RESET0 + 0.7)
        if ra > 0:
            p = self.off + (self.ipos - self.off) * e
            self._bodies(ctx, p, self.imass, self.icol, np.full(len(self.imass), ra), font)
        # star
        sa = 1 - ramp(d, RESET0 + 0.6, RESET0 + 1.5)
        if sa > 0:
            s = ease_out_back(ramp(d, 0.0, 0.5)) * (1 - 0.25 * ramp(d, RESET0 + 0.6, RESET0 + 1.5))
            rs = float(radius(self.fin[self.win])) * 1.05 * s
            c = COL[self.win]
            glow(ctx, sx, sy, rs * 4.5, c, 0.55 * sa)
            glow(ctx, sx, sy, rs * 2.4, lighten(c, 0.6), 0.55 * sa)
            ray_a = ramp(d, 0.0, 0.25) * (1 - ramp(d, 2.0, 3.0)) * sa
            if ray_a > 0:
                for q in range(14):
                    ang = q * TAU / 14 + 0.25 * d
                    L = rs * (2.0 + 2.2 * ease_out(ramp(d, 0, 0.7))) * (1.0 if q % 2 == 0 else 0.62)
                    w_ = 0.09
                    polygon(ctx, [(sx + rs * 0.8 * math.cos(ang - w_), sy + rs * 0.8 * math.sin(ang - w_)),
                                  (sx + L * math.cos(ang), sy + L * math.sin(ang)),
                                  (sx + rs * 0.8 * math.cos(ang + w_), sy + rs * 0.8 * math.sin(ang + w_))])
                    ctx.set_source_rgba(*lighten(c, 0.7), 0.8 * ray_a)
                    ctx.fill()
            g = cairo.RadialGradient(sx - rs * 0.25, sy - rs * 0.3, rs * 0.1, sx, sy, rs)
            g.add_color_stop_rgba(0, 1, 1, 0.96, sa)
            g.add_color_stop_rgba(0.55, *lighten(c, 0.55), sa)
            g.add_color_stop_rgba(1, *c, sa)
            ctx.set_source(g)
            ctx.new_path()
            ctx.arc(sx, sy, rs, 0, TAU)
            ctx.fill()
            ctx.new_path()
            ctx.arc(sx, sy, rs + 3, 0, TAU)
            ctx.set_source_rgba(1, 1, 1, 0.9 * sa)
            ctx.set_line_width(4)
            ctx.stroke()
        # shock rings
        for dly, wd, al in ((0.0, 12.0, 0.85), (0.22, 6.0, 0.55)):
            R = SHOCK_V * max(0.0, d - dly)
            if 0 < R < 1300:
                ctx.new_path()
                ctx.arc(sx, sy, R, 0, TAU)
                ctx.set_source_rgba(*lighten(COL[self.win], 0.6), al * (1 - R / 1300))
                ctx.set_line_width(wd)
                ctx.stroke()

    def _bars(self, ctx, font, bars, lead, d):
        y0 = 1372
        for k in range(3):
            cx = BAR_CX[k]
            dead = bars[k] <= 0.0
            a = 0.35 if dead else 1.0
            badge(ctx, k, cx - BAR_W / 2 + 22, y0 + 4, 22, a)
            font.draw(ctx, NAMES[k], cx - BAR_W / 2 + 54, y0 + 20, 44, CREAM if not dead else MUTED, alpha=a,
                      stroke=INK, stroke_w=7)
            txt = "SWALLOWED" if dead else f"{int(round(bars[k]))}"
            font.draw(ctx, txt, cx + BAR_W / 2, y0 + 20, 44 if not dead else 28, COL[k] if not dead else MUTED,
                      anchor="r", alpha=1.0, stroke=INK, stroke_w=7)
            by = y0 + 46
            rounded_rect(ctx, cx - BAR_W / 2, by, BAR_W, 30, 10)
            ctx.set_source_rgba(0.12, 0.15, 0.30, 0.9)
            ctx.fill()
            fill = clamp(bars[k] / IGNITE)
            if fill > 0.01:
                rounded_rect(ctx, cx - BAR_W / 2, by, max(20.0, BAR_W * fill), 30, 10)
                ctx.set_source_rgba(*COL[k], 1.0)
                ctx.fill()
            for tv in TIERS:                                    # tier ticks
                tx = cx - BAR_W / 2 + BAR_W * tv / IGNITE
                ctx.move_to(tx, by - 4)
                ctx.line_to(tx, by + 34)
                ctx.set_source_rgba(*CREAM, 0.8)
                ctx.set_line_width(3)
                ctx.stroke()
            rounded_rect(ctx, cx - BAR_W / 2, by, BAR_W, 30, 10)
            ctx.set_source_rgba(*CREAM, 0.55 if lead != k else 1.0)
            ctx.set_line_width(2.5 if lead != k else 4)
            ctx.stroke()
        # tier legend and the current leader
        ly = 1488
        items = [(0, "ROCK"), (1, "MOON 6"), (2, "PLANET 30"), (3, "GIANT 100")]
        xs = (30.0, 165.0, 320.0, 520.0)
        for (tr, lab), x in zip(items, xs):
            tier_icon(ctx, tr, x + 14, ly - 8, 11 if tr < 3 else 12)
            font.draw(ctx, lab, x + 36, ly, 27, CREAM, alpha=0.92, stroke=INK, stroke_w=5, tracking=0.6)
        if lead >= 0 and d < 0:
            font.draw(ctx, f"LEAD: {NAMES[lead]}", 920.0, ly, 27, COL[lead], anchor="r", stroke=INK, stroke_w=5,
                      tracking=0.6)

    def _result(self, ctx, d, font):
        k1 = ramp(d, 0.65, 0.95)
        fo = 1 - ramp(d, RESET1 - 0.55, RESET1 - 0.1)
        a = k1 * fo
        if a <= 0:
            return
        w = self.win
        oy = 420.0 if self.star[1] + DY < 980.0 else 0.0     # keep the panel off the star
        rounded_rect(ctx, HX - 410, 640 + oy, 820, 270, 26)
        ctx.set_source_rgba(0.02, 0.03, 0.09, 0.70 * a)
        ctx.fill()
        s = 0.8 + 0.2 * ease_out_back(k1)
        ctx.save()
        ctx.translate(HX, 748 + oy)
        ctx.scale(s, s)
        txt = f"{NAMES[w]} IGNITES"
        tw = font.width(txt, 104)
        badge(ctx, w, -tw / 2 - 14, -38, 32, a)
        font.draw(ctx, txt, 36, 0, 104, COL[w], anchor="c", alpha=a, stroke=INK, stroke_w=12, shadow=5)
        ctx.restore()
        k2 = ramp(d, 1.1, 1.4) * fo
        gap = IGNITE - self.fin[self.run]
        font.draw(ctx, f"{NAMES[self.run]} WAS {gap:.0f} MASS SHORT", HX, 812 + oy, 46, CREAM, anchor="c", alpha=k2,
                  stroke=INK, stroke_w=8)
        k3 = ramp(d, 1.6, 1.9) * fo
        font.draw(ctx, "WHICH CORE DID YOU PICK?", HX, 884 + oy, 50, AMBER, anchor="c", alpha=k3, stroke=INK, stroke_w=8)

    # ------------------------------------------------------------ sound
    def audio(self):
        sim = self.sim
        rng = np.random.default_rng(self.seed + 4242)
        sr = au.SR
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        bed = np.zeros((n + tail, 2))
        ti = self.t_ign

        def tt(dd):
            return np.arange(int(dd * sr)) / sr

        pent = (0, 2, 4, 7, 9)

        def note(i):
            return 60 + 12 * (i // 5) + pent[i % 5]

        def sine_pluck(f, dur=0.5):
            t_ = tt(dur)
            x = np.sin(TAU * f * t_) + 0.18 * np.sin(TAU * 2 * f * t_) * np.exp(-t_ / 0.05)
            return x * np.minimum(1.0, t_ / 0.003) * np.exp(-t_ / (0.07 + 0.10 * (dur / 0.5)))

        # merges: soft sine plucks, pitch falling with the merged mass, panned by x; one per 0.12 s, the biggest wins
        buckets = {}
        for e in sim["events"]:
            if e["t"] >= ti - 0.02:
                continue
            b = int(e["t"] / 0.12)
            if b not in buckets or e["m"] > buckets[b]["m"]:
                buckets[b] = e
        for b, e in buckets.items():
            m_ = e["m"]
            i = int(clamp(15 - 3.3 * math.log2(m_ / DUST_M), 0, 17))
            f0 = au.hz(note(i) + 12)
            g = (0.035 + 0.05 * min(1.0, math.sqrt(m_) / 7.0)) * (1.8 if e["c"] >= 0 else 1.0)
            au.place(fx, sine_pluck(f0, 0.5 if m_ > 20 else 0.3), e["t"], g, clamp((e["x"] - CX) / 420, -0.7, 0.7))
        # tier changes: a two-note rising interval in the core's own key
        base = (62, 65, 69)
        for e in self.tier_events:
            c = e["c"]
            r0 = base[c] + 5 * tier(e["m"]) - 12
            pan = clamp((e["x"] - CX) / 420, -0.7, 0.7)
            au.place(fx, au.marimba(au.hz(r0 + 12), 0.7), e["t"], 0.20, pan)
            au.place(fx, au.marimba(au.hz(r0 + 19), 0.9), e["t"] + 0.12, 0.17, pan)
        # a core swallows a rival: a falling glide and a low thump
        for e in self.core_events:
            au.place(fx, au.glide(au.hz(76), au.hz(52), 0.5, decay=0.3, harm=0.2), e["t"], 0.16)
            au.place(fx, au.boom(rng, 0.9, 80, 38), e["t"], 0.22)
        frs = sim["frames"]
        # a soft knock that speeds up as the leading core nears mass 250, so quiet stretches still tick
        tk = 0.4
        while tk < ti - 0.2:
            u = max(frs[min(self.ign, int(tk * FPS))][3]) / IGNITE
            au.place(fx, au.marimba(au.hz(45 + 7 * u), 0.35), tk, 0.16 + 0.06 * u)
            tk += lerp(1.25, 0.34, u ** 1.4)
        # a bell each time a core passes another 50 mass
        got = [0, 0, 0]
        for fi_ in range(self.ign + 1):
            for c in range(3):
                lvl = int(frs[fi_][3][c] // 50)
                if lvl > got[c]:
                    got[c] = lvl
                    au.place(fx, au.bell(au.hz(base[c] + 12 + 2 * lvl), 1.1), fi_ / FPS, 0.10,
                             (c - 1) * 0.35)
        # low pad: loudness and pitch follow the leading mass
        ft = np.arange(self.ign + 1) / FPS
        lead_m = np.array([max(fr[3]) for fr in frs[:self.ign + 1]])
        ts_ = np.arange(n) / sr
        lm = np.interp(ts_, ft, lead_m) / IGNITE
        lm = np.where(ts_ < ti, lm, 0.0)
        env = np.convolve(lm, np.ones(2400) / 2400, mode="same")
        ph = TAU * np.cumsum(55.0 * (1 + 0.5 * env)) / sr
        pad = (np.sin(ph) + 0.5 * np.sin(1.5 * ph * 1.002) + 0.25 * np.sin(2 * ph * 0.998))
        pad *= (0.012 + 0.055 * env ** 2) * (1 + 0.12 * np.sin(TAU * 0.35 * ts_))
        bed[:n, 0] += pad
        bed[:n, 1] += pad
        del lm, env, ph, pad, ts_
        au.place(bed, au.pad(au.hz(45), self.dur, 1.0, 0.8), 0.0, 0.010)
        au.place(bed, au.pad(au.hz(57), self.dur, 1.0, 0.8), 0.0, 0.006)
        # ignition: a rising sweep, then a flash, boom and a major chord with slow decay
        sw = au.swept_noise(rng, 1.0, 250, 6500)
        au.place(fx, sw * (np.arange(len(sw)) / len(sw)) ** 2.0, ti - 1.0, 0.32)
        rise = au.glide(au.hz(40), au.hz(76), 1.0, decay=1.2, harm=0.1)
        au.place(fx, rise * (np.arange(len(rise)) / len(rise)), ti - 1.0, 0.10)
        tb = tt(1.0)
        burst = au.noise(rng, 1.0, 120, 7000) * np.exp(-tb * 5.0) * np.minimum(1.0, tb / 0.004)
        au.place(fx, burst, ti, 0.50)
        au.place(fx, au.boom(rng, 2.0, 105, 32), ti, 0.62)
        for j, (m_, gain) in enumerate(((60, 0.20), (64, 0.17), (67, 0.17), (72, 0.14), (76, 0.10))):
            t_ = tt(3.4)
            x = np.sin(TAU * au.hz(m_) * t_) + 0.12 * np.sin(TAU * 2 * au.hz(m_) * t_)
            x *= np.minimum(1.0, t_ / 0.01) * np.exp(-t_ / 1.5)
            au.place(fx, x, ti + 0.03 + 0.05 * j, gain, (j - 2) * 0.25)
        # the reset: a falling sweep and two soft notes
        dd = 1.6
        au.place(fx, au.swept_noise(rng, dd, 5000, 400) * np.sin(np.pi * tt(dd) / dd), ti + RESET0, 0.14)
        au.place(fx, sine_pluck(au.hz(79), 0.6), ti + RESET1 - 0.6, 0.10, -0.3)
        au.place(fx, sine_pluck(au.hz(72), 0.7), ti + RESET1 - 0.25, 0.10, 0.3)
        mix_ = au.reverb(fx, rng, decay=1.4, wet=0.14) + bed
        out = mix_[:n].copy()
        out[:tail] += mix_[n:]
        out = au.eq(out, self.EQ)
        pk = np.abs(out).max()
        return np.tanh(out * (self.DRIVE / max(1e-9, pk)))

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        st = self.stats
        title = "3 Cores, 600 Dust Grains… Which Becomes a Star? ✨"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "Which core becomes a star: blue, red or gold? Call it before the dust starts to merge!\n\n"
                "Three colored cores sit in a rotating disc of 600 dust grains. This is a toy model, not real "
                "astrophysics: real stars do not form by pairwise merging, and the mass numbers are arbitrary. Every "
                "body pulls on every other body with softened inverse-square gravity. When two bodies touch they "
                "merge and keep their total mass and momentum. Dust has no color, so a merge with dust keeps the "
                "core's color; when two cores merge the heavier one keeps its color and the other is swallowed. "
                "Bodies change form at mass tiers: rock, then moon at 6, planet at 30 and giant at 100, and each "
                "step up pops a name tag. The three bars at the bottom show every core's mass from the first frame, "
                "and the first core to reach mass 250 ignites: it flashes into a star and a shockwave blows the rest "
                "away. Then the dust drifts back into the opening disc.\n"
                f"Gravity, picture and sound are generated by code (seed {self.seed}), with no footage, samples or "
                "stock media.\n"
                "Which core did you pick?\n"
                "#simulation #satisfying #space"),
            tags=["simulation", "star formation", "gravity simulation", "accretion", "n-body", "prediction",
                  "which one wins", "satisfying", "made with code", "generative animation"],
            pinned_comment=("Blue, red or gold: which core becomes a star? Pick before the dust starts to merge. "
                            "Touching bodies merge, a heavier core keeps its color, and the first core to mass 250 "
                            "ignites."),
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of abstract colored shapes; a toy "
                                              "gravity model that could not be mistaken for real footage, people, "
                                              "places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / FPS, 2),
            drama=st,
        )
