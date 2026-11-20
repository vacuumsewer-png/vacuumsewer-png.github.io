"""STORM RACE: four colored leaders grow from the cloud line toward the ground. First to touch the ground wins.

A stylized growth model (not an electrostatic simulation). Every leader gets the same 30 steps per second. On each
step one of its live tips, picked at random, extends 8 px at a random angle up to 68 degrees either side of
straight down; 12% of steps leave a fork behind (a new live tip). A tip that sits inside a drifting dry-air patch
stalls: that step is wasted. Tips that fall more than 60 px behind their leader's lowest tip die out, and a leader
keeps at most 6 live tips. The side walls reflect. The patches drift at constant speed and are on screen from frame 0.

simulate(seed) is deterministic and needs no rendering, so make.py can score many seeds (drama()) and render only
the best one (Scene).
"""
import math
import os
import random

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, darken, ease_in_out, ease_out, ease_out_back, glow, hex_rgb, lerp,
                        lighten, mix, polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30
S = 30                         # growth steps per second, the same for every leader
SPF = S // FPS
STEP = 8.0                     # px per step
ANG = math.radians(68.0)       # a step points up to this far either side of straight down
FORK = 0.12                    # share of steps that leave a fork
BAND = 60.0                    # tips this far above their leader's lowest tip die out
CAP = 6                        # most live tips per leader
NPATCH = 6
T_MAX = 27.0
PAYOFF = 4.6                   # seconds from the strike to the loop point
HYS = 4.0                      # px a challenger must be ahead by to count as the new lead

CLOUD_Y, GROUND_Y = 440.0, 1470.0
HGT = GROUND_Y - CLOUD_Y
XL, XR = 165.0, 940.0
X0 = (258.0, 454.0, 651.0, 847.0)
PY0, PY1 = 610.0, GROUND_Y - 12.0
NAMES = ("BLUE", "PINK", "GOLD", "MINT")      # symbols: circle, square, triangle, diamond


def reflect(u, lo, hi):
    span = hi - lo
    m = (u - lo) % (2 * span)
    return lo + (m if m <= span else 2 * span - m)


def make_patches(seed):
    rng = random.Random(seed * 7919 + 11)
    out = []
    for j in range(NPATCH):
        r = rng.uniform(58.0, 96.0)
        y0 = PY0 + (j + rng.uniform(0.1, 0.9)) * (PY1 - 96.0 - PY0) / NPATCH
        x0 = rng.uniform(XL + r, XR - r)
        vx = rng.choice((-1, 1)) * rng.uniform(34.0, 78.0)
        vy = rng.choice((-1, 1)) * rng.uniform(6.0, 20.0)
        out.append((x0, y0, r, vx, vy))
    return out


def patch_xy(p, t):
    x0, y0, r, vx, vy = p
    return reflect(x0 + vx * t, XL + r - 20, XR - r + 20), reflect(y0 + vy * t, PY0, PY1 - r)


def simulate(seed, record=False):
    patches = make_patches(seed)
    rngs = [random.Random(seed * 104729 + 977 * i + 5) for i in range(4)]
    segs = [[] for _ in range(4)]                 # (x0, y0, x1, y1, parent segment)
    tips = [[[X0[i], CLOUD_Y, -1]] for i in range(4)]
    low = [CLOUD_Y] * 4
    stalls, forks = [0] * 4, [0] * 4
    frames, ev = [], []
    winner = strike = win_seg = None
    tie = False
    nt = int(T_MAX * S)
    sin, cos = math.sin, math.cos

    def snap():
        frames.append((tuple(len(s) for s in segs), tuple(low),
                       tuple(tuple((a[0], a[1]) for a in tp) for tp in tips) if record
                       else tuple(len(tp) for tp in tips)))

    for k in range(nt):
        t = k / S
        if k % SPF == 0:
            snap()
        pos = [patch_xy(p, t) for p in patches]
        hit = []
        for i in range(4):
            rng, tp = rngs[i], tips[i]
            a = tp[rng.randrange(len(tp))]
            ang = rng.uniform(-ANG, ANG)
            fk = rng.random() < FORK
            x, y = a[0], a[1]
            stalled = False
            for (px, py), p in zip(pos, patches):
                if (x - px) ** 2 + (y - py) ** 2 < p[2] * p[2]:
                    stalled = True
                    break
            if stalled:
                stalls[i] += 1
                continue
            nx, ny = x + STEP * sin(ang), y + STEP * cos(ang)
            if nx < XL:
                nx = 2 * XL - nx
            elif nx > XR:
                nx = 2 * XR - nx
            if ny >= GROUND_Y:
                over = ny - GROUND_Y
                fr = (GROUND_Y - y) / (ny - y)
                nx, ny = x + (nx - x) * fr, GROUND_Y
                hit.append((over, i, len(segs[i])))
            segs[i].append((x, y, nx, ny, a[2]))
            idx = len(segs[i]) - 1
            if fk:
                tp.append([nx, ny, idx])
                forks[i] += 1
            else:
                a[0], a[1], a[2] = nx, ny, idx
            if ny > low[i]:
                low[i] = ny
            if len(tp) > 1:
                lim = low[i] - BAND
                tp[:] = [q for q in tp if q[1] >= lim]
                while len(tp) > CAP:
                    tp.remove(min(tp, key=lambda q: q[1]))
            if record:
                ev.append((t, i, nx, ny))
        if hit:
            hit.sort(reverse=True)
            _, winner, win_seg = hit[0]
            tie = len(hit) > 1
            strike = (k + 1) / S
            break
    snap()
    return dict(seed=seed, patches=patches, segs=segs, frames=frames, ev=ev, winner=winner, strike=strike,
                win_seg=win_seg, tie=tie, stalls=stalls, forks=forks, low=list(low))


def lead_series(frames):
    """Leader with the lowest tip per frame (-1 before anyone has moved); it changes only past HYS px."""
    out, lead = [], -1
    for fr in frames:
        low = fr[1]
        j = max(range(4), key=lambda i: low[i])
        if lead < 0:
            if low[j] > CLOUD_Y:
                lead = j
        elif j != lead and low[j] > low[lead] + HYS:
            lead = j
        out.append(lead)
    return out


def drama(sim):
    frames = sim["frames"]
    win, ts = sim["winner"], sim["strike"]
    valid = win is not None and not sim["tie"]
    lead = lead_series(frames)
    changes = []
    for f in range(1, len(frames)):
        if lead[f] != lead[f - 1] and lead[f - 1] >= 0:
            front = max(frames[f][1]) - CLOUD_Y
            if front >= 0.06 * HGT:
                changes.append((f, lead[f], front))
    f70 = next((f for f, fr in enumerate(frames) if max(fr[1]) - CLOUD_Y >= 0.7 * HGT), len(frames) - 1)
    low70 = frames[f70][1]
    order70 = sorted(range(4), key=lambda i: -low70[i])
    final = sim["low"]
    order = sorted(range(4), key=lambda i: -final[i])
    gaps = [round(GROUND_Y - final[i], 1) for i in order]
    gap = gaps[1]
    last = changes[-1] if changes else None
    last_to_strike = (ts - last[0] / FPS) if (last and ts) else None
    last_left = (HGT - last[2]) if last else None
    score = 0.0
    dur = None
    if not valid:
        score -= 100.0
    else:
        dur = ts + PAYOFF
        score += 4.0 if 24.3 <= dur <= 30.5 else -40.0 - 5 * abs(dur - 27.5)
        n = len(changes)
        score += 1.0 * min(n, 9) + (0.0 if n >= 5 else -20.0)
        score += 0.0 if lead[f70] != win else -20.0
        score += 1.5 if order70.index(win) >= 2 else 0.0
        score += (2.0 * (1 - abs(gap - 18.0) / 18.0)) if 10.0 <= gap <= 31.0 else -20.0
        if last is None or last[1] != win:
            score -= 20.0
        else:
            score += 4.0 if last_left <= 0.15 * HGT else (1.5 if last_left <= 0.25 * HGT else -2.0)
            score += 3.0 if 0.4 <= last_to_strike <= 2.5 else -3.0
        score += 1.5 if gaps[2] <= 0.15 * HGT else 0.0
        score += 1.0 if len({c[1] for c in changes}) >= 3 else 0.0      # at least three colors held the lead
    return dict(
        seed=sim["seed"], score=round(score, 3), valid=valid,
        winner=NAMES[win] if win is not None else None, winner_index=win,
        runner_up=NAMES[order[1]] if valid else None,
        strike_s=ts, duration_s=round(dur, 2) if dur else None,
        lead_changes=len(changes), lead_change_times_s=[round(c[0] / FPS, 2) for c in changes],
        lead_change_to=[NAMES[c[1]] for c in changes],
        leader_at_70pct=NAMES[lead[f70]] if lead[f70] >= 0 else None,
        winner_place_at_70pct=(order70.index(win) + 1) if valid else None,
        final_gap_px=gap, final_gap_pct_of_height=round(100 * gap / HGT, 2),
        gaps_px_by_place=gaps,
        last_lead_change_px_from_ground=round(last_left, 1) if last else None,
        last_lead_change_to_strike_s=round(last_to_strike, 2) if last_to_strike is not None else None,
        stalled_steps=dict(zip(NAMES, sim["stalls"])), forks=dict(zip(NAMES, sim["forks"])),
        steps_per_second=S,
    )


def score_seed(seed):
    return drama(simulate(seed))


# ====================================================================== look
COL = [hex_rgb("#45D4FF"), hex_rgb("#FF5DA2"), hex_rgb("#FFD23F"), hex_rgb("#7BF08C")]
INK = hex_rgb("#0B0A18")
CREAM = hex_rgb("#F1EEE6")
MUTED = hex_rgb("#A39FC4")
AMBER = hex_rgb("#FFC23D")
DRY = (0.82, 0.80, 0.93)
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
HX = 548.0                                         # text center (clear of the right margin)
LX = (34.0, 64.0, 94.0, 124.0)                     # altitude lanes on the left edge
Q1, Q2 = "WHICH BOLT", "STRIKES FIRST?"
RULE = "FIRST TO TOUCH THE GROUND WINS"
SUB = "SAME 30 STEPS PER SECOND EACH   •   TIPS STALL IN DRY AIR"
HOOK_Y0, HOOK_H = 690, 520                         # big hook block (first seconds)
HEAD_Y0, HEAD_H = 150, 215                         # docked header


def symbol(ctx, k, x, y, s):
    if k == 0:
        ctx.new_sub_path()
        ctx.arc(x, y, s, 0, TAU)
    elif k == 1:
        ctx.rectangle(x - s * 0.86, y - s * 0.86, s * 1.72, s * 1.72)
    elif k == 2:
        polygon(ctx, [(x, y - s * 1.15), (x + s * 1.1, y + s * 0.8), (x - s * 1.1, y + s * 0.8)])
    else:
        polygon(ctx, [(x, y - s * 1.25), (x + s * 1.05, y), (x, y + s * 1.25), (x - s * 1.05, y)])


def badge(ctx, k, x, y, r, alpha=1.0, ring=None):
    ctx.new_path()
    ctx.arc(x, y, r, 0, TAU)
    ctx.set_source_rgba(*COL[k], alpha)
    ctx.fill_preserve()
    ctx.set_source_rgba(*(ring or INK), alpha)
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


def make_background(font):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = np.clip((yy - CLOUD_Y) / HGT, 0, 1)[..., None]
    img = np.array([0.075, 0.065, 0.185], np.float32) * (1 - v) + np.array([0.230, 0.130, 0.300], np.float32) * v
    d = np.hypot((xx - 552) / 1.6, yy - GROUND_Y) / 520
    img += (np.clip(1 - d, 0, 1) ** 2)[..., None] * np.array([0.10, 0.05, 0.08], np.float32)
    vig = np.clip(1.10 - 0.42 * (((xx - 540) / 760) ** 2), 0.5, 1.0)
    img *= vig[..., None]
    img += np.random.default_rng(9).normal(0, 0.005, (H, W, 1)).astype(np.float32)
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    del img, yy, xx, v, d, vig
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    ctx = cairo.Context(surf)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    # altitude lines every 10%
    for j in range(1, 10):
        y = CLOUD_Y + HGT * j / 10
        ctx.move_to(150, y)
        ctx.line_to(952, y)
        ctx.set_source_rgba(0.8, 0.8, 1.0, 0.07)
        ctx.set_line_width(2)
        ctx.stroke()
    # cloud deck: a lumpy dark mass down to the cloud line
    rnd = random.Random(4)
    g = cairo.LinearGradient(0, 0, 0, CLOUD_Y)
    g.add_color_stop_rgb(0, 0.100, 0.095, 0.165)
    g.add_color_stop_rgb(0.7, 0.175, 0.165, 0.265)
    g.add_color_stop_rgb(1, 0.275, 0.250, 0.385)
    ctx.new_path()
    ctx.rectangle(0, 0, W, CLOUD_Y - 46)
    x = -40.0
    while x < W + 60:
        r = rnd.uniform(44, 86)
        ctx.new_sub_path()
        ctx.arc(x, CLOUD_Y - 8 - r - rnd.uniform(0, 14), r, 0, TAU)
        x += r * rnd.uniform(0.75, 1.15)
    ctx.set_source(g)
    ctx.fill()
    ctx.rectangle(0, CLOUD_Y - 10, W, 60)                 # soft haze under the deck
    g = cairo.LinearGradient(0, CLOUD_Y - 10, 0, CLOUD_Y + 50)
    g.add_color_stop_rgba(0, 0.30, 0.27, 0.42, 0.35)
    g.add_color_stop_rgba(1, 0.30, 0.27, 0.42, 0.0)
    ctx.set_source(g)
    ctx.fill()
    # ground
    g = cairo.LinearGradient(0, GROUND_Y, 0, H)
    g.add_color_stop_rgb(0, 0.085, 0.070, 0.120)
    g.add_color_stop_rgb(0.25, 0.050, 0.042, 0.078)
    g.add_color_stop_rgb(1, 0.028, 0.024, 0.045)
    ctx.rectangle(0, GROUND_Y, W, H - GROUND_Y)
    ctx.set_source(g)
    ctx.fill()
    ctx.rectangle(0, GROUND_Y - 2, W, 4)
    ctx.set_source_rgba(0.80, 0.74, 0.95, 0.9)
    ctx.fill()
    rnd = random.Random(8)
    for _ in range(140):                                   # sparse ground texture below the safe area
        gx, gy = rnd.uniform(0, W), rnd.uniform(GROUND_Y + 80, H)
        ctx.move_to(gx, gy)
        ctx.line_to(gx + rnd.uniform(14, 60), gy)
        ctx.set_source_rgba(0.6, 0.55, 0.8, rnd.uniform(0.03, 0.09))
        ctx.set_line_width(2)
        ctx.stroke()
    # altitude lanes
    for k in range(4):
        rounded_rect(ctx, LX[k] - 8, CLOUD_Y, 16, HGT, 4)
        ctx.set_source_rgba(0.02, 0.02, 0.06, 0.55)
        ctx.fill()
        for j in range(1, 10):
            y = CLOUD_Y + HGT * j / 10
            ctx.rectangle(LX[k] - 8, y - 1, 16, 2)
            ctx.set_source_rgba(0.8, 0.8, 1.0, 0.22)
            ctx.fill()
    # emitters on the cloud line, with names
    for k in range(4):
        glow(ctx, X0[k], CLOUD_Y, 70, COL[k], 0.30)
        font.draw(ctx, NAMES[k], X0[k], CLOUD_Y - 34, 30, COL[k], anchor="c", stroke=INK, stroke_w=6, tracking=1.5)
        badge(ctx, k, X0[k], CLOUD_Y, 19)
    surf.flush()
    return surf, buf


def make_text_layers(font):
    """The big opening hook and the docked header, each drawn once."""
    hook = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, HOOK_H)
    ctx = cairo.Context(hook)
    s1 = min(190.0, 770.0 / font.width(Q2, 1.0))
    font.draw(ctx, Q1, HX, s1 * 0.86 + 10, s1, CREAM, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    font.draw(ctx, Q2, HX, s1 * 1.92 + 10, s1, CREAM, anchor="c", stroke=INK, stroke_w=14, shadow=8)
    chip(ctx, font, RULE, HX, s1 * 1.92 + 98, 46, pad=24)
    font.draw(ctx, SUB, HX, s1 * 1.92 + 162, 29, CREAM, anchor="c", stroke=INK, stroke_w=6, tracking=0.8)
    hook.flush()
    head = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, HEAD_H)
    ctx = cairo.Context(head)
    q = Q1 + " " + Q2
    s2 = min(78.0, 770.0 / font.width(q, 1.0))
    font.draw(ctx, q, HX, 80, s2, CREAM, anchor="c", stroke=INK, stroke_w=9, shadow=5)
    chip(ctx, font, RULE, HX, 136, 35, pad=18)
    font.draw(ctx, SUB, HX, 190, 26, MUTED, anchor="c", stroke=INK, stroke_w=5, tracking=0.8)
    head.flush()
    return hook, head


# ====================================================================== the scene
class Scene:
    W, H, FPS = W, H, FPS
    EQ = ((50, -10.0), (120, -6.0), (300, -2.0), (900, 0.0), (3000, 3.0), (8000, 1.5), (16000, -2.0))
    DRIVE = 5.0

    def __init__(self, seed):
        self.seed = seed
        self.sim = simulate(seed, record=True)
        if self.sim["winner"] is None:
            raise ValueError(f"seed {seed}: nobody reaches the ground within {T_MAX} s")
        self.stats = drama(self.sim)
        self.win = self.sim["winner"]
        self.ts = self.sim["strike"]
        self.nframes = int(round((self.ts + PAYOFF) * FPS))
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
        self.lead = lead_series(frames)
        self.changes = [(f / FPS, self.lead[f]) for f in range(1, len(frames))
                        if self.lead[f] != self.lead[f - 1] and self.lead[f - 1] >= 0]
        # altitude lines crossed by the front
        self.miles = []
        j = 1
        for f, fr in enumerate(frames):
            while j <= 9 and max(fr[1]) - CLOUD_Y >= HGT * j / 10:
                self.miles.append((f / FPS, j))
                j += 1
        # winner's path, ground first
        segs = sim["segs"][self.win]
        k = sim["win_seg"]
        pts = [(segs[k][2], segs[k][3])]
        while k >= 0:
            pts.append((segs[k][0], segs[k][1]))
            k = segs[k][4]
        self.path = pts
        self.sx = pts[0][0]
        acc = [0.0]
        for a, b in zip(pts, pts[1:]):
            acc.append(acc[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        self.path_len = acc
        final = sim["low"]
        order = sorted(range(4), key=lambda i: -final[i])
        self.second = order[1]
        self.gap = GROUND_Y - final[self.second]
        tips = frames[-1][2][self.second]
        self.rx, self.ry = max(tips, key=lambda q: q[1])
        # segments near the runner-up's tip, for the magnifier
        x0, x1, y0 = self.rx - 60, self.rx + 60, GROUND_Y - 80
        self.near = [[s for s in sim["segs"][i] if max(s[1], s[3]) >= y0 and min(s[0], s[2]) <= x1
                      and max(s[0], s[2]) >= x0] for i in range(4)]

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            font = Font(FONT_PATH)
            bg, keep = make_background(font)
            hook, head = make_text_layers(font)
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep, font=font,
                             hook=hook, head=head)
        return self._res

    def frame(self, f):
        res = self._resources()
        ctx = res["ctx"]
        ctx.save()
        self.draw(ctx, f, res)
        ctx.restore()
        res["surf"].flush()
        return bytes(res["surf"].get_data())

    # ------------------------------------------------------------ helpers
    def fidx(self, f):
        return max(0, min(f, len(self.sim["frames"]) - 1))

    def patch_pos(self, t):
        home = ease_in_out(ramp(t, self.ts + 0.5, self.dur - 0.3))
        out = []
        for p in self.sim["patches"]:
            x, y = patch_xy(p, min(t, self.ts + 0.5) + max(0.0, t - self.ts - 0.5) * (1 - home))
            if home > 0:
                hx, hy = patch_xy(p, 0.0)
                x, y = lerp(x, hx, home), lerp(y, hy, home)
            out.append((x, y, p[2]))
        return out

    def dock(self, t):
        return ease_in_out(ramp(t, 2.4, 3.0)) * (1 - ease_in_out(ramp(t, self.dur - 0.62, self.dur - 0.04)))

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
        fi = self.fidx(f)
        fr = self.sim["frames"][fi]
        d = t - self.ts
        fo = 1 - ramp(t, self.dur - 1.05, self.dur - 0.6)      # everything from the race clears before the loop
        pp = self.patch_pos(t)
        self._miles(ctx, t)
        self._patches(ctx, t, pp, font)
        if d < 0 and self.lead[fi] >= 0:
            y = fr[1][self.lead[fi]]
            ctx.save()
            ctx.set_dash([14, 12])
            ctx.move_to(146, y)
            ctx.line_to(950, y)
            ctx.set_source_rgba(*lighten(COL[self.lead[fi]], 0.4), 0.30)
            ctx.set_line_width(2.5)
            ctx.stroke()
            ctx.restore()
        self._leaders(ctx, fr, d, fo)
        if d >= 0:
            self._strike(ctx, d, fo)
        self._tips(ctx, fr, t, d, fo, pp)
        self._lanes(ctx, fi, fr, t, d, fo)
        k = self.dock(t)
        self._hook(ctx, res, k)
        self._readout(ctx, fi, fr, t, d, k, font)
        if d >= 0:
            self._lens(ctx, d, fo, font)
            self._result(ctx, d, fo, font)
            w = ease_in_out(ramp(d, 0, 0.16)) * math.exp(-d / 0.45)     # one smooth bloom
            if w > 0.004:
                ctx.set_source_rgba(*lighten(COL[self.win], 0.75), 0.20 * w)
                ctx.paint()

    def _miles(self, ctx, t):
        for tm, j in self.miles:
            u = t - tm
            if 0 <= u < 0.7:
                y = CLOUD_Y + HGT * j / 10
                ctx.move_to(150, y)
                ctx.line_to(952, y)
                ctx.set_source_rgba(0.85, 0.85, 1.0, 0.30 * (1 - u / 0.7))
                ctx.set_line_width(3)
                ctx.stroke()

    def _patches(self, ctx, t, pp, font):
        for j, (x, y, r) in enumerate(pp):
            g = cairo.RadialGradient(x, y, 0, x, y, r)
            g.add_color_stop_rgba(0, *DRY, 0.20)
            g.add_color_stop_rgba(0.8, *DRY, 0.13)
            g.add_color_stop_rgba(1, *DRY, 0.06)
            ctx.new_path()
            ctx.arc(x, y, r, 0, TAU)
            ctx.set_source(g)
            ctx.fill()
            ctx.save()
            ctx.set_line_cap(cairo.LINE_CAP_BUTT)
            ctx.set_dash([13, 9], (t * 30 * (1 if j % 2 else -1)) % 22)
            ctx.arc(x, y, r - 2, 0, TAU)
            ctx.set_source_rgba(*DRY, 0.62)
            ctx.set_line_width(3.5)
            ctx.stroke()
            ctx.restore()
            font.draw(ctx, "DRY AIR", x, y + 9, 25, DRY, anchor="c", alpha=0.62, tracking=1.5)

    def _leaders(self, ctx, fr, d, fo):
        segs = self.sim["segs"]
        post = ramp(d, 0.05, 0.9)
        for i in range(4):
            n = fr[0][i]
            if n == 0:
                continue
            al = fo * (lerp(1.0, 0.55, post) if i == self.win else lerp(1.0, 0.26, post))
            ctx.new_path()
            px = py = None
            for s in segs[i][:n]:
                if s[0] != px or s[1] != py:
                    ctx.move_to(s[0], s[1])
                px, py = s[2], s[3]
                ctx.line_to(px, py)
            ctx.set_source_rgba(*COL[i], 0.14 * al)
            ctx.set_line_width(13)
            ctx.stroke_preserve()
            ctx.set_source_rgba(*COL[i], 0.95 * al)
            ctx.set_line_width(5)
            ctx.stroke_preserve()
            ctx.set_source_rgba(*lighten(COL[i], 0.7), 0.85 * al)
            ctx.set_line_width(1.8)
            ctx.stroke()
            if d < 0:                                          # the newest growth burns whiter
                ctx.new_path()
                px = py = None
                for s in segs[i][max(0, n - 14):n]:
                    if s[0] != px or s[1] != py:
                        ctx.move_to(s[0], s[1])
                    px, py = s[2], s[3]
                    ctx.line_to(px, py)
                ctx.set_source_rgba(*lighten(COL[i], 0.85), 0.9)
                ctx.set_line_width(3.2)
                ctx.stroke()

    def _strike(self, ctx, d, fo):
        col = COL[self.win]
        b = ease_out(ramp(d, 0, 0.14)) * (0.40 + 0.60 * math.exp(-d / 0.6)) * fo
        tot = self.path_len[-1]
        reach = tot * ease_out(ramp(d, 0, 0.24), 2)            # the return stroke runs up from the ground
        ctx.new_path()
        ctx.move_to(*self.path[0])
        for j in range(1, len(self.path)):
            if self.path_len[j] <= reach:
                ctx.line_to(*self.path[j])
            else:
                a, c = self.path[j - 1], self.path[j]
                u = (reach - self.path_len[j - 1]) / max(1e-9, self.path_len[j] - self.path_len[j - 1])
                ctx.line_to(lerp(a[0], c[0], u), lerp(a[1], c[1], u))
                break
        ctx.set_source_rgba(*col, 0.20 * b)
        ctx.set_line_width(40)
        ctx.stroke_preserve()
        ctx.set_source_rgba(*col, 0.55 * b)
        ctx.set_line_width(18)
        ctx.stroke_preserve()
        ctx.set_source_rgba(*lighten(col, 0.5), fo * (0.6 + 0.4 * b))
        ctx.set_line_width(9)
        ctx.stroke_preserve()
        ctx.set_source_rgba(1, 1, 1, fo * (0.75 + 0.25 * b))
        ctx.set_line_width(4)
        ctx.stroke()
        glow(ctx, self.sx, GROUND_Y, 460, lighten(col, 0.6), 0.85 * b)
        u = ramp(d, 0, 0.9)
        if 0 < u < 1:                                          # one ring running out along the ground
            ctx.save()
            ctx.translate(self.sx, GROUND_Y)
            ctx.scale(1, 0.16)
            ctx.arc(0, 0, 30 + 420 * ease_out(u), 0, TAU)
            ctx.restore()
            ctx.set_source_rgba(*lighten(col, 0.6), 0.8 * (1 - u) * fo)
            ctx.set_line_width(6)
            ctx.stroke()

    def _tips(self, ctx, fr, t, d, fo, pp):
        if fo <= 0:
            return
        for tc, k in self.changes:                             # one ring on the new leader's tip
            u = (t - tc) / 0.5
            if 0 <= u < 1 and d < 0:
                x, y = max(fr[2][k], key=lambda q: q[1])
                ctx.new_path()
                ctx.arc(x, y, 16 + 54 * ease_out(u), 0, TAU)
                ctx.set_source_rgba(*lighten(COL[k], 0.5), 0.9 * (1 - u))
                ctx.set_line_width(5)
                ctx.stroke()
        for i in range(4):
            tips = fr[2][i]
            al = fo * (1.0 if (d < 0 or i == self.win) else lerp(1.0, 0.45, ramp(d, 0.05, 0.9)))
            lx, ly = max(tips, key=lambda q: q[1])
            for x, y in tips:
                if (x, y) == (lx, ly):
                    continue
                stalled = any((x - px) ** 2 + (y - py) ** 2 < r * r for px, py, r in pp)
                ctx.new_path()
                ctx.arc(x, y, 5.5, 0, TAU)
                if stalled and d < 0:
                    ctx.set_source_rgba(*INK, 0.9 * al)
                    ctx.fill_preserve()
                    ctx.set_source_rgba(*DRY, 0.9 * al)
                    ctx.set_line_width(2.5)
                    ctx.stroke()
                else:
                    ctx.set_source_rgba(*lighten(COL[i], 0.75), al)
                    ctx.fill()
            if fr[0][i] or d < 0:
                stalled = d < 0 and any((lx - px) ** 2 + (ly - py) ** 2 < r * r for px, py, r in pp)
                if not (fr[0][i] == 0):
                    badge(ctx, i, lx, ly, 14, al, ring=DRY if stalled else None)

    def _lanes(self, ctx, fi, fr, t, d, fo):
        lead = self.lead[fi]
        for i in range(4):
            dep = (fr[1][i] - CLOUD_Y) * fo
            y = CLOUD_Y + dep
            if dep > 1:
                rounded_rect(ctx, LX[i] - 8, CLOUD_Y, 16, dep, 4)
                ctx.set_source_rgba(*COL[i], 0.85)
                ctx.fill()
            is_lead = i == lead and fo > 0.5
            if is_lead:
                ctx.new_path()
                ctx.arc(LX[i], y, 18.5, 0, TAU)
                ctx.set_source_rgba(*CREAM, 1)
                ctx.fill()
            badge(ctx, i, LX[i], y, 14)

    def _hook(self, ctx, res, k):
        if k < 0.997:
            ctx.save()
            s = 1 - 0.30 * k
            ctx.translate(HX, HOOK_Y0 + HOOK_H / 2 - 330 * k)
            ctx.scale(s, s)
            ctx.set_source_surface(res["hook"], -HX, -HOOK_H / 2)
            ctx.paint_with_alpha(1 - k)
            ctx.restore()
        if k > 0.003:
            ctx.set_source_surface(res["head"], 0, HEAD_Y0 + 40 * (1 - k))
            ctx.paint_with_alpha(k)

    def _readout(self, ctx, fi, fr, t, d, k, font):
        y = GROUND_Y + 47
        font.draw(ctx, f"SEED {self.seed}", 948, y - 2, 24, MUTED, anchor="r", alpha=0.8, tracking=1)
        lead = self.lead[fi]
        if k <= 0 or lead < 0:
            return
        x = 160.0
        if d < 0:
            x += font.draw(ctx, "LEAD", x, y, 34, MUTED, alpha=k, tracking=1.5) + 30
            badge(ctx, lead, x, y - 13, 14, k)
            x += 26
            x += font.draw(ctx, NAMES[lead], x, y, 34, COL[lead], alpha=k, tracking=1.5) + 22
            togo = max(0, int(round(GROUND_Y - fr[1][lead])))
            font.draw(ctx, f"{togo} PX TO GO", x, y, 34, CREAM, alpha=k, tracking=1)
        else:
            k *= 1 - ramp(t, self.dur - 1.05, self.dur - 0.6)
            if k <= 0:
                return
            badge(ctx, self.win, x + 14, y - 13, 14, k)
            x += 40
            x += font.draw(ctx, NAMES[self.win], x, y, 34, COL[self.win], alpha=k, tracking=1.5) + 22
            font.draw(ctx, f"TOUCHED DOWN AT {self.ts:.2f} S", x, y, 34, CREAM, alpha=k, tracking=1)

    def _lens(self, ctx, d, fo, font):
        """Magnifier on the runner-up's tip: how far short it was."""
        a = ramp(d, 0.7, 1.0) * fo
        if a <= 0:
            return
        R, Z = 118.0, 3.4
        cx = clamp(self.rx, 300.0, 810.0)
        if abs(cx - self.sx) < 150:                      # keep clear of the winning stroke
            cx = clamp(self.sx + (270 if self.rx >= self.sx else -270), 290.0, 820.0)
        cy = GROUND_Y - 262.0
        fx, fy = self.rx, GROUND_Y - 20.0
        s = 0.6 + 0.4 * ease_out_back(ramp(d, 0.7, 1.0))
        ctx.move_to(self.rx, self.ry - 18)
        ctx.line_to(cx, cy + R * s)
        ctx.set_source_rgba(*CREAM, 0.7 * a)
        ctx.set_line_width(3)
        ctx.stroke()
        ctx.new_path()
        ctx.arc(self.rx, fy, 26, 0, TAU)
        ctx.set_source_rgba(*CREAM, 0.8 * a)
        ctx.set_line_width(3)
        ctx.stroke()
        ctx.save()
        ctx.translate(cx, cy)
        ctx.scale(s, s)
        ctx.new_path()
        ctx.arc(0, 0, R, 0, TAU)
        ctx.set_source_rgba(0.05, 0.04, 0.12, 0.96 * a)
        ctx.fill_preserve()
        ctx.save()
        ctx.clip()
        ctx.scale(Z, Z)
        ctx.translate(-fx, -fy)
        ctx.rectangle(fx - 60, GROUND_Y, 120, 40)
        ctx.set_source_rgba(0.10, 0.085, 0.15, a)
        ctx.fill()
        ctx.rectangle(fx - 60, GROUND_Y - 1, 120, 2)
        ctx.set_source_rgba(0.80, 0.74, 0.95, a)
        ctx.fill()
        for i in range(4):
            if not self.near[i]:
                continue
            ctx.new_path()
            for q in self.near[i]:
                ctx.move_to(q[0], q[1])
                ctx.line_to(q[2], q[3])
            ctx.set_source_rgba(*COL[i], a * (1.0 if i in (self.win, self.second) else 0.45))
            ctx.set_line_width(2.6)
            ctx.stroke()
        ctx.new_path()
        ctx.arc(self.rx, self.ry, 3.2, 0, TAU)
        ctx.set_source_rgba(*lighten(COL[self.second], 0.6), a)
        ctx.fill()
        bx = self.rx + (9 if self.rx <= fx + 1 else -9)
        ctx.move_to(bx, self.ry)                          # the gap, bracketed
        ctx.line_to(bx, GROUND_Y - 1)
        ctx.move_to(bx - 3, self.ry)
        ctx.line_to(bx + 3, self.ry)
        ctx.move_to(bx - 3, GROUND_Y - 1)
        ctx.line_to(bx + 3, GROUND_Y - 1)
        ctx.set_source_rgba(*CREAM, a)
        ctx.set_line_width(1.2)
        ctx.stroke()
        ctx.restore()
        ctx.new_path()
        ctx.arc(0, 0, R, 0, TAU)
        ctx.set_source_rgba(*COL[self.second], a)
        ctx.set_line_width(6)
        ctx.stroke()
        font.draw(ctx, f"{self.gap:.0f} PX SHORT", 0, -R + 70, 40, CREAM, anchor="c", alpha=a, stroke=INK, stroke_w=8)
        badge(ctx, self.second, 0, -R, 21, a)
        ctx.restore()

    def _result(self, ctx, d, fo, font):
        k1 = ramp(d, 0.55, 0.85)
        if k1 <= 0 or fo <= 0:
            return
        w = self.win
        a = k1 * fo
        rounded_rect(ctx, HX - 400, 716, 800, 282, 26)
        ctx.set_source_rgba(0.03, 0.03, 0.08, 0.66 * a)
        ctx.fill()
        s = 0.8 + 0.2 * ease_out_back(k1)
        ctx.save()
        ctx.translate(HX, 826)
        ctx.scale(s, s)
        txt = f"{NAMES[w]} STRIKES FIRST"
        tw = font.width(txt, 96)
        badge(ctx, w, -tw / 2 - 12, -36, 30, a)
        font.draw(ctx, txt, 34, 0, 96, COL[w], anchor="c", alpha=a, stroke=INK, stroke_w=12, shadow=5)
        ctx.restore()
        k2 = ramp(d, 0.95, 1.25) * fo
        font.draw(ctx, f"{NAMES[self.second]} WAS {self.gap:.0f} PX FROM THE GROUND", HX, 896, 44, CREAM,
                  anchor="c", alpha=k2, stroke=INK, stroke_w=8)
        k3 = ramp(d, 1.5, 1.8) * fo
        font.draw(ctx, "WHO DID YOU BACK?", HX, 966, 50, AMBER, anchor="c", alpha=k3, stroke=INK, stroke_w=8)

    # ------------------------------------------------------------ sound
    def audio(self):
        sim = self.sim
        rng = np.random.default_rng(self.seed + 909)
        sr = au.SR
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        bed = np.zeros((n + tail, 2))
        ts = self.ts

        def tt(d):
            return np.arange(int(d * sr)) / sr

        # growth crackle: sparse clicks, one voice per leader, panned by x, a little louder near the ground
        fcs = (1500.0, 2150.0, 3000.0, 4200.0)
        t_ = tt(0.022)
        clicks = [[au.bandpass(rng.standard_normal(len(t_)), fc * 0.7, fc * 1.6) * np.exp(-t_ * 260)
                   * np.minimum(1.0, t_ / 0.0006) for _ in range(5)] for fc in fcs]
        pick = rng.random(len(sim["ev"]))
        var = rng.integers(0, 5, len(sim["ev"]))
        jit = rng.random(len(sim["ev"]))
        for j, (t, i, x, y) in enumerate(sim["ev"]):
            if pick[j] < 0.20:
                dep = (y - CLOUD_Y) / HGT
                au.place(fx, clicks[i][var[j]], t + jit[j] / S, 0.30 * (0.6 + 0.4 * dep),
                         clamp((x - 552) / 520, -0.6, 0.6))
        # tension tone and swell follow the front
        frs = sim["frames"]
        ft = np.arange(len(frs)) / FPS
        tsamp = np.arange(n) / sr
        front = np.interp(tsamp, ft, [(max(q[1]) - CLOUD_Y) / HGT for q in frs])
        live = (tsamp < ts).astype(np.float64)
        live = np.convolve(live, np.ones(960) / 960, mode="same")
        ph = TAU * np.cumsum(196.0 * 2 ** front) / sr
        tone = (np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.3 * np.sin(3 * ph) + 0.15 * np.sin(5 * ph))
        tone *= (0.010 + 0.030 * front ** 2) * live * (1 + 0.25 * np.sin(TAU * (2.0 + 5.0 * front) * tsamp))
        t0 = ts * 0.55
        sw = au.swept_noise(rng, ts - t0, 350, 3200)
        env = (np.arange(len(sw)) / max(1, len(sw))) ** 2.2
        au.place(bed, sw * env, t0, 0.08)
        bed[:n, 0] += tone
        bed[:n, 1] += tone
        del front, live, ph, tone, tsamp
        # pulse: a soft wooden knock that speeds up as the front nears the ground
        tk = 0.0
        while tk < ts - 0.12:
            fr_ = max(frs[min(len(frs) - 1, int(tk * FPS))][1]) - CLOUD_Y
            u = fr_ / HGT
            au.place(fx, au.marimba(au.hz(62 + 12 * u), 0.35), tk, 0.20 + 0.04 * u)
            tk += lerp(1.05, 0.30, u ** 1.5)
        # altitude lines: a rising run of soft bells
        scale = (57, 60, 62, 64, 67, 69, 72, 74, 76)
        for tm, j in self.miles:
            au.place(fx, au.bell(au.hz(scale[j - 1]), 1.2), tm, 0.17, -0.2)
        # lead changes: a pluck in the new leader's pitch
        notes = (64, 67, 71, 74)
        for tc, k in self.changes:
            if tc < ts - 0.02:
                pan = (X0[k] - 552) / 700
                au.place(fx, au.pluck(au.hz(notes[k]), 0.6), tc, 0.17, pan)
                au.place(fx, au.glide(au.hz(notes[k] + 12), au.hz(notes[k] + 19), 0.09, decay=0.05), tc, 0.07, pan)
        # the strike: broadband burst, boom, brown rumble with a falling sub
        pan = clamp((self.sx - 552) / 420, -0.6, 0.6)
        tb = tt(0.7)
        burst = au.noise(rng, 0.7, 150, 4500) * np.exp(-tb * 7.5) * np.minimum(1.0, tb / 0.004)
        au.place(fx, burst, ts, 0.62, pan)
        au.place(fx, au.boom(rng, 1.8, 120, 34), ts, 0.72, pan * 0.5)
        rl = 3.4
        br = np.cumsum(rng.standard_normal(int(rl * sr)))
        br = au.bandpass(br - np.convolve(br, np.ones(4800) / 4800, mode="same"), 45, 1300)
        br /= max(1e-9, np.abs(br).max())
        tr = tt(rl)
        br = br[:len(tr)] * np.minimum(1, tr / 0.03) * np.exp(-tr / 1.0) * (1 + 0.35 * np.sin(TAU * 3.1 * tr))
        au.place(fx, br, ts + 0.02, 0.60, pan * 0.4)
        au.place(fx, au.glide(96, 40, 2.2, decay=0.8, harm=0.5), ts, 0.35)
        # result: the winner's note, then the reset
        au.place(fx, au.bell(au.hz(notes[self.win] + 12), 2.0), ts + 0.6, 0.16)
        au.place(fx, au.bell(au.hz(notes[self.win] + 19), 1.6), ts + 0.98, 0.09, 0.2)
        au.place(fx, au.bell(au.hz(notes[self.win] + 24), 1.6), ts + 1.5, 0.07, -0.2)
        au.place(fx, au.pluck(au.hz(62), 0.7), ts + 2.6, 0.10, -0.3)
        au.place(fx, au.pluck(au.hz(57), 0.7), ts + 3.25, 0.10, 0.3)
        d = 0.6
        au.place(fx, au.swept_noise(rng, d, 2400, 500) * np.sin(np.pi * tt(d) / d), self.dur - 0.7, 0.22)
        # quiet bed
        au.place(bed, au.pad(au.hz(57), self.dur, 1.0, 0.8), 0.0, 0.014)
        au.place(bed, au.pad(au.hz(64), self.dur, 1.0, 0.8), 0.0, 0.010)
        au.place(bed, au.pad(au.hz(45), self.dur, 1.0, 0.8), 0.0, 0.008)
        mix_ = au.reverb(fx, rng, decay=1.2, wet=0.10) + bed
        out = mix_[:n].copy()
        out[:tail] += mix_[n:]
        out = au.eq(out, self.EQ)
        pk = np.abs(out).max()
        out = np.tanh(out * (self.DRIVE / max(1e-9, pk)))     # soft-clip the strike so mastering reaches -14 LUFS
        return out

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        st = self.stats
        title = f"4 Bolts Race to the Ground… Decided in the Last {int(round(st['last_lead_change_px_from_ground']))} Pixels ⚡"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "Blue, pink, gold or mint: which bolt touches the ground first? Call it now!\n\n"
                "Four colored leaders start on the cloud line and grow toward the ground. This is a stylized "
                "growth model, not a simulation of real lightning. Every leader gets exactly the same 30 steps "
                "per second. On each step one of its live tips, picked at random, grows 8 pixels at a random "
                "angle up to 68 degrees either side of straight down, and 12 percent of steps leave a fork "
                "behind, which becomes one more live tip. The pale circles are drifting dry-air patches: a tip "
                "inside one is stalled, and any step that lands on it is wasted. A leader with few forks drops "
                "fast but can be stopped dead by one patch; a heavily forked one spends its steps on many tips "
                "but usually has a branch that keeps going. Tips left more than 60 pixels behind their own "
                "leader's lowest tip die out, a leader keeps at most 6 live tips, and the side walls bounce tips "
                "back in. The bars on the left edge show each color's lowest tip, and the dashed line marks the "
                "current lead. The first leader to touch the ground wins, and the magnifier shows how far short "
                "the runner-up was.\n"
                f"Growth, picture and sound are generated by code (seed {self.seed}), with no footage, samples or "
                "stock media.\n"
                "Which color did you back?\n"
                "#simulation #satisfying #race"),
            tags=["simulation", "race simulation", "storm race", "branching", "random growth", "prediction",
                  "which one wins", "satisfying", "made with code", "generative animation"],
            pinned_comment=("Blue, pink, gold or mint: which one touches the ground first? Pick before the first "
                            "fork. Same 30 steps per second for all four, and dry air stalls any tip inside it."),
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of abstract colored lines; a "
                                              "stylized growth model that could not be mistaken for real "
                                              "footage, people, places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / FPS, 2),
            drama=st,
        )
