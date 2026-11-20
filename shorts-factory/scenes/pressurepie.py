"""PRESSURE PIE: four spokes split a disc into four colored wedges, one ball bounces in each.

A ball that hits a spoke rotates it away, by an amount proportional to that ball's own wedge angle,
and every push grows 3% per second. A wedge under 14 degrees pops and its neighbors close the gap.
Last color holding the full disc wins.

simulate(seed) is deterministic and needs no rendering, so make.py can score hundreds of seeds for
drama (drama()) and render only the best one (Scene).
"""
import bisect
import math
import os
import random

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, darken, ease_in_out, ease_out, ease_out_back, glow, hex_rgb, lerp,
                        lighten, mix, polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30
SUB = 6                       # physics substeps per frame
DT = 1.0 / (FPS * SUB)

R = 340.0                     # disc radius (px)
RB = 18.0                     # ball radius
RH = 54.0                     # hub radius: balls bounce between hub and rim
V = 1000.0                    # ball speed, constant
POP_DEG = 14.0
POP_ANG = math.radians(POP_DEG)
GROW = 1.03                   # push multiplier grows 3% per second
K = 0.22                      # spoke turn per hit = K * own wedge angle * multiplier (radians)
TS = 0.16                     # spoke motion time constant: a push is eased in over ~0.3 s
JIT = 0.10                    # bounce angle jitter (radians), keeps paths from locking into loops
T_MAX = 28.6                  # a round that has not ended by now is rejected by the seed search
CLOSE_T = 0.30                # popped wedge closing animation (render only)
PAYOFF = 3.1                  # seconds from the final pop to the loop point

NAMES = ("ORANGE", "TEAL", "PINK", "LIME")       # symbols: dot, square, triangle, cross


def mult(t):
    return GROW ** max(0.0, t)


def _jitter(rng, vx, vy, nx, ny):
    a = rng.uniform(-JIT, JIT)
    c, s = math.cos(a), math.sin(a)
    jx, jy = vx * c - vy * s, vx * s + vy * c
    if jx * nx + jy * ny > 0.25 * V:
        return jx, jy
    return vx, vy


def simulate(seed, record=False):
    rng = random.Random(seed * 7907 + 13)
    s = [math.radians(-45 + 90 * j) for j in range(4)]      # spoke angles; wedge j spans s[j] .. s[j+1]
    cols = [0, 1, 2, 3]
    om = [0.0] * 4
    fwd = [(-1, -1.0)] * 4        # last push of spoke i into wedge i (by wedge i-1): (color, t)
    bwd = [(-1, -1.0)] * 4        # last push of spoke i into wedge i-1 (by wedge i)
    balls = {}
    for j in range(4):
        a = s[j] + TAU / 8 + rng.uniform(-0.42, 0.42)
        r = rng.uniform(150, 280)
        d = rng.uniform(0, TAU)
        balls[j] = [r * math.cos(a), r * math.sin(a), V * math.cos(d), V * math.sin(d)]
    hits, pops, frames = [], [], []
    final, winner = None, None
    dec = math.exp(-DT / TS)
    step = 0
    nmax = int(T_MAX * FPS * SUB)
    while True:
        t = step * DT
        n = len(s)
        if n == 1:
            widths = [TAU]
        else:
            widths = [(s[(j + 1) % n] - s[j]) % TAU for j in range(n)]
        if step % SUB == 0:
            frames.append((tuple((cols[j], s[j], widths[j]) for j in range(n)),
                           tuple((c, b[0], b[1]) for c, b in balls.items())))
        if final is not None or step >= nmax:
            break
        m = mult(t)
        for i in range(n):
            s[i] += om[i] * DT
            om[i] *= dec
        widths = [(s[(j + 1) % n] - s[j]) % TAU for j in range(n)]
        for j in range(n):
            c = cols[j]
            b = balls[c]
            x, y, vx, vy = b
            x += vx * DT
            y += vy * DT
            w = widths[j]
            a0 = s[j]
            r = math.hypot(x, y)
            if r > R - RB:
                nx, ny = x / r, y / r
                vn = vx * nx + vy * ny
                if vn > 0:
                    vx, vy = vx - 2 * vn * nx, vy - 2 * vn * ny
                    vx, vy = _jitter(rng, vx, vy, -nx, -ny)
                x, y, r = nx * (R - RB), ny * (R - RB), R - RB
            rmin = max(RH + RB, RB / math.sin(min(w, math.pi - 0.01) / 2) + 0.5)
            if r < rmin:
                nx, ny = x / r, y / r
                vn = vx * nx + vy * ny
                if vn < 0:
                    vx, vy = vx - 2 * vn * nx, vy - 2 * vn * ny
                    vx, vy = _jitter(rng, vx, vy, nx, ny)
                x, y, r = nx * rmin, ny * rmin, rmin
            rel = (math.atan2(y, x) - a0) % TAU
            if rel > w:                                   # a moving spoke swept past the ball: put it back
                rel = w if rel - w < TAU - rel else 0.0
                x, y = r * math.cos(a0 + rel), r * math.sin(a0 + rel)
            if rel < math.pi / 2:
                d = r * math.sin(rel)
                if d < RB:
                    nx, ny = -math.sin(a0), math.cos(a0)
                    x += nx * (RB - d)
                    y += ny * (RB - d)
                    vn = vx * nx + vy * ny
                    if vn < 0:
                        vx, vy = vx - 2 * vn * nx, vy - 2 * vn * ny
                        vx, vy = _jitter(rng, vx, vy, nx, ny)
                        if n > 1:
                            om[j] -= K * w * m / TS
                            bwd[j] = (c, t)
                            hits.append((t, c, x, y, nx, ny, w))
            rel1 = w - rel
            if rel1 < math.pi / 2:
                a1 = a0 + w
                d = r * math.sin(rel1)
                if d < RB:
                    nx, ny = math.sin(a1), -math.cos(a1)
                    x += nx * (RB - d)
                    y += ny * (RB - d)
                    vn = vx * nx + vy * ny
                    if vn < 0:
                        vx, vy = vx - 2 * vn * nx, vy - 2 * vn * ny
                        vx, vy = _jitter(rng, vx, vy, nx, ny)
                        if n > 1:
                            i1 = (j + 1) % n
                            om[i1] += K * w * m / TS
                            fwd[i1] = (c, t)
                            hits.append((t, c, x, y, nx, ny, w))
            b[0], b[1], b[2], b[3] = x, y, vx, vy
        widths = [(s[(j + 1) % n] - s[j]) % TAU for j in range(n)]
        j = min(range(n), key=lambda q: widths[q])
        if widths[j] < POP_ANG:
            i1 = (j + 1) % n
            c = cols[j]
            by = max((fwd[j], bwd[i1]), key=lambda p: p[1])[0]
            bx, byy = balls[c][0], balls[c][1]
            pops.append(dict(t=round(t, 4), color=c, by=by, a0=s[j], w=widths[j], x=bx, y=byy,
                             shares=[0.0] * 4))
            for jj in range(n):
                pops[-1]["shares"][cols[jj]] = widths[jj] / TAU
            del balls[c]
            mid = s[j] + widths[j] / 2
            s[j] = mid
            om[j] = 0.0
            fwd[j] = bwd[j] = (-1, -1.0)
            if i1 > 0:
                del s[i1], om[i1], fwd[i1], bwd[i1], cols[j]
            else:
                del s[0], om[0], fwd[0], bwd[0]
                cols = cols[1:n - 1] + [cols[0]]
            if len(s) == 1:
                final, winner = t, cols[0]
        step += 1
    return dict(seed=seed, frames=frames, hits=hits if record else None, pops=pops, final=final,
                winner=winner, nhits=len(hits))


def shares_of(state):
    sh = [0.0] * 4
    for c, a0, w in state[0]:
        sh[c] = w / TAU
    return sh


def leaders(frames):
    """Leader per frame: the largest wedge; it changes only when another wedge is strictly bigger."""
    out, lead = [], -1
    for f, st in enumerate(frames):
        sh = shares_of(st)
        if f / FPS >= 1.0:
            best = max(range(4), key=lambda k: sh[k])
            if lead < 0 or sh[best] > sh[lead] + 0.004:
                lead = best if sh[best] > 0.255 or lead >= 0 else -1
        out.append(lead)
    return out


def drama(sim):
    frames = sim["frames"]
    pops = sim["pops"]
    lead = leaders(frames)
    changes = sum(1 for a, b in zip(lead, lead[1:]) if a >= 0 and b >= 0 and a != b)
    fin = sim["final"]
    first = pops[0]["t"] if pops else None
    f20 = min(len(frames) - 1, 20 * FPS)
    sh20 = sorted(shares_of(frames[f20]), reverse=True)
    split20 = sh20[0] / max(1e-9, sh20[0] + sh20[1])
    win = sim["winner"]
    half = lead[min(len(lead) - 1, int((fin or T_MAX) * FPS / 2))]
    wmin = min(shares_of(st)[win] for st in frames[FPS:]) if win is not None else 0.0
    alive20 = sum(1 for v in sh20 if v > 0)
    score = 0.0
    if fin is None:
        score -= 100.0
    else:
        score += 4.0 if 24.0 <= fin <= 28.6 else -40.0 - 5 * abs(fin - 26.3)
        score += 1.5 if 25.0 <= fin <= 28.0 else 0.0
    score += 1.2 * min(changes, 12) + (0.0 if changes >= 6 else -8.0)
    score += 3.0 if (first or 0) >= 10.0 else -12.0 + (first or 0)
    score += 3.0 - 20.0 * max(0.0, abs(split20 - 0.5) - 0.1) - 6.0 * abs(split20 - 0.5)
    score += 2.0 if (win is not None and half != win) else 0.0
    score += 0.04 * (100 * (0.25 - wmin)) if win is not None else 0.0
    score += 0.6 * len(set(x for x in lead if x >= 0))
    return dict(
        seed=sim["seed"], score=round(score, 3),
        winner=NAMES[win] if win is not None else None, winner_index=win,
        finished=fin is not None, final_pop_s=round(fin, 2) if fin is not None else None,
        first_pop_s=round(first, 2) if first is not None else None,
        pops=[dict(color=NAMES[p["color"]], at_s=round(p["t"], 2), pushed_by=NAMES[p["by"]] if p["by"] >= 0 else None)
              for p in pops],
        lead_changes=changes, colors_that_led=len(set(x for x in lead if x >= 0)),
        leader_at_half=NAMES[half] if half >= 0 else None, winner_led_at_half=half == win,
        top_two_split_at_20s=[round(100 * sh20[0], 1), round(100 * sh20[1], 1)],
        top_two_split_at_20s_ratio=round(split20, 3), colors_alive_at_20s=alive20,
        winner_smallest_share_percent=round(100 * wmin, 1), spoke_hits=sim["nhits"],
    )


def score_seed(seed):
    return drama(simulate(seed))


# ====================================================================== look
COLS = [hex_rgb(c) for c in ("#FF8A1F", "#1FD6C1", "#FF4F8E", "#B8EE3A")]
INK = hex_rgb("#170D1A")
CREAM = hex_rgb("#FFF1DC")
METAL = hex_rgb("#5A4660")
DANGER = hex_rgb("#FF2E3B")
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
CX, CY = 478.0, 880.0         # disc center: shifted left so the rim labels stay out of the right margin
X_SAFE = 950.0                # right edge of the safe area
QX = 60.0                     # left edge of the header and legend


def symbol(ctx, k, x, y, s):
    """Path of color k's symbol (dot, square, triangle, cross) centered on (x, y)."""
    if k == 0:
        ctx.new_sub_path()
        ctx.arc(x, y, s * 0.92, 0, TAU)
    elif k == 1:
        ctx.rectangle(x - s * 0.82, y - s * 0.82, s * 1.64, s * 1.64)
    elif k == 2:
        polygon(ctx, [(x + s * 1.12 * math.cos(a), y + s * 0.15 + s * 1.12 * math.sin(a))
                      for a in (-TAU / 4, -TAU / 4 + TAU / 3, -TAU / 4 + 2 * TAU / 3)])
    else:
        a, b = s * 0.34, s
        pts = [(-a, -b), (a, -b), (a, -a), (b, -a), (b, a), (a, a), (a, b), (-a, b), (-a, a), (-b, a), (-b, -a), (-a, -a)]
        c, sn = math.cos(TAU / 8), math.sin(TAU / 8)
        polygon(ctx, [(x + px * c - py * sn, y + px * sn + py * c) for px, py in pts])


def arrow(ctx, x, y, dx, dy, length, width):
    """Arrow path from (x, y) along unit direction (dx, dy)."""
    px, py = -dy, dx
    hl = min(length * 0.6, width * 2.0)
    bx, by = x + dx * (length - hl), y + dy * (length - hl)
    polygon(ctx, [(x + px * width / 2, y + py * width / 2), (bx + px * width / 2, by + py * width / 2),
                  (bx + px * width * 1.25, by + py * width * 1.25), (x + dx * length, y + dy * length),
                  (bx - px * width * 1.25, by - py * width * 1.25), (bx - px * width / 2, by - py * width / 2),
                  (x - px * width / 2, y - py * width / 2)])


def make_background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = (yy / H)[..., None]
    img = np.array([0.105, 0.062, 0.110], np.float32) * (1 - v) + np.array([0.040, 0.026, 0.048], np.float32) * v
    d = np.hypot(xx - CX, yy - CY) / 760
    img += (np.clip(1 - d, 0, 1) ** 2)[..., None] * np.array([0.150, 0.070, 0.070], np.float32)
    d = np.hypot(xx - 980, yy - 260) / 620
    img += (np.clip(1 - d, 0, 1) ** 2)[..., None] * np.array([0.050, 0.030, 0.075], np.float32)
    img += ((((xx + yy) // 14) % 2 == 0).astype(np.float32) * 0.006)[..., None]      # faint diagonal hatch
    vig = np.clip(1.15 - 0.55 * (((xx - 540) / 700) ** 2 + ((yy - 960) / 1150) ** 2), 0.3, 1.0)
    img *= vig[..., None]
    img += np.random.default_rng(11).normal(0, 0.006, (H, W, 1)).astype(np.float32)
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    ctx = cairo.Context(surf)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    for i in range(72):                                   # gauge ticks around the disc
        a = i * TAU / 72
        big = i % 9 == 0
        r0, r1 = R + 15, R + (29 if big else 22)
        ctx.move_to(CX + r0 * math.cos(a), CY + r0 * math.sin(a))
        ctx.line_to(CX + r1 * math.cos(a), CY + r1 * math.sin(a))
        ctx.set_source_rgba(1, 0.86, 0.75, 0.30 if big else 0.14)
        ctx.set_line_width(3 if big else 2)
        ctx.stroke()
    surf.flush()
    return surf, buf


# ====================================================================== the scene
class Scene:
    W, H, FPS = W, H, FPS

    def __init__(self, seed):
        self.seed = seed
        self.sim = simulate(seed, record=True)
        if self.sim["final"] is None:
            raise ValueError(f"seed {seed} does not finish within {T_MAX} s")
        self.stats = drama(self.sim)
        self.winner = self.sim["winner"]
        self.tf = self.sim["final"]
        self.dur = self.tf + PAYOFF
        self.nframes = int(round(self.dur * FPS))
        self.t_reset = self.dur - 0.62
        self._res = None
        self._prep()

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_res"] = None
        return d

    def _prep(self):
        sim = self.sim
        self.lead = leaders(sim["frames"])
        self.hit_t = [h[0] for h in sim["hits"]]
        self.shares = [shares_of(st) for st in sim["frames"]]
        rnd = random.Random(self.seed * 31 + 7)
        self.shards = [[(rnd.uniform(0, TAU), rnd.uniform(160, 620), rnd.uniform(5, 12), rnd.uniform(0, TAU),
                         rnd.uniform(-9, 9)) for _ in range(30)] for _ in sim["pops"]]
        self.win_share_at_pop = sim["pops"][-1]["shares"][self.winner]
        last = len(sim["pops"]) - 1
        self.shakes = [(p["t"], 15.0 if i == last else 9.0) for i, p in enumerate(sim["pops"])]

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            bg, keep = make_background()
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep,
                             font=Font(FONT_PATH))
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
    def state(self, f):
        fr = self.sim["frames"]
        return fr[max(0, min(f, len(fr) - 1))]

    def shake(self, t):
        sx = sy = 0.0
        for te, amp in self.shakes:
            d = t - te
            if 0 <= d < 0.5:
                a = amp * math.exp(-d / 0.10)
                sx += a * math.sin(d * 73 + te * 13)
                sy += a * math.cos(d * 61 + te * 7)
        return sx, sy

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, f, res):
        t = f / FPS
        font = res["font"]
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(res["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        tf = self.tf
        kr = ease_in_out(ramp(t, self.t_reset, self.dur - 1.5 / FPS))      # rewind to the opening state
        sx, sy = self.shake(t)
        ctx.save()
        ctx.translate(sx, sy)
        if kr < 1:
            self._disc(ctx, self.state(f), t, f, font)
        if kr > 0:
            ctx.push_group()
            self._disc(ctx, self.state(0), 0.0, 0, font, opening=True)
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(kr)
        fl = max(0.34 * math.exp(-t / 0.2), 0.34 * kr ** 3)      # opening pressure flash; rebuilt for the loop
        if fl > 0.004:
            ctx.arc(CX, CY, R, 0, TAU)
            ctx.set_source_rgba(1.0, 0.95, 0.88, fl)
            ctx.fill()
            ctx.arc(CX, CY, R + 4 + 30 * (1 - fl / 0.34), 0, TAU)
            ctx.set_source_rgba(1.0, 0.95, 0.88, fl * 1.5)
            ctx.set_line_width(14)
            ctx.stroke()
        self._pops(ctx, t, font)
        if t < tf + 0.3:
            self._labels(ctx, f, t, font, 1 - ramp(t, tf, tf + 0.25))
        elif kr > 0:
            self._labels(ctx, 0, 0.0, font, kr)
        self._finale(ctx, t, font, kr)
        ctx.restore()
        self._header(ctx, t, font)
        self._legend(ctx, t, font, max(1 - ramp(t, tf + 0.15, tf + 0.45), ease_in_out(ramp(t, self.t_reset + 0.22, self.dur - 1.5 / FPS))))

    def _wedge(self, ctx, c, a0, w, alpha=1.0, bright=0.0):
        col = COLS[c]
        ctx.move_to(CX, CY)
        ctx.arc(CX, CY, R, a0, a0 + w)
        ctx.close_path()
        g = cairo.RadialGradient(CX, CY, RH, CX, CY, R)
        g.add_color_stop_rgba(0, *darken(col, 0.62 - 0.3 * bright), alpha)
        g.add_color_stop_rgba(0.82, *darken(col, 0.22 - 0.2 * bright), alpha)
        g.add_color_stop_rgba(1, *lighten(col, 0.10 * bright), alpha)
        ctx.set_source(g)
        ctx.fill()
        if w > math.radians(26) and w < TAU - 0.01:     # faint watermark symbol so a wedge reads without color
            m = a0 + w / 2
            symbol(ctx, c, CX + 0.64 * R * math.cos(m), CY + 0.64 * R * math.sin(m), 30)
            ctx.set_source_rgba(*lighten(col, 0.25), 0.22 * alpha)
            ctx.fill()

    def _spoke(self, ctx, a, alpha=1.0, flash=None):
        if alpha <= 0.003:
            return
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.move_to(CX + (RH - 6) * math.cos(a), CY + (RH - 6) * math.sin(a))
        ctx.line_to(CX + (R + 8) * math.cos(a), CY + (R + 8) * math.sin(a))
        ctx.set_source_rgba(*INK, 0.85 * alpha)
        ctx.set_line_width(15)
        ctx.stroke_preserve()
        col = CREAM if flash is None else mix(CREAM, flash[0], 0.6 * flash[1])
        ctx.set_source_rgba(*col, alpha)
        ctx.set_line_width(8)
        ctx.stroke()

    def _disc(self, ctx, st, t, f, font, opening=False):
        wedges, balls = st
        tf = self.tf
        after = (not opening) and t >= tf
        glow(ctx, CX, CY + 18, R + 70, (0, 0, 0), 0.55)
        for c, a0, w in wedges:
            self._wedge(ctx, c, a0, w)
        if not opening:                                   # popped wedge closing: a shrinking sliver of its color
            for p in self.sim["pops"]:
                if 0 <= t - p["t"] < CLOSE_T:
                    e = ease_in_out(ramp(t, p["t"], p["t"] + CLOSE_T))
                    self._wedge(ctx, p["color"], p["a0"] + p["w"] / 2 * e, p["w"] * (1 - e), 1 - 0.6 * e, bright=1.0)
        if after:                                         # the disc floods to the winner
            kb = ramp(t, tf + 0.1, tf + 0.6)
            if kb > 0:
                self._wedge(ctx, self.winner, 0, TAU, kb * 0.85, bright=0.6)
            k = ramp(t, tf + 0.12, tf + 0.85)
            if 0 < k < 1:
                rr = RH + (R - RH) * ease_out(k, 2)
                g = cairo.RadialGradient(CX, CY, max(0, rr - 90), CX, CY, rr + 6)
                col = lighten(COLS[self.winner], 0.45)
                g.add_color_stop_rgba(0, *col, 0)
                g.add_color_stop_rgba(0.85, *col, 0.6 * (1 - k))
                g.add_color_stop_rgba(1, *col, 0)
                ctx.set_source(g)
                ctx.arc(CX, CY, min(R, rr + 6), 0, TAU)
                ctx.fill()
        # rim ring
        for rad, col, a, lw in ((R + 4, INK, 1, 16), (R + 4, METAL, 1, 9), (R + 2, (1, 0.9, 0.85), 0.25, 2)):
            ctx.arc(CX, CY, rad, 0, TAU)
            ctx.set_source_rgba(*col, a)
            ctx.set_line_width(lw)
            ctx.stroke()
        # spokes, tinted with the pusher's color right after a hit
        hits = []
        if not opening and t < tf:
            i0 = bisect.bisect_left(self.hit_t, t - 0.30)
            hits = self.sim["hits"][i0:bisect.bisect_right(self.hit_t, t)]
        if not after and len(wedges) > 1:
            for c, a0, w in wedges:
                fl = None
                for ht, hc, hx, hy, nx, ny, hw in hits:
                    a = math.atan2(hy - ny * RB, hx - nx * RB)
                    if t - ht < 0.25 and abs(((a0 - a + math.pi) % TAU) - math.pi) < 0.12:
                        fl = (COLS[hc], 1 - (t - ht) / 0.25)
                self._spoke(ctx, a0, 1.0, fl)
        if not opening:
            for p in self.sim["pops"]:
                if 0 <= t - p["t"] < CLOSE_T + 0.02:
                    e = ease_in_out(ramp(t, p["t"], p["t"] + CLOSE_T))
                    al = 1.0 if p["t"] < tf else 1 - e
                    self._spoke(ctx, p["a0"] + p["w"] / 2 * e, al)
                    self._spoke(ctx, p["a0"] + p["w"] - p["w"] / 2 * e, al)
        # push arrows: the spoke is shoved away from the ball; arrow length shows the size of that push
        for ht, hc, hx, hy, nx, ny, hw in hits:
            age = (t - ht) / 0.30
            px, py = CX + hx - nx * RB, CY + hy - ny * RB
            glow(ctx, px, py, 34, lighten(COLS[hc], 0.3), 0.75 * (1 - age))
            push = K * hw * mult(ht)
            ln = 18 + 72 * min(1.0, push / 0.9)
            arrow(ctx, px - nx * 10, py - ny * 10, -nx, -ny, ln * ease_out(min(1, age * 3 + 0.2)), 10)
            ctx.set_source_rgba(*INK, 0.9 * (1 - age))
            ctx.set_line_width(5)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)
            ctx.stroke_preserve()
            ctx.set_source_rgba(*lighten(COLS[hc], 0.2), 1 - age)
            ctx.fill()
        # hub: shows the live push multiplier
        ctx.arc(CX, CY, RH + 2, 0, TAU)
        g = cairo.RadialGradient(CX - 14, CY - 18, 4, CX, CY, RH + 2)
        g.add_color_stop_rgb(0, *lighten(METAL, 0.25))
        g.add_color_stop_rgb(1, *darken(METAL, 0.45))
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_source_rgba(*INK, 1)
        ctx.set_line_width(5)
        ctx.stroke()
        if after:                                         # the winning ball lingers a moment, then melts in
            fade = 1 - ramp(t, tf + 0.15, tf + 0.45)
            for c, x, y in balls:
                if fade > 0:
                    self._ball(ctx, c, CX + x, CY + y, fade)
        if not after:
            font.draw(ctx, f"×{mult(min(t, tf)):.2f}", CX, CY + 8, 34, CREAM, anchor="c")
            font.draw(ctx, "PUSH", CX, CY + 33, 18, lighten(METAL, 0.6), anchor="c", tracking=1.5)
            for c, x, y in balls:
                if not opening:
                    for back in range(4, 0, -1):
                        if f - back < 0:
                            continue
                        for c2, x2, y2 in self.state(f - back)[1]:
                            if c2 == c:
                                ctx.arc(CX + x2, CY + y2, RB * (1 - back * 0.13), 0, TAU)
                                ctx.set_source_rgba(*lighten(COLS[c], 0.3), 0.16 * (5 - back) / 4)
                                ctx.fill()
                self._ball(ctx, c, CX + x, CY + y)

    def _ball(self, ctx, c, x, y, alpha=1.0, rb=RB):
        col = COLS[c]
        glow(ctx, x, y, rb * 2.6, col, 0.45 * alpha)
        ctx.arc(x, y, rb + 3, 0, TAU)
        ctx.set_source_rgba(*INK, alpha)
        ctx.fill()
        g = cairo.RadialGradient(x - rb * 0.4, y - rb * 0.45, 1, x, y, rb)
        g.add_color_stop_rgba(0, *lighten(col, 0.6), alpha)
        g.add_color_stop_rgba(1, *col, alpha)
        ctx.arc(x, y, rb, 0, TAU)
        ctx.set_source(g)
        ctx.fill()
        symbol(ctx, c, x, y + (1 if c == 2 else 0), rb * 0.48)
        ctx.set_source_rgba(*INK, alpha)
        ctx.fill()

    def _pops(self, ctx, t, font):
        for p, shards in zip(self.sim["pops"], self.shards):
            d = t - p["t"]
            if not 0 <= d < 1.1:
                continue
            col = COLS[p["color"]]
            x, y = CX + p["x"], CY + p["y"]
            ctx.arc(x, y, 20 + 170 * ease_out(min(1, d / 0.45)), 0, TAU)
            ctx.set_source_rgba(*lighten(col, 0.4), max(0, 0.9 * (1 - d / 0.45)))
            ctx.set_line_width(10)
            ctx.stroke()
            if d < 0.2:
                glow(ctx, x, y, 160, lighten(col, 0.5), 0.9 * (1 - d / 0.2))
            for a, v, sz, rot, spin in shards:
                dist = v * (1 - math.exp(-d * 3.2)) / 3.2
                ctx.save()
                ctx.translate(x + math.cos(a) * dist, y + math.sin(a) * dist + 160 * d * d)
                ctx.rotate(rot + spin * d)
                ctx.rectangle(-sz, -sz * 0.5, sz * 2, sz)
                ctx.set_source_rgba(*lighten(col, 0.2), 1 - d / 1.1)
                ctx.fill()
                ctx.restore()
            if d < 0.9:
                s = 72 * ease_out_back(min(1, d / 0.22))
                font.draw(ctx, "POP!", x, y - 30 - 70 * d, s, CREAM, anchor="c", alpha=1 - ramp(d, 0.6, 0.9),
                          stroke=INK, stroke_w=10)

    def _labels(self, ctx, f, t, font, alpha):
        if alpha <= 0.003:
            return
        lead = self.lead[min(f, len(self.lead) - 1)] if t > 0 else -1
        items = [[a0 + w / 2, c, int(round(100 * w / TAU)), w, 1.0] for c, a0, w in self.state(f)[0]]
        live = {it[1] for it in items}
        for p in self.sim["pops"]:                          # popped wedge: its tag drops to 0% and fades
            d = t - p["t"]
            if 0 <= d < 0.55 and t < self.tf and p["color"] not in live:
                items.append([p["a0"] + p["w"] / 2, p["color"], 0, 0.0, 1 - d / 0.55])
        size = 42
        geo = []
        for it in items:
            txt = f"{it[2]}%"
            sc = 1.12 if it[1] == lead else 1.0
            geo.append([it[0] % TAU, (68 + font.width(txt, size)) * sc, 62 * sc, txt, it, sc])
        geo.sort(key=lambda g: g[0])
        RL = R + 34

        def ext(a, pw, ph):
            return abs(math.sin(a)) * pw / 2 + abs(math.cos(a)) * ph / 2

        n = len(geo)
        for _ in range(40 if n > 1 else 0):
            for i in range(n):
                g1, g2 = geo[i], geo[(i + 1) % n]
                gap = (g2[0] - g1[0]) % TAU
                need = (ext(g1[0], g1[1], g1[2]) + ext(g2[0], g2[1], g2[2])) / (RL + 40) + 0.03
                if gap < need:
                    g1[0] -= (need - gap) / 2
                    g2[0] += (need - gap) / 2
        for a, pw, ph, txt, it, sc in geo:
            c, fade = it[1], it[4]
            rad = RL + abs(math.cos(a)) * pw / 2 + abs(math.sin(a)) * ph / 2
            x = clamp(CX + rad * math.cos(a), 28 + pw / 2, X_SAFE - pw / 2)
            y = CY + rad * math.sin(a)
            al = alpha * fade
            danger = 0 < it[3] < math.radians(26)
            pulse = 0.5 + 0.5 * math.sin(t * 22) if danger else 0.0
            rounded_rect(ctx, x - pw / 2 + 3, y - ph / 2 + 5, pw, ph, ph / 2)
            ctx.set_source_rgba(0, 0, 0, 0.35 * al)
            ctx.fill()
            rounded_rect(ctx, x - pw / 2, y - ph / 2, pw, ph, ph / 2)
            ctx.set_source_rgba(*(COLS[c] if fade >= 1 else darken(COLS[c], 0.4)), al)
            ctx.fill_preserve()
            if c == lead:
                ctx.set_source_rgba(*CREAM, al)
                ctx.set_line_width(6)
            elif danger:
                ctx.set_source_rgba(*mix(DANGER, CREAM, pulse), al)
                ctx.set_line_width(4 + 3 * pulse)
            else:
                ctx.set_source_rgba(*INK, al)
                ctx.set_line_width(3)
            ctx.stroke()
            symbol(ctx, c, x - pw / 2 + 28 * sc, y, 11 * sc)
            ctx.set_source_rgba(*INK, al)
            ctx.fill()
            font.draw(ctx, txt, x - pw / 2 + 48 * sc, y + 16 * sc, size * sc, INK, alpha=al)

    def _finale(self, ctx, t, font, kr):
        tf = self.tf
        if t < tf:
            return
        out = 1 - ease_in_out(ramp(t, self.t_reset - 0.3, self.t_reset + 0.2))
        win = self.winner
        col = COLS[win]
        # share-over-time trace around the rim, swept clockwise from the top
        ks = ease_in_out(ramp(t, tf + 0.45, tf + 1.65))
        if ks > 0 and out > 0:
            sh = self.shares
            nfr = len(sh)
            N = 240
            upto = min(N, int(N * ks)) + 1
            r0, T = R + 12, 58
            samples = [sh[min(nfr - 1, int(i / N * (nfr - 1)))] for i in range(upto)]
            for c in range(4):
                inner, outer = [], []
                for i, s_ in enumerate(samples):
                    a = -math.pi / 2 + TAU * min(i / N, ks)
                    cum = sum(s_[:c])
                    ca, sa = math.cos(a), math.sin(a)
                    inner.append((CX + (r0 + T * cum) * ca, CY + (r0 + T * cum) * sa))
                    outer.append((CX + (r0 + T * (cum + s_[c])) * ca, CY + (r0 + T * (cum + s_[c])) * sa))
                polygon(ctx, inner + outer[::-1])
                ctx.set_source_rgba(*COLS[c], 0.95 * out)
                ctx.fill()
            a = -math.pi / 2 + TAU * ks                          # sweep head
            ctx.move_to(CX + (r0 - 4) * math.cos(a), CY + (r0 - 4) * math.sin(a))
            ctx.line_to(CX + (r0 + T + 8) * math.cos(a), CY + (r0 + T + 8) * math.sin(a))
            ctx.set_source_rgba(*CREAM, out * (1 - ramp(t, tf + 1.65, tf + 1.9)))
            ctx.set_line_width(4)
            ctx.stroke()
        # center badge: the winner and its share counting up to 100%
        kb = ramp(t, tf + 0.22, tf + 0.62)
        if kb > 0 and out > 0:
            s = ease_out_back(kb) * out
            rad = 150 * s
            glow(ctx, CX, CY, max(1, rad * 1.8), col, 0.5 * out)
            ctx.arc(CX, CY, max(0.1, rad), 0, TAU)
            ctx.set_source_rgba(*INK, 0.94 * out)
            ctx.fill_preserve()
            ctx.set_source_rgba(*col, out)
            ctx.set_line_width(max(0.1, 9 * s))
            ctx.stroke()
            kc = ease_out(ramp(t, tf + 0.35, tf + 1.15), 2)
            pct = int(round(lerp(100 * self.win_share_at_pop, 100, kc)))
            ctx.save()
            ctx.translate(CX, CY)
            ctx.scale(max(0.01, s), max(0.01, s))
            symbol(ctx, win, 0, -80, 20)
            ctx.set_source_rgba(*col, out)
            ctx.fill()
            font.draw(ctx, f"{pct}%", 0, 40, 104, CREAM, anchor="c", alpha=out)
            font.draw(ctx, NAMES[win], 0, 94, 38, col, anchor="c", alpha=out, tracking=3)
            ctx.restore()
        # result lines in the legend area
        k1 = ramp(t, tf + 0.5, tf + 0.8)
        if k1 > 0 and out > 0:
            font.draw(ctx, "SHARE OVER TIME, CLOCKWISE FROM TOP", CX, 1336, 28, lighten(METAL, 0.65), anchor="c",
                      alpha=out * ramp(t, tf + 1.0, tf + 1.3), tracking=1.5)
            s = 0.8 + 0.2 * ease_out_back(k1)
            ctx.save()
            ctx.translate(CX, 1412)
            ctx.scale(s, s)
            font.draw(ctx, f"{NAMES[win]} TAKES THE WHOLE CIRCLE", 0, 0, 62, col, anchor="c", alpha=out * k1,
                      stroke=INK, stroke_w=10, shadow=4)
            ctx.restore()
        k2 = ramp(t, tf + 1.15, tf + 1.4)
        if k2 > 0 and out > 0:
            font.draw(ctx, "DID YOU CALL IT?", CX, 1490, 52, CREAM, anchor="c", alpha=out * k2,
                      stroke=INK, stroke_w=8)

    def _header(self, ctx, t, font):
        font.draw(ctx, "WHICH COLOR TAKES", QX, 246, 88, CREAM, stroke=INK, stroke_w=10, shadow=6)
        font.draw(ctx, "THE WHOLE CIRCLE?", QX, 340, 88, CREAM, stroke=INK, stroke_w=10, shadow=6)
        txt = "LAST COLOR STANDING WINS"
        size = 38
        tw = font.width(txt, size, 1.5)
        x, y, h = QX, 362, 54
        rounded_rect(ctx, x, y, tw + 40, h, 12)
        ctx.set_source_rgba(*CREAM, 1)
        ctx.fill()
        font.draw(ctx, txt, x + 20, y + 43, size, INK, tracking=1.5)
        bx = x + tw + 74                               # the four contestants: symbol on color
        for c in range(4):
            cx = bx + c * 54
            ctx.arc(cx, y + h / 2, 22, 0, TAU)
            ctx.set_source_rgba(*COLS[c], 1)
            ctx.fill_preserve()
            ctx.set_source_rgba(*INK, 1)
            ctx.set_line_width(3)
            ctx.stroke()
            symbol(ctx, c, cx, y + h / 2 + (1 if c == 2 else 0), 10)
            ctx.set_source_rgba(*INK, 1)
            ctx.fill()

    def _legend(self, ctx, t, font, alpha):
        if alpha <= 0.003:
            return
        x0, y0, w, h = 40.0, 1330.0, 880.0, 200.0
        rounded_rect(ctx, x0, y0, w, h, 22)
        ctx.set_source_rgba(0.05, 0.02, 0.06, 0.72 * alpha)
        ctx.fill_preserve()
        ctx.set_source_rgba(*METAL, alpha)
        ctx.set_line_width(3)
        ctx.stroke()
        ix, size, tx = x0 + 72, 38, x0 + 164
        # row 1: a ball hits a spoke and shoves it
        y = y0 + 38
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.move_to(ix + 2, y - 26)
        ctx.line_to(ix + 2, y + 26)
        ctx.set_source_rgba(*CREAM, alpha)
        ctx.set_line_width(6)
        ctx.stroke()
        self._ball(ctx, 0, ix - 22, y, alpha, rb=15)
        arrow(ctx, ix + 12, y, 1, 0, 36, 8)
        ctx.set_source_rgba(*COLS[0], alpha)
        ctx.fill()
        font.draw(ctx, "A BALL HITTING A SPOKE SHOVES IT", tx, y + 14, size, CREAM, alpha=alpha)
        # row 2: bigger wedge, bigger push
        y = y0 + 102
        for wd, dx, c, ln in ((0.45, -44, 1, 10), (1.25, 20, 2, 30)):
            px, py = ix + dx, y + 30
            ctx.move_to(px, py)
            ctx.arc(px, py, 40, -math.pi / 2 - wd / 2, -math.pi / 2 + wd / 2)
            ctx.close_path()
            ctx.set_source_rgba(*COLS[c], alpha)
            ctx.fill_preserve()
            ctx.set_source_rgba(*INK, alpha)
            ctx.set_line_width(2)
            ctx.stroke()
            arrow(ctx, px + 26 if c == 2 else px + 12, py - 26, 1, 0, ln, 6)
            ctx.set_source_rgba(*CREAM, alpha)
            ctx.fill()
        font.draw(ctx, "BIGGER WEDGE = BIGGER PUSH", tx, y + 14, size, CREAM, alpha=alpha)
        # row 3: pop threshold and growth
        y = y0 + 164
        wd = POP_ANG * 1.6
        ctx.move_to(ix - 12, y + 26)
        ctx.arc(ix - 12, y + 26, 54, -math.pi / 2 - wd / 2, -math.pi / 2 + wd / 2)
        ctx.close_path()
        ctx.set_source_rgba(*COLS[3], alpha)
        ctx.fill()
        for i in range(7):
            a = i * TAU / 7 + 0.3
            bx, by_ = ix + 24, y - 10
            ctx.move_to(bx + 7 * math.cos(a), by_ + 7 * math.sin(a))
            ctx.line_to(bx + 16 * math.cos(a), by_ + 16 * math.sin(a))
        ctx.set_source_rgba(*CREAM, alpha)
        ctx.set_line_width(4)
        ctx.stroke()
        font.draw(ctx, "UNDER 14° POPS", tx, y + 14, size, CREAM, alpha=alpha)
        x2 = tx + font.width("UNDER 14° POPS", size) + 26
        font.draw(ctx, "•  PUSH GROWS 3%/S", x2, y + 14, size, lighten(METAL, 0.75), alpha=alpha)

    # ------------------------------------------------------------ sound
    def audio(self):
        sim = self.sim
        rng = np.random.default_rng(self.seed + 777)
        sr = au.SR
        n, tail = int(self.dur * sr), 3 * sr
        notes = np.zeros((n + tail, 2))
        bed = np.zeros((n + tail, 2))
        tf = self.tf

        def tt(dur):
            return np.arange(int(dur * sr)) / sr

        # spoke hits: plucked notes, pitch by the hitter's wedge size (small wedge = high, panicky)
        penta = (0, 2, 4, 7, 9)
        scale = [m for m in range(48, 97) if m % 12 in penta]
        cache = {}
        last, last_any = [-1.0] * 4, -1.0
        lo, hi = math.log(POP_ANG), math.log(TAU)
        for ht, c, x, y, nx, ny, w in sim["hits"]:
            if ht - last[c] < 0.055 or ht - last_any < 0.022:
                continue
            last[c] = last_any = ht
            k = (math.log(max(POP_ANG, w)) - lo) / (hi - lo)
            target = 91 - 36 * k
            m = min(scale, key=lambda q: abs(q - target))
            if m not in cache:
                cache[m] = au.pluck(au.hz(m), 0.7)
            g = 0.40 * (0.75 + 0.5 * min(1.0, mult(ht) - 1.0))
            au.place(notes, cache[m], ht, g, clamp(x / R * 0.75, -0.8, 0.8))
            au.place(notes, au.tick(2400, 0.03), ht, 0.05, clamp(x / R, -0.8, 0.8))
        # quiet bass pulse that speeds up with the push multiplier
        t, k = 0.0, 0
        roots = (48, 48, 43, 45)
        while t < tf:
            au.place(bed, au.bass(au.hz(roots[(k // 4) % 4]), 0.20), t, 0.34)
            au.place(bed, au.kick(0.22), t, 0.14)
            t += 0.52 / mult(t)
            k += 1
        au.place(bed, au.pad(au.hz(60), tf, 1.0, 0.4), 0.0, 0.035)
        au.place(bed, au.pad(au.hz(67), tf, 1.0, 0.4), 0.0, 0.035, 0.3)
        # pops: noise burst with a falling pitch sweep
        for i, p in enumerate(sim["pops"]):
            pn = clamp(p["x"] / R * 0.7, -0.7, 0.7)
            au.place(notes, au.noise(rng, 0.45, 300, 9000) * np.exp(-tt(0.45) * 9), p["t"], 0.55, pn)
            au.place(notes, au.glide(1400, 110, 0.55, 0.25), p["t"], 0.42, pn)
            au.place(notes, au.boom(rng, 0.8, 110, 40), p["t"], 0.35 + 0.15 * (i == len(sim["pops"]) - 1))
        # finish: count ticks, then a major chord built from additive sines
        for j in range(10):
            au.place(notes, au.tick(1100 * 2 ** (j / 12), 0.04), tf + 0.35 + j * 0.08, 0.16)
        tc = tf + 0.3
        chord_len = self.t_reset + 0.5 - tc
        tch = tt(chord_len)
        chord = np.zeros(len(tch))
        for m, a in ((48, 0.55), (60, 0.5), (64, 0.42), (67, 0.40), (72, 0.32), (76, 0.22), (79, 0.16)):
            f0 = au.hz(m)
            for hn, ha in ((1, 1.0), (2, 0.35), (3, 0.15), (4, 0.06)):
                chord += a * ha * np.sin(2 * np.pi * f0 * hn * tch) * np.exp(-tch * (0.5 + 0.4 * hn))
        chord *= np.minimum(1.0, tch / 0.01) * np.minimum(1.0, (chord_len - tch) / 0.3)
        au.place(notes, chord / 2.2, tc, 0.55)
        for j, m in enumerate((72, 76, 79, 84)):
            au.place(notes, au.bell(au.hz(m), 1.6), tc + 0.05 + j * 0.08, 0.20, (-0.4, -0.1, 0.1, 0.4)[j])
        au.place(notes, au.glide(520, 1040, 0.09, 0.05), tf + 1.15, 0.18)
        wl = self.dur - self.t_reset
        au.place(bed, au.swept_noise(rng, wl, 400, 6000) * (tt(wl) / wl) ** 2, self.t_reset, 0.45)
        mix_ = au.reverb(notes, rng, decay=0.9, wet=0.14) + bed
        out = mix_[:n].copy()
        out[:tail] += mix_[n:]                     # tails past the end wrap to the start: seamless loop
        return au.eq(out, ((60, -3.0), (250, -2.0), (900, 0.0), (3000, 3.5), (8000, 5.0), (16000, 3.0)))

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        title = "Which color takes the whole circle? Pressure pie"
        assert len(title) <= 60
        order = ", then ".join(p["color"].lower() for p in self.stats["pops"])
        return dict(
            title=title,
            description=(
                "Four spokes split a disc into four colored wedges, and one ball bounces inside each wedge. "
                "Every time a ball hits a spoke it shoves that spoke away, and the shove is proportional to the "
                "size of the ball's own wedge: bigger wedge, bigger push. A small wedge gets hit more often but "
                "pushes weakly, so the borders swing back and forth instead of snowballing. The push multiplier "
                "on the hub grows 3% every second. A wedge squeezed under 14 degrees pops and its neighbors close "
                "the gap. The last color holding the whole circle wins.\n"
                f"In this run the wedges popped in this order: {order}. Physics, picture and sound are "
                f"generated by code (seed {self.seed}), with no footage, samples or stock media.\n"
                "Which color did you pick?"),
            tags=["physics simulation", "color battle", "pie chart battle", "bouncing balls", "prediction",
                  "satisfying", "made with code"],
            pinned_comment="Which color did you pick in the first second? Say it before you rewatch, "
                           "and tell us when you knew.",
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of abstract shapes; nothing "
                                              "realistic that could be mistaken for real people, places or events.",
            seed=self.seed,
            drama=self.stats,
        )
