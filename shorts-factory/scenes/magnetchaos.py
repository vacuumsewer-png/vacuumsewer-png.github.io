"""MAGNET CHAOS: top view of a pendulum bob hanging over three magnets at the corners of a triangle.

Numerical model (not a lab measurement): a spring pull to the center, an inverse-square pull to each magnet
softened by the hanging height, and linear friction, integrated with fixed-step RK4 (DT = 1/54, three steps
per video frame). The bob is caught once its energy is under the lowest pass between the magnets: from then
on it can never leave that magnet, so that is where it stops.

Act 1 is the real trajectory from the release point (seed = release point, a pixel of the map grid).
Act 2 releases a second bob one map pixel (4 screen pixels) away. Then every start point of a 270 x 480 grid
is run with the SAME integrator and step and colored by its winner, and the camera zooms into the release
point through nested maps until one color fills the frame. The Scene asserts that the winner shown equals
the map pixel under the release point.
"""
import math
import os
import random
import sys
import time

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, darken, ease_in_out, ease_out, ease_out_back, glow, hex_rgb, lerp,
                        lighten, mix, polygon, ramp, rounded_rect)

W, H, FPS = 1080, 1920, 30
SUB = 3                       # integrator steps per video frame
DT = 1.0 / 54.0               # fixed RK4 step (model time units)
FPU = 18                      # video frames per model time unit (0.6 s per unit)
KS = 0.5                      # spring to the center
BF = 0.17                     # linear friction
GM = 1.0                      # magnet strength
H2 = 0.25 ** 2                # softening: hanging height squared
MAG = [(math.sin(a), -math.cos(a)) for a in (0.0, TAU / 3, 2 * TAU / 3)]   # top, lower right, lower left
NAMES = ("GOLD", "AQUA", "ROSE")            # symbols: triangle, circle, square
DIRS = ((-1, 0, "LEFT"), (1, 0, "RIGHT"), (0, -1, "HIGHER"), (0, 1, "LOWER"))

GW, GH = 270, 480             # map grid; one map pixel = 4 screen pixels at zoom 1
CX, CY, SC = 490.0, 830.0, 200.0            # board center and scale (px per unit)
MAXF = 1620                   # map: give up after 90 time units (never reached in practice)
REST_F = 51                   # act 1 keeps running this long after the bob is caught
GSPEED = 3                    # act 2 plays at 3x
G_INTRO, G_REST, G_OUT = 15, 42, 9
FILL_F, HOLD_F, ZOOM_F, END_F = 114, 15, 138, 72
MAXLEV = 8


def _acc(x, y, vx, vy):
    ax = -KS * x - BF * vx
    ay = -KS * y - BF * vy
    for mx, my in MAG:
        dx = mx - x
        dy = my - y
        q = dx * dx + dy * dy + H2
        inv = GM / (q * np.sqrt(q))
        ax = ax + dx * inv
        ay = ay + dy * inv
    return ax, ay


def _step(x, y, vx, vy):
    """One RK4 step. Only + - * / sqrt, so every element gets the same result whatever the array size."""
    h = DT
    a1x, a1y = _acc(x, y, vx, vy)
    x2 = x + 0.5 * h * vx
    y2 = y + 0.5 * h * vy
    vx2 = vx + 0.5 * h * a1x
    vy2 = vy + 0.5 * h * a1y
    a2x, a2y = _acc(x2, y2, vx2, vy2)
    x3 = x + 0.5 * h * vx2
    y3 = y + 0.5 * h * vy2
    vx3 = vx + 0.5 * h * a2x
    vy3 = vy + 0.5 * h * a2y
    a3x, a3y = _acc(x3, y3, vx3, vy3)
    x4 = x + h * vx3
    y4 = y + h * vy3
    vx4 = vx + h * a3x
    vy4 = vy + h * a3y
    a4x, a4y = _acc(x4, y4, vx4, vy4)
    return (x + h / 6 * (vx + 2 * vx2 + 2 * vx3 + vx4), y + h / 6 * (vy + 2 * vy2 + 2 * vy3 + vy4),
            vx + h / 6 * (a1x + 2 * a2x + 2 * a3x + a4x), vy + h / 6 * (a1y + 2 * a2y + 2 * a3y + a4y))


def _pot(x, y):
    u = 0.5 * KS * (x * x + y * y)
    for mx, my in MAG:
        dx = mx - x
        dy = my - y
        u = u - GM / np.sqrt(dx * dx + dy * dy + H2)
    return u


def _energy(x, y, vx, vy):
    return 0.5 * (vx * vx + vy * vy) + _pot(x, y)


# The three nearest-magnet sectors meet along rays from the center pointing away from each magnet. A bob whose
# energy is under the lowest potential on those rays can never cross one again: it is caught.
_r = np.linspace(0.0, 4.0, 40001)
U_PASS = float(_pot(0 * _r, _r).min()) - 0.02
U_MIN = float(_pot(np.array([MAG[0][0]]), np.array([MAG[0][1]]))[0])
del _r


def _nearest(x, y):
    d = np.stack([(x - mx) ** 2 + (y - my) ** 2 for mx, my in MAG])
    return np.argmin(d, axis=0)


def integrate(x0, y0, maxf=MAXF):
    """Run every start point until it is caught. Returns winner (0..2, 255 = not caught) and the catch frame."""
    n = x0.size
    x, y = x0.astype(np.float64).copy(), y0.astype(np.float64).copy()
    vx, vy = np.zeros(n), np.zeros(n)
    win = np.full(n, 255, np.uint8)
    tc = np.full(n, maxf, np.int32)
    idx = np.arange(n)
    for f in range(1, maxf + 1):
        for _ in range(SUB):
            x, y, vx, vy = _step(x, y, vx, vy)
        done = _energy(x, y, vx, vy) < U_PASS
        if done.any():
            w = _nearest(x, y)
            win[idx[done]] = w[done]
            tc[idx[done]] = f
            keep = ~done
            x, y, vx, vy, idx = x[keep], y[keep], vx[keep], vy[keep], idx[keep]
            if idx.size == 0:
                break
    return win, tc


def trace(x0, y0, nf):
    """One trajectory with the same stepper: positions at every step, energy and catch frame per video frame."""
    x, y = np.array([x0], np.float64), np.array([y0], np.float64)
    vx, vy = np.zeros(1), np.zeros(1)
    px, py = np.empty(nf * SUB + 1), np.empty(nf * SUB + 1)
    en = np.empty(nf + 1)
    px[0], py[0], en[0] = x[0], y[0], _energy(x, y, vx, vy)[0]
    seal = win = None
    k = 0
    for f in range(1, nf + 1):
        for _ in range(SUB):
            x, y, vx, vy = _step(x, y, vx, vy)
            k += 1
            px[k], py[k] = x[0], y[0]
        en[f] = _energy(x, y, vx, vy)[0]
        if seal is None and en[f] < U_PASS:
            seal, win = f, int(_nearest(x, y)[0])
    return dict(px=px, py=py, en=en, seal=seal, win=win)


def release_pixel(seed):
    """Seed -> release point: a pixel of the 270 x 480 map grid, away from the magnets, inside the board."""
    rng = random.Random(seed * 7919 + 11)
    while True:
        i, j = rng.randrange(20, 226), rng.randrange(98, 318)
        x, y = (4 * i + 2 - CX) / SC, (4 * j + 2 - CY) / SC
        if 1.45 <= math.hypot(x, y) <= 2.2 and min(math.hypot(x - mx, y - my) for mx, my in MAG) >= 0.8:
            return i, j


def grid_axis(k, n, i0, c):
    """World coordinates of map level k along one axis. Level k is the frame zoomed 4^k about the release
    pixel; index i0 is the release point itself at every level (offset exactly zero)."""
    p = 4 * i0 + 2
    return (p - c) / SC + (4 * np.arange(n) + 2 - p) / (SC * 4.0 ** k)


def lead_series(px, py, nf):
    """Nearest magnet per video frame; it only switches after 12 frames (0.4 s) with a new nearest."""
    near = _nearest(px[::SUB][:nf + 1], py[::SUB][:nf + 1])
    out, lead, run, cand = [], -1, 0, -1
    for v in near:
        v = int(v)
        if v == cand:
            run += 1
        else:
            cand, run = v, 1
        if run >= 12 and lead != cand:
            lead = cand
        out.append(lead)
    return out


def simulate(seed):
    i0, j0 = release_pixel(seed)
    xs0, ys0 = grid_axis(0, GW, i0, CX), grid_axis(0, GH, j0, CY)
    pts = [(xs0[i0], ys0[j0])] + [(xs0[i0 + di], ys0[j0 + dj]) for di, dj, _ in DIRS]
    probes = []                                   # frame border at zoom 4^k: does one color fill it?
    for k in range(1, 8):
        for ox, oy in ((-540, -960), (0, -960), (540, -960), (-540, 0), (540, 0), (-540, 960), (0, 960), (540, 960)):
            probes.append((xs0[i0] + ox / (SC * 4.0 ** k), ys0[j0] + oy / (SC * 4.0 ** k)))
    allp = np.array(pts + probes)
    win, tc = integrate(allp[:, 0], allp[:, 1], maxf=420)
    sim = dict(seed=seed, i0=i0, j0=j0, win=int(win[0]), seal=int(tc[0]), ghost=None, kest=None)
    if win[0] == 255:
        return sim
    for k in range(1, 8):
        if all(win[5 + 8 * (k - 1) + q] == win[0] for q in range(8)):
            sim["kest"] = k
            break
    for d in range(4):
        if win[1 + d] != 255 and win[1 + d] != win[0] and tc[1 + d] <= 405:
            sim["ghost"] = dict(d=d, win=int(win[1 + d]), seal=int(tc[1 + d]))
            break
    if sim["ghost"] is None or not (9.0 * FPS <= sim["seal"] <= 12.6 * FPS):
        return sim
    g = sim["ghost"]
    f1 = sim["seal"] + REST_F
    gshow = g["seal"] + G_REST
    gdur = -(-gshow // GSPEED)
    f2 = f1 + G_INTRO + gdur + G_OUT
    f3 = f2 + FILL_F
    f4 = f3 + HOLD_F + ZOOM_F
    sim.update(f1=f1, fg=f1 + G_INTRO, gdur=gdur, f2=f2, f3=f3, f4=f4, nf=f4 + END_F)
    sim["hero"] = trace(pts[0][0], pts[0][1], f2 + 2)
    sim["gpath"] = trace(pts[1 + g["d"]][0], pts[1 + g["d"]][1], gshow + GSPEED + 2)
    assert sim["hero"]["seal"] == sim["seal"] and sim["hero"]["win"] == sim["win"]
    assert sim["gpath"]["seal"] == g["seal"] and sim["gpath"]["win"] == g["win"]
    return sim


def drama(sim):
    out = dict(seed=sim["seed"], score=-100.0, valid=False,
               winner=NAMES[sim["win"]] if sim["win"] < 3 else None,
               release_px=[4 * sim["i0"] + 2, 4 * sim["j0"] + 2], caught_s=round(sim["seal"] / FPS, 2))
    if "hero" not in sim:
        return out
    h, g, gp = sim["hero"], sim["ghost"], sim["gpath"]
    seal = sim["seal"]
    lead = lead_series(h["px"], h["py"], seal)
    ch = [round(f / FPS, 2) for f in range(1, len(lead)) if lead[f] != lead[f - 1] and lead[f - 1] >= 0]
    # near captures: 0.8 s or longer within 0.5 of a magnet that did not win
    near, run, who = [], 0, -1
    fx, fy = h["px"][::SUB], h["py"][::SUB]
    for f in range(seal):
        d = [math.hypot(fx[f] - mx, fy[f] - my) for mx, my in MAG]
        m = int(np.argmin(d))
        if d[m] < 0.5 and m == who:
            run += 1
        else:
            if run >= 24 and who != sim["win"] and who >= 0:
                near.append(dict(magnet=NAMES[who], until_s=round(f / FPS, 2), stayed_s=round(run / FPS, 2)))
            who, run = (m, 1) if d[m] < 0.5 else (-1, 0)
    n = min(len(h["px"]), len(gp["px"]))
    sep = np.hypot(h["px"][:n] - gp["px"][:n], h["py"][:n] - gp["py"][:n])
    far = np.nonzero(sep > 0.3)[0]
    split = float(far[0]) / (SUB * FPS) if len(far) else None
    sx = np.concatenate([h["px"], gp["px"]]) * SC + CX
    sy = np.concatenate([h["py"], gp["py"]]) * SC + CY
    inside = bool(sx.min() >= 30 and sx.max() <= 950 and sy.min() >= 345 and sy.max() <= 1330)
    dur = sim["nf"] / FPS
    share = sum(1 for v in lead if v == sim["win"]) / max(1, len(lead))
    valid = (len(ch) >= 4 and inside and 24.5 <= dur <= 31.5 and sim["kest"] is not None and sim["kest"] <= 6)
    score = (0.0 if valid else -100.0) + 1.5 * min(len(ch), 7) + 2.0 * min(len(near), 2)
    score += 2.0 if (split is not None and split >= 2.5) else 0.0
    score += 2.0 if share < 0.45 else 0.0
    score += 1.5 if (ch and ch[-1] >= 6.0) else 0.0
    score -= 0.5 * (sim["kest"] or 8)
    out.update(
        score=round(score, 3), valid=bool(valid), winner_index=sim["win"],
        apparent_winner_changes=len(ch), change_times_s=ch, near_captures=near,
        winner_was_nearest_share=round(share, 2),
        ghost_offset_px=[4 * DIRS[g["d"]][0], 4 * DIRS[g["d"]][1]], ghost_direction=DIRS[g["d"]][2],
        ghost_winner=NAMES[g["win"]], ghost_caught_s=round(g["seal"] / FPS, 2),
        ghost_splits_from_first_at_s=round(split, 2) if split is not None else None,
        zoom_levels_estimate=sim["kest"], path_inside_board=inside, duration_s=round(dur, 2),
        integrator=f"RK4 fixed step 1/54, {SUB} steps per frame, same for the map",
    )
    return out


def score_seed(seed):
    return drama(simulate(seed))


# ====================================================================== look
COL = [hex_rgb("#F5B833"), hex_rgb("#19C3B1"), hex_rgb("#E8507A")]
INK = hex_rgb("#0E0A15")
CREAM = hex_rgb("#F3EEE4")
MUTED = hex_rgb("#9A90AC")
STEEL = hex_rgb("#C9CED6")
PLUM = hex_rgb("#1B1426")
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", "cache")
HX = 490.0
MXY = [(CX + SC * mx, CY + SC * my) for mx, my in MAG]
CHIP_X = (40.0, 345.0, 650.0)
CHIP_Y, CHIP_W, CHIP_H = 172.0, 280.0, 84.0
FREQ = (au.hz(64), au.hz(71), au.hz(76))          # root, fifth, octave


def symbol(ctx, k, x, y, s):
    if k == 0:
        polygon(ctx, [(x, y - s * 1.15), (x + s * 1.08, y + s * 0.78), (x - s * 1.08, y + s * 0.78)])
    elif k == 1:
        ctx.new_sub_path()
        ctx.arc(x, y, s, 0, TAU)
    else:
        ctx.rectangle(x - s * 0.88, y - s * 0.88, s * 1.76, s * 1.76)


def badge(ctx, k, x, y, r, alpha=1.0):
    ctx.new_sub_path()
    ctx.arc(x, y, r, 0, TAU)
    ctx.set_source_rgba(*COL[k], alpha)
    ctx.fill_preserve()
    ctx.set_source_rgba(*INK, alpha)
    ctx.set_line_width(max(2.0, r * 0.13))
    ctx.stroke()
    symbol(ctx, k, x, y, r * 0.46)
    ctx.set_source_rgba(*INK, alpha)
    ctx.fill()


def make_background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.hypot(xx - CX, yy - CY) / 1100
    img = np.array(PLUM, np.float32)[None, None, :] * (1.25 - 0.75 * np.clip(d, 0, 1))[..., None]
    img += (np.clip(1 - d * 2.2, 0, 1) ** 2)[..., None] * np.array([0.05, 0.035, 0.08], np.float32)
    img += np.random.default_rng(9).normal(0, 0.004, (H, W, 1)).astype(np.float32)
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    del img, yy, xx, d
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    ctx = cairo.Context(surf)
    for i in range(1, 13):                               # the spring bowl: rings around the center
        ctx.arc(CX, CY, i * SC * 0.5, 0, TAU)
        ctx.set_source_rgba(0.75, 0.68, 0.95, 0.10 if i % 2 == 0 else 0.05)
        ctx.set_line_width(2)
        ctx.stroke()
    for a in range(12):
        c, s = math.cos(a * TAU / 12), math.sin(a * TAU / 12)
        ctx.move_to(CX + 60 * c, CY + 60 * s)
        ctx.line_to(CX + 2400 * c, CY + 2400 * s)
        ctx.set_source_rgba(0.75, 0.68, 0.95, 0.045)
        ctx.set_line_width(2)
        ctx.stroke()
    surf.flush()
    return surf, buf


# ====================================================================== the scene
class Scene:
    W, H, FPS = W, H, FPS
    EQ = ((80, -8.0), (200, -2.0), (1000, 0.0), (3500, 2.0), (9000, 0.0), (16000, -3.0))
    DRIVE = 3.0

    def __init__(self, seed):
        self.seed = seed
        sim = simulate(seed)
        if "hero" not in sim:
            raise ValueError(f"seed {seed}: no usable release (catch time or neighbor outcome out of range)")
        self.sim = sim
        self.stats = drama(sim)
        self.i0, self.j0 = sim["i0"], sim["j0"]
        self.PX, self.PY = 4.0 * self.i0 + 2, 4.0 * self.j0 + 2
        self.win, self.seal = sim["win"], sim["seal"]
        g = sim["ghost"]
        self.gwin, self.gseal, self.gd = g["win"], g["seal"], DIRS[g["d"]]
        self.F1, self.FG, self.GDUR, self.F2, self.F3, self.F4 = (sim[k] for k in ("f1", "fg", "gdur", "f2", "f3", "f4"))
        self.nframes = sim["nf"]
        self.dur = self.nframes / FPS
        self.T1, self.T2, self.T3, self.T4 = self.F1 / FPS, self.F2 / FPS, self.F3 / FPS, self.F4 / FPS
        self.tseal = self.seal / FPS
        self.tgcap = (self.FG + self.gseal / GSPEED) / FPS
        h, gp = sim["hero"], sim["gpath"]
        self.hx, self.hy = (h["px"] * SC + CX).tolist(), (h["py"] * SC + CY).tolist()
        self.gx, self.gy = (gp["px"] * SC + CX).tolist(), (gp["py"] * SC + CY).tolist()
        self.hlead = lead_series(h["px"], h["py"], len(h["en"]) - 1)
        self.glead = lead_series(gp["px"], gp["py"], len(gp["en"]) - 1)
        self._maps()
        self.stats["zoom_levels"] = self.K
        self.stats["zoom_factor"] = 4 ** self.K
        self.stats["map_agrees_with_shown_winner"] = True
        self.stats["map_share_percent"] = {NAMES[k]: round(100 * float((self.lev[0][0] == k).mean()), 1) for k in range(3)}
        self._res = None

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_res"] = None
        return d

    # ------------------------------------------------------------ the maps (cached per level)
    def _maps(self):
        os.makedirs(CACHE, exist_ok=True)
        tag = f"{KS}_{BF}_{GM}_{H2}_{DT}_{SUB}_{CX}_{CY}_{SC}".replace(".", "p")
        self.lev = []
        for k in range(MAXLEV + 1):
            path = os.path.join(CACHE, f"magnetchaos_seed{self.seed}_L{k}_{tag}.npz")
            if os.path.exists(path):
                z = np.load(path)
                win, tc = z["win"], z["tc"]
            else:
                t0 = time.time()
                xs, ys = grid_axis(k, GW, self.i0, CX), grid_axis(k, GH, self.j0, CY)
                X, Y = np.meshgrid(xs, ys)
                win, tc = integrate(X.ravel(), Y.ravel())
                win, tc = win.reshape(GH, GW), tc.reshape(GH, GW).astype(np.int16)
                np.savez_compressed(path, win=win, tc=tc)
                print(f"map level {k}: {time.time() - t0:.0f} s, not caught {(win == 255).sum()}", file=sys.stderr)
            # the winner shown in act 1 must be the map's pixel, computed with the same integrator and step
            assert win[self.j0, self.i0] == self.win and tc[self.j0, self.i0] == self.seal, "map disagrees with act 1"
            if k == 0:
                assert win[self.j0 + self.gd[1], self.i0 + self.gd[0]] == self.gwin, "map disagrees with act 2"
            self.lev.append((win, tc))
            if (win == self.win).all():
                break
        else:
            raise ValueError(f"seed {self.seed}: still mixed at zoom 4^{MAXLEV}")
        self.K = len(self.lev) - 1
        tc0 = self.lev[0][1].astype(np.float32)
        self.tsort = np.sort(tc0.ravel())

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            bg, keep = make_background()
            pal = np.array([[c[2], c[1], c[0]] for c in COL] + [[0.2, 0.2, 0.2]], np.float32) * 255
            maps = []
            for win, tc in self.lev:
                shade = 0.55 + 0.45 * np.exp(-tc.astype(np.float32) / (FPU * 9.0))
                buf = np.zeros((GH, GW, 4), np.uint8)
                buf[..., :3] = (pal[np.minimum(win, 3)] * shade[..., None]).astype(np.uint8)
                buf[..., 3] = 255
                maps.append((cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, GW, GH, GW * 4), buf))
            fbuf = np.zeros((GH, GW, 4), np.uint8)
            fsurf = cairo.ImageSurface.create_for_data(memoryview(fbuf), cairo.FORMAT_ARGB32, GW, GH, GW * 4)
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep,
                             font=Font(FONT_PATH), maps=maps, fbuf=fbuf, fsurf=fsurf,
                             col0=maps[0][1][..., :3].astype(np.float32), tc0=self.lev[0][1].astype(np.float32))
        return self._res

    def frame(self, f):
        res = self._resources()
        ctx = res["ctx"]
        ctx.save()
        self.draw(ctx, f, res)
        ctx.restore()
        res["surf"].flush()
        return bytes(res["surf"].get_data())

    # ------------------------------------------------------------ timeline helpers
    def zoom_levels(self, t):
        """Zoom as a number of 4x levels (0 = whole board)."""
        z0 = self.T3 + HOLD_F / FPS
        u = ramp(t, z0, self.T4)
        return self.K * (u * u * (3 - 2 * u)) + 0.3 * ramp(t, self.T4, self.dur)

    def draw(self, ctx, f, res):
        t = f / FPS
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(res["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        self._main(ctx, f, t, res)
        back = ramp(t, self.dur - 0.95, self.dur - 0.4)
        if back > 0:                                   # back to the untouched board for the loop
            ctx.push_group()
            ctx.set_source_surface(res["bg"], 0, 0)
            ctx.paint()
            self._start_state(ctx, t - self.dur, res)
            ctx.pop_group_to_source()
            ctx.paint_with_alpha(ease_in_out(back))

    def _start_state(self, ctx, tt, res):
        font = res["font"]
        self._pull_rings(ctx, tt, 1.0)
        self._pivot(ctx, 1.0)
        self._magnets(ctx, tt, 1.0, None, None)
        self._string(ctx, self.PX, self.PY, 1.0)
        self._bob(ctx, self.PX, self.PY, "1", 1.0, False)
        self._marker(ctx, tt, font, 1.0, 1.0)
        self._chips(ctx, font, -1, tt, start=True)
        self._energy_bar(ctx, font, 1.0, 1.0)
        self._hook(ctx, font, tt, 1.0)

    def _main(self, ctx, f, t, res):
        font = res["font"]
        lv = self.zoom_levels(t)
        s = 4.0 ** lv
        # ---- map
        if t >= self.T2:
            if t < self.T3:
                self._fill(ctx, t, res)
            else:
                self._zoom_maps(ctx, lv, res)
        # ---- board (moves with the zoom)
        fade = 1 - ramp(t, self.T2 + 0.3, self.T2 + 1.3)     # trails and bobs leave as the map arrives
        if s < 40:
            ctx.save()
            ctx.translate(self.PX, self.PY)
            ctx.scale(s, s)
            ctx.translate(-self.PX, -self.PY)
            if t < self.T2:
                self._pull_rings(ctx, t, 1.0)
            self._pivot(ctx, 1.0 if t < self.T2 else 0.8)
            nh = min(f, len(self.hx) // SUB - 1) * SUB
            ng = None
            if f >= self.FG:
                ng = min((f - self.FG) * GSPEED, self.gseal + G_REST) * SUB
                ng = min(ng, len(self.gx) - 1)
            if fade > 0:
                dim = 1 - 0.55 * ramp(t, self.T1, self.T1 + 0.5)
                self._trail(ctx, self.hx, self.hy, nh, CREAM, fade * dim, 200 if t < self.T1 else 60)
                if ng is not None:
                    self._trail(ctx, self.gx, self.gy, ng, hex_rgb("#9FD8FF"), fade, 420, dash=True)
            self._magnets(ctx, t, 1.0, self.win if t >= self.tseal + 0.5 else None,
                          self.gwin if t >= self.tgcap else None)
            if fade > 0:
                self._string(ctx, self.hx[nh], self.hy[nh], fade * (1 - 0.5 * ramp(t, self.T1, self.T1 + 0.5)))
                self._bob(ctx, self.hx[nh], self.hy[nh], "1", fade, False)
                if ng is not None:
                    self._string(ctx, self.gx[ng], self.gy[ng], fade)
                    self._bob(ctx, self.gx[ng], self.gy[ng], "2", fade, True)
                elif t >= self.T1:                   # second bob waits at its release point, 4 px over
                    a = ramp(t, self.T1, self.T1 + 0.3) * fade
                    self._bob(ctx, self.gx[0], self.gy[0], "2", a, True)
            ctx.restore()
        self._marker(ctx, t, font, 1.0, s)
        # ---- interface
        if f < self.F1:
            lead = self.hlead[min(f, len(self.hlead) - 1)]
            en = self.sim["hero"]["en"]
            e = en[min(f, len(en) - 1)]
        elif self.FG <= f < self.F2:
            q = min((f - self.FG) * GSPEED, self.gseal + G_REST)
            lead = self.glead[min(q, len(self.glead) - 1)]
            en = self.sim["gpath"]["en"]
            e = en[min(q, len(en) - 1)]
        else:
            lead = -1
            e = self.sim["gpath"]["en"][0] if f < self.FG else None
        self._chips(ctx, font, lead, t)
        if e is not None:
            e0 = self.sim["hero"]["en"][0]
            self._energy_bar(ctx, font, clamp((e - U_MIN) / (e0 - U_MIN)), 1 - ramp(t, self.T2 - 0.3, self.T2))
        self._texts(ctx, font, t, s)

    # ------------------------------------------------------------ map drawing
    def _paint_level(self, ctx, surf, sc):
        ctx.save()
        ctx.translate(self.PX, self.PY)
        ctx.scale(4 * sc, 4 * sc)
        ctx.translate(-(self.i0 + 0.5), -(self.j0 + 0.5))
        ctx.set_source_surface(surf, 0, 0)
        pat = ctx.get_source()
        pat.set_filter(cairo.FILTER_BILINEAR)
        pat.set_extend(cairo.EXTEND_PAD)
        ctx.rectangle(0, 0, GW, GH)
        ctx.fill()
        ctx.restore()

    def _fill(self, ctx, t, res):
        """Each start point lights up at the moment its own run is caught (quantile-paced)."""
        u = ramp(t, self.T2, self.T3 - 0.35)
        q = u * u * (3 - 2 * u)
        thr = float(np.interp(q * (len(self.tsort) - 1), np.arange(len(self.tsort)), self.tsort)) + 2 * q
        a = np.clip((thr - res["tc0"]) / 3.0, 0, 1)
        if q >= 1:
            a[:] = 1
        fb = res["fbuf"]
        res["fsurf"].flush()
        fb[..., :3] = (res["col0"] * a[..., None]).astype(np.uint8)
        fb[..., 3] = (a * 255).astype(np.uint8)
        res["fsurf"].mark_dirty()
        self._paint_level(ctx, res["fsurf"], 1.0)

    def _zoom_maps(self, ctx, lv, res):
        k = min(self.K, int(lv))
        self._paint_level(ctx, res["maps"][k][0], 4.0 ** (lv - k))
        if k + 1 <= self.K:
            self._paint_level(ctx, res["maps"][k + 1][0], 4.0 ** (lv - k - 1))

    # ------------------------------------------------------------ board pieces
    def _pull_rings(self, ctx, t, alpha):
        for k, (mx, my) in enumerate(MXY):
            for j in range(2):
                ph = (t / 1.5 + j * 0.5 + k * 0.17) % 1.0
                r = lerp(150, 58, ph)
                ctx.new_sub_path()
                ctx.arc(mx, my, r, 0, TAU)
                ctx.set_source_rgba(*COL[k], 0.30 * alpha * math.sin(math.pi * ph))
                ctx.set_line_width(4)
                ctx.stroke()

    def _pivot(self, ctx, alpha):
        ctx.new_sub_path()
        ctx.arc(CX, CY, 9, 0, TAU)
        ctx.set_source_rgba(*INK, alpha)
        ctx.fill_preserve()
        ctx.set_source_rgba(*MUTED, alpha)
        ctx.set_line_width(3)
        ctx.stroke()

    def _magnets(self, ctx, t, alpha, w1, w2):
        for k, (mx, my) in enumerate(MXY):
            glow(ctx, mx, my, 120, COL[k], 0.28 * alpha)
            for w, t0 in ((w1, self.tseal + 0.5), (w2, self.tgcap)):
                if w == k:
                    d = t - t0
                    if 0 <= d < 0.9:                 # one ring when it catches the bob
                        ctx.new_sub_path()
                        ctx.arc(mx, my, 56 + 150 * ease_out(d / 0.9), 0, TAU)
                        ctx.set_source_rgba(*lighten(COL[k], 0.4), 0.9 * (1 - d / 0.9) * alpha)
                        ctx.set_line_width(8)
                        ctx.stroke()
                    ctx.new_sub_path()
                    ctx.arc(mx, my, 62, 0, TAU)
                    ctx.set_source_rgba(*CREAM, 0.9 * alpha)
                    ctx.set_line_width(5)
                    ctx.stroke()
            badge(ctx, k, mx, my, 48, alpha)

    def _string(self, ctx, x, y, alpha):
        if alpha <= 0:
            return
        ctx.move_to(CX, CY)
        ctx.line_to(x, y)
        ctx.set_source_rgba(*STEEL, 0.30 * alpha)
        ctx.set_line_width(3)
        ctx.stroke()

    def _bob(self, ctx, x, y, label, alpha, ghost):
        if alpha <= 0:
            return
        r = 21.0
        ctx.new_sub_path()
        ctx.arc(x, y, r, 0, TAU)
        if ghost:
            ctx.set_source_rgba(0.62, 0.85, 1.0, 0.55 * alpha)
            ctx.fill_preserve()
            ctx.set_source_rgba(*INK, alpha)
            ctx.set_line_width(3)
            ctx.stroke()
            ctx.save()
            ctx.set_dash([7, 6])
            ctx.new_sub_path()
            ctx.arc(x, y, r + 6, 0, TAU)
            ctx.set_source_rgba(0.75, 0.9, 1.0, 0.9 * alpha)
            ctx.set_line_width(3)
            ctx.stroke()
            ctx.restore()
        else:
            g = cairo.RadialGradient(x - 7, y - 8, 2, x, y, r)
            g.add_color_stop_rgba(0, 1, 1, 1, alpha)
            g.add_color_stop_rgba(1, *darken(STEEL, 0.35), alpha)
            ctx.set_source(g)
            ctx.fill_preserve()
            ctx.set_source_rgba(*INK, alpha)
            ctx.set_line_width(4)
            ctx.stroke()
        self._res["font"].draw(ctx, label, x, y + 10, 26, INK, anchor="c", alpha=alpha)

    def _trail(self, ctx, xs, ys, n, col, alpha, recent, dash=False):
        if n < 1 or alpha <= 0:
            return
        ctx.save()
        if dash:
            ctx.set_dash([10, 8])
        ctx.move_to(xs[0], ys[0])                      # whole path so far, faint
        for i in range(1, n + 1):
            ctx.line_to(xs[i], ys[i])
        ctx.set_source_rgba(*col, 0.16 * alpha)
        ctx.set_line_width(3)
        ctx.stroke()
        ctx.restore()
        a0 = max(0, n - recent)
        parts = 10
        for c in range(parts):                         # the last stretch, brighter toward the bob
            i0 = a0 + (n - a0) * c // parts
            i1 = a0 + (n - a0) * (c + 1) // parts
            if i1 <= i0:
                continue
            ctx.move_to(xs[i0], ys[i0])
            for i in range(i0 + 1, i1 + 1):
                ctx.line_to(xs[i], ys[i])
            k = (c + 1) / parts
            ctx.set_source_rgba(*col, 0.85 * alpha * k * k)
            ctx.set_line_width(3 + 6 * k)
            ctx.stroke()

    def _marker(self, ctx, t, font, alpha, s):
        """Release point (screen space, fixed size) and, in the zoom, the second release 4 px away."""
        x, y = self.PX, self.PY
        big = t >= self.T2 + 0.6
        r = 30.0 if big else 27.0
        for col, lw in ((INK, 9), (CREAM, 4)):
            ctx.new_sub_path()
            ctx.arc(x, y, r, 0, TAU)
            for a in range(4):
                c, sn = math.cos(a * TAU / 4), math.sin(a * TAU / 4)
                ctx.move_to(x + c * (r - 9), y + sn * (r - 9))
                ctx.line_to(x + c * (r + 13), y + sn * (r + 13))
            ctx.set_source_rgba(*col, alpha)
            ctx.set_line_width(lw)
            ctx.stroke()
        if big:
            ctx.new_sub_path()
            ctx.arc(x, y, 5, 0, TAU)
            ctx.set_source_rgba(*INK, alpha)
            ctx.fill_preserve()
            ctx.set_source_rgba(*CREAM, alpha)
            ctx.set_line_width(2.5)
            ctx.stroke()
        lab = 1 - ramp(t, 1.6, 2.0) if t < self.T2 else ramp(t, self.T2 + 0.6, self.T2 + 1.0)
        if lab > 0:
            up = y > CY
            font.draw(ctx, "RELEASE POINT", clamp(x, 150, 830), y + (72 if up else -52), 30, CREAM, anchor="c",
                      alpha=lab * alpha, stroke=INK, stroke_w=7, tracking=1)
        # contracting ring: the release itself (runs across the loop point)
        tt = t if t < 1 else t - self.dur
        if -0.5 <= tt <= 0.4:
            k = ramp(tt, -0.5, 0.4)
            ctx.new_sub_path()
            ctx.arc(x, y, lerp(230, 34, k), 0, TAU)
            ctx.set_source_rgba(*CREAM, 0.9 * math.sin(math.pi * clamp(k * 0.9 + 0.1)) * alpha)
            ctx.set_line_width(9)
            ctx.stroke()
        if t >= self.T3:                               # the neighbor release separates as the zoom grows
            gx, gy = x + 4 * s * self.gd[0], y + 4 * s * self.gd[1]
            sep = 4 * s
            a = ramp(sep, 40, 70) * alpha
            if a > 0 and -80 < gx < W + 80 and -80 < gy < H + 80:
                polygon(ctx, [(gx, gy - 24), (gx + 24, gy), (gx, gy + 24), (gx - 24, gy)])
                ctx.set_source_rgba(*INK, a)
                ctx.set_line_width(9)
                ctx.stroke_preserve()
                ctx.set_source_rgba(0.75, 0.9, 1.0, a)
                ctx.set_line_width(4)
                ctx.stroke()
                font.draw(ctx, "DROP 2", clamp(gx, 110, 870), gy - 38, 30, CREAM, anchor="c", alpha=a,
                          stroke=INK, stroke_w=7)

    # ------------------------------------------------------------ interface
    def _chips(self, ctx, font, lead, t, start=False):
        for k in range(3):
            x0 = CHIP_X[k]
            on = lead == k
            rounded_rect(ctx, x0, CHIP_Y, CHIP_W, CHIP_H, 20)
            if on:
                ctx.set_source_rgba(*COL[k], 1)
            else:
                ctx.set_source_rgba(*INK, 0.86)
            ctx.fill_preserve()
            ctx.set_source_rgba(*(INK if on else COL[k]), 1)
            ctx.set_line_width(5)
            ctx.stroke()
            ctx.new_sub_path()
            ctx.arc(x0 + 50, CHIP_Y + CHIP_H / 2, 27, 0, TAU)
            ctx.set_source_rgba(*(INK if on else COL[k]), 1)
            ctx.fill()
            symbol(ctx, k, x0 + 50, CHIP_Y + CHIP_H / 2, 12.5)
            ctx.set_source_rgba(*(COL[k] if on else INK), 1)
            ctx.fill()
            font.draw(ctx, NAMES[k], x0 + 92, CHIP_Y + 62, 54, INK if on else COL[k], tracking=1.5)
            if on:
                font.draw(ctx, "NEAREST", x0 + CHIP_W - 14, CHIP_Y + 34, 22, INK, anchor="r", tracking=1)
            if start:
                continue
            out = 1 - ramp(t, self.dur - 0.95, self.dur - 0.6)
            for n, (w, t0) in enumerate(((self.win, self.tseal + 0.5), (self.gwin, self.tgcap))):
                a = ramp(t, t0, t0 + 0.25) * out
                if w == k and a > 0:
                    xx = x0 + 14 + n * 132
                    rounded_rect(ctx, xx, CHIP_Y + CHIP_H + 8, 120, 40, 10)
                    ctx.set_source_rgba(*CREAM, a)
                    ctx.fill()
                    font.draw(ctx, f"DROP {n + 1}", xx + 60, CHIP_Y + CHIP_H + 39, 28, INK, anchor="c", alpha=a,
                              tracking=1)

    def _energy_bar(self, ctx, font, frac, alpha):
        if alpha <= 0:
            return
        x0, x1, y = 190.0, 920.0, 322.0
        font.draw(ctx, "ENERGY", 60, y + 20, 30, CREAM, alpha=alpha, stroke=INK, stroke_w=6, tracking=1)
        rounded_rect(ctx, x0, y, x1 - x0, 20, 8)
        ctx.set_source_rgba(*INK, 0.85 * alpha)
        ctx.fill()
        e0 = self.sim["hero"]["en"][0]
        tk = x0 + (x1 - x0) * (U_PASS - U_MIN) / (e0 - U_MIN)
        caught = x0 + (x1 - x0) * frac <= tk
        rounded_rect(ctx, x0 + 3, y + 3, max(6.0, (x1 - x0 - 6) * frac), 14, 6)
        ctx.set_source_rgba(*(MUTED if caught else CREAM), alpha)
        ctx.fill()
        ctx.move_to(tk, y - 8)
        ctx.line_to(tk, y + 28)
        ctx.set_source_rgba(*INK, alpha)
        ctx.set_line_width(9)
        ctx.stroke_preserve()
        ctx.set_source_rgba(*CREAM, alpha)
        ctx.set_line_width(4)
        ctx.stroke()
        font.draw(ctx, "CAUGHT BELOW THIS LINE", tk + 14, y + 52, 24, CREAM, alpha=alpha, stroke=INK,
                  stroke_w=6, tracking=1)

    def _headline(self, ctx, font, parts, y, size, alpha, scale=1.0):
        """Centered line of text pieces and magnet badges."""
        if alpha <= 0:
            return

        def width(sz):
            w = 0.0
            for p in parts:
                w += sz * 0.95 if p[0] == "sym" else font.width(p[1], sz, 1.0)
            return w
        while width(size) > 850 and size > 30:
            size -= 2
        ctx.save()
        ctx.translate(HX, y)
        ctx.scale(scale, scale)
        x = -width(size) / 2
        for p in parts:
            if p[0] == "sym":
                badge(ctx, p[1], x + size * 0.40, -size * 0.36, size * 0.40, alpha)
                x += size * 0.95
            else:
                font.draw(ctx, p[1], x, 0, size, p[2], alpha=alpha, stroke=INK, stroke_w=size * 0.13, shadow=5,
                          tracking=1.0)
                x += font.width(p[1], size, 1.0)
        ctx.restore()

    def _pill(self, ctx, font, txt, y, alpha, size=38):
        if alpha <= 0:
            return
        while font.width(txt, size, 1.2) > 800 and size > 24:
            size -= 2
        tw = font.width(txt, size, 1.2)
        rounded_rect(ctx, HX - tw / 2 - 24, y - size - 8, tw + 48, size + 24, 14)
        ctx.set_source_rgba(*CREAM, alpha)
        ctx.fill_preserve()
        ctx.set_source_rgba(*INK, alpha)
        ctx.set_line_width(4)
        ctx.stroke()
        font.draw(ctx, txt, HX, y, size, INK, anchor="c", alpha=alpha, tracking=1.2)

    def _hook(self, ctx, font, tt, alpha):
        pop = 1 + 0.06 * (1 - ease_out(ramp(tt, -0.3, 0.45)))
        self._headline(ctx, font, [("txt", "WHICH MAGNET CATCHES IT?", CREAM)], 1432, 90, alpha, pop)
        self._pill(ctx, font, "WHERE IT STOPS WINS", 1508, alpha, 44)

    def _texts(self, ctx, font, t, s):
        def vis(t0, t1):
            return ramp(t, t0, t0 + 0.2) * (1 - ramp(t, t1 - 0.2, t1))
        w, g = self.win, self.gwin
        a = 1 - ramp(t, self.tseal + 0.3, self.tseal + 0.5)
        if a > 0:
            self._hook(ctx, font, t, a)
        a = vis(self.tseal + 0.5, self.T1 + 0.2)
        self._headline(ctx, font, [("sym", w), ("txt", f"{NAMES[w]} CATCHES IT", COL[w])], 1432, 92, a)
        self._pill(ctx, font, "IT CAN'T ESCAPE NOW: THAT'S WHERE IT STOPS", 1508, a)
        a = vis(self.T1 + 0.2, self.tgcap)
        self._headline(ctx, font, [("txt", f"SAME DROP, 4 PIXELS {self.gd[2]}", CREAM)], 1432, 86, a)
        self._pill(ctx, font, f"SAME RULES  •  {GSPEED}× SPEED", 1508, a)
        a = vis(self.tgcap, self.T2 + 0.1)
        self._headline(ctx, font, [("txt", "THIS TIME: ", CREAM), ("sym", g), ("txt", NAMES[g], COL[g])], 1432, 92, a)
        self._pill(ctx, font, "4 PIXELS MOVED THE FINISH", 1508, a)
        a = vis(self.T2 + 0.1, self.T3 + 0.4)
        self._headline(ctx, font, [("txt", "EVERY POSSIBLE START POINT", CREAM)], 1432, 84, a)
        self._pill(ctx, font, "COLORED BY THE MAGNET THAT CATCHES IT", 1508, a)
        a = vis(self.T3 + 0.4, self.T4)
        self._headline(ctx, font, [("txt", "ZOOM IN ON THE RELEASE", CREAM)], 1432, 86, a)
        self._pill(ctx, font, f"ZOOM ×{max(1, int(round(s))):,}", 1508, a, 44)
        a = vis(self.T4, self.dur - 0.7)
        self._headline(ctx, font, [("sym", w), ("txt", f"{NAMES[w]} CATCHES IT", COL[w])], 1432, 92, a,
                       0.85 + 0.15 * ease_out_back(ramp(t, self.T4, self.T4 + 0.3)))
        self._pill(ctx, font, f"4 PIXELS {self.gd[2]}: {NAMES[g]} INSTEAD", 1508, a, 42)

    # ------------------------------------------------------------ sound
    def audio(self):
        rng = np.random.default_rng(self.seed + 77)
        sr = au.SR
        n, tail = int(self.dur * sr), 3 * sr
        fx = np.zeros((n + tail, 2))
        bed = np.zeros((n + tail, 2))
        mpan = (0.0, 0.55, -0.55)

        def tt(d):
            return np.arange(int(d * sr)) / sr

        def voice(path, rate, t0, dur_v, fmul, gain, env):
            """Bob: a sine whose pitch follows speed and whose pan follows x. Each magnet owns a note that
            swells when the bob is near, and is plucked at every close pass."""
            m = int(dur_v * sr)
            pos = np.minimum(np.arange(m) / sr * rate, len(path["px"]) - 1.001)
            idx = np.arange(len(path["px"]))
            x, y = np.interp(pos, idx, path["px"]), np.interp(pos, idx, path["py"])
            sp = np.hypot(np.diff(path["px"]), np.diff(path["py"])) / DT
            spd = np.interp(pos, idx[:-1] + 0.5, sp)
            ph = TAU * np.cumsum(190.0 * fmul * 2 ** (1.5 * np.clip(spd / 3.5, 0, 1))) / sr
            e = env(np.arange(m) / sr + t0)
            tone = (np.sin(ph) + 0.28 * np.sin(2 * ph) + 0.1 * np.sin(3 * ph)) * (0.2 + 0.8 * np.clip(spd / 3, 0, 1))
            pan = np.clip(x / 2.4, -0.8, 0.8)
            out = np.zeros((m, 2))
            ang = (pan + 1) * np.pi / 4
            out[:, 0] += tone * np.cos(ang) * 0.085 * gain * e
            out[:, 1] += tone * np.sin(ang) * 0.085 * gain * e
            tsec = np.arange(m) / sr
            for k, (mx, my) in enumerate(MAG):
                d = np.hypot(x - mx, y - my)
                amp = 1 / (1 + (d / 0.38) ** 2)
                f = FREQ[k] * fmul
                note = (np.sin(TAU * f * tsec) + 0.25 * np.sin(TAU * 2 * f * tsec)) * amp * 0.12 * gain * e
                a2 = (mpan[k] + 1) * np.pi / 4
                out[:, 0] += note * np.cos(a2)
                out[:, 1] += note * np.sin(a2)
                dd = np.hypot(path["px"] - mx, path["py"] - my)        # close passes -> plucks
                last, prevmax = -1e9, dd[0]
                gap = 0.14 * rate
                for i in range(1, len(dd) - 1):
                    prevmax = max(prevmax, dd[i])
                    if dd[i] < 0.5 and dd[i] <= dd[i - 1] and dd[i] < dd[i + 1]:
                        depth = clamp((prevmax - dd[i]) / 0.35)
                        tp = t0 + i / rate
                        if i - last >= gap and depth > 0.15 and tp < t0 + dur_v:
                            gg = (0.10 + 0.20 * (1 - dd[i] / 0.5)) * depth * gain * float(env(np.array([tp]))[0])
                            au.place(fx, au.pluck(f, 0.7), tp, gg, mpan[k])
                            last = i
                        prevmax = dd[i]
            i0 = int(t0 * sr)
            bed[i0:i0 + m] += out[:max(0, min(m, len(bed) - i0))]

        T1, T2, T3, T4 = self.T1, self.T2, self.T3, self.T4
        voice(self.sim["hero"], SUB * FPS, 0.0, T2,  1.0, 1.0,
              lambda t: np.minimum(1.0, t / 0.05) * (1 - 0.8 * np.clip((t - T1) / 0.5, 0, 1)) * np.clip((T2 - t) / 0.4, 0, 1))
        tg0 = self.FG / FPS
        gd = self.GDUR / FPS
        voice(self.sim["gpath"], SUB * FPS * GSPEED, tg0, gd + 0.25, 1.5, 0.85,
              lambda t: np.minimum(1.0, (t - tg0) / 0.05).clip(0, 1) * np.clip((tg0 + gd + 0.25 - t) / 0.3, 0, 1))
        # release
        au.place(fx, au.pluck(au.hz(52), 0.8), 0.0, 0.30, clamp((self.PX - CX) / 480, -0.7, 0.7))
        au.place(fx, au.tick(1500, 0.05), 0.0, 0.10)
        au.place(fx, au.pluck(au.hz(59), 0.6), tg0, 0.22)
        # catches: that magnet's note rings out
        for w, t0, g_ in ((self.win, self.tseal + 0.5, 0.24), (self.gwin, self.tgcap, 0.20)):
            au.place(fx, au.bell(FREQ[w], 2.2), t0, g_, mpan[w])
            au.place(fx, au.bell(FREQ[w] * 2, 1.6), t0 + 0.02, g_ * 0.4, mpan[w])
        for t0 in (T1 + 0.2, T2 + 0.05, T3 + HOLD_F / FPS):
            au.place(fx, au.swept_noise(rng, 0.35, 700, 3200) * np.sin(np.pi * tt(0.35) / 0.35), t0, 0.10)
        # map fill: grains of the three notes, paced like the fill, each one the winner of a start point
        order = np.argsort(self.lev[0][1].ravel(), kind="stable")
        wflat = self.lev[0][0].ravel()
        fill = (self.F3 - self.F2) / FPS - 0.35
        tg = tt(0.32)
        for u in np.sort(rng.random(170)):
            q = u * u * (3 - 2 * u)
            w = int(wflat[order[min(len(order) - 1, int(q * len(order)) + int(rng.integers(0, 400)))]]) % 3
            f = FREQ[w] * (2 if rng.random() < 0.6 else 4)
            grain = np.sin(TAU * f * tg) * np.exp(-tg / 0.07) * np.minimum(1, tg / 0.002)
            au.place(fx, grain, T2 + fill * u, 0.05 + 0.05 * rng.random(), rng.uniform(-0.8, 0.8))
        for mnote, g_ in ((64, 0.035), (71, 0.028)):
            au.place(bed, au.pad(au.hz(mnote), fill + 1.2, 2.4, 0.8), T2, g_)
        # zoom: falling Shepard-style glide, one tick per 4x level
        z0 = T3 + HOLD_F / FPS
        zd = T4 - z0
        tz = tt(zd + 1.2)
        lvl = np.array([self.zoom_levels(z0 + v) for v in tz[::480]])
        lvl = np.interp(np.arange(len(tz)), np.arange(len(lvl)) * 480, lvl)
        glide = np.zeros(len(tz))
        for j in range(7):
            o = (j - 0.7 * lvl) % 7.0
            f = 55.0 * 2 ** o
            glide += np.sin(TAU * np.cumsum(f) / sr) * np.exp(-0.5 * ((o - 3.2) / 1.1) ** 2)
        glide *= np.minimum(1, tz / 0.4) * np.clip((zd + 1.2 - tz) / 1.0, 0, 1)
        au.place(bed, glide, z0, 0.045)
        for k in range(1, self.K + 1):
            i = int(np.argmax(lvl >= k))
            au.place(fx, au.marimba(au.hz(88 - 2 * k), 0.5), z0 + i / sr, 0.16, 0.3 * (-1) ** k)
            au.place(fx, au.tick(1700, 0.05), z0 + i / sr, 0.08)
        # one color fills the frame: the winner's note
        au.place(fx, au.bell(FREQ[self.win], 2.4), T4, 0.26)
        au.place(fx, au.bell(FREQ[self.win] * 1.5, 2.0), T4 + 0.03, 0.10, 0.3)
        au.place(fx, au.bell(FREQ[self.win] / 2, 2.4), T4, 0.14, -0.3)
        au.place(fx, au.marimba(FREQ[self.gwin] * 2, 0.6), T4 + 0.75, 0.10, 0.4)
        # back to the board
        sw = tt(0.7)
        au.place(fx, au.swept_noise(rng, 0.7, 500, 4000) * (sw / 0.7) ** 2, self.dur - 0.72, 0.12)
        au.place(bed, au.pad(au.hz(52), self.dur, 1.0, 0.8), 0.0, 0.012)
        mix_ = au.reverb(fx, rng, decay=0.9, wet=0.12) + bed
        out = mix_[:n].copy()
        out[:tail] += mix_[n:]
        out = au.eq(out, self.EQ)
        out = np.tanh(out * (self.DRIVE / max(1e-9, np.abs(out).max())))
        return out

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        title = "Three Magnets, One Pendulum… Which One Catches It? 🧲"
        assert len(title) <= 60
        d = self.gd[2].lower()
        return dict(
            title=title,
            description=(
                "Gold, aqua or rose: which magnet catches the pendulum? Call it before the bob lets go!\n\n"
                "Top view: a steel bob hangs on a string above three magnets at the corners of a triangle: gold "
                "(triangle), aqua (circle) and rose (square). It is released from rest at the marked release "
                "point. Three things act on it: the string pulls it back toward the center like a spring, each "
                "magnet pulls harder the closer the bob gets (inverse-square, softened by the height the bob "
                "hangs at), and friction drains its energy, shown on the energy bar. Once the energy drops under "
                "the marked line the bob can no longer get away from the magnet it is circling, so that is where "
                "it stops, and where it stops wins. Then a second bob is released 4 pixels "
                f"{'to the ' + d if d in ('left', 'right') else d} under the same rules at 3x speed. Finally "
                "every one of the 129,600 start points on a 270 by 480 grid is run and colored by the magnet "
                "that catches it, and the camera zooms in on the release point.\n"
                "This is a numerical model, not a lab measurement: fixed-step RK4, the same integrator and step "
                "for the bob you watch and for every pixel of the map. A different step or method can move the "
                f"fine detail. Motion, picture and sound are generated by code (seed {self.seed}), with no "
                "footage, samples or stock media.\n"
                "Which magnet did you pick?\n"
                "#simulation #satisfying #chaos"),
            tags=["magnetic pendulum", "chaos theory", "fractal", "physics simulation", "basins of attraction",
                  "prediction", "satisfying", "made with code", "simulation"],
            pinned_comment=("Gold, aqua or rose: which magnet did you call? Pick before the release, then watch "
                            "what 4 pixels do to the answer."),
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of abstract shapes; nothing "
                                              "realistic that could be mistaken for real people, places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / FPS, 2),
            drama=self.stats,
        )
