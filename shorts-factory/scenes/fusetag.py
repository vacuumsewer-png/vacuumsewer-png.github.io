"""FUSE TAG: six balls bounce in a box and one carries a lit charge with a visible countdown ring.

Touching another ball passes the charge (the passer is immune for half a second), the carrier
moves 30% faster and steers at the nearest ball it may tag, and at zero the carrier bursts and
the charge jumps to the nearest ball. Last ball wins.

simulate(seed) is deterministic and needs no rendering, so make.py can score hundreds of seeds
for drama (drama()) and render only the best one (Scene). Nothing is edited after the fact.
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
SUB = 2                              # physics steps per frame
DT = 1.0 / (FPS * SUB)
NB = 6
AW, AH = 800.0, 1030.0               # arena size in arena units (= pixels at zoom 1)
AX, AY = 110.0, 470.0                # arena top-left on screen in the main layout
AXC = AX + AW / 2
R = 48.0                             # ball radius
V = 500.0                            # ball speed, px/s
BOOST = 1.3                          # the carrier is 30% faster
TURN = 2.8                           # how fast the carrier can steer, rad/s
IMMUNE_S = 0.5
IMMUNE = int(round(IMMUNE_S / DT))
ROUND_S = (4.0, 4.4, 4.8, 5.2, 5.6)  # the five fuse lengths; the seed decides their order
ROUND_STEPS = tuple(int(round(s / DT)) for s in ROUND_S)
NR = len(ROUND_S)
JUMP_S = 0.35                        # the charge takes this long to jump to the nearest ball after a burst
JUMP = int(round(JUMP_S / DT))
PLAY_STEPS = sum(ROUND_STEPS) + (NR - 1) * JUMP
PLAY = PLAY_STEPS * DT               # 25.4 s: the last burst
T_WIN = PLAY + 0.30                  # winner banner
T_LINE = PLAY + 1.00                 # the question back to the viewer
T_RESET = 27.45                      # everything rewinds so the last frame cuts back to the first
DUR = 28.0
NF = int(DUR * FPS)
TOTAL_STEPS = NF * SUB
SHOW = (AW / 2, AH * 0.60)           # where the winner settles for the result

NAMES = ("YELLOW", "PINK", "VIOLET", "BLUE", "LIME", "WHITE")   # symbols: triangle, ring, plus, diamond, square, star


# ====================================================================== simulation
def simulate(seed, record=False):
    rnd = random.Random(seed * 9973 + 7)
    px, py = [], []
    while len(px) < NB:
        x, y = rnd.uniform(R + 14, AW - R - 14), rnd.uniform(R + 14, AH - R - 14)
        if all((x - a) ** 2 + (y - b) ** 2 > (3.4 * R) ** 2 for a, b in zip(px, py)):
            px.append(x)
            py.append(y)
    vx, vy = [], []
    for _ in range(NB):
        a = rnd.uniform(0, TAU)
        q = a % (TAU / 4)                      # keep headings off the axes: nobody ping-pongs along a wall
        if q < 0.26:
            a += 0.26 - q
        elif q > TAU / 4 - 0.26:
            a -= q - (TAU / 4 - 0.26)
        vx.append(V * math.cos(a))
        vy.append(V * math.sin(a))
    timers = list(ROUND_STEPS)
    rnd.shuffle(timers)
    carrier = first = rnd.randrange(NB)
    vx[carrier] *= BOOST
    vy[carrier] *= BOOST
    start = [(px[i], py[i], vx[i], vy[i]) for i in range(NB)]

    ids = list(range(NB))                      # balls still in
    immune = [0] * NB                          # step index until which a ball cannot be tagged
    left = timers[0]
    rnd_i, jump_left, jump_to = 0, 0, -1
    passes, bursts, walls, hits, jumps, rounds = [], [], [], [], [], [[0.0, None, first]]
    if record:
        P = np.zeros((TOTAL_STEPS + 2, NB, 2), np.float32)
        C = np.full(TOTAL_STEPS + 2, -1, np.int8)
        G = np.full(TOTAL_STEPS + 2, -1, np.int8)
    lo_x, hi_x, lo_y, hi_y = R, AW - R, R, AH - R
    D2 = (2 * R) ** 2
    max_turn = TURN * DT
    relax = min(1.0, 10 * DT)
    VB = V * BOOST

    def move(i, t):
        x, y = px[i] + vx[i] * DT, py[i] + vy[i] * DT
        if x < lo_x:
            x, vx[i] = 2 * lo_x - x, -vx[i]
            walls.append((t, i, 0.0, y, 0))
        elif x > hi_x:
            x, vx[i] = 2 * hi_x - x, -vx[i]
            walls.append((t, i, AW, y, 1))
        if y < lo_y:
            y, vy[i] = 2 * lo_y - y, -vy[i]
            walls.append((t, i, x, 0.0, 2))
        elif y > hi_y:
            y, vy[i] = 2 * hi_y - y, -vy[i]
            walls.append((t, i, x, AH, 3))
        px[i], py[i] = x, y

    for s in range(PLAY_STEPS):
        t1 = (s + 1) * DT
        if record:
            for i in range(NB):
                P[s, i, 0], P[s, i, 1] = px[i], py[i]
            C[s] = carrier
        if carrier >= 0:                       # the carrier steers at the nearest ball it is allowed to tag
            cx, cy = px[carrier], py[carrier]
            best, bd, anyb, ad = -1, 1e18, -1, 1e18
            for j in ids:
                if j == carrier:
                    continue
                d = (px[j] - cx) ** 2 + (py[j] - cy) ** 2
                if d < ad:
                    ad, anyb = d, j
                if immune[j] <= s and d < bd:
                    bd, best = d, j
            tg = best if best >= 0 else anyb
            if record:
                G[s] = tg
            lead = min(0.45, math.sqrt(bd if best >= 0 else ad) / VB) * 0.7
            dx = min(hi_x, max(lo_x, px[tg] + vx[tg] * lead)) - cx
            dy = min(hi_y, max(lo_y, py[tg] + vy[tg] * lead)) - cy
            cvx, cvy = vx[carrier], vy[carrier]
            ang = math.atan2(cvx * dy - cvy * dx, cvx * dx + cvy * dy)
            ang = max_turn if ang > max_turn else -max_turn if ang < -max_turn else ang
            ca, sa = math.cos(ang), math.sin(ang)
            vx[carrier], vy[carrier] = cvx * ca - cvy * sa, cvx * sa + cvy * ca
        for i in ids:
            move(i, t1)
        passed = False
        n = len(ids)
        for a_ in range(n):
            i = ids[a_]
            for b_ in range(a_ + 1, n):
                j = ids[b_]
                dx, dy = px[j] - px[i], py[j] - py[i]
                d2 = dx * dx + dy * dy
                if d2 >= D2 or d2 < 1e-9:
                    continue
                d = math.sqrt(d2)
                nx, ny = dx / d, dy / d
                ov = (2 * R - d) / 2
                px[i] -= nx * ov
                py[i] -= ny * ov
                px[j] += nx * ov
                py[j] += ny * ov
                rv = (vx[j] - vx[i]) * nx + (vy[j] - vy[i]) * ny
                if rv < 0:                     # equal masses: swap the velocity components along the normal
                    vx[i] += rv * nx
                    vy[i] += rv * ny
                    vx[j] -= rv * nx
                    vy[j] -= rv * ny
                mx, my = (px[i] + px[j]) / 2, (py[i] + py[j]) / 2
                is_pass = False
                if not passed and carrier in (i, j):
                    other = j if carrier == i else i
                    if immune[other] <= s:
                        passes.append((t1, carrier, other, mx, my, (left - 1) * DT))
                        immune[carrier] = s + 1 + IMMUNE
                        carrier = other
                        passed = is_pass = True
                if rv < 0 or is_pass:
                    hits.append((t1, i, j, mx, my, is_pass))
        for i in ids:
            sp = math.hypot(vx[i], vy[i])
            if sp > 1e-6:
                k = (sp + ((VB if i == carrier else V) - sp) * relax) / sp
                vx[i] *= k
                vy[i] *= k
        if carrier >= 0:
            left -= 1
            if left == 0:                      # zero: the carrier bursts, the charge jumps to the nearest ball
                bx, by, out = px[carrier], py[carrier], carrier
                ids.remove(out)
                carrier = -1
                rounds[-1][1] = t1
                jump_to = -1
                if len(ids) > 1:
                    jump_to = min(ids, key=lambda j: (px[j] - bx) ** 2 + (py[j] - by) ** 2)
                    jump_left = JUMP
                    jumps.append((t1, t1 + JUMP * DT, bx, by, jump_to))
                bursts.append((t1, out, bx, by, jump_to))
        elif jump_left > 0:
            jump_left -= 1
            if jump_left == 0:
                carrier = jump_to
                rnd_i += 1
                left = timers[rnd_i]
                rounds.append([t1, None, carrier])
    winner = ids[0]
    sim = dict(seed=seed, start=start, first=first, timers=[k * DT for k in timers], passes=passes, bursts=bursts,
               walls=walls, hits=hits, jumps=jumps, rounds=[tuple(r) for r in rounds], winner=winner,
               order=[b[1] for b in bursts])
    if record:                                 # after the result the winner glides to its spot (damped spring)
        w = winner
        for s in range(PLAY_STEPS, TOTAL_STEPS + 2):
            for i in range(NB):
                P[s, i, 0], P[s, i, 1] = px[i], py[i]
            vx[w] += (-16.0 * (px[w] - SHOW[0]) - 7.0 * vx[w]) * DT
            vy[w] += (-16.0 * (py[w] - SHOW[1]) - 7.0 * vy[w]) * DT
            move(w, (s + 1) * DT)
        sim.update(pos=P, car=C, tgt=G)
    return sim


def drama(sim):
    passes, bursts, rounds, w = sim["passes"], sim["bursts"], sim["rounds"], sim["winner"]
    per = []
    for r, (t0, t1, c0) in enumerate(rounds):
        ps = [p for p in passes if t0 < p[0] <= t1 + 1e-9]
        h1 = c0                                # who held it one second before zero
        for p in ps:
            if p[0] <= t1 - 1.0:
                h1 = p[2]
        per.append(dict(n=len(ps), gap=(t1 - ps[-1][0]) if ps else (t1 - t0), holders={c0} | {p[2] for p in ps},
                        swing=h1 != bursts[r][1], length=t1 - t0))
    fin = per[-1]
    w_rounds = sum(1 for p in per if w in p["holders"])
    w_escapes = [p[5] for p in passes if p[1] == w]
    closest = min(w_escapes) if w_escapes else None
    held = set()
    for p in per:
        held |= p["holders"]
    first_out_first = sim["first"] == bursts[0][1]

    sc = 1.2 * min(fin["n"], 6) + (4.0 if fin["n"] >= 3 else 0.0)      # a real duel in the last round
    g = fin["gap"]
    if fin["n"]:                                                        # the last pass lands just before zero
        if 0.1 <= g <= 0.7:
            sc += 10.0 - 3.0 * abs(g - 0.3) / 0.4
        elif g < 0.1:                                                   # same instant as zero: unreadable
            sc += 1.0
        else:
            sc += max(0.0, 6.0 - 5.0 * (g - 0.7))
    sc += -4.0 if first_out_first else 3.0
    for p in per[:-1]:
        if p["n"]:                                                      # late decision in every round
            sc += 1.5 if 0.1 <= p["gap"] <= 1.0 else -2.5 if p["gap"] < 0.1 else 0.0
        sc += 0.8 if p["swing"] else 0.0                                # the holder changed in the last second
        sc -= 6.0 if p["n"] == 0 else 0.0
        sc -= 0.4 * max(0, p["n"] - 9)                                  # too many passes reads as noise
    sc += 1.0 * w_rounds + (2.5 if closest is not None and closest < 0.8 else 0.0)   # the winner was in danger
    sc += 0.4 * len(held)
    return dict(
        seed=sim["seed"], score=round(sc, 3), winner=NAMES[w], winner_index=w,
        elimination_order=[NAMES[i] for i in sim["order"]],
        first_carrier=NAMES[sim["first"]], first_carrier_out_first=first_out_first,
        fuse_seconds_per_round=[round(p["length"], 2) for p in per],
        passes_total=len(passes), passes_per_round=[p["n"] for p in per],
        last_pass_before_zero_s=[round(p["gap"], 3) if p["n"] else None for p in per],
        holder_changed_in_last_second=[p["swing"] for p in per],
        final_round_passes=fin["n"], final_last_pass_before_zero_s=round(g, 3) if fin["n"] else None,
        winner_held_charge_in_rounds=w_rounds, winner_times_holding=sum(1 for p in passes if p[2] == w)
        + (1 if sim["first"] == w else 0) + sum(1 for b in bursts if b[4] == w),
        winner_closest_escape_s=None if closest is None else round(closest, 3),
        balls_that_held_charge=len(held))


def score_seed(seed):
    return drama(simulate(seed))


# ====================================================================== look
COLS = [hex_rgb(c) for c in ("#FFD84A", "#FF6FB1", "#B48CFF", "#4D9BFF", "#A6EE5A", "#F4F7F2")]
INK = hex_rgb("#03161B")
WHITE = (1.0, 1.0, 1.0)
MINT = hex_rgb("#9CFFE9")
HOT = hex_rgb("#FF8A1F")             # danger orange
HOT2 = hex_rgb("#FFD27A")
RED = hex_rgb("#FF4B2B")
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")

HOOK_ZOOM = 0.86                     # arena starts smaller to make room for the big question
T_HOOK = 1.9
CORNER = 40.0
M = 70                               # margin of the arena sprite (room for glow and shadow)
RING_R, RING_W = R + 26.0, 17.0
CHIP_W, CHIP_H, CHIP_GAP, CHIP_Y = 118.0, 116.0, 136.0, 258.0
STRIP_Y = 441.0
QA, QB = "WHO'S HOLDING IT", "AT ZERO?"
IT, RULE, ASK = "THE CHARGE", "LAST BALL LEFT WINS", "DID YOUR PICK SURVIVE?"
ADD, OVER = cairo.OPERATOR_ADD, cairo.OPERATOR_OVER


def hook(t):
    """1 while the opening layout is up (big question, smaller arena), 0 in the main layout."""
    if t < 14:
        return 1 - ease_in_out(ramp(t, T_HOOK, T_HOOK + 0.7))
    return ease_in_out(ramp(t, T_RESET, T_RESET + 0.36))


def fold(x, lo, hi):
    """Reflect x into [lo, hi] (a ball path unfolded across the walls)."""
    span = hi - lo
    u = (x - lo) % (2 * span)
    return lo + (u if u <= span else 2 * span - u)


def fit(font, text, width, max_size):
    return min(max_size, width / font.width(text, 1.0))


def tick_times(t0, t1):
    """Metronome clicks for one round: slow at a full ring, fast near zero. Shared by picture and sound."""
    out, t, dur = [], t0, t1 - t0
    while t < t1 - 0.04:
        left = (t1 - t) / dur
        out.append((t, 1.0 - left))
        t += lerp(0.09, 0.50, left ** 0.8)
    return out


def symbol(ctx, k, x, y, s, a=1.0, col=INK):
    ctx.new_path()
    ctx.set_source_rgba(col[0], col[1], col[2], a)
    if k == 0:                                   # triangle
        polygon(ctx, [(x, y - s * 1.08), (x + s * 1.0, y + s * 0.72), (x - s * 1.0, y + s * 0.72)])
    elif k == 1:                                 # ring
        ctx.set_line_width(s * 0.48)
        ctx.arc(x, y, s * 0.74, 0, TAU)
        ctx.stroke()
        return
    elif k == 2:                                 # plus
        w = s * 0.36
        ctx.rectangle(x - s, y - w, 2 * s, 2 * w)
        ctx.rectangle(x - w, y - s, 2 * w, 2 * s)
    elif k == 3:                                 # diamond
        polygon(ctx, [(x, y - s * 1.15), (x + s * 0.82, y), (x, y + s * 1.15), (x - s * 0.82, y)])
    elif k == 4:                                 # square
        ctx.rectangle(x - s * 0.78, y - s * 0.78, s * 1.56, s * 1.56)
    else:                                        # star
        pts = []
        for j in range(10):
            rr = s * 1.18 if j % 2 == 0 else s * 0.50
            pts.append((x + rr * math.cos(-TAU / 4 + j * TAU / 10), y + rr * math.sin(-TAU / 4 + j * TAU / 10)))
        polygon(ctx, pts)
    ctx.fill()


def draw_ball(ctx, i, x, y, r, a=1.0):
    """Flat arcade puck: dark rim, colour face, bevel line, dark symbol."""
    col = COLS[i]
    ctx.new_path()
    ctx.set_source_rgba(0, 0.03, 0.04, 0.36 * a)
    ctx.arc(x + r * 0.10, y + r * 0.24, r * 1.03, 0, TAU)
    ctx.fill()
    ctx.set_source_rgba(INK[0], INK[1], INK[2], a)
    ctx.arc(x, y, r, 0, TAU)
    ctx.fill()
    g = cairo.LinearGradient(x, y - r, x, y + r)
    c0, c1 = lighten(col, 0.30), darken(col, 0.30)
    g.add_color_stop_rgba(0, c0[0], c0[1], c0[2], a)
    g.add_color_stop_rgba(0.5, col[0], col[1], col[2], a)
    g.add_color_stop_rgba(1, c1[0], c1[1], c1[2], a)
    ctx.set_source(g)
    ctx.arc(x, y, r * 0.885, 0, TAU)
    ctx.fill()
    ctx.set_source_rgba(1, 1, 1, 0.30 * a)
    ctx.set_line_width(max(1.0, r * 0.07))
    ctx.arc(x, y, r * 0.78, 0, TAU)
    ctx.stroke()
    symbol(ctx, i, x, y, r * 0.40, 0.93 * a, INK)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_source_rgba(1, 1, 1, 0.62 * a)
    ctx.set_line_width(r * 0.085)
    ctx.arc(x, y, r * 0.66, -2.55, -1.80)
    ctx.stroke()
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)


def draw_ring(ctx, x, y, frac, sc=1.0, a=1.0):
    """The countdown ring: a dark track, the bright remaining arc, and a white-hot burning end."""
    if sc <= 0.03 or a <= 0.01:
        return None
    rr, lw = RING_R * sc, RING_W * sc
    frac = clamp(frac)
    col = mix(RED, HOT, clamp(frac / 0.25)) if frac < 0.25 else mix(HOT, HOT2, clamp((frac - 0.25) / 0.45))
    a0, a1 = -TAU / 4 + (1 - frac) * TAU, -TAU / 4 + TAU
    ctx.new_path()
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    if frac < 0.3:                               # nearly empty: the whole ring glows red
        ctx.set_operator(ADD)
        ctx.set_source_rgba(RED[0], RED[1], RED[2], 0.32 * (1 - frac / 0.3) * a)
        ctx.set_line_width(lw + 42 * sc)
        ctx.arc(x, y, rr, 0, TAU)
        ctx.stroke()
        ctx.set_operator(OVER)
    ctx.set_source_rgba(INK[0], INK[1], INK[2], 0.9 * a)
    ctx.set_line_width(lw + 12 * sc)
    ctx.arc(x, y, rr, 0, TAU)
    ctx.stroke()
    ctx.set_source_rgba(1, 1, 1, 0.22 * a)
    ctx.set_line_width(lw - 4 * sc)
    ctx.arc(x, y, rr, 0, TAU)
    ctx.stroke()
    ex, ey = x + rr * math.cos(a0), y + rr * math.sin(a0)
    if frac > 0.002:
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_operator(ADD)
        for wdt, al in ((lw + 36 * sc, 0.10), (lw + 18 * sc, 0.22)):
            ctx.set_source_rgba(col[0], col[1], col[2], al * a)
            ctx.set_line_width(wdt)
            ctx.arc(x, y, rr, a0, a1)
            ctx.stroke()
        ctx.set_operator(OVER)
        ctx.set_source_rgba(col[0], col[1], col[2], a)
        ctx.set_line_width(lw)
        ctx.arc(x, y, rr, a0, a1)
        ctx.stroke()
        hc = lighten(col, 0.7)
        ctx.set_source_rgba(hc[0], hc[1], hc[2], 0.95 * a)
        ctx.set_line_width(lw * 0.34)
        ctx.arc(x, y, rr, a0, a1)
        ctx.stroke()
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    ctx.set_operator(ADD)
    glow(ctx, ex, ey, 50 * sc, HOT2, 0.95 * a)
    ctx.set_operator(OVER)
    ctx.set_source_rgba(1, 1, 1, a)
    ctx.arc(ex, ey, lw * 0.64, 0, TAU)
    ctx.fill()
    return ex, ey


def make_background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = (yy / H)[..., None]
    img = np.array([0.022, 0.235, 0.275], np.float32) * (1 - v) + np.array([0.006, 0.090, 0.115], np.float32) * v
    for cx, cy, rad, col in ((AXC, 900, 980, (0.020, 0.130, 0.125)), (40, 280, 620, (0.000, 0.090, 0.130)),
                             (1030, 1640, 720, (0.030, 0.085, 0.060))):       # soft light pools for depth
        d = np.hypot(xx - cx, yy - cy) / rad
        img += (np.clip(1 - d, 0, 1) ** 2)[..., None] * np.array(col, np.float32)
    stripe = (0.5 + 0.5 * np.sin((xx - yy) * (TAU / 72.0))) ** 8                 # faint diagonal arcade stripes
    img += stripe[..., None] * np.array([0.004, 0.022, 0.022], np.float32)
    vig = np.clip(1.12 - 0.5 * (((xx - 540) / 720) ** 2 + ((yy - 960) / 1180) ** 2), 0.3, 1.0)
    img *= vig[..., None]
    img += np.random.default_rng(11).normal(0, 0.006, (H, W, 1)).astype(np.float32)   # grain hides banding
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    return cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4), buf


def make_arena():
    """The static arena (shadow, bright floor, court markings, neon wall) as one sprite."""
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(AW + 2 * M), int(AH + 2 * M))
    c = cairo.Context(s)
    c.translate(M, M)
    for k in range(7):
        rounded_rect(c, -10 - k * 5, 4 - k * 2, AW + 20 + k * 10, AH + 30 + k * 7, CORNER + 10 + k * 5)
        c.set_source_rgba(0, 0.02, 0.03, 0.085)
        c.fill()
    rounded_rect(c, 0, 0, AW, AH, CORNER)
    g = cairo.RadialGradient(AW / 2, AH * 0.46, 40, AW / 2, AH * 0.5, AH * 0.72)
    g.add_color_stop_rgb(0, 0.075, 0.420, 0.445)
    g.add_color_stop_rgb(0.55, 0.045, 0.310, 0.345)
    g.add_color_stop_rgb(1, 0.022, 0.205, 0.245)
    c.set_source(g)
    c.fill_preserve()
    c.save()
    c.clip()
    for iy in range(21):
        for ix in range(16):
            c.arc(25 + ix * 50, 15 + iy * 50, 3.0, 0, TAU)
            c.close_path()
    c.set_source_rgba(MINT[0], MINT[1], MINT[2], 0.17)
    c.fill()
    c.set_source_rgba(MINT[0], MINT[1], MINT[2], 0.12)
    c.set_line_width(5)
    c.arc(AW / 2, AH / 2, 220, 0, TAU)
    c.stroke()
    c.set_line_width(2.5)
    c.arc(AW / 2, AH / 2, 304, 0, TAU)
    c.stroke()
    for cx, cy in ((0, 0), (AW, 0), (0, AH), (AW, AH)):
        c.arc(cx, cy, 150, 0, TAU)
        c.stroke()
    lg = cairo.LinearGradient(0, 0, 0, 80)
    lg.add_color_stop_rgba(0, 0, 0.03, 0.04, 0.40)
    lg.add_color_stop_rgba(1, 0, 0.03, 0.04, 0)
    c.rectangle(0, 0, AW, 80)
    c.set_source(lg)
    c.fill()
    c.restore()
    for wdt, a in ((52, 0.045), (34, 0.08), (20, 0.15)):
        rounded_rect(c, -5, -5, AW + 10, AH + 10, CORNER + 5)
        c.set_line_width(wdt)
        c.set_source_rgba(MINT[0], MINT[1], MINT[2], a)
        c.stroke()
    rounded_rect(c, -5, -5, AW + 10, AH + 10, CORNER + 5)
    c.set_line_width(10)
    c.set_source_rgb(*lighten(MINT, 0.25))
    c.stroke()
    rounded_rect(c, -5, -5, AW + 10, AH + 10, CORNER + 5)
    c.set_line_width(3)
    c.set_source_rgba(1, 1, 1, 0.9)
    c.stroke()
    s.flush()
    return s


# ====================================================================== the scene
class Scene:
    W, H, FPS, nframes = W, H, FPS, NF

    def __init__(self, seed):
        self.seed = seed
        self.sim = simulate(seed, record=True)
        self.stats = drama(self.sim)
        self.winner = self.sim["winner"]
        self.font = Font(FONT_PATH)
        self._res = None
        self._prep()

    # ------------------------------------------------------------ precomputed animation state
    def _prep(self):
        sim, font = self.sim, self.font
        self.pl = sim["pos"].tolist()
        self.car = sim["car"].tolist()
        self.tgt = sim["tgt"].tolist()
        self.rounds, self.jumps = sim["rounds"], sim["jumps"]
        self.passes, self.bursts, self.walls = sim["passes"], sim["bursts"], sim["walls"]
        self.pass_t = [p[0] for p in self.passes]
        self.wall_t = [w[0] for w in self.walls]
        self.out = [math.inf] * NB
        for b in self.bursts:
            self.out[b[1]] = b[0]
        self.ticks = []
        for t0, t1, _ in self.rounds:
            self.ticks += tick_times(t0, t1)
        self.tick_t = [x[0] for x in self.ticks]
        nb = len(self.bursts)
        self.shakes = sorted([(b[0], 26.0 if j == nb - 1 else 13.0) for j, b in enumerate(self.bursts)]
                             + [(p[0], 2.5) for p in self.passes])
        rnd = random.Random(self.seed + 31)
        self.parts = [[(rnd.uniform(0, TAU), rnd.uniform(180, 980), rnd.uniform(5, 15), rnd.random(),
                        rnd.uniform(0.55, 1.15)) for _ in range(54)] for _ in self.bursts]
        self.zig = [[rnd.uniform(-1, 1) for _ in range(5)] for _ in self.passes]
        self.spk = [[(rnd.uniform(0, TAU), rnd.uniform(160, 520), rnd.uniform(3, 7)) for _ in range(9)]
                    for _ in self.passes]
        self.ember = [(rnd.uniform(0, TAU), rnd.uniform(30, 150), rnd.uniform(2.5, 6.0), rnd.uniform(0.28, 0.5))
                      for _ in range(NF)]
        self.twinkle = [(rnd.uniform(0, TAU), rnd.uniform(95, 300), rnd.uniform(4, 10), rnd.uniform(0, TAU),
                         rnd.choice((2, 3, 4))) for _ in range(22)]
        self.dust = [(rnd.uniform(0, W), rnd.uniform(0, H), rnd.uniform(3, 9), rnd.uniform(0.05, 0.16),
                      rnd.choice((1, 1, 2)), rnd.uniform(10, 46), rnd.choice((1, 2, 3)), rnd.uniform(0, TAU))
                     for _ in range(28)]
        # text layout: the question is two big lines in the hook and one compact line afterwards
        cap = self.cap = font.cap_height
        self.s1, self.s2 = fit(font, QA, 730, 150), fit(font, QB, 600, 235)
        self.y1 = 172 + cap * self.s1
        self.y2 = self.y1 + 26 + cap * self.s2
        self.chip_y_hook = self.y2 + 26
        self.sm = fit(font, QA + " " + QB, 790, 86)
        wa, wb = font.width(QA, self.sm), font.width(QB, self.sm)
        gap = font.width(QA + " " + QB, self.sm) - wa - wb
        self.xa_m = AXC - (wa + gap + wb) / 2
        self.xb_m = self.xa_m + wa + gap
        self.ym = 232.0
        self.xa_h, self.xb_h = AXC - font.width(QA, self.s1) / 2, AXC - font.width(QB, self.s2) / 2
        self.win_text = f"{NAMES[self.winner]} SURVIVES!"
        self.win_size = fit(font, self.win_text, 700, 150)
        self.line_size = fit(font, ASK, 660, 124)

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            bg, keep = make_background()
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep,
                             arena=make_arena())
        return self._res

    def frame(self, f):
        res = self._resources()
        self.draw(res["ctx"], f, res)
        res["surf"].flush()
        return bytes(res["surf"].get_data())

    # ------------------------------------------------------------ state helpers
    def P(self, i, t):
        """Position of ball i at time t. Before 0 and after the rewind it is the launch path run
        backwards from the start position, so the last frame flows into the first."""
        if t < 0 or t >= T_RESET:
            tau = t if t < 0 else t - DUR
            sx, sy, vx, vy = self.sim["start"][i]
            return fold(sx + vx * tau, R, AW - R), fold(sy + vy * tau, R, AH - R)
        s = t * FPS * SUB
        k = int(s)
        a, b = self.pl[k][i], self.pl[k + 1][i]
        u = s - k
        return a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u

    def scale(self, i, t):
        if t >= T_RESET:
            return ease_out_back(ramp(t, T_RESET + 0.05 + 0.02 * i, T_RESET + 0.30 + 0.02 * i))
        return 1.0 if t < self.out[i] else 0.0

    def state(self, t):
        """Who holds the charge at time t, how much of the ring is left and how close zero is."""
        if t >= T_RESET:
            return dict(mode="hold", c=self.sim["first"], tg=self.tgt[0], frac=1.0, left=self.rounds[0][1], rnd=0,
                        danger=0.0,
                        pop=ease_out_back(ramp(t, T_RESET + 0.10, T_RESET + 0.38)))
        if t < PLAY:
            for r, (t0, t1, c0) in enumerate(self.rounds):
                if t < t1:
                    if t >= t0:
                        k = int(t * FPS * SUB + 1e-6)
                        c, left = self.car[k], t1 - t
                        return dict(mode="hold", c=c if c >= 0 else c0, tg=self.tgt[k] if c >= 0 else -1,
                                    frac=left / (t1 - t0), left=left, rnd=r,
                                    danger=clamp(1 - left / 1.6), pop=1.0)
                    ta, tb, bx, by, to = self.jumps[r - 1]
                    return dict(mode="jump", c=-1, to=to, k=clamp((t - ta) / (tb - ta)), src=(bx, by),
                                left=t1 - t0, frac=0.0, rnd=r, danger=0.0, pop=1.0)
        return dict(mode="none", c=-1, frac=0.0, left=0.0, rnd=NR - 1, danger=0.0, pop=0.0)

    def ring_pos(self, t, st=None):
        """(x, y, fraction left, scale) of the countdown ring, or None when no charge is live."""
        st = st or self.state(t)
        if st["mode"] == "hold":
            c = st["c"]
            x, y = self.P(c, t)
            j = bisect.bisect_right(self.pass_t, t) - 1
            if j >= 0 and t < PLAY:                       # the ring snaps across from the passer in 0.1 s
                p = self.passes[j]
                if t - p[0] < 0.10 and p[2] == c:
                    k = ease_out((t - p[0]) / 0.10)
                    ox, oy = self.P(p[1], t)
                    x, y = lerp(ox, x, k), lerp(oy, y, k)
            return x, y, st["frac"], st["pop"]
        if st["mode"] == "jump":
            e = ease_in_out(st["k"])
            bx, by = st["src"]
            tx, ty = self.P(st["to"], t)
            arc = 0.22 * math.sin(math.pi * e)
            return lerp(bx, tx, e) - (ty - by) * arc, lerp(by, ty, e) + (tx - bx) * arc, e, 0.5 + 0.5 * e
        return None

    def tick_pulse(self, t):
        j = bisect.bisect_right(self.tick_t, t) - 1
        return math.exp(-(t - self.tick_t[j]) / 0.07) if j >= 0 and t < PLAY else 0.0

    def shake(self, t):
        sx = sy = 0.0
        for te, amp in self.shakes:
            if te > t:
                break
            d = t - te
            if d < 0.6:
                a = amp * math.exp(-d / 0.11)
                sx += a * math.sin(d * 71 + te * 13)
                sy += a * math.cos(d * 59 + te * 7)
        return sx, sy

    def wpos(self, t):
        return self.P(self.winner, min(t, T_RESET - 0.001))

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, f, res):
        t = f / FPS
        font = self.font
        h = hook(t)
        e1, e2 = ease_in_out(clamp(2 * (1 - h))), ease_in_out(clamp(1 - 2 * h))   # two-stage layout change
        st = self.state(t)
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(res["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(OVER)
        self._ambient(ctx, t, st["danger"])
        z = lerp(HOOK_ZOOM, 1.0, e2)
        sx, sy = self.shake(t)
        ctx.save()
        ctx.translate(AXC - AW / 2 * z + sx, AY + AH - AH * z + sy)
        ctx.scale(z, z)
        ctx.set_source_surface(res["arena"], -M, -M)
        ctx.paint()
        self._floor(ctx, t, st, font)
        self._marks(ctx, t)
        self._trails(ctx, t, st)
        self._fx_under(ctx, t)
        self._aim(ctx, t, st)
        self._balls(ctx, t, st)
        self._ring(ctx, f, t, st)
        self._fx_over(ctx, t, font)
        self._finale(ctx, t, font)
        if h > 0.01:                                       # hook: the winning rule, on the arena floor
            font.draw(ctx, RULE, AW / 2, AH - 46, 66, WHITE, "c", h, INK, 11)
        ctx.restore()
        self._titles(ctx, t, h, e1, e2, st, font)
        self._chips(ctx, t, e2, st)
        fl = 0.0
        for j, b in enumerate(self.bursts):
            if 0 <= t - b[0] < 0.3:
                fl += (0.16 if j == len(self.bursts) - 1 else 0.09) * (1 - (t - b[0]) / 0.3) ** 2
        if fl > 0.004:
            ctx.set_operator(ADD)
            ctx.set_source_rgba(1.0, 0.93, 0.82, fl)
            ctx.paint()
            ctx.set_operator(OVER)

    def _ambient(self, ctx, t, danger):
        ctx.set_operator(ADD)
        ph = TAU * t / DUR
        glow(ctx, AXC + 300 * math.cos(ph), 760 + 240 * math.sin(ph), 820, (0.10, 0.55, 0.52), 0.15)
        glow(ctx, AXC - 340 * math.cos(ph), 1400 - 260 * math.sin(ph), 640, (0.05, 0.35, 0.60), 0.12)
        if danger > 0.02:
            glow(ctx, AXC, 1640, 640, HOT, 0.17 * danger)
        for x0, y0, r, a, loops, sway, cyc, p0 in self.dust:          # drifting motes, periodic over the loop
            y = (y0 - loops * H * t / DUR) % H
            x = x0 + sway * math.sin(cyc * ph + p0)
            g = cairo.RadialGradient(x, y, 0, x, y, r)
            g.add_color_stop_rgba(0, 0.62, 1.0, 0.92, a)
            g.add_color_stop_rgba(1, 0.62, 1.0, 0.92, 0)
            ctx.set_source(g)
            ctx.arc(x, y, r, 0, TAU)
            ctx.fill()
        ctx.set_operator(OVER)

    def _floor(self, ctx, t, st, font):
        if st["mode"] != "none":                           # big seconds-left numeral painted on the floor
            n = max(1, math.ceil(st["left"] - 1e-6))
            hot = st["mode"] == "hold" and st["left"] <= 3.0 and t < PLAY
            pop = 1.0
            if st["mode"] == "hold" and t < PLAY:
                pop = 1 + 0.10 * (1 - ease_out(ramp(n - st["left"], 0.0, 0.25)))
            size = 540 * pop
            col = HOT if hot else MINT
            a = (0.46 if hot else 0.32) * clamp(st["pop"])
            font.draw(ctx, str(n), AW / 2, AH / 2 + self.cap * size / 2, size, col, "c", a)
        d = st["danger"]
        if d > 0.01:                                       # the wall heats up in the last second and a half
            pulse = 0.55 + 0.45 * self.tick_pulse(t)
            ctx.set_operator(ADD)
            for wdt, a in ((46, 0.12), (26, 0.24)):
                rounded_rect(ctx, -5, -5, AW + 10, AH + 10, CORNER + 5)
                ctx.set_line_width(wdt)
                ctx.set_source_rgba(HOT[0], HOT[1], HOT[2], a * d * pulse)
                ctx.stroke()
            ctx.set_operator(OVER)
            rounded_rect(ctx, -5, -5, AW + 10, AH + 10, CORNER + 5)
            ctx.set_line_width(10)
            ctx.set_source_rgba(HOT[0], HOT[1], HOT[2], min(1.0, 0.95 * d * pulse))
            ctx.stroke()

    def _marks(self, ctx, t):
        fade = 1 - ramp(t, T_RESET, T_RESET + 0.25)
        for tb, i, bx, by, _ in self.bursts:               # scorch mark where each ball went out
            age = t - tb
            if age < 0.12 or fade <= 0.01:
                continue
            a = ramp(age, 0.12, 0.5) * fade
            col = COLS[i]
            ctx.new_path()
            ctx.set_source_rgba(INK[0], INK[1], INK[2], 0.42 * a)
            ctx.arc(bx, by, R * 1.2, 0, TAU)
            ctx.fill()
            ctx.set_source_rgba(col[0], col[1], col[2], 0.42 * a)
            ctx.set_line_width(4)
            ctx.set_dash([10, 9])
            ctx.arc(bx, by, R * 0.95, 0, TAU)
            ctx.stroke()
            ctx.set_dash([])
            symbol(ctx, i, bx, by, R * 0.36, 0.48 * a, col)

    def _trails(self, ctx, t, st):
        ctx.set_operator(ADD)
        for i in range(NB):
            sc = self.scale(i, t)
            if sc <= 0.05:
                continue
            lit = st["mode"] == "hold" and st["c"] == i
            n = 13 if lit else 8
            col = HOT if lit else COLS[i]
            for k in range(n, 0, -1):
                ts = t - k * 0.02
                if t >= T_RESET > ts:
                    continue
                x, y = self.P(i, ts)
                u = 1 - k / (n + 1)
                ctx.set_source_rgba(col[0], col[1], col[2], ((0.07 + 0.20 * u) if lit else (0.03 + 0.11 * u)) * sc)
                ctx.arc(x, y, R * (0.42 + 0.52 * u) * sc, 0, TAU)
                ctx.fill()
        ctx.set_operator(OVER)

    def _fx_under(self, ctx, t):
        ctx.set_operator(ADD)
        j = bisect.bisect_right(self.pass_t, t) - 1
        while j >= 0 and t - self.passes[j][0] < 0.4 and t < PLAY + 0.4:       # pulse on the new carrier
            p = self.passes[j]
            k = (t - p[0]) / 0.4
            x, y = self.P(p[2], t)
            ctx.set_source_rgba(HOT2[0], HOT2[1], HOT2[2], 0.75 * (1 - k) ** 2)
            ctx.set_line_width(9 * (1 - k) + 2)
            ctx.arc(x, y, RING_R + 14 + 95 * ease_out(k), 0, TAU)
            ctx.stroke()
            j -= 1
        for jb, (tb, i, bx, by, _) in enumerate(self.bursts):                  # shock ring
            age = t - tb
            if not 0 <= age < 0.75:
                continue
            k = age / 0.75
            big = 1.35 if jb == len(self.bursts) - 1 else 1.0
            glow(ctx, bx, by, 420 * big, HOT, 0.60 * (1 - k) ** 2)
            ctx.set_source_rgba(HOT2[0], HOT2[1], HOT2[2], 0.85 * (1 - k))
            ctx.set_line_width(28 * (1 - k) + 2)
            ctx.arc(bx, by, R + 440 * big * ease_out(k), 0, TAU)
            ctx.stroke()
        if t >= PLAY:                                                          # the winner's slow glow
            k = ease_out(ramp(t, PLAY + 0.1, PLAY + 0.9)) * (1 - ramp(t, T_RESET, T_RESET + 0.2))
            if k > 0.01:
                x, y = self.wpos(t)
                col = COLS[self.winner]
                glow(ctx, x, y, 330, col, 0.42 * k)
                rot = TAU * t / 14.0
                for q in range(12):
                    a = rot + q * TAU / 12
                    ctx.move_to(x, y)
                    ctx.line_to(x + 380 * math.cos(a - 0.085), y + 380 * math.sin(a - 0.085))
                    ctx.line_to(x + 380 * math.cos(a + 0.085), y + 380 * math.sin(a + 0.085))
                    ctx.close_path()
                g = cairo.RadialGradient(x, y, R, x, y, 380)
                g.add_color_stop_rgba(0, col[0], col[1], col[2], 0.22 * k)
                g.add_color_stop_rgba(1, col[0], col[1], col[2], 0)
                ctx.set_source(g)
                ctx.fill()
                for q in range(3):
                    phs = ((t - PLAY) / 1.5 + q / 3) % 1.0
                    ctx.set_source_rgba(col[0], col[1], col[2], 0.50 * k * (1 - phs))
                    ctx.set_line_width(5)
                    ctx.arc(x, y, R + 12 + 230 * phs, 0, TAU)
                    ctx.stroke()
                for ang, rad, size, p0, cyc in self.twinkle:
                    tw = max(0.0, math.sin(TAU * (t - PLAY) * cyc / 2 + p0)) ** 2
                    glow(ctx, x + rad * math.cos(ang + rot * 0.5), y + rad * math.sin(ang + rot * 0.5),
                         size * 3.2, WHITE, 0.85 * k * tw)
        ctx.set_operator(OVER)

    def _aim(self, ctx, t, st):
        """Dashed line and brackets: which ball the carrier is steering at right now."""
        tg = st.get("tg", -1)
        a = 0.62 * clamp(st["pop"])
        if st["mode"] != "hold" or tg < 0 or a <= 0.02:
            return
        cx, cy = self.P(st["c"], t)
        tx, ty = self.P(tg, t)
        dx, dy = tx - cx, ty - cy
        d = math.hypot(dx, dy)
        ctx.new_path()
        ctx.set_source_rgba(HOT[0], HOT[1], HOT[2], a)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        if d > RING_R + R + 50:
            ux, uy = dx / d, dy / d
            ctx.set_line_width(4)
            ctx.set_dash([14, 12], -t * 84.5)
            ctx.move_to(cx + ux * (RING_R + 20), cy + uy * (RING_R + 20))
            ctx.line_to(tx - ux * (R + 22), ty - uy * (R + 22))
            ctx.stroke()
            ctx.set_dash([])
        rot = TAU * t * 0.5
        ctx.set_line_width(5)
        for q in range(4):
            ctx.new_sub_path()
            ctx.arc(tx, ty, R + 14, rot + q * TAU / 4 - 0.32, rot + q * TAU / 4 + 0.32)
        ctx.stroke()
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)

    def _balls(self, ctx, t, st):
        c = st["c"] if st["mode"] == "hold" else -1
        jp = bisect.bisect_right(self.pass_t, t) - 1 if t < PLAY else -1
        lastp = self.passes[jp] if jp >= 0 else None
        if T_RESET <= t < T_RESET + 0.2:                   # the winner shrinks away as the six pop back in
            x, y = self.wpos(t)
            draw_ball(ctx, self.winner, x, y, R * (1 - ease_out(ramp(t, T_RESET, T_RESET + 0.2))) + 0.01)
        for i in [b for b in range(NB) if b != c] + ([c] if c >= 0 else []):
            sc = self.scale(i, t)
            if sc <= 0.03:
                continue
            x, y = self.P(i, t)
            r = R * sc
            if i == c:
                fl = 0.5 + 0.5 * math.sin(TAU * t * 6) * math.sin(TAU * t * 3.75)
                ctx.set_operator(ADD)
                glow(ctx, x, y, R * 3.4, HOT, (0.50 + 0.12 * fl) * clamp(st["pop"]))
                ctx.set_operator(OVER)
                if lastp and lastp[2] == i:
                    r *= 1 + 0.16 * (1 - ramp(t, lastp[0], lastp[0] + 0.18))
            draw_ball(ctx, i, x, y, r)
        j = jp
        while j >= 0 and t - self.passes[j][0] < IMMUNE_S:   # dashed shield: the passer cannot be tagged back yet
            p = self.passes[j]
            i = p[1]
            if t < self.out[i] and i != c:
                k = (t - p[0]) / IMMUNE_S
                x, y = self.P(i, t)
                ctx.new_path()
                ctx.set_source_rgba(MINT[0], MINT[1], MINT[2], 0.95 * (1 - 0.5 * k))
                ctx.set_line_width(5)
                ctx.set_dash([9, 7])
                ctx.arc(x, y, R + 10, -TAU / 4, -TAU / 4 + (1 - k) * TAU)
                ctx.stroke()
                ctx.set_dash([])
            j -= 1

    def _ring(self, ctx, f, t, st):
        ctx.set_operator(ADD)
        for j in range(1, 13):                             # embers shed by the burning end of the ring
            te = t - j / FPS
            ang, spd, size, life = self.ember[(f - j) % NF]
            age = j / FPS
            if age >= life:
                continue
            rp = self.ring_pos(te % DUR)
            if rp is None:
                continue
            a0 = -TAU / 4 + (1 - rp[2]) * TAU
            k = age / life
            d = spd * age * (1 - 0.4 * k)
            x = rp[0] + RING_R * rp[3] * math.cos(a0) + d * math.cos(ang)
            y = rp[1] + RING_R * rp[3] * math.sin(a0) + d * math.sin(ang) - 40 * age
            glow(ctx, x, y, size * 2.6 * (1 - 0.5 * k), HOT2, 0.9 * (1 - k) * clamp(rp[3]))
        ctx.set_operator(OVER)
        rp = self.ring_pos(t, st)
        if rp is None:
            return
        x, y, frac, sc = rp
        if st["mode"] == "jump":                           # the charge in flight: a hot spark with a tail
            ctx.set_operator(ADD)
            for q in range(1, 9):
                tq = t - q * 0.012
                if tq < self.jumps[st["rnd"] - 1][0]:
                    break
                r2 = self.ring_pos(tq)
                if r2:
                    glow(ctx, r2[0], r2[1], 46 * (1 - q / 10), HOT, 0.55 * (1 - q / 9))
            glow(ctx, x, y, 120, HOT, 0.8)
            glow(ctx, x, y, 44, WHITE, 0.95)
            ctx.set_operator(OVER)
        else:
            sc *= 1 + (0.03 + 0.07 * st["danger"]) * self.tick_pulse(t)
        draw_ring(ctx, x, y, frac, sc, clamp(sc * 1.5))
        if st["mode"] == "hold" and (t < PLAY or t >= T_RESET):       # seconds left, in solid white on the carrier
            a = clamp(st["pop"])
            n = max(1, math.ceil(st["left"] - 1e-6))
            cap = self.cap
            up = y - RING_R - 96 > 0                                   # under the ring when the carrier is at the top wall
            base = y - RING_R - 20 if up else y + RING_R + 20 + 72 * cap
            self.font.draw(ctx, str(n), x, base, 72, WHITE, "c", a, INK, 13)
            ha = hook(t) * a
            if ha > 0.01:
                lb = base - 72 * cap - 22 if up else base + 22 + 48 * cap
                lx = clamp(x, 150, AW - 150)
                self.font.draw(ctx, IT, lx, lb, 48, HOT2, "c", ha, INK, 10)

    def _fx_over(self, ctx, t, font):
        j = bisect.bisect_right(self.wall_t, t) - 1
        while j >= 0 and t - self.walls[j][0] < 0.25:      # wall ripple in the ball's colour
            tw, i, x, y, side = self.walls[j]
            k = (t - tw) / 0.25
            ln = 30 + 70 * ease_out(k)
            col = lighten(COLS[i], 0.3)
            ctx.new_path()
            ctx.set_source_rgba(col[0], col[1], col[2], 0.9 * (1 - k))
            ctx.set_line_width(7 * (1 - k) + 1)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            if side < 2:
                ctx.move_to(x, max(CORNER, y - ln))
                ctx.line_to(x, min(AH - CORNER, y + ln))
            else:
                ctx.move_to(max(CORNER, x - ln), y)
                ctx.line_to(min(AW - CORNER, x + ln), y)
            ctx.stroke()
            ctx.set_line_cap(cairo.LINE_CAP_BUTT)
            j -= 1
        j = bisect.bisect_right(self.pass_t, t) - 1
        ctx.set_operator(ADD)
        while j >= 0 and t - self.passes[j][0] < 0.36 and t < PLAY + 0.36:     # pass: zip between the two balls
            p = self.passes[j]
            age = t - p[0]
            k = age / 0.36
            if age < 0.17:
                ka = age / 0.17
                ax, ay = self.P(p[1], t)
                bx, by = self.P(p[2], t)
                dx, dy = bx - ax, by - ay
                d = math.hypot(dx, dy) or 1.0
                nx, ny = -dy / d, dx / d
                pts = [(ax, ay)] + [(ax + dx * q / 6 + nx * 20 * self.zig[j][q - 1],
                                     ay + dy * q / 6 + ny * 20 * self.zig[j][q - 1]) for q in range(1, 6)] + [(bx, by)]
                for wdt, col, al in ((20, HOT, 0.45), (6, WHITE, 1.0)):
                    ctx.new_path()
                    ctx.move_to(*pts[0])
                    for q in pts[1:]:
                        ctx.line_to(*q)
                    ctx.set_source_rgba(col[0], col[1], col[2], al * (1 - ka))
                    ctx.set_line_width(wdt)
                    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                    ctx.stroke()
                glow(ctx, p[3], p[4], 150, HOT2, 0.85 * (1 - ka))
            for ang, spd, size in self.spk[j]:
                d = spd * age * (1 - 0.5 * k)
                glow(ctx, p[3] + d * math.cos(ang), p[4] + d * math.sin(ang), size * 2.4 * (1 - k) + 1, HOT2, 1 - k)
            j -= 1
        ctx.set_operator(OVER)
        for jb, (tb, i, bx, by, _) in enumerate(self.bursts):                   # burst: flash, chunks, sparks, label
            age = t - tb
            if not 0 <= age < 1.3:
                continue
            col = COLS[i]
            big = 1.3 if jb == len(self.bursts) - 1 else 1.0
            if age < 0.16:
                k = age / 0.16
                ctx.new_path()
                ctx.set_source_rgba(1, 0.97, 0.9, (1 - k) ** 1.5)
                ctx.arc(bx, by, R * (1 + 2.4 * big * ease_out(k)), 0, TAU)
                ctx.fill()
            for ang, spd, size, sel, life in self.parts[jb]:
                if age >= life:
                    continue
                k = age / life
                d = spd * big * (1 - math.exp(-3.4 * age)) / 3.4
                ca, sa = math.cos(ang), math.sin(ang)
                x, y = bx + ca * d, by + sa * d
                a = 1 - k * k
                ctx.new_path()
                if sel < 0.55:
                    ctx.set_source_rgba(INK[0], INK[1], INK[2], 0.8 * a)
                    ctx.arc(x, y, size * (1 - 0.6 * k) + 2.5, 0, TAU)
                    ctx.fill()
                    ctx.set_source_rgba(col[0], col[1], col[2], a)
                    ctx.arc(x, y, size * (1 - 0.6 * k), 0, TAU)
                    ctx.fill()
                else:
                    cc = HOT2 if sel < 0.8 else WHITE
                    tail = min(d, 30 + spd * 0.04) * (1 - k)
                    ctx.set_operator(ADD)
                    ctx.set_source_rgba(cc[0], cc[1], cc[2], 0.9 * a)
                    ctx.set_line_width(size * 0.5 * (1 - k) + 1.5)
                    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                    ctx.move_to(x, y)
                    ctx.line_to(x - ca * tail, y - sa * tail)
                    ctx.stroke()
                    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
                    ctx.set_operator(OVER)
            if 0.08 < age < 1.0:
                size = 72 * ease_out_back(ramp(age, 0.08, 0.3))
                if size > 3:
                    font.draw(ctx, "OUT!", clamp(bx, 95, AW - 95), clamp(by - 80 - 46 * ease_out(age), 84, AH - 30),
                              size, col, "c", 1 - ramp(age, 0.7, 1.0), INK, 10, shadow=4)

    def _finale(self, ctx, t, font):
        if t < T_WIN:
            return
        fade = 1 - ramp(t, T_RESET, T_RESET + 0.18)
        if fade <= 0.004:
            return
        col = COLS[self.winner]
        size = self.win_size * ease_out_back(ramp(t, T_WIN, T_WIN + 0.34))
        if size > 3:
            font.draw(ctx, self.win_text, AW / 2, AH * 0.235, size, col, "c", fade, INK, size * 0.11,
                      shadow=size * 0.06)
        size = 62 * ease_out_back(ramp(t, T_WIN + 0.35, T_WIN + 0.6))
        if size > 3:
            font.draw(ctx, f"{len(self.passes)} PASSES", AW / 2, AH * 0.235 + 88, size, MINT, "c", fade, INK, 9,
                      shadow=4)
        k = ramp(t, T_LINE, T_LINE + 0.32)
        size = self.line_size * ease_out_back(k) * (1 + 0.025 * math.sin(TAU * (t - T_LINE) * 2.0))
        if k > 0 and size > 3:
            font.draw(ctx, ASK, AW / 2, AH * 0.885, size, WHITE, "c", fade, INK, size * 0.12,
                      shadow=size * 0.06)

    def _titles(self, ctx, t, h, e1, e2, st, font):
        # stage 1: both lines shrink where they are; stage 2: the orange words rise onto the first line
        sa, sb = lerp(self.s1, self.sm, e1), lerp(self.s2, self.sm, e1)
        xa, xb = lerp(self.xa_h, self.xa_m, e1), lerp(self.xb_h, self.xb_m, e1)
        ya, yb = lerp(self.y1, self.ym, e1), lerp(self.y2, self.ym, e2)
        font.draw(ctx, QA, xa, ya, sa, WHITE, "l", 1.0, INK, sa * 0.10, shadow=sa * 0.05)
        wb = font.width(QB, sb)
        pulse = 0.5 + 0.5 * math.sin(TAU * t * 1.5)
        ctx.set_operator(ADD)
        glow(ctx, xb + wb / 2, yb - self.cap * sb / 2, wb * 0.8, HOT, 0.14 + 0.10 * pulse * h)
        ctx.set_operator(OVER)
        g = cairo.LinearGradient(0, yb - self.cap * sb, 0, yb)
        g.add_color_stop_rgb(0, *HOT2)
        g.add_color_stop_rgb(1, *HOT)
        font.draw(ctx, QB, xb, yb, sb, g, "l", 1.0, INK, sb * 0.10, shadow=sb * 0.05)
        a = e2
        if a > 0.01 and st["mode"] != "none":              # the fuse burning down, under the chips
            fr = clamp(st["frac"]) if st["mode"] == "hold" else 0.0
            x0, y0, bw, bh = AX, 380.0, AW, 20.0
            ctx.new_path()
            rounded_rect(ctx, x0, y0, bw, bh, 10)
            ctx.set_source_rgba(INK[0], INK[1], INK[2], 0.80 * a)
            ctx.fill()
            if fr > 0.004:
                col = mix(RED, HOT, clamp(fr / 0.25)) if fr < 0.25 else mix(HOT, HOT2, clamp((fr - 0.25) / 0.45))
                rounded_rect(ctx, x0, y0, max(20.0, bw * fr), bh, 10)
                ctx.set_source_rgba(col[0], col[1], col[2], a)
                ctx.fill()
                ctx.set_operator(ADD)
                glow(ctx, x0 + bw * fr, y0 + bh / 2, 34, HOT2, 0.85 * a)
                glow(ctx, x0 + bw * fr, y0 + bh / 2, 13, WHITE, 0.95 * a)
                ctx.set_operator(OVER)
        if a > 0.01:                                       # round and pass counters above the arena
            font.draw(ctx, f"ROUND {st['rnd'] + 1}/{NR}", AX + 4, STRIP_Y, 38, MINT, "l", 0.92 * a, INK, 7)
            n = bisect.bisect_right(self.pass_t, t)
            pop = 1.0
            if n and t < PLAY + 0.2:
                pop = 1 + 0.22 * (1 - ramp(t, self.pass_t[n - 1], self.pass_t[n - 1] + 0.16))
            font.draw(ctx, f"PASSES {n}", AX + AW - 4, STRIP_Y, 38 * pop, WHITE, "r", 0.92 * a, INK, 7)

    def _chips(self, ctx, t, e2, st):
        """Status row: one chip per ball. Bright = still in, crossed and dark = out, orange with a ring = holding."""
        y0 = lerp(self.chip_y_hook, CHIP_Y, e2)
        for i in range(NB):
            cx = AXC + (i - 2.5) * CHIP_GAP
            col = COLS[i]
            dead = self.out[i] <= t
            back = ramp(t, T_RESET + 0.02 * i, T_RESET + 0.12 + 0.02 * i)      # chips relight for the loop
            dk = 0.0 if not dead else 1.0 - back
            holding = st["mode"] == "hold" and st["c"] == i and dk < 0.5
            won = i == self.winner and PLAY <= t < T_RESET
            x0, y = cx - CHIP_W / 2, y0
            if dead and t < T_RESET:
                age = t - self.out[i]
                x0 += 8 * math.exp(-age / 0.12) * math.sin(age * 70)
                y += 7 * ease_out(ramp(age, 0, 0.3))
            elif dead:
                y += 7 * (1 - back)
            if holding or won:
                gc = HOT if holding else col
                ctx.set_operator(ADD)
                glow(ctx, cx, y + CHIP_H / 2, 120, gc, 0.34 * clamp(st["pop"] if holding else 1.0))
                ctx.set_operator(OVER)
            ctx.new_path()
            rounded_rect(ctx, x0, y, CHIP_W, CHIP_H, 24)
            if holding:
                fc = mix(INK, HOT, 0.26)
                ctx.set_source_rgba(fc[0], fc[1], fc[2], 0.96)
            else:
                fc = mix((0.030, 0.200, 0.235), INK, dk * 0.8)
                ctx.set_source_rgba(fc[0], fc[1], fc[2], 0.94)
            ctx.fill_preserve()
            if holding:
                ctx.set_source_rgba(HOT[0], HOT[1], HOT[2], 1.0)
                ctx.set_line_width(5)
            else:
                bc = mix(col, (0.25, 0.36, 0.38), dk)
                ctx.set_source_rgba(bc[0], bc[1], bc[2], lerp(0.9, 0.5, dk))
                ctx.set_line_width(5 if won else 3.5)
            ctx.stroke()
            icx, icy = x0 + CHIP_W / 2, y + CHIP_H / 2
            draw_ball(ctx, i, icx, icy, 31, lerp(1.0, 0.26, dk))
            if dk > 0.02:
                for wdt, cc in ((14, INK), (8, RED)):
                    ctx.set_source_rgba(cc[0], cc[1], cc[2], dk)
                    ctx.set_line_width(wdt)
                    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
                    ctx.move_to(icx - 25, icy - 25)
                    ctx.line_to(icx + 25, icy + 25)
                    ctx.move_to(icx + 25, icy - 25)
                    ctx.line_to(icx - 25, icy + 25)
                    ctx.stroke()
                ctx.set_line_cap(cairo.LINE_CAP_BUTT)
            if holding:
                draw_ring(ctx, icx, icy, st["frac"], 0.64 * st["pop"], clamp(st["pop"] * 1.5))

    # ------------------------------------------------------------ sound
    def audio(self):
        """Stereo float mix at au.SR. Palette: chip voices (one per ball, they drop out with their ball),
        a metronome that speeds up and rises as the ring empties, two-tone sine zips for passes,
        noise-and-sub bursts, a riser into the last zero and a held additive-sine chord for the winner."""
        sim = self.sim
        rng = np.random.default_rng(self.seed + 2024)
        sr = au.SR
        n, tail = int(DUR * sr), 3 * sr
        fx = np.zeros((n + tail, 2))             # events (these get reverb)
        bed = np.zeros((n + tail, 2))            # voices, pulse, pads
        tt_ = lambda dur: np.arange(int(dur * sr)) / sr
        pan = lambda x: clamp((x / AW - 0.5) * 1.5, -0.8, 0.8)
        note = (64, 67, 69, 71, 74, 76)          # E minor pentatonic, one note per ball

        # hocket: every 0.2 s one ball's note; a ball that is out leaves a rest, so the texture thins
        order, cache, k = (0, 3, 1, 4, 2, 5), {}, 0
        while k * 0.2 < PLAY - 0.05:
            t, b = k * 0.2, order[k % NB]
            if t < self.out[b]:
                lit = self.state(t)["c"] == b
                m = note[b] + (12 if lit else 0)             # the carrier's note jumps an octave
                if m not in cache:
                    cache[m] = au.chip(au.hz(m), 0.34)
                au.place(bed, cache[m], t, 0.30 if lit else 0.17, pan(self.P(b, t)[0]))
            k += 1

        # pulse: gets denser every round while the voices thin out
        kick, beat, j = au.kick(0.28), 0.4, 0
        while j * beat < PLAY - 0.02:
            t = j * beat
            r = self.state(t)["rnd"]
            lvl = 0.55 + 0.11 * r
            if r >= 2 or j % 2 == 0:
                au.place(bed, kick, t, 0.44 * lvl)
            if r >= 1:
                au.place(bed, au.hat(rng), t + beat / 2, 0.20 * lvl, 0.3)
            if r >= 3:
                for q in (0.25, 0.75):
                    au.place(bed, au.shaker(rng), t + beat * q, 0.12 * lvl, -0.3)
            if r >= 4:
                au.place(bed, au.bass(au.hz(40 if j % 4 < 3 else 43), 0.16), t, 0.30)
                au.place(bed, au.bass(au.hz(52), 0.12), t + beat / 2, 0.20)
            j += 1
        for r, (t0, t1, _) in enumerate(self.rounds):        # low drone, a little louder each round
            d = t1 - t0 + JUMP_S + 0.3
            for m, g, pn in ((40, 0.55, 0.0), (47, 0.38, -0.3)) + (((52, 0.30, 0.3),) if r >= 2 else ()) \
                    + (((55, 0.24, 0.0),) if r >= 4 else ()):
                au.place(bed, au.pad(au.hz(m), d, 0.3, 0.5), t0, (0.10 + 0.035 * r) * g * 2, pn)

        # metronome: rate and pitch rise as the ring empties (same clicks pulse the ring on screen)
        for tk, kk in self.ticks:
            au.place(fx, au.tick(880 * 2 ** (0.9 * kk), 0.05), tk, 0.10 + 0.12 * kk)

        last = -1.0
        for p in self.passes:                                # pass: two-tone sine zip, passer note then receiver note
            if p[0] - last < 0.04:
                continue
            last = p[0]
            fa, fb = au.hz(note[p[1]] + 24), au.hz(note[p[2]] + 24)
            z = np.concatenate([au.glide(fa, fa * 1.03, 0.045, None, 0.15), au.glide(fb * 0.97, fb, 0.085, 0.05, 0.15)])
            au.place(fx, z, p[0], 0.30, pan(p[3]))
        last = -1.0
        for wl in self.walls:                                # soft wall knocks
            if wl[0] >= PLAY or wl[0] - last < 0.06:
                continue
            last = wl[0]
            au.place(fx, au.glide(210, 140, 0.06, 0.03, 0.2), wl[0], 0.10, pan(wl[2]))
        for ht in sim["hits"]:
            if not ht[5]:
                au.place(fx, au.tick(620, 0.05), ht[0], 0.07, pan(ht[3]))

        for jb, (tb, i, bx, by, to) in enumerate(self.bursts):   # burst: noise with a falling filter + sub thump
            fin = jb == len(self.bursts) - 1
            g, d = (1.3, 1.1) if fin else (1.0, 0.75)
            au.place(fx, au.swept_noise(rng, d, 7500, 170) * np.exp(-tt_(d) * (3.2 if fin else 5.0)), tb, 0.55 * g,
                     pan(bx) * 0.6)
            au.place(fx, au.boom(rng, 1.1, 115, 34), tb, 0.80 * g)
            au.place(fx, au.noise(rng, 0.10, 2500, 10000) * np.exp(-tt_(0.10) * 45), tb, 0.35 * g, pan(bx))
            if to >= 0:                                      # the charge jumping to the nearest ball
                au.place(fx, au.glide(480, 1900, JUMP_S, None, 0.2) * np.linspace(0.3, 1.0, int(JUMP_S * sr)), tb,
                         0.14, pan(self.P(to, tb + JUMP_S)[0]))
                au.place(fx, au.tick(2400, 0.06), tb + JUMP_S, 0.22, pan(self.P(to, tb + JUMP_S)[0]))

        rl = 3.2                                             # riser into the last zero
        env = (tt_(rl) / rl) ** 2
        au.place(bed, au.swept_noise(rng, rl, 300, 5500) * env, PLAY - rl, 0.26)
        au.place(bed, au.glide(165, 660, rl) * env, PLAY - rl, 0.09)

        # winner: held additive-sine chord (E major), a short chip arpeggio on top
        t0 = PLAY + 0.10
        cd = T_RESET + 0.35 - t0
        tc = tt_(cd)
        env = np.minimum(1.0, tc / 0.05) * np.minimum(1.0, np.maximum(0.0, (cd - tc) / 0.7)) \
            * (0.86 + 0.14 * np.sin(au.TAU * 5.0 * tc))
        for m, g, pn in ((40, 0.5, 0.0), (52, 0.8, -0.2), (59, 0.7, 0.3), (64, 0.8, -0.4), (68, 0.75, 0.4),
                         (71, 0.6, 0.0), (76, 0.45, 0.2)):
            f = au.hz(m)
            x = sum(a * np.sin(au.TAU * f * hn * tc + ph)
                    for hn, a, ph in ((1, 1.0, 0.0), (2, 0.35, 1.0), (3, 0.15, 2.0), (4, 0.06, 0.5)))
            au.place(bed, x * env, t0, 0.085 * g, pn)
        for q, m in enumerate((76, 80, 83, 88)):
            au.place(fx, au.chip(au.hz(m), 0.6), PLAY + 0.14 + q * 0.07, 0.26, (-0.4, -0.1, 0.2, 0.45)[q])
        au.place(fx, au.glide(660, 1320, 0.10, 0.06), T_LINE, 0.20)
        wl = DUR - T_RESET                                   # rewind whoosh into the loop point, then launch blip
        au.place(bed, au.swept_noise(rng, wl, 500, 6000) * (tt_(wl) / wl) ** 2, T_RESET, 0.50)
        au.place(fx, au.glide(330, 990, 0.12, 0.07), 0.0, 0.26)

        mix_ = au.reverb(fx, rng, decay=0.9, wet=0.14) + bed
        out = mix_[:n].copy()
        out[:tail] += mix_[n:]                    # tails past the end wrap to the start: seamless loop
        return au.eq(out, ((60, -2.0), (250, -2.5), (900, 0.0), (3000, 3.5), (8000, 5.0), (16000, 3.0)))

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        title = "Who's holding the charge at zero? 6-ball fuse tag"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "Six balls bounce in an arena and one carries a lit charge with a countdown ring. Touching another "
                "ball passes it on; at zero the holder bursts and the charge jumps to the nearest ball. The "
                "carrier is not passive: it steers toward the nearest ball and moves 30% faster than the rest. "
                "Last ball left wins.\n"
                "Five fuses, five bursts, one survivor. The big number on the floor, the digit on the carrier and "
                f"the bar under the chips all count the same fuse. Simulated and drawn by code (seed {self.seed}), "
                "sound synthesized, with no footage, samples or stock media.\n"
                "Did your pick survive?"),
            tags=["simulation", "tag game", "bouncing balls", "hot potato", "elimination",
                  "last one standing", "generative animation", "made with code"],
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylised 2D animation of abstract shapes; nothing "
                                              "realistic that could be mistaken for real people, places or events.",
            seed=self.seed,
            drama=self.stats,
        )
