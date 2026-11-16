"""EVENT HORIZON: five planets orbit a black hole; drag shrinks every orbit; the last planet left wins.

Rules (game rules, not astrophysics): inverse-square gravity, a weak linear drag that differs per planet
(shown on screen from the first frame as 1 to 5 pips), a planet whose edge touches the horizon is swallowed,
and the horizon radius grows 15% with each meal. The run ends when one planet is left; the video keeps
simulating at full speed (no slow motion) and a seed only counts if the survivor then completes a full orbit. simulate(seed) is deterministic and needs no rendering, so make.py
can score hundreds of seeds for drama (drama()) and render only the best one (Scene).
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
NP = 5
CX, CY = 490.0, 765.0        # the hole on screen: centred in the area the Shorts UI leaves free
R0 = 54.0                    # starting horizon radius (world units = pixels at zoom 1)
PR = 34.0                    # planet radius; a planet is swallowed when its edge touches the horizon
GROW, GROW_T = 1.15, 0.30    # the horizon grows 15% per meal, over 0.3 s
MU = 2.8e7                   # gravitational parameter
PRE = 2.4                    # seconds simulated before frame 0 (trails exist on the first frame; loop hand-over)
SPS = 4                      # recorded samples per frame
HS = 1.0 / (FPS * SPS)
S0 = int(round(PRE * FPS * SPS))      # sample index of t = 0
CSUB = 7200.0                # substeps per sample ~ CSUB / r^1.5 (finer near the hole)
T_EARLY = 4.0                # a meal before this makes the seed invalid (no early blowout)
END_MIN, END_MAX = 18.5, 24.0         # the last meal must land in this window (video = last meal + POST)
REVEAL_AFTER, LINE_AFTER, RESET_AFTER, RESET_LEN = 0.90, 0.75, 2.55, 0.60
POST = REVEAL_AFTER + RESET_AFTER + RESET_LEN         # shortest ending; it is held until the survivor's orbit closes
ORBIT_HOLD, ORBIT_MAX = 0.20, 5.6     # the reset waits this long after the full orbit; longer orbits are rejected
DUR_MIN, DUR_MAX = 24.1, 30.0
SIM_POST = ORBIT_MAX + ORBIT_HOLD + RESET_LEN + 0.1   # the whole ending is simulated and shown at full speed
GAP_W, GAP_ANY = 6.0, 2.5    # seed gates: the winner's (and anyone's) survived dives must clear visibly
TAU_BASE = (6.8, 11.5, 16.5, 23.0, 32.0)   # nominal fall times the per-planet drag is drawn around
TAU_JIT = (1.0, 1.3, 1.5, 1.5, 2.5)
RP_F = (1.08, 1.08, 1.08, 1.08, 1.25)   # smallest starting periapsis per fall rank, as a multiple of its kill radius

NAMES = ("YELLOW", "BLUE", "GREEN", "RED", "PINK")      # symbols: triangle, ring, plus, diamond, bars


# ====================================================================== simulation
def initial(seed):
    """Seeded starting orbits. Shape, orientation, phase and drag are all drawn independently of each other."""
    rng = random.Random(seed * 9973 + 4)
    dirn = rng.choice((-1.0, 1.0))
    th0 = rng.uniform(0, TAU)
    slots = list(range(NP))
    rng.shuffle(slots)
    ranks = list(range(NP))
    rng.shuffle(ranks)
    out = []
    for i in range(NP):
        j = ranks[i]
        thr = R0 * GROW ** j + PR                  # the kill radius this planet meets after j meals
        while True:
            ra = rng.uniform(325.0, 430.0)
            e = rng.uniform(0.26, 0.66)
            rp = ra * (1 - e) / (1 + e)
            if RP_F[j] * thr <= rp <= 240.0:
                break
        tau = TAU_BASE[j] + rng.uniform(-TAU_JIT[j], TAU_JIT[j])
        k = clamp(0.5 * math.log(rp / thr) / tau, 0.0008, 0.09)    # linear drag: periapsis decays as exp(-2kt)
        a = 0.5 * (ra + rp)
        om = th0 + slots[i] * TAU / NP + rng.uniform(-0.36, 0.36)
        M = rng.uniform(0, TAU)
        E = M
        for _ in range(80):
            E = M + e * math.sin(E)
        nu = 2 * math.atan2(math.sqrt(1 + e) * math.sin(E / 2), math.sqrt(1 - e) * math.cos(E / 2))
        r = a * (1 - e * math.cos(E))
        s = math.sqrt(MU / (a * (1 - e * e)))
        xo, yo = r * math.cos(nu), r * math.sin(nu) * dirn
        vxo, vyo = -s * math.sin(nu), s * (e + math.cos(nu)) * dirn
        c, sn = math.cos(om), math.sin(om)
        out.append(dict(x=xo * c - yo * sn, y=xo * sn + yo * c, vx=vxo * c - vyo * sn, vy=vxo * sn + vyo * c,
                        k=k, e=e, ra=ra, rp=rp))
    return dirn, out


def simulate(seed, record=False):
    dirn, P = initial(seed)
    x = [p["x"] for p in P]
    y = [p["y"] for p in P]
    vx = [p["vx"] for p in P]
    vy = [p["vy"] for p in P]
    kd = [p["k"] for p in P]
    alive = [True] * NP
    mins = [1e9] * NP                     # closest gap so far (edge of planet to horizon)
    pmin, pang, pt, prh = [1e9] * NP, [0.0] * NP, [0.0] * NP, [R0] * NP
    prd = [x[i] * vx[i] + y[i] * vy[i] for i in range(NP)]
    meals, passes, ms, fra, gapn, rh_s = [], [], [], [], [], []
    xs = [[] for _ in range(NP)]
    ys = [[] for _ in range(NP)]
    s, s_end, reason = 0, None, None
    swept, t_orbit = 0.0, None            # angle the survivor travels after the last meal
    s_limit = S0 + int(END_MAX * FPS * SPS) + 2
    sqrt, atan2 = math.sqrt, math.atan2
    while True:
        t = (s - S0) * HS
        Rh = R0
        for m in meals:
            u = (t - m[0]) / GROW_T
            Rh *= GROW if u >= 1 else 1 + (GROW - 1) * u * u * (3 - 2 * u)
        ms.append(tuple(mins))
        if record:
            rh_s.append(Rh)
            for i in range(NP):
                if alive[i]:
                    xs[i].append(x[i])
                    ys[i].append(y[i])
        if s >= S0 and (s - S0) % SPS == 0:    # once per video frame: far point and next-dive gap of each orbit
            rowa, rowg = [], []
            for i in range(NP):
                if not alive[i]:
                    rowa.append(0.0)
                    rowg.append(0.0)
                    continue
                rr = sqrt(x[i] * x[i] + y[i] * y[i])
                en = 0.5 * (vx[i] * vx[i] + vy[i] * vy[i]) - MU / rr
                if en >= 0:
                    rowa.append(900.0)
                    rowg.append(900.0)
                    continue
                a = -MU / (2 * en)
                L = x[i] * vy[i] - y[i] * vx[i]
                ecc = sqrt(max(0.0, 1 + 2 * en * L * L / (MU * MU)))
                rowa.append(a * (1 + ecc))
                rowg.append(max(0.0, a * (1 - ecc) - Rh - PR))
            fra.append(rowa)
            gapn.append(rowg)
        if s_end is not None and s >= s_end:
            break
        if s_end is None and s > s_limit:
            reason = "too long"
            break
        lim = Rh + PR
        for i in range(NP):
            if not alive[i]:
                continue
            xi, yi, vxi, vyi = x[i], y[i], vx[i], vy[i]
            x0, y0 = xi, yi
            r2 = xi * xi + yi * yi
            rr = sqrt(r2)
            sp = sqrt(vxi * vxi + vyi * vyi)
            reff = max(rr - sp * HS, 0.55 * rr, 40.0)
            n = int(CSUB / (reff * sqrt(reff))) + 1
            n = 2 if n < 2 else 24 if n > 24 else n
            dt = HS / n
            hdt = 0.5 * dt
            drag = 1.0 - kd[i] * dt
            inv = MU / (r2 * rr)
            hit = False
            for _ in range(n):
                vxi -= xi * inv * hdt
                vyi -= yi * inv * hdt
                xi += vxi * dt
                yi += vyi * dt
                r2 = xi * xi + yi * yi
                rr = sqrt(r2)
                inv = MU / (r2 * rr)
                vxi = (vxi - xi * inv * hdt) * drag
                vyi = (vyi - yi * inv * hdt) * drag
                if rr <= lim:
                    hit = True
                    break
            x[i], y[i], vx[i], vy[i] = xi, yi, vxi, vyi
            if hit:
                alive[i] = False
                mins[i] = 0.0
                meals.append((t + HS, i, xi, yi, vxi, vyi, Rh))
                if t + HS < T_EARLY:
                    reason = "early meal"
                continue
            if s_end is not None:
                swept += abs(atan2(x0 * yi - y0 * xi, x0 * xi + y0 * yi))
                if t_orbit is None and swept >= TAU:        # orbit closed: the ending needs no more than this
                    t_orbit = t + HS
                    s_end = min(s_end, s + 1 + int((max(POST - (t_orbit - meals[3][0]), ORBIT_HOLD + RESET_LEN)
                                                    + 0.1) * FPS * SPS))
            g = rr - lim
            if g < mins[i]:
                mins[i] = g
            if g < pmin[i]:
                pmin[i], pang[i], pt[i], prh[i] = g, atan2(yi, xi), t + HS, Rh
            rd = xi * vxi + yi * vyi
            if prd[i] < 0 <= rd:                       # periapsis survived: one dive
                passes.append((pt[i], i, pmin[i], pang[i], prh[i]))
            elif rd < 0 <= prd[i]:                     # apoapsis: a new dive begins
                pmin[i] = 1e9
            prd[i] = rd
        if reason:
            break
        if not any(alive):
            reason = "no survivor"
            break
        if s_end is None and len(meals) == NP - 1:
            s_end = s + 1 + int(SIM_POST * FPS * SPS)
        s += 1
    if reason is None and not (END_MIN <= meals[3][0] <= END_MAX):
        reason = "ends too early"
    if reason is None:
        w = alive.index(True)
        seen = [p for p in passes if p[0] >= 0]
        if t_orbit is None or t_orbit > meals[3][0] + ORBIT_MAX:
            reason = "survivor does not complete a full orbit on screen"
        elif not DUR_MIN <= meals[3][0] + ending(meals[3][0], t_orbit) <= DUR_MAX:
            reason = "video length out of range"
        elif min([p[2] for p in seen if p[1] == w] or [999.0]) < GAP_W:
            reason = "winner clearance too small to see"
        elif min([p[2] for p in seen] or [999.0]) < GAP_ANY:
            reason = "a survived dive too close to read"
    sim = dict(seed=seed, valid=reason is None, reason=reason, dirn=dirn, params=P, meals=meals, passes=passes,
               ms=ms, gapn=gapn, t_orbit=t_orbit, winner=alive.index(True) if any(alive) else -1)
    if record:
        sim.update(xs=xs, ys=ys, rh=rh_s, fra=fra)
    return sim


def ending(t_end, t_orbit):
    """Seconds of video after the last meal: at least POST, and long enough to show the survivor's full orbit."""
    return max(POST, t_orbit - t_end + ORBIT_HOLD + RESET_LEN)


def drag_pips(params):
    """1 to 5 pips per planet: its rank by drag (1 = least drag). The same pips are drawn on screen."""
    by = sorted(range(NP), key=lambda i: params[i]["k"])
    return [by.index(i) + 1 for i in range(NP)]


def drama(sim):
    """Score one simulated run. Higher = more worth watching."""
    if not sim["valid"]:
        return dict(seed=sim["seed"], score=-1000.0, valid=False, reason=sim["reason"])
    meals, passes, gn, w = sim["meals"], sim["passes"], sim["gapn"], sim["winner"]
    T = [m[0] for m in meals]
    order = [m[1] for m in meals]
    ru = order[3]
    death = [1e9] * NP
    for m in meals:
        death[m[1]] = m[0]
    f_end = int(T[3] * FPS)
    lead, risk = [], []
    for f in range(f_end + 1):              # the on-screen standings measure: gap on each orbit's next dive
        row = gn[f]
        al = [i for i in range(NP) if death[i] > f / FPS]
        lead.append(max(al, key=lambda i: row[i]))
        risk.append(min(al, key=lambda i: row[i]) if len(al) >= 3 else -1)
    f0 = int(2.0 * FPS)
    runs = []
    for f in range(f0, f_end + 1):
        if runs and runs[-1][0] == lead[f]:
            runs[-1][2] += 1
        else:
            runs.append([lead[f], f, 1])
    solid = []
    for r in runs:
        if r[2] < int(0.5 * FPS) and r is not runs[-1]:
            continue
        if solid and solid[-1][0] == r[0]:
            continue
        solid.append(r)
    changes = len(solid) - 1
    half = lead[f_end // 2]
    led = sum(1 for f in range(f0, f_end + 1) if lead[f] == w) / (f_end + 1 - f0)
    was_risk = any(risk[f] == w for f in range(int(3 * FPS), int(T[2] * FPS)))
    wp = [(p[2] / p[4], p[0], p[2]) for p in passes if p[1] == w and p[0] >= 0]
    w_ratio, w_t, w_gap = min(wp) if wp else (9.0, 0.0, 999.0)
    duel_p = [q for q in wp if T[2] < q[1] < T[3]]
    close_n = sum(1 for p in passes if p[0] >= 0 and p[2] / p[4] < 0.12)
    duel = T[3] - T[2]
    fav_pre_duel = lead[max(0, int(T[2] * FPS) - 3)]
    max_wait = max(T[j + 1] - T[j] for j in range(3))
    pips = drag_pips(sim["params"])
    by_drag = sorted(range(NP), key=lambda i: -pips[i])                                 # most drag first
    drag_order = order + [w] == by_drag
    score = 0.0
    score += 3.0 if 4.0 <= T[0] <= 7.0 else -3.0                                        # an early first meal
    score += 4.0 * (1 - clamp((duel - 3.0) / 3.0)) * (0.6 if duel < 0.8 else 1.0)       # short final duel
    score += 4.5 * clamp((0.30 - w_ratio) / 0.22)                                       # winner's close call
    score += 1.5 if (w_ratio < 0.20 and w_t >= T[2] - 5.0) else 0.0
    if duel_p:                                                                          # winner dived mid-duel
        score += 1.5 + 1.5 * clamp((0.30 - min(duel_p)[0]) / 0.25)
    score += 0.8 * min(changes, 6)                                                      # favourite changes
    score += 2.5 if half != w else 0.0
    score += 1.5 if lead[0] != w else 0.0
    score += 2.0 if was_risk else 0.0                                                   # winner once looked doomed
    score += 1.5 if fav_pre_duel == ru else 0.0
    score -= 4.0 * max(0.0, led - 0.5)                                                  # no obvious favourite
    score -= 1.5 * max(0.0, max_wait - 5.5)                                             # no long lulls
    score -= 2.5 if drag_order else 0.0                                                 # drag alone must not decide
    score += 2.0 if pips[w] > 1 else 0.0
    score += 0.4 * min(close_n, 6)
    return dict(
        seed=sim["seed"], score=round(score, 3), valid=True, winner=NAMES[w], winner_index=w,
        swallowed_in_order=[NAMES[i] for i in order], meal_times_s=[round(t, 2) for t in T],
        first_meal_s=round(T[0], 2), final_duel_s=round(duel, 2), last_meal_s=round(T[3], 2),
        survivor_completes_full_orbit_after_last_meal_s=round(sim["t_orbit"] - T[3], 2),
        survivor_dives_after_last_meal=sum(1 for q in wp if q[1] > T[3]),
        winner_closest_pass_gap=round(w_gap, 2), winner_closest_pass_percent_of_horizon=round(100 * w_ratio, 1),
        winner_closest_pass_at_s=round(w_t, 2), winner_dives_during_final_duel=len(duel_p),
        closest_survived_pass_gap_any_planet=round(min(p[2] for p in passes if p[0] >= 0), 2),
        favourite_changes=changes, favourite_at_start=NAMES[lead[0]], favourite_at_half=NAMES[half],
        winner_was_favourite_at_half=half == w,
        favourite_before_final_duel=NAMES[fav_pre_duel], winner_share_of_time_as_favourite=round(led, 2),
        winner_was_most_at_risk_at_some_point=was_risk, close_passes_survived_under_12_percent=close_n,
        longest_wait_between_meals_s=round(max_wait, 2), dives_survived=sum(1 for p in passes if p[0] >= 0),
        drag_per_planet=dict(zip(NAMES, (round(p["k"], 4) for p in sim["params"]))),
        drag_pips_shown=dict(zip(NAMES, pips)), swallow_order_equals_drag_order=drag_order,
        least_drag_planet_wins=pips[w] == 1,
    )


def score_seed(seed):
    return drama(simulate(seed))


# ====================================================================== look
COLS = [hex_rgb(c) for c in ("#FFD83D", "#3DB8FF", "#2EE6A6", "#FF5A3C", "#F47BE6")]
INK = hex_rgb("#070914")
WHITE = (1.0, 1.0, 1.0)
AMBER = hex_rgb("#FFC078")
WARN = hex_rgb("#FF7A5C")
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
CXT = 490.0                  # text centre line
ZFIT, ZMAX = 425.0, 1.8      # the camera pushes in as the surviving orbits shrink
TAIL_N = 150                 # bright tail length in samples (1.25 s)
HIST_N = 840                 # faint history length in samples (7 s)
GAPMAX = 120.0               # gap that fills a standings bar
STAR_W = 0.02                # background drift, rad/s
QA, QB = "WHICH PLANET", "SURVIVES?"
RULE, RESULT, ASK = "LAST ONE LEFT WINS", "IS THE LAST ONE LEFT", "WHO DID YOU PICK?"


def arc_text(ctx, font, text, cx, cy, r, mid, size, fill, alpha=1.0, stroke=None, stroke_w=0.0, tracking=0.0,
             bottom=False):
    """Text set on a circle of radius r round (cx, cy), centred on screen angle `mid`. A top arc has its
    baseline on the circle with the letters pointing outward; a bottom arc (bottom=True) reads the other way
    round with the letter tops pointing at the centre, so both stay upright. Outlines first, fills on top."""
    if alpha <= 0.003:
        return
    widths = [font.width(ch, size) for ch in text]
    tot = sum(widths) + tracking * (len(text) - 1)
    for layer in ((0, 1) if stroke is not None and stroke_w else (1,)):
        pos = -tot / 2
        for ch, w in zip(text, widths):
            if ch != " ":
                a = (pos + w / 2) / r
                th = mid - a if bottom else mid + a
                ctx.save()
                ctx.translate(cx + r * math.cos(th), cy + r * math.sin(th))
                ctx.rotate(th - TAU / 4 if bottom else th + TAU / 4)
                if layer == 0:
                    font.draw(ctx, ch, 0, 0, size, stroke, anchor="c", alpha=alpha, stroke=stroke, stroke_w=stroke_w)
                else:
                    font.draw(ctx, ch, 0, 0, size, fill, anchor="c", alpha=alpha)
                ctx.restore()
            pos += w + tracking


def fmt_gap(g):
    g = max(0.0, g)
    return f"{g:.1f}" if g < 9.95 else f"{g:.0f}"


def symbol(ctx, k, x, y, s):
    """Contestant mark k centred on (x, y), half-size s: triangle, ring, plus, diamond, bars."""
    ctx.new_path()
    if k == 0:
        polygon(ctx, [(x + 1.2 * s * math.cos(a), y + 0.12 * s + 1.2 * s * math.sin(a))
                      for a in (-TAU / 4, TAU / 12, TAU * 5 / 12)])
        ctx.fill()
    elif k == 1:
        ctx.set_line_width(0.46 * s)
        ctx.arc(x, y, 0.80 * s, 0, TAU)
        ctx.stroke()
    elif k == 2:
        ctx.rectangle(x - s, y - 0.28 * s, 2 * s, 0.56 * s)
        ctx.rectangle(x - 0.28 * s, y - s, 0.56 * s, 2 * s)
        ctx.fill()
    elif k == 3:
        polygon(ctx, [(x, y - 1.15 * s), (x + 0.95 * s, y), (x, y + 1.15 * s), (x - 0.95 * s, y)])
        ctx.fill()
    else:
        ctx.rectangle(x - 0.98 * s, y - 0.78 * s, 1.96 * s, 0.56 * s)
        ctx.rectangle(x - 0.98 * s, y + 0.22 * s, 1.96 * s, 0.56 * s)
        ctx.fill()


def draw_planet(ctx, i, x, y, r, lx, ly, glow_a=0.40):
    """Planet i as a lit sphere; (lx, ly) is the unit direction the light comes from."""
    col = COLS[i]
    if glow_a > 0.004:
        ctx.set_operator(cairo.OPERATOR_ADD)
        glow(ctx, x, y, r * 3.4, col, glow_a)
        ctx.set_operator(cairo.OPERATOR_OVER)
    g = cairo.RadialGradient(x + lx * r * 0.45, y + ly * r * 0.45, r * 0.05, x, y, r)
    g.add_color_stop_rgb(0, *lighten(col, 0.55))
    g.add_color_stop_rgb(0.55, *col)
    g.add_color_stop_rgb(1.0, *darken(col, 0.48))
    ctx.new_path()
    ctx.arc(x, y, r, 0, TAU)
    ctx.set_source(g)
    ctx.fill()
    ctx.set_source_rgba(INK[0], INK[1], INK[2], 0.90)
    symbol(ctx, i, x, y, r * 0.50)
    ctx.new_path()
    ctx.arc(x, y, r - 0.6, 0, TAU)
    ctx.set_source_rgba(1, 1, 1, 0.30)
    ctx.set_line_width(max(1.0, r * 0.06))
    ctx.stroke()


def make_background():
    """Deep space: near-black gradient, soft nebula pools, a darker well around the hole, dust stars, grain."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = (yy / H)[..., None]
    img = np.array([0.022, 0.030, 0.070], np.float32) * (1 - v) + np.array([0.014, 0.014, 0.044], np.float32) * v
    rng = np.random.default_rng(11)
    pal = ((0.00, 0.105, 0.135), (0.075, 0.035, 0.170), (0.125, 0.020, 0.095), (0.020, 0.055, 0.150))
    for n in range(26):
        cx, cy = rng.uniform(-120, W + 120), rng.uniform(-120, H + 120)
        rad, amp = rng.uniform(200, 620), rng.uniform(0.35, 1.0)
        d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (rad * rad)
        img += (np.exp(-2.2 * d2) * amp)[..., None] * np.array(pal[n % 4], np.float32)
    d = np.hypot(xx - CX, yy - CY)
    img *= (0.50 + 0.50 * np.clip(d / 560, 0, 1) ** 1.2)[..., None]
    img *= np.clip(1.10 - 0.42 * (((xx - 540) / 760) ** 2 + ((yy - 900) / 1250) ** 2), 0.35, 1.0)[..., None]
    for _ in range(1100):                                             # static dust stars
        px, py = int(rng.integers(1, W - 1)), int(rng.integers(1, H - 1))
        b = float(rng.uniform(0.05, 0.30)) ** 1.3
        img[py, px] += b
        img[py, px + 1] += b * 0.35
        img[py + 1, px] += b * 0.35
    img += rng.normal(0, 0.006, (H, W, 1)).astype(np.float32)          # grain hides banding
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    return cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4), buf


# ====================================================================== the scene
class Scene:
    W, H, FPS, nframes = W, H, FPS, 0

    def __init__(self, seed):
        self.seed = seed
        self.sim = simulate(seed, record=True)
        if not self.sim["valid"]:
            raise ValueError(f"seed {seed} is not a valid run: {self.sim['reason']}")
        self.stats = drama(self.sim)
        self.w = self.sim["winner"]
        self.dirn = self.sim["dirn"]
        self._res = None
        self._prep()

    # ------------------------------------------------------------ precomputed animation state
    def _prep(self):
        sim = self.sim
        meals = sim["meals"]
        self.T = [m[0] for m in meals]
        self.T_END = self.T[3]
        self.DUR = math.ceil((self.T_END + ending(self.T_END, sim["t_orbit"])) * FPS - 1e-6) / FPS
        self.NF = self.nframes = int(round(self.DUR * FPS))
        self.T_RESET = self.DUR - RESET_LEN
        self.T_REVEAL = self.T_END + REVEAL_AFTER
        self.T_LINE = self.T_REVEAL + LINE_AFTER
        NF = self.NF
        self.death = [1e9] * NP
        for m in meals:
            self.death[m[1]] = m[0]
        NS = len(sim["rh"])
        self.nlive = [len(a) for a in sim["xs"]]
        self.pos = np.zeros((NP, NS, 2))
        for i in range(NP):
            n = self.nlive[i]
            self.pos[i, :n, 0] = sim["xs"][i]
            self.pos[i, :n, 1] = sim["ys"][i]
            self.pos[i, n:] = self.pos[i, n - 1]
        self.rh = np.array(sim["rh"])
        self.ms = np.array(sim["ms"])
        # video frame -> sample index: real time from the first frame to the last (nothing is slowed down)
        self.sf = np.minimum(S0 + np.arange(NF) * float(SPS), NS - 1.001)
        k = [self.kreset(f / FPS) for f in range(NF)]
        rh_f = np.interp(self.sf, np.arange(NS), self.rh)
        self.rhv = [lerp(float(rh_f[f]), R0, k[f]) for f in range(NF)]
        # camera
        fra = sim["fra"]
        a = 1 - math.exp(-1 / (FPS * 0.45))
        z1 = z2 = None
        self.zf = []
        for f in range(NF):
            t = f / FPS
            row = fra[min(len(fra) - 1, f)]
            far = max([row[i] for i in range(NP) if self.death[i] > t] or [ZFIT])
            zt = clamp((ZFIT if t < self.T_END + 0.25 else 385.0) / far, 1.0, ZMAX)   # the ending stays close
            if z1 is None:
                z1 = z2 = zt
            z1 += (zt - z1) * a
            z2 += (z1 - z2) * a
            z = lerp(z2, 0.86 + 0.02 * t, self.hook(t))
            self.zf.append(lerp(z, 0.86 + 0.02 * (t - self.DUR), k[f]))
        # standings: alive planets by the gap of their next dive (safest first), then the swallowed, latest
        # first. The order is frozen for the first 2 s and rows swap only when the difference clears a margin.
        self.gn = np.array(sim["gapn"])
        self.pips = drag_pips(sim["params"])
        self.tw = [0.60 + 0.16 * (p - 1) for p in self.pips]       # tail thickness follows drag
        self.order, self.nalive, self.rowy = [], [], []
        cur = None
        al = sorted(range(NP), key=lambda i: (-self.gn[0][i], i))
        for f in range(NF):
            t = f / FPS
            row = self.gn[min(f, len(self.gn) - 1)]
            al = [i for i in al if self.death[i] > t]
            if t >= 2.0:
                for j in range(len(al) - 1):
                    hi, lo = al[j], al[j + 1]
                    if row[lo] > row[hi] + max(5.0, 0.12 * row[hi]):
                        al[j], al[j + 1] = lo, hi
            dead = sorted((i for i in range(NP) if self.death[i] <= t), key=lambda i: -self.death[i])
            order = al + dead
            tgt = [float(order.index(i)) for i in range(NP)]
            cur = tgt[:] if cur is None else [c + (g - c) * 0.30 for c, g in zip(cur, tgt)]
            self.order.append(order)
            self.nalive.append(len(al))
            self.rowy.append(cur[:])
        self.calls = [p for p in sim["passes"] if p[0] >= 0.05 and p[2] <= 0.85 * p[4]]
        rg = np.random.default_rng(31)
        self.streaks = []                                           # accretion swirl; whole turns per loop
        for _ in range(62):
            rho = 1.14 + 2.7 * rg.random() ** 1.5
            nrev = max(1, int(round(self.DUR * 1.9 * rho ** -1.5 / TAU)))
            fall = 1 - (rho - 1.14) / 2.8
            self.streaks.append((rho, rg.uniform(0, TAU), nrev, rg.uniform(0.35, 1.5), rg.uniform(0.03, 0.10),
                                 rg.uniform(0.07, 0.22) * (0.30 + 0.70 * fall),
                                 mix((1.0, 0.86, 0.66), (0.95, 0.42, 0.16), clamp((rho - 1.1) / 2.0))))
        self.motes = [(int(rg.integers(3, 8)), rg.random(), rg.uniform(0, TAU), rg.uniform(0.6, 1.5),
                       rg.uniform(1.4, 2.8), rg.uniform(0.35, 0.9)) for _ in range(36)]
        ns = 230
        u = rg.random(ns)
        self.stars = dict(d=np.sqrt(rg.uniform(90.0 ** 2, 1350.0 ** 2, ns)), a=rg.uniform(0, TAU, ns),
                          size=0.7 + 1.9 * u ** 3, br=0.30 + 0.70 * rg.random(ns), n=rg.integers(1, 10, ns),
                          p=rg.uniform(0, TAU, ns),
                          col=[((1, 1, 1), (0.80, 0.88, 1.0), (1.0, 0.90, 0.78))[j] for j in rg.integers(0, 3, ns)])
        self.mfx = []                                               # per meal: contact angle, spin sign, sparks
        for m in meals:
            rc2 = m[2] * m[2] + m[3] * m[3]
            om = (m[2] * m[5] - m[3] * m[4]) / rc2
            sparks = [(rg.uniform(3.0, 10.0), rg.uniform(1.03, 1.32), rg.uniform(0.5, 1.0), rg.uniform(-0.3, 0.3))
                      for _ in range(16)]
            self.mfx.append((math.atan2(m[3], m[2]), math.sqrt(rc2), om, sparks))

    def kreset(self, t):
        """0 during the run, 1 on the last frames: blends everything back to the opening state."""
        return ease_in_out(ramp(t, self.T_RESET, self.DUR - 2.0 / FPS))

    @staticmethod
    def hook(t):
        """1 while the opening layout is up (big question, wide shot), 0 in the main layout."""
        return 1 - ease_in_out(ramp(t, 2.0, 2.8))

    def P(self, i, sfr):
        """Position of planet i at fractional sample index sfr (clamped to its last sample alive)."""
        k = int(sfr)
        n = self.nlive[i]
        if k >= n - 1:
            return self.pos[i, n - 1]
        u = sfr - k
        return self.pos[i, k] * (1 - u) + self.pos[i, k + 1] * u

    def shake(self, t):
        sx = sy = 0.0
        for tm in self.T:
            d = t - tm
            if 0 <= d < 0.6:
                a = 9.0 * math.exp(-d / 0.10)
                sx += a * math.sin(d * 67 + tm * 13)
                sy += a * math.cos(d * 53 + tm * 7)
        return sx, sy

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            bg, keep = make_background()
            font = Font(FONT_PATH)
            wa, wb = font.width(QA, 100), font.width(QB, 100)
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep, font=font,
                             s1=min(136.0, 100 * 820 / max(wa, wb)), s2=min(80.0, 100 * 850 / (wa + wb + 26)),
                             wa=wa / 100, wb=wb / 100)
        return self._res

    def frame(self, f):
        R = self._resources()
        self.draw(R["ctx"], f, R)
        R["surf"].flush()
        return bytes(R["surf"].get_data())

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, f, R):
        t = f / FPS
        font = R["font"]
        k = self.kreset(t)
        z, Rh, sfr = self.zf[f], self.rhv[f], float(self.sf[f])
        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(R["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_OVER)
        self._stars(ctx, t, z, Rh, 1 - k)
        if k > 0:
            self._stars(ctx, t - self.DUR, z, Rh, k)
        sx, sy = self.shake(t)
        ctx.save()
        ctx.translate(CX + sx, CY + sy)
        ctx.scale(z, z)
        self._hole_glow(ctx, t, Rh)
        if k <= 0:
            self._trails(ctx, t, sfr, z)
            self._dying(ctx, t, Rh)
            self._horizon(ctx, t, Rh)
            self._meal_fx(ctx, t, Rh)
            self._planets(ctx, t, sfr)
        else:                                   # loop hand-over: this run fades out, the next one fades in
            self._horizon(ctx, t, Rh)
            t2 = t - self.DUR
            for tt, ss, a in ((t, sfr, 1 - k), (t2, S0 + t2 * FPS * SPS, k)):
                if a <= 0.003:
                    continue
                ctx.push_group()
                self._trails(ctx, tt, ss, z)
                self._planets(ctx, tt, ss)
                ctx.pop_group_to_source()
                ctx.paint_with_alpha(a)
        ctx.restore()
        self._callouts(ctx, t, z, Rh, k, font)
        hk = self.hook(t)
        if k > 0.003:
            self._dragtags(ctx, S0 + (t - self.DUR) * FPS * SPS, z, k, font)
        elif hk > 0.003:
            self._dragtags(ctx, sfr, z, hk, font)
        self._titles(ctx, t, k, z, Rh, R)
        self._board(ctx, f, t, k, font)
        fl = 0.0
        for tm in self.T:
            if 0 <= t - tm < 0.3:
                fl += 0.11 * (1 - (t - tm) / 0.3) ** 2
        if fl > 0.004:
            ctx.set_operator(cairo.OPERATOR_ADD)
            ctx.set_source_rgba(1.0, 0.90, 0.80, fl)
            ctx.paint()
            ctx.set_operator(cairo.OPERATOR_OVER)

    def _stars(self, ctx, t, z, Rh, alpha):
        if alpha <= 0.003:
            return
        S = self.stars
        d = S["d"] * (1 + (z - 1) * 0.10)                           # slight parallax under the zoom
        d = d + (1.5 * Rh * z) ** 2 / d                             # stars bend away from the hole
        ang = S["a"] + STAR_W * self.dirn * t
        x, y = CX + d * np.cos(ang), CY + d * np.sin(ang)
        al = S["br"] * (0.70 + 0.30 * np.sin(TAU * S["n"] * t / self.DUR + S["p"])) * alpha
        size, cols = S["size"], S["col"]
        for j in np.nonzero((x > -6) & (x < W + 6) & (y > -6) & (y < H + 6))[0]:
            c = cols[j]
            if size[j] > 1.9:
                ctx.set_operator(cairo.OPERATOR_ADD)
                glow(ctx, x[j], y[j], size[j] * 6, c, al[j] * 0.45)
                ctx.set_operator(cairo.OPERATOR_OVER)
            ctx.set_source_rgba(c[0], c[1], c[2], al[j])
            ctx.arc(x[j], y[j], size[j], 0, TAU)
            ctx.fill()

    def _hole_glow(self, ctx, t, Rh):
        ctx.set_operator(cairo.OPERATOR_ADD)
        br = 1 + 0.07 * math.sin(TAU * 3 * t / self.DUR)
        glow(ctx, 0, 0, Rh * 8.5, (0.55, 0.22, 0.10), 0.24 * br)
        glow(ctx, -0.3 * Rh * self.dirn, 0.15 * Rh, Rh * 3.8, (1.0, 0.62, 0.30), 0.30 * br)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        dn = self.dirn
        for rho, th0, nrev, ln, w, a, c in self.streaks:
            th = th0 + dn * TAU * nrev * t / self.DUR
            ctx.set_line_width(w * Rh)
            for part, aa in ((1.0, 0.55), (0.4, 0.75)):             # brighter leading end
                ctx.set_source_rgba(c[0], c[1], c[2], a * aa)
                ctx.new_path()
                if dn > 0:
                    ctx.arc(0, 0, rho * Rh, th - ln * part, th)
                else:
                    ctx.arc(0, 0, rho * Rh, th, th + ln * part)
                ctx.stroke()
        for n, ph, a0, turns, size, bright in self.motes:           # motes spiralling in
            u = (t * n / self.DUR + ph) % 1.0
            r = Rh * (1.0 + 5.4 * (1 - u) ** 1.7)
            ang = a0 + dn * TAU * turns * u * u
            ln = 0.04 + 0.30 * u
            ctx.set_source_rgba(1.0, 0.86, 0.62, bright * min(1.0, u * 6) * min(1.0, (1 - u) * 9))
            ctx.set_line_width(size)
            ctx.new_path()
            if dn > 0:
                ctx.arc(0, 0, r, ang - ln, ang)
            else:
                ctx.arc(0, 0, r, ang, ang + ln)
            ctx.stroke()
        ctx.set_operator(cairo.OPERATOR_OVER)

    def _horizon(self, ctx, t, Rh):
        boost = 1.0
        for tm in self.T:
            if t >= tm:
                boost += 0.8 * math.exp(-(t - tm) / 0.25)
        ctx.set_operator(cairo.OPERATOR_ADD)
        g = cairo.RadialGradient(0, 0, Rh * 0.96, 0, 0, Rh * 2.1)
        g.add_color_stop_rgba(0.0, 1.0, 0.95, 0.85, 0.0)
        g.add_color_stop_rgba(0.04, 1.0, 0.93, 0.80, min(1.0, 0.80 * boost))
        g.add_color_stop_rgba(0.16, 1.0, 0.70, 0.35, min(1.0, 0.40 * boost))
        g.add_color_stop_rgba(0.45, 0.9, 0.40, 0.15, 0.12)
        g.add_color_stop_rgba(1.0, 0.9, 0.40, 0.15, 0.0)
        ctx.set_source(g)
        ctx.new_path()
        ctx.arc(0, 0, Rh * 2.1, 0, TAU)
        ctx.fill()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.new_path()
        ctx.arc(0, 0, Rh, 0, TAU)                                   # the horizon itself: exactly the kill radius
        ctx.set_source_rgb(0, 0, 0)
        ctx.fill()
        rw = max(1.8, 0.04 * Rh)
        ctx.arc(0, 0, Rh - rw * 0.5, 0, TAU)                        # bright rim: its outer edge is the kill radius
        ctx.set_source_rgba(1.0, 0.97, 0.90, 0.95)
        ctx.set_line_width(rw)
        ctx.stroke()

    def _trails(self, ctx, t, sfr, z):
        ks = int(sfr)
        fin = ease_in_out(ramp(t, self.T_END + 0.3, self.T_END + 1.4))
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_operator(cairo.OPERATOR_ADD)
        for i in range(NP):
            gone = t - self.death[i]
            if gone > 0.8:
                continue
            fade = 1.0 if gone <= 0 else 1 - gone / 0.8
            end = min(ks, self.nlive[i] - 1)
            col = COLS[i]
            pts = self.pos[i]
            head = self.P(i, sfr)
            ctx.set_line_cap(cairo.LINE_CAP_BUTT)
            if i == self.w and fin > 0:                             # the survivor's path since the last meal, lit
                i0 = min(end, int(S0 + self.T_END * FPS * SPS))
                for wd, al in ((9.0, 0.10), (2.6, 0.55)):
                    ctx.set_line_width(wd / z)
                    ctx.set_source_rgba(col[0], col[1], col[2], al * fin)
                    ctx.new_path()
                    for p in pts[i0:end + 1:2]:
                        ctx.line_to(p[0], p[1])
                    ctx.line_to(head[0], head[1])
                    ctx.stroke()
            h0, h1 = max(0, end - HIST_N), end - TAIL_N // 2        # faint history: the laps show the shrink
            if h1 - h0 > 12:
                ctx.set_line_width(1.8 / z)
                G = 6
                for g in range(G):
                    a0 = h0 + (h1 - h0) * g // G
                    a1 = h0 + (h1 - h0) * (g + 1) // G
                    al = 0.24 * ((g + 1) / G) ** 1.3 * fade * clamp((a0 + a1) / 2 / 150.0)
                    ctx.set_source_rgba(col[0], col[1], col[2], al)
                    ctx.new_path()
                    for p in pts[a0:a1 + 2:2]:
                        ctx.line_to(p[0], p[1])
                    ctx.line_to(pts[min(a1 + 1, end)][0], pts[min(a1 + 1, end)][1])
                    ctx.stroke()
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            for j in range(8):                                      # comet tail: nested strokes, fat at the head
                u = j / 7
                a0 = max(0, end - int(TAIL_N * (1 - 0.86 * u ** 0.8)))
                ctx.set_line_width(lerp(2.2, PR * 1.25 * self.tw[i], u ** 1.4))
                ctx.set_source_rgba(col[0], col[1], col[2], 0.080 * fade)
                ctx.new_path()
                for p in pts[a0:end + 1]:
                    ctx.line_to(p[0], p[1])
                ctx.line_to(head[0], head[1])
                ctx.stroke()
        ctx.set_operator(cairo.OPERATOR_OVER)

    def _planets(self, ctx, t, sfr):
        for i in range(NP):
            if t >= self.death[i]:
                continue
            x, y = self.P(i, sfr)
            r = math.hypot(x, y)
            draw_planet(ctx, i, x, y, PR, -x / r, -y / r)

    def _dying(self, ctx, t, Rh):
        for j, m in enumerate(self.sim["meals"]):
            d = t - m[0]
            if not 0 <= d < 0.45:
                continue
            ang0, rc, om, _ = self.mfx[j]
            u = d / 0.45
            r = lerp(rc, Rh * 0.3, u ** 1.5)
            ang = ang0 + om * d * (1 + 1.5 * u)
            draw_planet(ctx, m[1], r * math.cos(ang), r * math.sin(ang), PR * (1 - 0.5 * u),
                        -math.cos(ang), -math.sin(ang), 0.40 * (1 - u))

    def _meal_fx(self, ctx, t, Rh):
        ctx.set_operator(cairo.OPERATOR_ADD)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for j, m in enumerate(self.sim["meals"]):
            d = t - m[0]
            if not 0 <= d < 1.1:
                continue
            col = COLS[m[1]]
            ang0, _, om, sparks = self.mfx[j]
            sg = 1.0 if om >= 0 else -1.0
            u = d / 0.9
            if u < 1:                                               # shock ring
                c = mix(col, WHITE, 0.4)
                ctx.set_source_rgba(c[0], c[1], c[2], 0.55 * (1 - u) ** 2)
                ctx.set_line_width(1.5 + 6 * (1 - u))
                ctx.new_path()
                ctx.arc(0, 0, Rh * (1.05 + 2.8 * ease_out(u)), 0, TAU)
                ctx.stroke()
            v = d / 0.8
            if v < 1:                                               # the meal's colour smears round the ring
                ctx.set_source_rgba(col[0], col[1], col[2], 0.85 * (1 - v) ** 1.5)
                ctx.set_line_width(max(2.5, Rh * 0.07))
                ctx.new_path()
                ctx.arc(0, 0, Rh * 1.035, 0, TAU)
                ctx.stroke()
            lc = lighten(col, 0.3)
            for dw, rho0, life, off in sparks:
                q = d / life
                if q >= 1:
                    continue
                ang = ang0 + off + sg * dw * d
                r = Rh * lerp(rho0, 1.01, ease_out(q))
                ln = 0.10 + 0.5 * q
                ctx.set_source_rgba(lc[0], lc[1], lc[2], 0.8 * (1 - q) ** 1.4)
                ctx.set_line_width(3.2 * (1 - 0.5 * q))
                ctx.new_path()
                if sg > 0:
                    ctx.arc(0, 0, r, ang - ln, ang)
                else:
                    ctx.arc(0, 0, r, ang, ang + ln)
                ctx.stroke()
        ctx.set_operator(cairo.OPERATOR_OVER)

    def _callouts(self, ctx, t, z, Rh, k, font):
        """Dive read-outs: a gauge between horizon and planet at closest approach, and the gap as a number."""
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)
        pop = 0.0                                                   # HORIZON +15% is up: gap labels wait
        for tm in self.T:
            if 0.0 <= t - tm < 1.5:
                pop = max(pop, 1 - ramp(t - tm, 0.95, 1.5))
        for tp, i, gap, ang, rhp in self.calls:
            d = t - tp
            if not 0 <= d < 1.2:
                continue
            u = d / 1.2
            a = min(1.0, u * 12) * (1 - ramp(u, 0.65, 1.0)) * (1 - k)
            col = COLS[i]
            lc = lighten(col, 0.35)
            c, s = math.cos(ang), math.sin(ang)
            close = gap < 0.15 * rhp
            r0, r1 = max(rhp, Rh), rhp + gap
            x0, y0, x1, y1 = CX + z * r0 * c, CY + z * r0 * s, CX + z * r1 * c, CY + z * r1 * s
            ctx.set_source_rgba(lc[0], lc[1], lc[2], a)
            ctx.set_line_width(3.0)
            ctx.new_path()
            if r1 > r0:
                ctx.move_to(x0, y0)
                ctx.line_to(x1, y1)
            for px, py in ((x0, y0), (x1, y1)):
                ctx.move_to(px - s * 9, py + c * 9)
                ctx.line_to(px + s * 9, py - c * 9)
            ctx.stroke()
            rl = z * (r1 + 2 * PR) + 46
            lx, ly = clamp(CX + rl * c, 100, 850), clamp(CY + rl * s + 14, 300, 1195)
            size = (54 if close else 40) * (0.6 + 0.4 * ease_out_back(ramp(u, 0, 0.16)))
            font.draw(ctx, "GAP " + fmt_gap(gap), lx, ly, size, WHITE if close else lc, anchor="c", alpha=a * (1 - pop),
                      stroke=darken(col, 0.55) if close else INK, stroke_w=size * 0.18)
        for tm in self.T:
            d = t - tm
            if 0.1 <= d < 1.5:
                u = (d - 0.1) / 1.4
                a = min(1.0, u * 10) * (1 - ramp(u, 0.6, 1.0))
                font.draw(ctx, "HORIZON +15%", CX, CY - Rh * z - 46 - 34 * ease_out(u), 52, WHITE, anchor="c",
                          alpha=a, stroke=INK, stroke_w=9)

    def _question(self, ctx, R, h, alpha):
        """The question rides two arcs round the hole in the hook (h = 1), then settles into one shallow arc
        above the orbits (h = 0). The rule sits on the lower arc while the hook is up."""
        if alpha <= 0.003:
            return
        font = R["font"]
        top = -TAU / 4
        hm, hb = h * h, h ** 0.45                    # the words part sideways first, then the second one rises
        rH, sH = 560.0, 56.0
        tot = (R["wa"] + R["wb"]) * sH + 24
        ma = top - (1 - hm) * (tot / 2 - R["wa"] * sH / 2) / rH
        mb = top + (1 - hm) * (tot / 2 - R["wb"] * sH / 2) / rH
        sa, sb = lerp(sH, 92.0, h), lerp(sH, 116.0, hb)
        arc_text(ctx, font, QA, CX, CY, lerp(rH, 530.0, h), ma, sa, WHITE, alpha, INK, sa * 0.10)
        arc_text(ctx, font, QB, CX, CY, lerp(rH, 410.0, hb), mb, sb, AMBER, alpha, INK, sb * 0.10)
        arc_text(ctx, font, RULE, CX, CY, 446.0, TAU / 4, 58, WHITE, alpha * ramp(h, 0.35, 1.0), INK, 7, 1.0,
                 bottom=True)

    def _titles(self, ctx, t, k, z, Rh, R):
        font = R["font"]
        if t < self.T_REVEAL:
            self._question(ctx, R, self.hook(t), 1 - ramp(t, self.T_REVEAL - 0.35, self.T_REVEAL - 0.05))
            return
        if k > 0:
            self._question(ctx, R, 1.0, ramp(k, 0.4, 1.0))
        if k >= 1:
            return
        ctx.push_group()
        col = COLS[self.w]
        top = -TAU / 4
        u = ramp(t, self.T_REVEAL, self.T_REVEAL + 0.5)
        swing = 0.60 * (1 - ease_out(u))                            # the result slides round the ring into place
        name, s1, r1, ir = NAMES[self.w], 100.0, 522.0, 36.0
        wn = sum(font.width(ch, s1) for ch in name)
        m1 = top - swing + (ir + 10) / r1
        arc_text(ctx, font, name, CX, CY, r1, m1, s1, lighten(col, 0.25), 1.0, INK, s1 * 0.10)
        ai, rc = m1 - (wn / 2 + ir + 18) / r1, r1 + s1 * font.cap_height / 2
        draw_planet(ctx, self.w, CX + rc * math.cos(ai), CY + rc * math.sin(ai), ir, -0.6, -0.6, 0.35)
        arc_text(ctx, font, RESULT, CX, CY, 440.0, top + swing, 60, WHITE, 1.0, INK, 7, 1.0)
        u2 = ramp(t, self.T_LINE, self.T_LINE + 0.35)
        if u2 > 0:
            arc_text(ctx, font, ASK, CX, CY, 446.0, TAU / 4 + 0.45 * (1 - ease_out(u2)), 58, AMBER,
                     min(1.0, u2 * 4), INK, 7, 1.0, bottom=True)
        u3 = ramp(t, self.T_REVEAL + 0.45, self.T_REVEAL + 0.8)
        if u3 > 0:
            txt = "CLOSEST GAP: " + fmt_gap(float(self.ms[-1, self.w]))
            tw3, y3 = font.width(txt, 42), CY + Rh * z + 78
            ctx.new_path()
            rounded_rect(ctx, CX - tw3 / 2 - 20, y3 - 42, tw3 + 40, 56, 16)
            ctx.set_source_rgba(INK[0], INK[1], INK[2], 0.78 * u3)
            ctx.fill()
            font.draw(ctx, txt, CX, y3, 42, lighten(col, 0.3), anchor="c", alpha=u3)
        ctx.pop_group_to_source()
        ctx.paint_with_alpha(min(1.0, u * 5) * (1 - ramp(k, 0.0, 0.45)))

    def _pips(self, ctx, i, x0, cy, w, h, pitch, a, dim=0.0):
        """Drag of planet i as 1 to 5 filled pips."""
        col = COLS[i]
        for p in range(NP):
            ctx.new_path()
            rounded_rect(ctx, x0 + p * pitch, cy - h / 2, w, h, 3)
            if p < self.pips[i]:
                ctx.set_source_rgba(col[0], col[1], col[2], a * (1 - 0.65 * dim))
            else:
                ctx.set_source_rgba(1, 1, 1, 0.14 * a)
            ctx.fill()

    def _dragtags(self, ctx, sfr, z, a, font):
        """Hook only: every planet carries its drag as pips (the same pips stay in the standings)."""
        for i in range(NP):
            x, y = self.P(i, sfr)
            px, py = CX + z * x, CY + z * y + PR * z + 30
            ctx.new_path()
            rounded_rect(ctx, px - 84, py - 20, 168, 40, 12)
            ctx.set_source_rgba(INK[0], INK[1], INK[2], 0.82 * a)
            ctx.fill()
            font.draw(ctx, "DRAG", px - 74, py + 11, 28, WHITE, alpha=0.95 * a)
            self._pips(ctx, i, px - 8, py, 11, 24, 16, a)

    def _board(self, ctx, f, t, k, font):
        """Live standings: the gap each planet will have on its next dive (from its current orbit), safest on
        top, swallowed planets at the bottom, and every planet's drag as pips."""
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.new_path()
        rounded_rect(ctx, 38, 1220, 914, 322, 24)
        ctx.set_source_rgba(0.012, 0.016, 0.045, 0.66)
        ctx.fill()
        rounded_rect(ctx, 38.5, 1220.5, 913, 321, 24)
        ctx.set_source_rgba(1, 1, 1, 0.07)
        ctx.set_line_width(1.0)
        ctx.stroke()
        tp = t - self.DUR if k > 0 else t
        pulse = max(self.hook(t), k) * (0.5 - 0.5 * math.cos(TAU * 1.5 * tp))
        font.draw(ctx, "DRAG", 214, 1263 + 2 * pulse, 36 + 6 * pulse, AMBER, anchor="c", alpha=0.95)
        font.draw(ctx, "GAP ON NEXT DIVE", 284, 1263, 36, WHITE, alpha=0.85, tracking=1.0)
        font.draw(ctx, "HORIZON +15% PER MEAL", 928, 1263, 36, AMBER, anchor="r", alpha=0.90, tracking=0.5)
        late = k > 0.5
        order = self.order[0] if late else self.order[f]
        nal = NP if late else self.nalive[f]
        fi = min(f, len(self.gn) - 1)
        hl = ramp(t, self.T_REVEAL, self.T_REVEAL + 0.3) * (1 - k)
        Rh = self.rhv[f]
        for i in reversed(order):
            cy = 1296 + 54 * lerp(self.rowy[f][i], self.rowy[0][i], k)
            dl = ramp(t, self.death[i], self.death[i] + 0.4) * (1 - k)
            g = lerp(float(self.gn[fi, i]), float(self.gn[0, i]), k)
            col = COLS[i]
            rank = order.index(i)
            win = hl if i == self.w else 0.0
            ctx.new_path()
            rounded_rect(ctx, 54, cy - 24, 882, 48, 12)
            ctx.set_source_rgba(1, 1, 1, 0.055 * (1 - 0.5 * dl) + 0.06 * win)
            ctx.fill()
            if win > 0:
                rounded_rect(ctx, 54, cy - 24, 882, 48, 12)
                ctx.set_source_rgba(col[0], col[1], col[2], 0.9 * win)
                ctx.set_line_width(2.5)
                ctx.stroke()
            font.draw(ctx, str(rank + 1), 84, cy + 15, 40, WHITE, anchor="c", alpha=0.70 - 0.35 * dl)
            draw_planet(ctx, i, 136, cy, 21, -0.6, -0.6, 0.22 * (1 - dl))
            if dl > 0:
                ctx.new_path()
                ctx.arc(136, cy, 21.6, 0, TAU)
                ctx.set_source_rgba(0.02, 0.03, 0.07, 0.62 * dl)
                ctx.fill()
            self._pips(ctx, i, 172, cy, 13, 28, 18, 1.0, dl)
            ctx.new_path()
            rounded_rect(ctx, 284, cy - 11, 470, 22, 11)
            ctx.set_source_rgba(1, 1, 1, 0.06 * (1 - dl))
            ctx.fill()
            if dl < 1:
                bw = max(12.0, 470 * min(1.0, g / GAPMAX))
                rounded_rect(ctx, 284, cy - 11, bw, 22, 11)
                ctx.set_source_rgba(col[0], col[1], col[2], 1 - dl)
                ctx.fill()
                rounded_rect(ctx, 286, cy - 9, max(4.0, bw - 4), 6, 3)
                ctx.set_source_rgba(1, 1, 1, 0.28 * (1 - dl))
                ctx.fill()
                danger = nal >= 2 and rank == nal - 1 and g < 0.25 * Rh
                font.draw(ctx, fmt_gap(g), 284 + bw + 14, cy + 17, 46, WARN if danger else WHITE, alpha=1 - dl)
            if dl > 0:
                font.draw(ctx, "SWALLOWED", 284, cy + 15, 38, mix(col, (0.6, 0.62, 0.7), 0.35), alpha=0.80 * dl,
                          tracking=1.5)
            if win > 0:
                font.draw(ctx, "LAST ONE LEFT", 924, cy + 15, 38, lighten(col, 0.2), anchor="r", alpha=win)

    # ------------------------------------------------------------ sound
    def audio(self):
        """Stereo float mix at au.SR: a quiet sub drone that drops a semitone per meal, a sine per planet whose
        pitch follows its speed, a tick on every dive (a ping on the close ones), and on every meal a percussive
        hit with a sub drop, a falling glissando and a hush; an open fifth to end."""
        sr = au.SR
        DUR, NF, T = self.DUR, self.NF, self.T
        n, tail = int(round(DUR * sr)), 3 * sr
        rng = np.random.default_rng(self.seed + 777)
        wet = np.zeros((n + tail, 2))            # tones and events (reverb)
        bed = np.zeros((n + tail, 2))            # pads and pulse (ducked after each meal)
        dry = np.zeros((n + tail, 2))
        tt = np.arange(n) / sr
        tf = np.arange(NF) / FPS
        loc = lambda dur: np.arange(int(dur * sr)) / sr
        nr = lambda a, b: np.clip((tt - a) / (b - a), 0, 1)
        duck = np.ones(n)                        # the short hush after every swallow
        for tm in T:
            duck *= 1 - 0.86 * nr(tm + 0.50, tm + 0.62) * (1 - nr(tm + 0.92, tm + 1.25))
        edge = np.minimum(1.0, np.minimum(tt, DUR - tt) / 0.05)       # continuous layers meet at zero on the loop
        live = duck * edge

        semi = np.zeros(n)                       # the hole: drone one semitone lower after each meal
        for tm in T:
            u = nr(tm + 0.70, tm + 1.15)
            semi += u * u * (3 - 2 * u)
        ph = TAU * np.cumsum(55.0 * 2 ** (-semi / 12)) / sr
        build = 0.45 + 0.55 * np.clip(tt / self.T_END, 0, 1)
        dr = (0.80 * np.sin(ph) + 0.60 * np.sin(2 * ph + 0.4) + 0.42 * np.sin(3 * ph + 1.1)
              + 0.22 * np.sin(4 * ph) + 0.10 * np.sin(6 * ph))
        d2 = 0.50 * np.sin(1.007 * ph + 2.0) + 0.34 * np.sin(2.014 * ph)
        trem = (1 + 0.18 * np.sin(TAU * 0.31 * tt)) * build * live
        dry[:n, 0] += 0.022 * (dr + d2) * trem
        dry[:n, 1] += 0.022 * (dr - 0.6 * d2) * trem

        base = [au.hz(m) for m in (57, 60, 62, 64, 67)]               # one pitch class per planet (A C D E G)
        prox_sum = np.zeros(n)
        for i in range(NP):
            p = np.array([self.P(i, float(self.sf[f])) for f in range(NF)])
            r = np.hypot(p[:, 0], p[:, 1])
            v = np.hypot(np.gradient(p[:, 0]), np.gradient(p[:, 1])) * FPS
            al = (tf < self.death[i]).astype(float)
            fr = np.interp(tt, tf, base[i] * (np.maximum(v, 20.0) / 420.0) ** 0.75)
            prox = np.interp(tt, tf, np.minimum(1.0, 140.0 / r) ** 2 * al)
            gate = np.interp(tt, tf, al)
            pp = TAU * np.cumsum(fr) / sr
            tone = (np.sin(pp) + 0.25 * np.sin(2 * pp) + 0.08 * np.sin(3 * pp)) * (0.06 + 0.94 * prox) * gate * live
            a = (np.interp(tt, tf, np.clip(p[:, 0] / 430.0, -0.8, 0.8)) + 1) * np.pi / 4
            wet[:n, 0] += 0.035 * tone * np.cos(a)
            wet[:n, 1] += 0.035 * tone * np.sin(a)
            prox_sum += prox ** 1.5
        nz = au.noise(rng, DUR, 700, 5200)[:n] * np.minimum(1.0, prox_sum[:len(tt)]) * live      # dive whoosh
        wet[:n, 0] += 0.05 * nz
        wet[:n, 1] += 0.05 * nz

        for tp, i, gap, ang, rhp in self.calls:                       # sonar ping per dive read-out
            if tp >= self.T_RESET:
                continue
            close = gap < 0.15 * rhp
            pan = clamp(math.cos(ang) * 0.7, -0.7, 0.7)
            au.place(wet, au.glide(base[i] * 2, base[i] * 1.97, 0.7, decay=0.16, harm=0.12), tp,
                     0.20 if close else 0.11, pan)
            if close:
                au.place(wet, au.glide(base[i] * 4, base[i] * 3.94, 0.9, decay=0.22, harm=0.08), tp + 0.07, 0.16, pan)
                au.place(wet, au.noise(rng, 0.5, 5000, 11000) * np.exp(-loc(0.5) * 9), tp, 0.10, pan)

        for tp, i, gap, ang, rhp in self.sim["passes"]:               # a dry tick on every dive
            if 0 <= tp < self.T_RESET:
                pan = clamp(math.cos(ang) * 0.6, -0.6, 0.6)
                au.place(dry, au.tick(1250.0 + 190.0 * i, 0.07), tp, 1.50, pan)
                au.place(dry, au.glide(760.0, 360.0, 0.07, decay=0.022, harm=0.3), tp, 1.20, pan)

        for j, m in enumerate(self.sim["meals"]):                     # swallow: hit, sub drop, glissando, hush
            au.place(dry, au.kick(0.34), m[0], 2.00)
            au.place(dry, au.clap(rng, 0.20), m[0] + 0.004, 2.20, clamp(m[2] / 200.0, -0.6, 0.6))
            au.place(dry, au.noise(rng, 0.12, 1500, 7000) * np.exp(-loc(0.12) * 40), m[0], 1.00)
            au.place(dry, au.glide(520.0, 85.0, 0.30, decay=0.09, harm=0.5), m[0], 1.80)
            au.place(dry, au.glide(150.0, 36.0, 1.0, decay=0.40, harm=0.6), m[0] + 0.03, 0.90)
            pan = clamp(m[2] / 200.0, -0.6, 0.6)
            au.place(wet, au.glide(base[m[1]] * 2.2, 45.0, 0.75, decay=0.45, harm=0.4), m[0], 0.60, pan)
            au.place(wet, au.swept_noise(rng, 0.7, 3600, 180) * np.exp(-loc(0.7) * 3.5), m[0], 0.50, pan)
            au.place(dry, au.boom(rng, 1.3, 78, 32), m[0] + 0.02, 0.80)

        bounds = [0.0] + [tm + 0.9 for tm in T[:3]] + [self.T_END + 0.6]
        for j in range(4):                                            # pads thicken, the pulse quickens
            t0, t1 = bounds[j], bounds[j + 1]
            root = 45 - j
            voices = [(root, 0.9, 0.0), (root + 7, 0.7, -0.3)]
            if j >= 1:
                voices.append((root + 12, 0.6, 0.3))
            if j >= 2:
                voices.append((root + 15, 0.5, -0.2))
            if j >= 3:
                voices += [(root + 19, 0.5, 0.2), (root + 13, 0.30, 0.0)]
            for mnote, g, pn in voices:
                au.place(bed, au.pad(au.hz(mnote), t1 - t0 + 0.4, 1.0, 0.6), max(0.0, t0 - 0.2),
                         0.035 * (0.45 + 0.25 * j) * g, pn)
            if j >= 1:
                step = (1.0, 0.75, 0.5)[j - 1]
                kick = au.kick()
                tb = t0
                while tb < T[j] - 0.05:
                    au.place(bed, kick, tb, 0.20 + 0.07 * j)
                    au.place(bed, kick, tb + 0.17, 0.12 + 0.05 * j)
                    if j == 3:
                        au.place(bed, au.tick(880, 0.05), tb + step / 2, 0.07)
                    tb += step
        rl = min(3.0, T[3] - T[2] - 1.3)                              # riser into the last meal
        if rl > 0.7:
            env = (loc(rl) / rl) ** 2
            au.place(bed, au.swept_noise(rng, rl, 300, 4200) * env, T[3] - rl, 0.22)
            au.place(bed, au.glide(110, 330, rl) * env, T[3] - rl, 0.07)

        for mnote, g, pn in ((41, 0.50, 0.0), (48, 0.44, -0.3), (53, 0.38, 0.3), (60, 0.32, -0.15),
                             (65, 0.24, 0.15), (72, 0.15, 0.0)):      # result: an open fifth, held
            au.place(dry, au.pad(au.hz(mnote), self.T_RESET + 0.5 - self.T_REVEAL, 0.12, 1.0), self.T_REVEAL,
                     0.32 * g, pn)
        au.place(wet, au.bell(au.hz(65), 2.6), self.T_REVEAL, 0.30, -0.2)
        au.place(wet, au.bell(au.hz(72), 2.6), self.T_REVEAL + 0.12, 0.26, 0.2)
        au.place(wet, au.bell(au.hz(77), 2.2), self.T_REVEAL + 0.24, 0.16)
        au.place(dry, au.boom(rng, 1.2, 70, 35), self.T_REVEAL, 0.50)
        au.place(wet, au.glide(660, 990, 0.12, 0.06), self.T_LINE, 0.18)
        wl = DUR - self.T_RESET
        au.place(dry, au.swept_noise(rng, wl, 500, 6000) * (loc(wl) / wl) ** 2, self.T_RESET, 0.50)

        bed[:n] *= duck[:, None]
        mix_ = au.reverb(wet, rng, decay=1.8, wet=0.28) + bed + dry
        out = mix_[:n].copy()
        out[:tail] += mix_[n:]                    # tails past the end wrap to the start: seamless loop
        return au.eq(out, ((40, -7.0), (90, -3.0), (250, -2.0), (600, 0.0), (1500, 2.0), (3500, 4.5), (9000, 5.0), (16000, 2.0)))

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        title = "Which planet survives the black hole? 5 orbits, 1 survivor"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "Five planets orbit a black hole. Drag shrinks every orbit, and each planet has its own drag: the "
                "pips next to its symbol show how much. A planet that touches the horizon is swallowed, and the "
                "horizon grows 15% with every meal. The last planet left wins.\n"
                "The standings show the gap each planet will have on its next dive, so you can see who is in "
                "trouble before it happens. Nothing is slowed down: after the last meal the survivor flies at "
                f"least one more full orbit at full speed. Motion, picture and sound are generated by code (seed "
                f"{self.seed}), with no footage, samples or stock media. Drag and the growing horizon are game "
                "rules, not astrophysics.\n"
                "Who did you pick?"),
            tags=["black hole", "orbit simulation", "gravity simulation", "space simulation", "prediction",
                  "last one standing", "simulation", "made with code"],
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylised 2D animation of abstract shapes with game rules; "
                                              "nothing realistic that could be mistaken for real people, places, "
                                              "events or scientific footage.",
            seed=self.seed,
            duration_seconds=self.DUR,
            drama=self.stats,
        )
