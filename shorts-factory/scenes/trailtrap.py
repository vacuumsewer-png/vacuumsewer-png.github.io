"""TRAIL TRAP: four lines grow one cell per tick on a 36 x 64 grid and may never cross a trail or a wall.

Each line follows one fixed rule (straight, wall hugger, space seeker, random). The last line still
moving wins. simulate(seed) is deterministic and needs no rendering, so make.py can score hundreds of
seeds for drama (drama()) and render only the best one (Scene).
"""
import bisect
import math
import os
import random

import cairo
import numpy as np

from engine import audio as au
from engine.gfx import (TAU, Font, clamp, ease_in, ease_in_out, ease_out, ease_out_back, glow, hex_rgb, lerp,
                        lighten, mix, polygon, ramp)

W, H, FPS = 1080, 1920, 30
GW, GH = 36, 64                 # playing grid, cells
PW = GW + 2                     # padded width: a one-cell wall all round removes every bounds check
OFF = (1, PW, -1, -PW)          # east, south, west, north (a right turn is +1)
DXY = ((1, 0), (0, 1), (-1, 0), (0, -1))
INTERIOR = [(y + 1) * PW + x + 1 for y in range(GH) for x in range(GW)]
NCELL = GW * GH
EMPTY = bytes(NCELL)
RULES = ("STRAIGHT", "HUGGER", "SEEKER", "RANDOM")
NAMES = ("BLUE", "PINK", "GREEN", "ORANGE")       # symbols: triangle, square, cross, hexagon
QUADS = ((9, 16), (27, 16), (9, 48), (27, 48))

R0 = 16.0                       # ticks per second at the start
PRE = 24                        # ticks simulated before frame 0, so the trails are already on screen (1.5 s)
STEP_S, STEP_K = 3.0, 1.05      # speed rises 5% every three seconds
ROUND_K = (1.0, 1.15, 2.5)      # ... and jumps at every board clear: x1.15 with three left, x2.5 for the last two
P_TURN = 0.04                   # RANDOM: chance per tick of an unforced turn
SAFE = True                     # every rule refuses a one-cell dead end when it has another move
ZIG = 0.6                       # RANDOM: extra chance that a free turn goes the opposite way to its last one
SEEK_CAP = 0                    # SEEKER look-ahead: regions larger than this count as equal (0 = no limit)
# Winner by rule over seeds 1-600 with these settings: SEEKER 34%, STRAIGHT 32%, HUGGER 22%, RANDOM 5%,
# no single survivor 7%. The larger-region rule does not dominate, so its look-ahead is left unlimited.
SW_MIN, SW_FRAC = 100, 0.30     # a "swing": a line loses this many cells and this share of its room in one tick
E_MIN, E_MAX = 19.6, 26.6       # accepted time of the deciding crash, seconds
DUEL_MAX, SHARE_MIN, LULL_MAX = 6.5, 0.5, 9.0   # seed gates: short final duel in a shared region, no long lull
OUTRO = 4.5                     # result, DID YOU CALL IT?, wipe back to the start


# ====================================================================== clock
def s_of_t(t):
    """Ticks elapsed (fractional) at video time t. The video opens PRE ticks into the run."""
    if t <= 0:
        return PRE + R0 * t
    k = int(t // STEP_S)
    return PRE + R0 * STEP_S * (STEP_K ** k - 1) / (STEP_K - 1) + R0 * STEP_K ** k * (t - k * STEP_S)


def t_of_s(s):
    s -= PRE
    if s <= 0:
        return s / R0
    k = 0
    while True:
        seg = R0 * STEP_S * STEP_K ** k
        if s <= seg:
            return k * STEP_S + s / (R0 * STEP_K ** k)
        s -= seg
        k += 1


def make_clock(clear_ticks):
    """Tick position <-> video time for one run. The rate starts at R0, rises by STEP_K every STEP_S seconds
    and is multiplied by the next ROUND_K factor at every board clear. Returns (s_of_t, t_of_s, segments),
    segments being (start time, start tick, ticks per second)."""
    clears = sorted(clear_ticks)
    segs, t, s, n, j = [], 0.0, float(PRE), 0, 0
    while t < 90.0:
        rate = R0 * STEP_K ** n * ROUND_K[min(j, len(ROUND_K) - 1)]
        segs.append((t, s, rate))
        t_step = (n + 1) * STEP_S
        t_clear = t + max(0.0, clears[j] - s) / rate if j < len(clears) else 1e9
        if t_clear <= t_step:
            t, s, j = t_clear, max(s, float(clears[j])), j + 1
        else:
            s, t, n = s + (t_step - t) * rate, t_step, n + 1
    T0, S0 = [g[0] for g in segs], [g[1] for g in segs]

    def s_at(t):
        if t <= 0:
            return PRE + R0 * t
        i = bisect.bisect_right(T0, t) - 1
        return S0[i] + (t - T0[i]) * segs[i][2]

    def t_at(s):
        if s <= PRE:
            return (s - PRE) / R0
        i = bisect.bisect_right(S0, s) - 1
        return T0[i] + (s - S0[i]) / segs[i][2]

    return s_at, t_at, segs


# ====================================================================== simulation
def label(grid, comp):
    """4-connected components of the free cells: writes ids (1..) into comp, returns the size of each id."""
    for c in INTERIOR:
        comp[c] = 0
    sizes = [0]
    for c0 in INTERIOR:
        if grid[c0] or comp[c0]:
            continue
        cid = len(sizes)
        comp[c0] = cid
        stack = [c0]
        n = 0
        while stack:
            c = stack.pop()
            n += 1
            d = c + 1
            if not grid[d] and not comp[d]:
                comp[d] = cid
                stack.append(d)
            d = c - 1
            if not grid[d] and not comp[d]:
                comp[d] = cid
                stack.append(d)
            d = c + PW
            if not grid[d] and not comp[d]:
                comp[d] = cid
                stack.append(d)
            d = c - PW
            if not grid[d] and not comp[d]:
                comp[d] = cid
                stack.append(d)
        sizes.append(n)
    return sizes


def may_split(grid, c):
    """Could filling cell c have cut a region in two? (free neighbours not joined around the corner)"""
    n, e, s, w = not grid[c - PW], not grid[c + 1], not grid[c + PW], not grid[c - 1]
    f = n + e + s + w
    if f <= 1:
        return False
    p = ((n and e and not grid[c - PW + 1]) + (e and s and not grid[c + PW + 1])
         + (s and w and not grid[c + PW - 1]) + (w and n and not grid[c - PW - 1]))
    return f - p >= 2


def ray(grid, p, d):
    o, n = OFF[d], 0
    p += o
    while not grid[p]:
        n += 1
        p += o
    return n


def longer_ray(grid, p, opts, rng):
    r = [ray(grid, p, d) for d in opts]
    m = max(r)
    best = [d for d, v in zip(opts, r) if v == m]
    return best[0] if len(best) == 1 else rng.choice(best)


def exits(grid, c):
    return (not grid[c + 1]) + (not grid[c - 1]) + (not grid[c + PW]) + (not grid[c - PW])


def decide(h, grid, comp, sizes):
    """Direction a head takes this tick under its rule. With no free neighbour it drives on and crashes."""
    p, d = h["pos"], h["dir"]
    L, R = (d - 1) & 3, (d + 1) & 3
    fs, fl, fr = not grid[p + OFF[d]], not grid[p + OFF[L]], not grid[p + OFF[R]]
    if not (fs or fl or fr):
        return d
    rule = h["rule"]
    if rule == 0:                                   # STRAIGHT: straight until blocked, then the longer way
        c = d if fs else longer_ray(grid, p, [x for x, f in ((L, fl), (R, fr)) if f], h["rng"])
    elif rule == 1:                                 # HUGGER: keep a wall or trail on one hand
        sd, od = (d + h["hand"]) & 3, (d - h["hand"]) & 3
        if not grid[p + OFF[sd]] and grid[p + OFF[sd] - OFF[d]]:
            c = sd                                  # the wall just ended: wrap round its corner
        elif fs:
            c = d
        else:
            c = od if not grid[p + OFF[od]] else sd
    elif rule == 2:                                 # SEEKER: always into the larger open region
        opts = [x for x, f in ((d, fs), (L, fl), (R, fr)) if f]
        sz = [sizes[comp[p + OFF[x]]] for x in opts]
        if SEEK_CAP:
            sz = [min(v, SEEK_CAP) for v in sz]
        m = max(sz)
        top = [x for x, v in zip(opts, sz) if v == m]
        c = top[0] if top[0] == d or len(top) == 1 else longer_ray(grid, p, top, h["rng"])
    else:                                           # RANDOM: turns whenever it feels like it
        sides = [x for x, f in ((L, fl), (R, fr)) if f]
        c = d
        if sides and (not fs or h["rng"].random() < P_TURN):
            c = h["rng"].choice(sides)
            if len(sides) == 2 and h.get("turn") is not None and h["rng"].random() < ZIG:
                c = (d - h["turn"]) & 3             # leans away from its previous turn, so it wanders
    if SAFE and not exits(grid, p + OFF[c]):        # nobody steps into a one-cell dead end by choice
        alt = [x for x, f in ((d, fs), (L, fl), (R, fr)) if f and exits(grid, p + OFF[x])]
        if alt:
            c = alt[0] if len(alt) == 1 else longer_ray(grid, p, alt, h["rng"])
    if c != d:
        h["turn"] = 1 if c == R else -1
    return c


def simulate(seed, record=False):
    rnd = random.Random(seed * 7919 + 13)
    grid = bytearray(PW * (GH + 2))
    for x in range(PW):
        grid[x] = grid[(GH + 1) * PW + x] = 1
    for y in range(GH + 2):
        grid[y * PW] = grid[y * PW + PW - 1] = 1
    order, rules = list(range(4)), list(range(4))
    rnd.shuffle(order)                              # which corner each colour starts in
    rnd.shuffle(rules)                              # which rule each colour follows
    heads = []
    for i in range(4):
        qx, qy = QUADS[order[i]]
        x, y, d = qx + rnd.randint(-4, 4), qy + rnd.randint(-7, 7), rnd.randrange(4)
        c = (y + 1) * PW + x + 1
        grid[c] = i + 1
        heads.append(dict(pos=c, dir=d, alive=True, path=[c], dirs=[d], rule=rules[i], hand=rnd.choice((-1, 1)),
                          rng=random.Random(seed * 31 + i), death=None, death_s=None, half=False))
    comp = [0] * len(grid)
    sizes = label(grid, comp)

    def touching(h):
        p = h["pos"]
        return {comp[p + o] for o in OFF if not grid[p + o]}

    def own_map(ids):
        cnt = {}
        for i, s in enumerate(ids):
            for c in s:
                cnt[c] = 0 if c in cnt else i + 1   # 0: more than one line can still get there
        if not any(cnt.values()):
            return EMPTY
        return bytes(cnt.get(comp[c], 0) for c in INTERIOR)

    ids = [touching(h) for h in heads]
    reach = [tuple(sum(sizes[c] for c in s) for s in ids)]
    own = [own_map(ids)] if record else None
    swings, elims, joint = [], [], [0]              # joint[k]: the last two can still reach a common region
    alive, k, end, end_s, winner = 4, 0, None, None, None
    while alive > 1 and k < 4000:
        k += 1
        prev = comp[:] if record else None
        targets = {}
        for i, h in enumerate(heads):
            if h["alive"]:
                h["dir"] = decide(h, grid, comp, sizes)
                targets.setdefault(h["pos"] + OFF[h["dir"]], []).append(i)
        new, out = [], []
        for tgt, who in targets.items():
            if grid[tgt]:                           # trail or wall: stopped at the edge of the cell
                for i in who:
                    heads[i].update(alive=False, death=k, death_s=k - 0.5, half=True)
                    out.append(i)
            elif len(who) > 1:                      # two heads into one cell: both out
                grid[tgt] = 9
                new.append(tgt)
                for i in who:
                    h = heads[i]
                    h["path"].append(tgt)
                    h["dirs"].append(h["dir"])
                    h.update(alive=False, death=k, death_s=float(k), half=False)
                    out.append(i)
            else:
                h = heads[who[0]]
                grid[tgt] = who[0] + 1
                h["pos"] = tgt
                h["path"].append(tgt)
                h["dirs"].append(h["dir"])
                new.append(tgt)
        relabel = False
        newset = set(new)
        for c in new:
            cid = comp[c]
            comp[c] = 0
            if cid:
                sizes[cid] -= 1
            if not relabel and (may_split(grid, c) or any(c + o in newset for o in OFF)):
                relabel = True
        if relabel:
            sizes = label(grid, comp)
        if out:
            alive -= len(out)
            elims.append((k, max(heads[i]["death_s"] for i in out), sorted(out)))
            if alive >= 2:                          # rule: every crash clears the board; the lines left carry on
                for c in INTERIOR:                  # from where they are, all in one open region again
                    grid[c] = 0
                for i, h in enumerate(heads):
                    if h["alive"]:
                        grid[h["pos"]] = i + 1
                sizes = label(grid, comp)
        prev_ids = ids
        ids = [touching(h) if h["alive"] else set() for h in heads]
        r = tuple(sum(sizes[c] for c in s) for s in ids)
        two = [i for i, h in enumerate(heads) if h["alive"]] if alive == 2 else None
        joint.append(1 if two and ids[two[0]] & ids[two[1]] else 0)
        pr = reach[-1]
        for i, h in enumerate(heads):
            if h["alive"] and pr[i] - r[i] >= max(SW_MIN, SW_FRAC * pr[i]):
                sw = dict(tick=k, head=i, before=pr[i], after=r[i])
                if record:
                    sw["lost"] = [j for j, c in enumerate(INTERIOR)
                                  if not grid[c] and prev[c] in prev_ids[i] and comp[c] not in ids[i]]
                swings.append(sw)
        reach.append(r)
        if record:
            own.append(own_map(ids))
    end = k
    end_s = elims[-1][1] if elims else float(k)
    left = [i for i, h in enumerate(heads) if h["alive"]]
    winner = left[0] if len(left) == 1 else None
    unused = []
    if record and winner is not None and len(elims[-1][2]) == 1:
        loser = elims[-1][2][0]                     # cells that were the runner-up's alone and never got used
        last = {}
        for kk in range(elims[-2][0] if len(elims) > 1 else 0, end):
            if own[kk] is not EMPTY:
                for j, o in enumerate(own[kk]):
                    if o:
                        last[j] = o
        unused = sorted(j for j, o in last.items() if o == loser + 1 and not grid[INTERIOR[j]])
    # How much longer would the winner have lasted? Not shown; it measures how close the finish was.
    extra = None
    if winner is not None:
        h = dict(heads[winner])
        extra = 0
        while extra < 1500:
            d = decide(h, grid, comp, sizes)
            tgt = h["pos"] + OFF[d]
            if grid[tgt]:
                break
            grid[tgt] = winner + 1
            h["pos"], h["dir"] = tgt, d
            extra += 1
            sizes[comp[tgt]] -= 1
            comp[tgt] = 0
            if may_split(grid, tgt):
                sizes = label(grid, comp)
    return dict(seed=seed, heads=heads, reach=reach, own=own, swings=swings, elims=elims, joint=joint, end=end, end_s=end_s,
                winner=winner, extra=extra, unused=unused)


def drama(sim):
    """Score a finished simulation: late decision, counter swings, favourite changes, no early blowout."""
    seed, end, w = sim["seed"], sim["end"], sim["winner"]
    heads, reach, elims = sim["heads"], sim["reach"], sim["elims"]
    if w is None:
        return dict(seed=seed, score=-1.0, rejected="no single survivor")
    s_of_t, t_of_s, _ = make_clock([e[0] for e in elims[:-1]])
    E = t_of_s(sim["end_s"])
    order, left = [], 4
    for k, _, who in elims:
        left -= len(who)
        for i in who:
            heads[i]["place"] = left + 1
            order.append(dict(line=NAMES[i], rule=RULES[heads[i]["rule"]],
                              at_s=round(t_of_s(heads[i]["death_s"]), 2), tick=k, place=left + 1))
    heads[w]["place"] = 1
    t_first = order[0]["at_s"]
    lead = []
    for k in range(end + 1):
        al = [i for i in range(4) if heads[i]["death"] is None or heads[i]["death"] > k]
        m = max(reach[k][i] for i in al)
        top = [i for i in al if reach[k][i] == m]
        lead.append(top[0] if len(top) == 1 else -1)
    runs = []
    for v in lead:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    held = [v for v, n in runs if v >= 0 and n >= 4]
    lead_changes = sum(1 for a, b in zip(held, held[1:]) if a != b)
    decided = [v for v in lead if v >= 0]
    led_share = decided.count(w) / len(decided) if decided else 0.0
    fav_half = lead[end // 2]
    swings = sim["swings"]
    biggest = max(swings, key=lambda s: s["before"] - s["after"]) if swings else None

    duel = None
    ratio = late = duel_ok = share = 0.0
    if len(elims) >= 2 and len(elims[-1][2]) == 1:
        k2, loser = elims[-2][0], elims[-1][2][0]
        duel_s = E - t_of_s(k2)
        rw, rl = reach[k2][w], reach[k2][loser]
        ratio = min(rw, rl) / max(rw, rl, 1)
        behind = [k for k in range(k2, end) if reach[k][w] < reach[k][loser]]
        took = None
        if behind:
            took = t_of_s(behind[-1] + 1)
            late = clamp(1 - (E - took) / 6.0)
        duel_ok = clamp(duel_s / 2.0) * clamp((7.5 - duel_s) / 2.0)
        share = sum(sim["joint"][k2:end]) / max(1, end - k2)
        duel = dict(seconds=round(duel_s, 2), share_of_duel_in_one_shared_region=round(share, 2),
                    room_at_start={NAMES[w]: rw, NAMES[loser]: rl},
                    winner_had_less_room_at_start=rw < rl,
                    winner_last_behind_on_room_at_s=None if took is None else round(took, 2))
    extra = sim["extra"]
    score = (6.0 * clamp(1 - extra / 40.0)              # the winner was almost out of room as well
             + 3.0 * clamp(1 - reach[end][w] / 150.0)   # ... and its counter shows it
             + 2.0 * min(len(swings), 4)                # violent swings of the room counters
             + 1.2 * min(lead_changes, 5)               # the favourite keeps changing
             + 4.0 * clamp((t_first - 3.0) / 5.0)       # nobody out before about 8 s
             + 3.0 * ratio                              # final two start the duel with similar room
             + 4.0 * late                               # the winner was behind on room until late
             + 2.0 * duel_ok
             + 3.0 * share                              # the last two fight over the same room
             + (2.0 if fav_half != w else 0.0)
             - 4.0 * clamp((led_share - 0.6) / 0.4))    # no start-to-finish favourite
    rejected = None
    if not E_MIN <= E <= E_MAX:
        rejected = "length"
    elif t_first < 4.0:
        rejected = "early exit"
    elif duel is None or duel["seconds"] > DUEL_MAX:
        rejected = "long duel"
    elif share < SHARE_MIN:
        rejected = "last two never share a region"
    elif t_first > 9.5 or max(b["at_s"] - a["at_s"] for a, b in zip(order, order[1:])) > LULL_MAX:
        rejected = "long lull"
    if rejected:
        score = -1.0
    return dict(
        seed=seed, score=round(score, 3), rejected=rejected,
        winner=NAMES[w], winner_index=w, winner_rule=RULES[heads[w]["rule"]],
        rules={NAMES[i]: RULES[heads[i]["rule"]] for i in range(4)},
        deciding_crash_at_s=round(E, 2), ticks=end, eliminations=order, first_elimination_at_s=t_first,
        final_duel=duel, winner_room_left_at_end=reach[end][w], winner_moves_left_after_end=extra,
        runner_up_unused_cells=len(sim["unused"]) if sim["own"] is not None else None,
        big_room_swings=len(swings),
        biggest_room_swing=None if biggest is None else dict(
            line=NAMES[biggest["head"]], before=biggest["before"], after=biggest["after"],
            at_s=round(t_of_s(biggest["tick"]), 2)),
        room_lead_changes=lead_changes, most_room_at_half="TIED" if fav_half < 0 else NAMES[fav_half],
        winner_share_of_time_with_most_room=round(led_share, 3),
    )


def score_seed(seed):
    return drama(simulate(seed))


# ====================================================================== look
COLS = [hex_rgb(c) for c in ("#22D3FF", "#FF3FA4", "#8CFF3A", "#FF9A1F")]
INK = hex_rgb("#04070F")
WHITE = (1.0, 1.0, 1.0)
HUDC = (0.55, 0.78, 0.90)
GREY = (0.15, 0.18, 0.26)
FONT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Anton-Regular.ttf")
ADD, OVER = cairo.OPERATOR_ADD, cairo.OPERATOR_OVER

C = 19                           # cell size, px
AW, AH = GW * C, GH * C          # arena 684 x 1216
AX, AY = 22, 290
HX, HW = AX + AW + 14, 200       # status column, ends at x = 920 (30 px inside the right-hand safe line)
GAP = 14
CH = (AH - 3 * GAP) / 4
ANCX, ANCY = 490.0, float(AY + AH)
CX_T = 490                       # centre line of the titles
HOOK_ZOOM = 0.86                 # the board starts smaller to make room for the big question
T_HOOK = 2.0
BM = 48                          # margin of the cached board sprite
FLOW = 620.0                     # speed of the signal pulses along live trails, px/s
PERIOD = 8                       # each line sounds every 8th tick
VOICE = (62, 69, 65, 72)         # D4 A4 F4 C5: one pitch of D minor pentatonic per line
PLACE = {4: "4TH", 3: "3RD", 2: "2ND", 1: "1ST"}


def cell_xy(j):
    return AX + (j % GW) * C, AY + (j // GW) * C


def chamfer(ctx, x, y, w, h, c):
    ctx.move_to(x + c, y)
    ctx.line_to(x + w, y)
    ctx.line_to(x + w, y + h - c)
    ctx.line_to(x + w - c, y + h)
    ctx.line_to(x, y + h)
    ctx.line_to(x, y + c)
    ctx.close_path()


def symbol_path(ctx, k, x, y, r, rot=0.0):
    ctx.new_path()
    if k == 0:
        polygon(ctx, [(x + r * 1.18 * math.cos(rot + a), y + r * 1.18 * math.sin(rot + a))
                      for a in (-TAU / 4, -TAU / 4 + TAU / 3, -TAU / 4 + 2 * TAU / 3)])
    elif k == 1:
        ctx.rectangle(x - r * 0.86, y - r * 0.86, r * 1.72, r * 1.72)
    elif k == 2:
        q = r * 0.9
        ctx.move_to(x - q, y - q)
        ctx.line_to(x + q, y + q)
        ctx.move_to(x + q, y - q)
        ctx.line_to(x - q, y + q)
    else:
        polygon(ctx, [(x + r * 1.06 * math.cos(a * TAU / 6), y + r * 1.06 * math.sin(a * TAU / 6))
                      for a in range(6)])


def neon(ctx, col, lw, a=1.0, halo=1.0):
    """Stroke the current path as a neon tube: two additive halos, a saturated core, a hot centre line."""
    ctx.set_line_join(cairo.LINE_JOIN_MITER)
    if halo > 0.01:
        ctx.set_operator(ADD)
        ctx.set_source_rgba(col[0], col[1], col[2], 0.10 * a * halo)
        ctx.set_line_width(lw * 3.6)
        ctx.stroke_preserve()
        ctx.set_source_rgba(col[0], col[1], col[2], 0.22 * a * halo)
        ctx.set_line_width(lw * 2.0)
        ctx.stroke_preserve()
        ctx.set_operator(OVER)
    ctx.set_source_rgba(col[0], col[1], col[2], a)
    ctx.set_line_width(lw)
    ctx.stroke_preserve()
    hi = lighten(col, 0.7)
    ctx.set_source_rgba(hi[0], hi[1], hi[2], 0.9 * a)
    ctx.set_line_width(lw * 0.36)
    ctx.stroke()


def fit(font, text, size, max_w):
    w = font.width(text, size)
    return size if w <= max_w else size * max_w / w


def decor(ctx):
    """Faint circuit traces in the margins (non-essential areas). Returns the polylines for the data packets."""
    rnd = random.Random(11)
    traces = []
    for j in range(12):                                   # bottom zone, runs from alternate sides
        y, sgn = 1580 + j * 28, (1 if j % 2 == 0 else -1)
        x = -20 if sgn > 0 else W + 20
        pts = [(x, y)]
        for _ in range(rnd.randint(2, 4)):
            x += sgn * rnd.uniform(110, 300)
            pts.append((x, y))
            jog = 28 * rnd.choice((-1, 1))
            if not 1566 <= y + jog <= 1900:
                jog = -jog
            x += sgn * 28
            y += jog
            pts.append((x, y))
        x += sgn * rnd.uniform(50, 160)
        pts.append((x, y))
        traces.append(pts)
    for j in range(4):                                    # right-hand strip, runs downwards
        x, y = 974 + j * 30, -20
        pts = [(x, y)]
        for _ in range(rnd.randint(3, 5)):
            y += rnd.uniform(170, 380)
            pts.append((x, y))
            jog = 15 * rnd.choice((-1, 1))
            if not 966 <= x + jog <= 1072:
                jog = -jog
            x += jog
            y += 15
            pts.append((x, y))
        y += rnd.uniform(40, 120)
        pts.append((x, y))
        traces.append(pts)
    for j in range(3):                                    # top zone
        x, y = -20, 38 + j * 36
        pts = [(x, y)]
        for _ in range(rnd.randint(1, 3)):
            x += rnd.uniform(140, 320)
            pts.append((x, y))
            jog = 18 * rnd.choice((-1, 1))
            if not 20 <= y + jog <= 130:
                jog = -jog
            x += 18
            y += jog
            pts.append((x, y))
        x += rnd.uniform(60, 200)
        pts.append((x, y))
        traces.append(pts)
    for cx, cy, w, h in ((250, 1800, 170, 76), (790, 1716, 130, 96)):     # two chip outlines
        ctx.rectangle(cx - w / 2, cy - h / 2, w, h)
        ctx.set_source_rgb(0.020, 0.032, 0.066)
        ctx.fill_preserve()
        ctx.set_source_rgb(0.055, 0.185, 0.25)
        ctx.set_line_width(3)
        ctx.stroke()
        for q in range(int(w // 20)):
            px = cx - w / 2 + 10 + q * 20
            for sy in (-1, 1):
                ctx.move_to(px, cy + sy * h / 2)
                ctx.line_to(px, cy + sy * (h / 2 + 12))
        ctx.stroke()
    ctx.set_line_join(cairo.LINE_JOIN_MITER)
    for pts in traces:
        ctx.move_to(*pts[0])
        for p in pts[1:]:
            ctx.line_to(*p)
        ctx.set_source_rgb(0.045, 0.165, 0.225)
        ctx.set_line_width(3)
        ctx.stroke()
        ex, ey = pts[-1]
        ctx.rectangle(ex - 7, ey - 7, 14, 14)
        ctx.set_source_rgb(0.012, 0.02, 0.045)
        ctx.fill_preserve()
        ctx.set_source_rgb(0.07, 0.25, 0.33)
        ctx.stroke()
    return traces


def make_background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    v = (yy / H)[..., None]
    img = np.array([0.018, 0.038, 0.075], np.float32) * (1 - v) + np.array([0.012, 0.014, 0.040], np.float32) * v
    for cx, cy, rad, col in ((380, 900, 1000, (0.000, 0.050, 0.075)), (1000, 1650, 700, (0.060, 0.000, 0.070)),
                             (60, 180, 620, (0.010, 0.030, 0.100))):       # soft light pools for depth
        d = np.hypot(xx - cx, yy - cy) / rad
        img += (np.clip(1 - d, 0, 1) ** 2)[..., None] * np.array(col, np.float32)
    vig = np.clip(1.10 - 0.45 * (((xx - 540) / 720) ** 2 + ((yy - 960) / 1180) ** 2), 0.35, 1.0)
    img *= vig[..., None]
    img += np.random.default_rng(9).normal(0, 0.005, (H, W, 1)).astype(np.float32)    # grain hides banding
    buf = np.zeros((H, W, 4), np.uint8)
    buf[..., :3] = (np.clip(img[..., ::-1], 0, 1) * 255).astype(np.uint8)
    buf[..., 3] = 255
    surf = cairo.ImageSurface.create_for_data(memoryview(buf), cairo.FORMAT_RGB24, W, H, W * 4)
    traces = decor(cairo.Context(surf))
    surf.flush()
    return surf, buf, traces


def make_board():
    """The empty arena (panel, grid, frame) as a sprite, so the hook zoom can scale it in one blit."""
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, AW + 2 * BM, AH + 2 * BM)
    c = cairo.Context(s)
    c.translate(BM, BM)
    for k in range(6):                                    # soft outer glow
        e = 4 + k * 6
        c.rectangle(-e, -e, AW + 2 * e, AH + 2 * e)
        c.set_source_rgba(0.05, 0.55, 0.85, 0.028)
        c.fill()
    g = cairo.LinearGradient(0, 0, 0, AH)
    g.add_color_stop_rgb(0, 0.026, 0.046, 0.090)
    g.add_color_stop_rgb(1, 0.016, 0.024, 0.058)
    c.rectangle(0, 0, AW, AH)
    c.set_source(g)
    c.fill()
    for x in range(1, GW):
        c.rectangle(x * C - 0.5, 0, 1, AH)
        c.set_source_rgba(0.25, 0.75, 1.0, 0.27 if x % 4 == 0 else 0.115)
        c.fill()
    for y in range(1, GH):
        c.rectangle(0, y * C - 0.5, AW, 1)
        c.set_source_rgba(0.25, 0.75, 1.0, 0.27 if y % 4 == 0 else 0.115)
        c.fill()
    for y in range(4, GH, 4):                             # solder points where the heavy lines cross
        for x in range(4, GW, 4):
            c.rectangle(x * C - 2, y * C - 2, 4, 4)
    c.set_source_rgba(0.50, 0.92, 1.0, 0.85)
    c.fill()
    c.rectangle(-1.5, -1.5, AW + 3, AH + 3)
    c.set_source_rgba(0.30, 0.80, 1.0, 0.85)
    c.set_line_width(3)
    c.stroke()
    c.set_line_width(7)                                   # corner brackets
    c.set_source_rgba(0.65, 0.95, 1.0, 1.0)
    for cx, cy, sx, sy in ((0, 0, 1, 1), (AW, 0, -1, 1), (0, AH, 1, -1), (AW, AH, -1, -1)):
        c.move_to(cx + sx * 54, cy - sy * 3.5)
        c.line_to(cx - sx * 3.5, cy - sy * 3.5)
        c.line_to(cx - sx * 3.5, cy + sy * 54)
        c.stroke()
    s.flush()
    return s


# ====================================================================== the scene
class Scene:
    W, H, FPS = W, H, FPS

    def __init__(self, seed):
        self.seed = seed
        self.sim = simulate(seed, record=True)
        self.stats = drama(self.sim)
        if self.sim["winner"] is None:
            raise ValueError(f"seed {seed}: no single survivor, pick another seed")
        self.winner = self.sim["winner"]
        self.END = self.sim["end"]
        self.s_of_t, self.t_of_s, self.segs = make_clock([e[0] for e in self.sim["elims"][:-1]])
        self.seg_t = [g[0] for g in self.segs]
        self.E = self.t_of_s(self.sim["end_s"])                # the deciding crash
        self.nframes = int(round((self.E + OUTRO) * FPS))
        self.DUR = self.nframes / FPS
        self.T_WIN = self.E + 0.45                        # winner line
        self.T_LINE = self.E + 1.55                       # "DID YOU CALL IT?"
        self.T_RESET = self.DUR - 0.6                     # wipe, then the next round is already running
        self._res = None
        self._prep()

    # ------------------------------------------------------------ precomputed animation state
    def _prep(self):
        sim = self.sim
        rnd = random.Random(self.seed + 77)
        self.total0 = max(sim["reach"][0])
        self.hd, self.crashes, self.shakes = [], [], []
        for i, h in enumerate(sim["heads"]):
            pts = [(AX + (c % PW - 0.5) * C, AY + (c // PW - 0.5) * C) for c in h["path"]]
            dirs = h["dirs"]
            d = dict(pts=pts, dirs=dirs, dir=h["dir"], rule=h["rule"], place=h.get("place", 1),
                     vanish=h["death"] is not None and h["death"] != self.END,
                     corners=[j for j in range(1, len(pts) - 1) if dirs[j + 1] != dirs[j]],
                     death=h["death"], death_s=h["death_s"], half=h["half"], death_t=None, crash=None)
            if h["death_s"] is not None:
                dx, dy = DXY[h["dir"]]
                k = 0.5 * C if h["half"] else 0.0
                d["crash"] = (pts[-1][0] + dx * k, pts[-1][1] + dy * k)
                d["death_t"] = self.t_of_s(h["death_s"])
                big = h["death"] == self.END
                sparks = [(rnd.randrange(16) * TAU / 16, rnd.uniform(160, 620) * (1.4 if big else 1.0),
                           rnd.uniform(0.45, 0.95), rnd.uniform(3.0, 7.0)) for _ in range(40 if big else 26)]
                self.crashes.append((d["death_t"], d["crash"][0], d["crash"][1], i, big, sparks))
                self.shakes.append((d["death_t"], 17.0 if big else 9.0))
            self.hd.append(d)
        self.tagpos = []                                  # hook labels: one per quadrant, board coordinates
        for d in self.hd:
            x0, y0 = d["pts"][0]
            topq = y0 < AY + AH / 2
            self.tagpos.append((AX + AW * (0.25 if x0 < AX + AW / 2 else 0.75),
                                AY + AH * (0.25 if topq else 0.75) - 190))
        el = sim["elims"]
        self.clear_k = [e[0] for e in el[:-1]]            # ticks at which the board was cleared
        self.clears = [(self.t_of_s(e[1]), e[0], ([0] + self.clear_k)[j], 3 - j) for j, e in enumerate(el[:-1])]
        self.num_a, a = [], None                          # room counters show only while they differ
        for f in range(self.nframes):
            kk = min(int(self.s_of_t(f / FPS)), self.END)
            r = sim["reach"][kk]
            vals = {r[i] for i, h in enumerate(sim["heads"]) if h["death"] is None or h["death"] > kk}
            tgt = 1.0 if len(vals) > 1 or f / FPS >= self.E else 0.0
            a = tgt if a is None else a + (tgt - a) * 0.18
            self.num_a.append(a)
        self.swings = []
        for sw in sim["swings"]:
            sw = dict(sw, t=self.t_of_s(sw["tick"]))
            if sw.get("lost"):                            # where on the board the room was lost
                pts = [cell_xy(j) for j in sw["lost"]]
                sw["at"] = (sum(p[0] for p in pts) / len(pts) + C / 2, sum(p[1] for p in pts) / len(pts) + C / 2)
            self.swings.append(sw)
            self.shakes.append((sw["t"], 4.5))
        self.shakes.sort()
        self.speedups = [g for g in self.seg_t[1:] if g <= self.E]
        self.unused = sim["unused"]
        el = sim["elims"]
        self.duel = None                                  # (start time, left line, right line) of the last two
        if len(el) >= 2 and len(el[-1][2]) == 1:
            pair = sorted((self.winner, el[-1][2][0]))
            self.duel = (self.t_of_s(el[-2][1]), pair[0], pair[1])

    def _resources(self):
        if self._res is None or self._res["pid"] != os.getpid():
            surf = cairo.ImageSurface(cairo.FORMAT_RGB24, W, H)
            bg, keep, traces = make_background()
            rnd = random.Random(23)
            packets = []
            for pts in traces:
                cum = [0.0]
                for a, b in zip(pts, pts[1:]):
                    cum.append(cum[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
                packets.append((pts, cum, rnd.choice((2, 3, 3, 4)), rnd.random()))
            font = Font(FONT_PATH)
            self._res = dict(pid=os.getpid(), surf=surf, ctx=cairo.Context(surf), bg=bg, keep=keep, font=font,
                             board=make_board(), packets=packets,
                             slot=max(font.width(ch, 88) for ch in "0123456789") + 2,
                             times="×" if 0xD7 in font.cmap else "x")
        return self._res

    def frame(self, f):
        R = self._resources()
        self.draw(R["ctx"], f, R)
        R["surf"].flush()
        return bytes(R["surf"].get_data())

    # ------------------------------------------------------------ state helpers
    def _alive(self, i, s):
        ds = self.hd[i]["death_s"]
        return ds is None or s < ds

    def _since(self, s):
        """Tick of the last board clear at or before tick position s (0 if none yet)."""
        k0 = 0
        for kc in self.clear_k:
            if kc <= s:
                k0 = kc
        return k0

    def _seg(self, i, k0, k1):
        """Polyline of line i from tick k0 to tick k1."""
        d = self.hd[i]
        pts, cor = d["pts"], d["corners"]
        k1 = min(k1, len(pts) - 1)
        k0 = min(k0, k1)
        out = [pts[k0]] + [pts[j] for j in cor[bisect.bisect_right(cor, k0):bisect.bisect_left(cor, k1)]]
        if k1 > k0:
            out.append(pts[k1])
        return out

    def _path(self, i, s):
        """Polyline of line i since the last board clear at tick position s, its head point (None once
        crashed) and heading."""
        d = self.hd[i]
        pts, cor = d["pts"], d["corners"]
        n = len(pts)
        if d["death_s"] is not None and s >= d["death_s"]:
            out = self._seg(i, self._since(d["death"] - 1), n - 1)
            if d["half"]:
                out.append(d["crash"])
            return out, None, d["dir"]
        k = min(int(s), n - 1)
        fr = s - k
        k0 = min(self._since(s), k)
        out = [pts[k0]] + [pts[j] for j in cor[bisect.bisect_right(cor, k0):bisect.bisect_right(cor, k)]]
        if k + 1 < n:
            a, b = pts[k], pts[k + 1]
            hp, hd = (a[0] + (b[0] - a[0]) * fr, a[1] + (b[1] - a[1]) * fr), d["dirs"][k + 1]
        else:
            dx, dy = DXY[d["dir"]]
            hp, hd = (pts[k][0] + dx * C * fr, pts[k][1] + dy * C * fr), d["dir"]
        if hp != out[-1]:
            out.append(hp)
        return out, hp, hd

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

    def hook(self, t):
        """1 in the opening and closing layout (big lines, smaller board), 0 in the main layout."""
        if t < 10:
            return 1 - ease_in_out(ramp(t, T_HOOK + 0.1, T_HOOK + 0.75))
        return ease_in_out(ramp(t, self.E + 0.25, self.E + 0.95))

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, f, R):
        t = f / FPS
        font = R["font"]
        reset = t >= self.T_RESET
        tau = t - self.DUR if reset else t                # time relative to the loop point
        s = min(self.s_of_t(t), float(self.END))
        k = int(s)
        z = lerp(1.0, HOOK_ZOOM, self.hook(t))
        sx, sy = self.shake(t)
        wipe = ease_in_out(ramp(t, self.T_RESET + 0.04, self.T_RESET + 0.42)) if reset else 0.0
        fresh = reset and t >= self.T_RESET + 0.24        # the status column already shows the next round

        ctx.set_operator(cairo.OPERATOR_SOURCE)
        ctx.set_source_surface(R["bg"], 0, 0)
        ctx.paint()
        ctx.set_operator(OVER)
        ctx.set_dash([])
        self._ambient(ctx, t, R)
        ctx.save()
        ctx.translate(ANCX + sx, ANCY + sy)
        ctx.scale(z, z)
        ctx.translate(-ANCX, -ANCY)
        ctx.set_source_surface(R["board"], AX - BM, AY - BM)
        ctx.paint()
        wy = AY + AH * wipe
        tagp = None
        if wipe < 1:
            ctx.save()
            if wipe > 0:                                  # the finished round is erased from the top down
                ctx.rectangle(AX - 90, wy, AW + 180, AY + AH + 90 - wy)
                ctx.clip()
            paths = [self._path(i, s) for i in range(4)]
            self._regions(ctx, t, k, font)
            self._trails(ctx, t, s, paths)
            self._heads(ctx, t, s, paths)
            self._fx(ctx, t)
            ctx.restore()
            if not reset and tau < T_HOOK + 0.3:
                tagp = paths
        if reset and wipe > 0:                            # above the wipe line the next round is already running
            s2 = self.s_of_t(tau)
            ctx.save()
            ctx.rectangle(AX - 90, AY - 90, AW + 180, wy - AY + 90)
            ctx.clip()
            tagp = [self._path(i, s2) for i in range(4)]
            self._trails(ctx, tau, s2, tagp)
            self._heads(ctx, tau, s2, tagp)
            ctx.restore()
        self._sweeps(ctx, t, wipe, wy)
        if tau < 0.9:
            self._launch(ctx, tau)
        self._hud(ctx, font, R, t, s, k, fresh)
        ctx.restore()
        if tagp:
            self._rule_tags(ctx, font, tau, z, tagp)
        for tc, kc, kp, left in self.clears:              # the rule, said out loud at every crash
            age = t - tc
            if 0.08 <= age < 1.35:
                a = min(1.0, (age - 0.08) * 9) * (1 - ramp(age, 1.0, 1.35))
                cx, cy = AX + AW / 2, AY + AH * 0.40
                sc = 0.7 + 0.3 * ease_out_back(ramp(age, 0.08, 0.36))
                font.draw(ctx, f"{left} LEFT", cx, cy, 170 * sc, WHITE, anchor="c", alpha=a, stroke=INK, stroke_w=16)
                font.draw(ctx, "BOARD CLEARS, FASTER", cx, cy + 92, 66 * sc, (0.45, 0.90, 1.0), anchor="c", alpha=a,
                          stroke=INK, stroke_w=12)
        for sw in self.swings:                            # room cut off: the loss, written on the board
            age = t - sw["t"]
            if 0 <= age < 0.9 and "at" in sw and not reset:
                n = sw["before"] - sw["after"]
                size = clamp(44 + 2.2 * math.sqrt(n), 60, 136) * (1 + 0.22 * (1 - ramp(age, 0.0, 0.2)))
                bx = clamp(sw["at"][0], AX + 110, AX + AW - 110)
                by = clamp(sw["at"][1] + size * 0.4, AY + 120, AY + AH - 40)
                font.draw(ctx, f"-{n}", ANCX + z * (bx - ANCX), ANCY + z * (by - ANCY), size,
                          lighten(COLS[sw["head"]], 0.45), anchor="c", alpha=1 - ramp(age, 0.55, 0.9), stroke=INK,
                          stroke_w=12)
        ha = ramp(tau, -0.34, -0.14) * (1 - ramp(tau, T_HOOK - 0.15, T_HOOK + 0.15))
        if ha > 0.004:                                    # hook: the rule in one line, inside the arena
            text = "EVERY CRASH CLEARS THE BOARD"                # (and speeds the rest up: said at each crash)
            size = fit(font, text, 50, z * AW - 60)
            cx, by = ANCX + z * (AX + AW / 2 - ANCX), ANCY - z * 40
            tw = font.width(text, size)
            chamfer(ctx, cx - tw / 2 - 18, by - size * 0.86 - 14, tw + 36, size * 0.86 + 28, 10)
            ctx.set_source_rgba(*INK, 0.88 * ha)
            ctx.fill()
            font.draw(ctx, text, cx, by, size, (0.45, 0.90, 1.0), anchor="c", alpha=ha)
        self._titles(ctx, t, tau, font, k)
        fl = 0.11 * (1 - ramp(t, self.E, self.E + 0.35)) ** 2 if t >= self.E else 0.0
        if fl > 0.004:
            ctx.set_operator(ADD)
            ctx.set_source_rgba(1, 1, 1, fl)
            ctx.paint()
            ctx.set_operator(OVER)

    def _ambient(self, ctx, t, R):
        ph = TAU * t / self.DUR
        ctx.set_operator(ADD)
        glow(ctx, 380 + 260 * math.cos(ph), 820 + 320 * math.sin(ph), 760, (0.05, 0.45, 0.65), 0.10)
        glow(ctx, 600 - 300 * math.cos(ph), 1300 - 260 * math.sin(ph), 620, (0.55, 0.10, 0.55), 0.07)
        for pts, cum, loops, p0 in R["packets"]:          # data packets on the margin traces (periodic)
            a = ((p0 + loops * t / self.DUR) % 1.0) * cum[-1]
            j = min(len(pts) - 2, bisect.bisect_right(cum, a) - 1)
            q = (a - cum[j]) / max(1e-6, cum[j + 1] - cum[j])
            x, y = lerp(pts[j][0], pts[j + 1][0], q), lerp(pts[j][1], pts[j + 1][1], q)
            glow(ctx, x, y, 26, (0.2, 0.8, 1.0), 0.35)
            ctx.rectangle(x - 4, y - 4, 8, 8)
            ctx.set_source_rgba(0.55, 0.95, 1.0, 0.85)
            ctx.fill()
        ctx.set_operator(OVER)

    def _regions(self, ctx, t, k, font):
        """Cells only one line can still reach are tinted in its colour: the room each counter is counting."""
        own = self.sim["own"][min(k, self.END)]
        if own is not EMPTY:
            groups = ([], [], [], [])
            for j, o in enumerate(own):
                if o:
                    groups[o - 1].append(j)
            a = 0.20 + 0.05 * math.sin(t * 5.0)
            for i, g in enumerate(groups):
                if g:
                    for j in g:
                        x, y = cell_xy(j)
                        ctx.rectangle(x + 5, y + 5, C - 10, C - 10)
                    ctx.set_source_rgba(*COLS[i], a)
                    ctx.fill()
        ctx.set_operator(ADD)
        for sw in self.swings:                            # room that was just cut off flashes and fades
            age = t - sw["t"]
            if 0 <= age < 0.8 and sw["lost"]:
                for j in sw["lost"]:
                    x, y = cell_xy(j)
                    ctx.rectangle(x + 2, y + 2, C - 4, C - 4)
                ctx.set_source_rgba(*mix(COLS[sw["head"]], WHITE, 0.5 * (1 - age / 0.8)), 0.30 * (1 - age / 0.8) ** 2)
                ctx.fill()
        ctx.set_operator(OVER)
        if self.unused and t >= self.E + 0.5:             # the runner-up's wasted room
            lo = self.sim["elims"][-1][2][0]
            col = COLS[lo]
            a = ramp(t, self.E + 0.5, self.E + 0.7) * (0.65 + 0.35 * math.sin((t - self.E) * 11))
            for j in self.unused:
                x, y = cell_xy(j)
                ctx.rectangle(x + 3.5, y + 3.5, C - 7, C - 7)
            ctx.set_source_rgba(*col, 0.30 * a)
            ctx.fill_preserve()
            ctx.set_source_rgba(*lighten(col, 0.3), 0.95 * a)
            ctx.set_line_width(2)
            ctx.stroke()
            mx = sum(cell_xy(j)[0] for j in self.unused) / len(self.unused) + C / 2
            my = sum(cell_xy(j)[1] for j in self.unused) / len(self.unused) + C / 2
            font.draw(ctx, f"{len(self.unused)} UNUSED", clamp(mx, AX + 110, AX + AW - 110),
                      clamp(my - 34, AY + 60, AY + AH - 30), 44, lighten(col, 0.35), anchor="c",
                      alpha=ramp(t, self.E + 0.6, self.E + 0.8), stroke=INK, stroke_w=10)

    def _trails(self, ctx, t, s, paths):
        ctx.set_line_join(cairo.LINE_JOIN_MITER)
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)
        for tc, kc, kp, _ in self.clears:                 # the survivors' trails wiped by a crash dissolve
            age = t - tc
            if not 0 <= age < 0.9 or s < kc:
                continue
            va = 1 - ease_in_out(ramp(age, 0.20, 0.90))
            for i in range(4):
                d = self.hd[i]
                if d["death"] is not None and d["death"] <= kc:
                    continue
                pts = self._seg(i, kp, kc)
                if len(pts) > 1:
                    ctx.new_path()
                    ctx.move_to(*pts[0])
                    for p in pts[1:]:
                        ctx.line_to(p[0], p[1])
                    ctx.set_operator(ADD)
                    ctx.set_source_rgba(*COLS[i], 0.16 * va)
                    ctx.set_line_width(13)
                    ctx.stroke_preserve()
                    ctx.set_operator(OVER)
                    ctx.set_source_rgba(*mix((0.05, 0.07, 0.12), COLS[i], 0.75), va)
                    ctx.set_line_width(7)
                    ctx.stroke()
        for i in sorted(range(4), key=lambda i: self._alive(i, s)):      # live lines on top
            pts, hp, _ = paths[i]
            d, col = self.hd[i], COLS[i]
            br, hot, va = 1.0, 0.0, 1.0
            if hp is None:
                age = t - d["death_t"]
                br = lerp(1.0, 0.36, ease_out(ramp(age, 0.05, 0.55)))
                hot = 1 - ramp(age, 0.0, 0.18)
                if d["vanish"]:                           # rule: the trail of a crashed line disappears
                    va = 1 - ease_in_out(ramp(age, 0.25, 0.90))
                    if va <= 0.004:
                        continue
            x0, y0 = pts[0]
            if len(pts) > 1:
                ctx.new_path()
                ctx.move_to(x0, y0)
                for p in pts[1:]:
                    ctx.line_to(p[0], p[1])
                ctx.set_operator(ADD)
                ctx.set_source_rgba(*col, 0.07 * br * va)
                ctx.set_line_width(24)
                ctx.stroke_preserve()
                ctx.set_source_rgba(*col, 0.16 * br * va)
                ctx.set_line_width(13)
                ctx.stroke_preserve()
                ctx.set_operator(OVER)
                ctx.set_source_rgba(*mix(mix((0.05, 0.07, 0.12), col, br), WHITE, 0.7 * hot), va)
                ctx.set_line_width(7)
                ctx.stroke_preserve()
                if hp is not None:
                    ctx.set_source_rgba(*lighten(col, 0.72), 0.9)
                    ctx.set_line_width(2.4)
                    ctx.stroke_preserve()
                    ctx.set_operator(ADD)                 # signal pulses running towards the head
                    ctx.set_dash([16, 138], 154 - (FLOW * t + i * 37) % 154)
                    ctx.set_source_rgba(1, 1, 1, 0.55)
                    ctx.set_line_width(4.5)
                    ctx.stroke_preserve()
                    ctx.set_dash([])
                    if i == self.winner and t >= self.E + 0.3:        # the whole winning trail lights up
                        age = t - (self.E + 0.3)
                        total = max(C, (len(d["pts"]) - 1 - self._since(self.END)) * C)
                        a1 = total * ease_in_out(ramp(age, 0.0, 1.15))
                        a0 = max(0.0, a1 - 0.22 * total)
                        if a1 > 1:
                            ctx.set_dash([a1, 3 * total])             # lit from the start to the wave
                            ctx.set_source_rgba(*col, 0.30)
                            ctx.set_line_width(15)
                            ctx.stroke_preserve()
                        if 1 < a1 < total and a1 - a0 > 1:
                            ctx.set_dash([a1 - a0, 3 * total], (a1 - a0) + 3 * total - a0)   # the wave itself
                            ctx.set_source_rgba(1, 1, 1, 0.85)
                            ctx.set_line_width(10)
                            ctx.stroke_preserve()
                            ctx.set_source_rgba(*col, 0.5)
                            ctx.set_line_width(26)
                            ctx.stroke_preserve()
                        ctx.set_dash([])
                        if age > 1.15:
                            ctx.set_source_rgba(*col, 0.22 * (0.6 + 0.4 * math.sin((age - 1.15) * 9)))
                            ctx.set_line_width(20)
                            ctx.stroke_preserve()
                    ctx.set_operator(OVER)
                ctx.new_path()
            polygon(ctx, [(x0, y0 - 9), (x0 + 9, y0), (x0, y0 + 9), (x0 - 9, y0)])    # start pad
            ctx.set_source_rgba(*INK, va)
            ctx.fill_preserve()
            ctx.set_source_rgba(*mix((0.07, 0.10, 0.16), col, br), va)
            ctx.set_line_width(3)
            ctx.stroke()

    def _head(self, ctx, i, x, y, hd, a=1.0, k=1.0):
        col = COLS[i]
        ctx.set_operator(ADD)
        glow(ctx, x, y, 98 * k, col, 0.55 * a)
        glow(ctx, x, y, 30 * k, lighten(col, 0.5), 0.65 * a)
        ctx.set_operator(OVER)
        rot = math.atan2(DXY[hd][1], DXY[hd][0]) + TAU / 4
        symbol_path(ctx, i, x, y, 16 * k, rot)
        ctx.set_line_join(cairo.LINE_JOIN_MITER)
        ctx.set_source_rgba(*INK, 0.9 * a)
        ctx.set_line_width(13 * k)
        ctx.stroke_preserve()
        if i != 2:
            ctx.fill_preserve()
        neon(ctx, lighten(col, 0.22), 4.8 * k, a)
        ctx.rectangle(x - 2.5, y - 2.5, 5, 5)
        ctx.set_source_rgba(1, 1, 1, a)
        ctx.fill()

    def _heads(self, ctx, t, s, paths):
        for i in range(4):
            pts, hp, hd = paths[i]
            d = self.hd[i]
            if hp is None:                                # where it stopped (fades with a trail that disappears)
                va = 1 - ramp(t - d["death_t"], 0.25, 0.90) if d["vanish"] else 1.0
                if va > 0.004:
                    x, y = d["crash"]
                    symbol_path(ctx, i, x, y, 9)
                    neon(ctx, mix(GREY, COLS[i], 0.45), 3.0, 0.9 * va, 0.2)
                continue
            k = 1.0 + 0.07 * math.sin(s * 0.9 + i * 1.7)
            if i == self.winner and t >= self.E:
                k = 1.25 + 0.2 * math.sin((t - self.E) * 9)
            self._head(ctx, i, hp[0], hp[1], hd, 1.0, k)

    def _spawn(self, ctx, tau):
        """The next round's heads arrive at their start cells just before the cut back to frame 0."""
        a = ramp(tau, -0.26, -0.09)
        if a <= 0:
            return
        s = self.s_of_t(tau)
        for i, d in enumerate(self.hd):
            x0, y0 = d["pts"][0]
            hd = d["dirs"][1] if len(d["dirs"]) > 1 else d["dir"]
            polygon(ctx, [(x0, y0 - 9), (x0 + 9, y0), (x0, y0 + 9), (x0 - 9, y0)])
            ctx.set_source_rgba(*INK, a)
            ctx.fill_preserve()
            ctx.set_source_rgba(*COLS[i], a)
            ctx.set_line_width(3)
            ctx.stroke()
            self._head(ctx, i, x0 + DXY[hd][0] * C * s, y0 + DXY[hd][1] * C * s, hd, a,
                       1.0 + 0.07 * math.sin(s * 0.9 + i * 1.7))

    def _launch(self, ctx, tau):
        """Start-up pulses from the four start pads. They straddle the loop cut, so frame 0 is already moving."""
        ctx.set_operator(ADD)
        ctx.set_line_join(cairo.LINE_JOIN_MITER)
        for j in range(2):
            p = (tau + 0.14 - j * 0.24) / 0.75
            if 0 < p < 1:
                r = 12 + 125 * ease_out(p, 2)
                for i, d in enumerate(self.hd):
                    x, y = d["pts"][min(PRE, len(d["pts"]) - 1)]          # where each head is on frame 0
                    polygon(ctx, [(x, y - r), (x + r, y), (x, y + r), (x - r, y)])
                    ctx.set_source_rgba(*COLS[i], 0.8 * (1 - p) ** 1.5)
                    ctx.set_line_width(5 * (1 - p) + 1)
                    ctx.stroke()
        ctx.set_operator(OVER)

    def _fx(self, ctx, t):
        ctx.set_operator(ADD)
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)
        for tc, x, y, i, big, sparks in self.crashes:
            age = t - tc
            if age < 0 or age > 1.0:
                continue
            col, m = COLS[i], (1.5 if big else 1.0)
            if age < 0.25:
                glow(ctx, x, y, (110 + 520 * age) * m, lighten(col, 0.6), 0.95 * (1 - age / 0.25))
            p = age / 0.55
            if p < 1:                                     # square shock ring
                r = (18 + 150 * ease_out(p)) * m
                polygon(ctx, [(x, y - r), (x + r, y), (x, y + r), (x - r, y)])
                ctx.set_source_rgba(*lighten(col, 0.4), (1 - p) ** 1.5)
                ctx.set_line_width(7 * (1 - p) + 1)
                ctx.stroke()
            for ang, sp, life, size in sparks:
                q = age / life
                if q >= 1:
                    continue
                dist = sp * life * 0.55 * (1 - (1 - q) ** 2.4)
                ca, sa = math.cos(ang), math.sin(ang)
                px, py = x + ca * dist, y + sa * dist
                ln = size * 4.0 * (1 - q) + 2
                ctx.move_to(px, py)
                ctx.line_to(px - ca * ln, py - sa * ln)
                ctx.set_source_rgba(*lighten(col, 0.35 * (1 - q)), (1 - q) ** 0.7)
                ctx.set_line_width(size * (1 - 0.5 * q))
                ctx.stroke()
        ctx.set_operator(OVER)

    @staticmethod
    def _band(ctx, y, h, col, a):
        """Scan band: bright edge at y, tail of height |h| behind it (below for h > 0, above for h < 0)."""
        g = cairo.LinearGradient(0, y, 0, y + h)
        g.add_color_stop_rgba(0, col[0], col[1], col[2], a)
        g.add_color_stop_rgba(1, col[0], col[1], col[2], 0)
        ctx.set_source(g)
        ctx.rectangle(AX, min(y, y + h), AW, abs(h))
        ctx.fill()
        ctx.set_source_rgba(0.85, 1.0, 1.0, min(1.0, a * 2.4))
        ctx.rectangle(AX, y - 1.5, AW, 3)
        ctx.fill()

    def _sweeps(self, ctx, t, wipe, wy):
        ctx.save()
        ctx.rectangle(AX, AY, AW, AH)
        ctx.clip()
        ctx.set_operator(ADD)
        for ts in self.speedups:                          # every speed step sends a scan line up the grid
            p = (t - ts) / 0.6
            if 0 <= p < 1:
                self._band(ctx, AY + AH * (1 - ease_out(p, 2)), 150, (0.25, 0.8, 1.0), 0.30 * (1 - p) ** 0.7)
        if 0 < wipe < 1:
            self._band(ctx, wy, -170, (0.35, 0.85, 1.0), 0.42)
        ctx.restore()
        ctx.set_operator(OVER)

    # ------------------------------------------------------------ status column
    def _digits(self, ctx, font, val, cx, base, size, fill, slot, alpha=1.0):
        txt = str(val)
        slot *= size / 88
        x0 = cx - slot * len(txt) / 2
        for j, ch in enumerate(txt):
            font.draw(ctx, ch, x0 + slot * (j + 0.5), base, size, fill, anchor="c", alpha=alpha, stroke=INK, stroke_w=8)

    def _hud(self, ctx, font, R, t, s, k, fresh):
        sim = self.sim
        if fresh:
            vals, alive = sim["reach"][0], [True] * 4
        else:
            vals, alive = sim["reach"][k], [self._alive(i, s) for i in range(4)]
        al = [i for i in range(4) if alive[i]]
        lead = -1
        if len(al) == 1:
            lead = al[0]
        elif al:
            m = max(vals[i] for i in al)
            top = [i for i in al if vals[i] == m]
            lead = top[0] if len(top) == 1 else -1
        # header row: speed on the left, what the numbers mean on the right
        base = AY - 16
        n_up = 0 if fresh else bisect.bisect_right(self.seg_t, min(t, self.E)) - 1
        pop = 0.0
        if n_up and not fresh:
            pop = 1 - ease_out(ramp(t - self.seg_t[n_up], 0.0, 0.45))
        font.draw(ctx, "SPEED", AX + 2, base, 38, HUDC)
        x = AX + 2 + font.width("SPEED", 38) + 14
        font.draw(ctx, f"{R['times']}{self.segs[n_up][2] / R0:.2f}", x, base, 38 * (1 + 0.22 * pop), mix((0.8, 0.96, 1.0), WHITE, pop))
        x += font.width(f"{R['times']}1.00", 38) + 30
        ctx.set_line_join(cairo.LINE_JOIN_MITER)
        for j in range(n_up):
            g = 1 + 0.6 * pop if j == n_up - 1 else 1.0
            ctx.move_to(x + j * 19, base - 14 - 12 * g)
            ctx.line_to(x + j * 19 + 10 * g, base - 14)
            ctx.line_to(x + j * 19, base - 14 + 12 * g)
            ctx.set_source_rgba(0.3, 0.85, 1.0, 0.95)
            ctx.set_line_width(4.5)
            ctx.stroke()
        na = self._numbers(t, fresh)                      # the room counters appear when they first differ
        font.draw(ctx, "ROOM LEFT", HX + HW / 2, base, 38, HUDC, anchor="c", alpha=na)
        font.draw(ctx, "4 RULES", HX + HW / 2, base, 38, HUDC, anchor="c", alpha=1 - na)
        for i in range(4):
            self._card(ctx, font, R, i, alive[i], vals[i], lead == i, t, fresh)

    def _numbers(self, t, fresh):
        return self.num_a[0] if fresh else self.num_a[min(len(self.num_a) - 1, int(round(t * FPS)))]

    def _card(self, ctx, font, R, i, alive, val, lead, t, fresh):
        d, col = self.hd[i], COLS[i]
        x, y, w, h = HX, AY + i * (CH + GAP), HW, CH
        live = alive or fresh
        won = not fresh and t >= self.E and i == self.winner
        dead_age = None if live else t - d["death_t"]
        dim = 1.0 if live else lerp(1.0, 0.0, ramp(dead_age, 0.1, 0.5))
        fl, dx, pop = 0.0, 0.0, None
        if not fresh:
            for sw in self.swings:
                age = t - sw["t"]
                if sw["head"] == i and 0 <= age < 1.0:
                    fl = max(fl, 1 - age / 0.45)
                    dx += 8 * math.sin(age * 70) * math.exp(-age / 0.10)
                    pop = (sw, age)
            if dead_age is not None and dead_age < 0.45:
                fl = max(fl, 1 - dead_age / 0.45)
                dx += 10 * math.sin(dead_age * 70) * math.exp(-dead_age / 0.12)
        ra = t - (self.T_RESET + 0.24)
        if 0 <= ra < 0.3:
            fl = max(fl, 0.7 * (1 - ra / 0.3))
        fl = max(0.0, fl)
        x += dx
        cc = mix(GREY, col, 0.18 + 0.82 * dim)
        hot = (lead and alive) or won
        for e in (11, 5):                                 # soft shadow
            chamfer(ctx, x - e, y - e + 8, w + 2 * e, h + 2 * e, 18 + e)
            ctx.set_source_rgba(0, 0, 0.02, 0.17)
            ctx.fill()
        chamfer(ctx, x, y, w, h, 18)
        g = cairo.LinearGradient(0, y, 0, y + h)
        g.add_color_stop_rgba(0, 0.055, 0.080, 0.145, 0.96)
        g.add_color_stop_rgba(1, 0.026, 0.036, 0.080, 0.96)
        ctx.set_source(g)
        ctx.fill_preserve()
        ctx.set_line_join(cairo.LINE_JOIN_MITER)
        if hot:
            pul = 0.75 + 0.25 * math.sin(t * (10 if won else 5))
            ctx.set_operator(ADD)
            ctx.set_source_rgba(*col, 0.15 * pul)
            ctx.set_line_width(18)
            ctx.stroke_preserve()
            ctx.set_source_rgba(*col, 0.30 * pul)
            ctx.set_line_width(9)
            ctx.stroke_preserve()
            ctx.set_operator(OVER)
        ctx.set_source_rgba(*mix(cc, WHITE, 0.8 * fl), 1.0 if hot else 0.8)
        ctx.set_line_width(4 if hot else 2.5)
        ctx.stroke()
        if fl > 0.01:
            chamfer(ctx, x, y, w, h, 18)
            ctx.set_source_rgba(*mix(col, WHITE, 0.5), 0.24 * fl)
            ctx.fill()
        tag = RULES[d["rule"]]
        na = self._numbers(t, fresh) if live else 1.0     # while the counters are equal a live card is just symbol and rule
        symbol_path(ctx, i, x + w / 2, y + lerp(70, 40, na), lerp(28, 19, na))
        neon(ctx, cc, 4.6, 1.0, dim)
        font.draw(ctx, tag, x + w / 2, y + lerp(178, 112, na), fit(font, tag, lerp(58, 50, na), w - 18),
                  mix(cc, WHITE, 0.45), anchor="c")
        if na <= 0.004:
            return
        base = y + 192
        if live:
            self._digits(ctx, font, val, x + w / 2, base, 74 * (1 + 0.14 * fl), mix(lighten(col, 0.74), WHITE, fl),
                         R["slot"], na)
        else:
            font.draw(ctx, "OUT", x + w / 2, base, 74, mix(cc, WHITE, 0.15 + 0.6 * fl), anchor="c", stroke=INK,
                      stroke_w=8)
        nseg = 14
        bw = (w - 28 - (nseg - 1) * 4) / nseg
        lit = min(nseg, math.ceil(nseg * math.sqrt(min(1.0, val / self.total0)))) if live and val > 0 else 0
        for q in range(nseg):
            ctx.rectangle(x + 14 + q * (bw + 4), y + 204, bw, 16)
            if q < lit:
                ctx.set_source_rgba(*mix(col, WHITE, 0.12 + 0.6 * fl), na)
            else:
                ctx.set_source_rgba(0.085, 0.115, 0.185, na)
            ctx.fill()
        sy = y + 230
        if pop:                                           # room just lost, in place of the status line
            sw, age = pop
            a = 1 - ramp(age, 0.7, 1.0)
            chamfer(ctx, x + 12, sy, w - 24, 46, 10)
            ctx.set_source_rgba(0.32, 0.03, 0.09, 0.92 * a)
            ctx.fill()
            font.draw(ctx, f"-{sw['before'] - sw['after']}", x + w / 2, sy + 37,
                      38 * (1 + 0.25 * max(0.0, 1 - age / 0.2)), (1.0, 0.45, 0.50), anchor="c", alpha=a)
        elif hot:
            chamfer(ctx, x + 12, sy, w - 24, 46, 10)
            ctx.set_source_rgba(*mix(col, WHITE, 0.10 + (0.25 + 0.25 * math.sin(t * 10) if won else 0.0)), na)
            ctx.fill()
            font.draw(ctx, "WINNER" if won else "MOST ROOM", x + w / 2, sy + 35, 33, INK, anchor="c", alpha=na)
        elif not live:
            font.draw(ctx, PLACE[d["place"]] + " PLACE", x + w / 2, sy + 35, 33, mix(cc, WHITE, 0.4), anchor="c")

    def _rule_tags(self, ctx, font, tau, z, paths):
        """Hook: each head is named by its rule in large type, tied to the head by a leader line."""
        a = ramp(tau, -0.34, -0.14) * (1 - ramp(tau, T_HOOK - 0.15, T_HOOK + 0.15))
        if a <= 0.004:
            return
        for i in range(4):
            hp = paths[i][1]
            if hp is None:
                continue
            col = COLS[i]
            bx, by = self.tagpos[i]
            lx, cy = ANCX + z * (bx - ANCX), ANCY + z * (by - ANCY)
            hx, hy = ANCX + z * (hp[0] - ANCX), ANCY + z * (hp[1] - ANCY)
            text = RULES[self.hd[i]["rule"]]
            w = font.width(text, 64)
            ctx.new_path()
            ctx.move_to(lx, cy - 44 if hy < cy else cy + 44)
            ctx.line_to(hx, hy)
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
            ctx.set_source_rgba(*INK, 0.8 * a)
            ctx.set_line_width(9)
            ctx.stroke_preserve()
            ctx.set_source_rgba(*lighten(col, 0.35), a)
            ctx.set_line_width(4)
            ctx.stroke()
            ctx.new_path()
            ctx.arc(hx, hy, 24, 0, TAU)
            ctx.set_source_rgba(*lighten(col, 0.35), a)
            ctx.set_line_width(4)
            ctx.stroke()
            ctx.new_path()
            chamfer(ctx, lx - w / 2 - 20, cy - 44, w + 40, 88, 14)
            ctx.set_source_rgba(*INK, 0.88 * a)
            ctx.fill_preserve()
            ctx.set_source_rgba(*col, a)
            ctx.set_line_width(3.5)
            ctx.stroke()
            font.draw(ctx, text, lx, cy + 64 * font.cap_height / 2, 64, lighten(col, 0.30), anchor="c", alpha=a)

    # ------------------------------------------------------------ titles
    @staticmethod
    def _line(ctx, font, text, x, base, size, fill, a, halo, anchor="c"):
        """Display line with a neon halo behind an ink outline."""
        if a <= 0.004:
            return
        ctx.new_path()
        font.path(ctx, text, x, base, size, anchor)
        ctx.set_operator(ADD)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_source_rgba(halo[0], halo[1], halo[2], 0.13 * a)
        ctx.set_line_width(34)
        ctx.stroke_preserve()
        ctx.set_source_rgba(halo[0], halo[1], halo[2], 0.16 * a)
        ctx.set_line_width(22)
        ctx.stroke()
        ctx.set_operator(OVER)
        font.draw(ctx, text, x, base, size, fill, anchor=anchor, alpha=a, stroke=INK, stroke_w=12, shadow=5)

    def _typed(self, ctx, font, text, x, base, size, fill, a, halo, u, col):
        """A line written from the left by a running trail: the text is revealed up to the trail's head."""
        if a <= 0.004 or u <= 0:
            return
        xh = x - 10 + (font.width(text, size) + 34) * u
        ctx.save()
        ctx.rectangle(0, base - size * 1.3, xh, size * 1.8)
        ctx.clip()
        self._line(ctx, font, text, x, base, size, fill, a, halo, anchor="l")
        ctx.restore()
        yl = base + 0.16 * size + 4
        ctx.new_path()
        ctx.move_to(x - 10, yl)
        ctx.line_to(xh, yl)
        ctx.set_line_cap(cairo.LINE_CAP_BUTT)
        ctx.set_operator(ADD)
        ctx.set_source_rgba(col[0], col[1], col[2], 0.16 * a)
        ctx.set_line_width(20)
        ctx.stroke_preserve()
        ctx.set_operator(OVER)
        ctx.set_source_rgba(col[0], col[1], col[2], a)
        ctx.set_line_width(7)
        ctx.stroke_preserve()
        ctx.set_source_rgba(1, 1, 1, 0.75 * a)
        ctx.set_line_width(2.2)
        ctx.stroke()
        ctx.rectangle(xh - 8, yl - 8, 16, 16)
        ctx.set_source_rgba(1, 1, 1, a)
        ctx.fill()
        ctx.set_operator(ADD)
        glow(ctx, xh, yl, 46, col, 0.5 * a)
        ctx.set_operator(OVER)

    def _titles(self, ctx, t, tau, font, k):
        E = self.E
        # opening question, written along two trails: the first line is complete on frame 0 (it is written
        # over the loop cut), the second one finishes a third of a second in
        if tau < T_HOOK + 0.4:
            a = 1 - ramp(tau, T_HOOK, T_HOOK + 0.3)
            dy = -40 * ease_in(ramp(tau, T_HOOK, T_HOOK + 0.3))
            l1, l2 = "WHICH LINE OUTLASTS", "THE OTHERS?"
            s1, s2 = fit(font, l1, 104, 800), fit(font, l2, 112, 800)
            self._typed(ctx, font, l1, 64, 248 + dy, s1, WHITE, a, (0.1, 0.6, 1.0), ramp(tau, -0.42, -0.10), COLS[0])
            fill = cairo.LinearGradient(64, 0, 64 + font.width(l2, s2), 0)
            for s_, c in zip((0.0, 0.34, 0.67, 1.0), COLS):
                c = lighten(c, 0.18)
                fill.add_color_stop_rgba(s_, c[0], c[1], c[2], a)
            self._typed(ctx, font, l2, 64, 388 + dy, s2, fill, a, (0.1, 0.6, 1.0), ramp(tau, -0.22, 0.30), COLS[1])
        # compact question above the arena during play
        q_end = self.duel[0] + 0.35 if self.duel else E + 0.15
        a = ramp(t, T_HOOK + 0.25, T_HOOK + 0.45) * (1 - ramp(t, q_end, q_end + 0.2))
        if a > 0.004:
            dy = 24 * (1 - ease_out(ramp(t, T_HOOK + 0.25, T_HOOK + 0.55))) - 20 * ease_in(ramp(t, q_end, q_end + 0.2))
            text = "WHICH LINE OUTLASTS?"
            self._line(ctx, font, text, CX_T, 226 + dy, fit(font, text, 78, 800), WHITE, a, (0.1, 0.6, 1.0))
        # last two: their room counters head to head, in place of the question
        if self.duel and t < E + 0.4:
            td, ia, ib = self.duel
            a = ramp(t, td + 0.55, td + 0.8) * (1 - ramp(t, E + 0.15, E + 0.35))
            if a > 0.004:
                va, vb = self.sim["reach"][k][ia], self.sim["reach"][k][ib]
                x0, x1, yb, hb = 256.0, 724.0, 170.0, 54.0
                xs = x0 + (x1 - x0) * clamp(va / (va + vb) if va + vb else 0.5, 0.05, 0.95)
                ctx.save()
                ctx.translate(0, 22 * (1 - ease_out(ramp(t, td + 0.55, td + 0.85))))
                chamfer(ctx, x0 - 8, yb - 8, x1 - x0 + 16, hb + 16, 12)
                ctx.set_source_rgba(*INK, 0.9 * a)
                ctx.fill_preserve()
                ctx.set_source_rgba(0.3, 0.8, 1.0, 0.7 * a)
                ctx.set_line_width(2.5)
                ctx.stroke()
                for xa, xb, i, v, o in ((x0, xs - 3, ia, va, vb), (xs + 3, x1, ib, vb, va)):
                    ctx.rectangle(xa, yb, max(0.0, xb - xa), hb)
                    ctx.set_source_rgba(*mix(COLS[i], WHITE, 0.14 if v > o else 0.0), (1.0 if v > o else 0.55) * a)
                    ctx.fill()
                ctx.rectangle(xs - 2.5, yb - 8, 5, hb + 16)
                polygon(ctx, [(xs, yb + hb / 2 - 15), (xs + 15, yb + hb / 2), (xs, yb + hb / 2 + 15), (xs - 15, yb + hb / 2)])
                ctx.set_source_rgba(1, 1, 1, a)
                ctx.fill()
                for i, v, o, sx_, tx, anc in ((ia, va, vb, 112, 146, "l"), (ib, vb, va, 868, 834, "r")):
                    symbol_path(ctx, i, sx_, yb + hb / 2, 19)
                    neon(ctx, COLS[i], 4.6, a)
                    font.draw(ctx, str(v), tx, yb + hb - 2, fit(font, str(v), 60, 96),
                              lighten(COLS[i], 0.6 if v > o else 0.15), anchor=anc, alpha=a, stroke=INK, stroke_w=8)
                ctx.restore()
        # result and the question back to the viewer, written the same way
        if t >= self.T_WIN:
            fade = 1 - ramp(t, self.T_RESET, self.T_RESET + 0.15)
            col = COLS[self.winner]
            text = f"{NAMES[self.winner]} OUTLASTS ALL!"
            size = fit(font, text, 94, 716)
            u = ramp(t, self.T_WIN, self.T_WIN + 0.40)
            if fade > 0.004:
                symbol_path(ctx, self.winner, 64 + 30, 246 - size * font.cap_height / 2, 27)
                neon(ctx, col, 7.0, fade * min(1.0, u * 6))
            self._typed(ctx, font, text, 64 + 84, 246, size, lighten(col, 0.18), fade, col, u, col)
            if t >= self.T_LINE:
                text = "WAS IT YOUR PICK?"
                self._typed(ctx, font, text, 64, 388, fit(font, text, 112, 800), WHITE, fade, (0.1, 0.6, 1.0),
                            ramp(t, self.T_LINE, self.T_LINE + 0.40), WHITE)

    # ------------------------------------------------------------ sound
    def audio(self):
        """Stereo float mix at au.SR. Each live line is a square-wave step at its own pitch, so the pattern
        thins from four voices to a duet to a solo; turns bend the note; a crash is a bit-crushed burst."""
        sim, sr = self.sim, au.SR
        rng = np.random.default_rng(self.seed + 515)
        rnd = random.Random(self.seed + 99)
        E, END, DUR = self.E, self.END, self.DUR
        n, tail = int(round(DUR * sr)), 3 * sr
        fx = np.zeros((n + tail, 2))             # steps and events (these get reverb)
        bed = np.zeros((n + tail, 2))            # drone, pulse, risers
        tt = lambda dur: np.arange(int(dur * sr)) / sr
        tick_t = [self.t_of_s(k) for k in range(END + 2)]
        cache = {}

        def square(m, bend=0.0, dur=0.13, decay=0.045):
            key = (m, bend, dur, decay)
            if key not in cache:
                t = tt(dur)
                f0 = au.hz(m)
                ph = 2 * np.pi * np.cumsum(f0 * 2 ** (bend / 12 * np.exp(-t / 0.028))) / sr
                x = sum(np.sin(q * ph) / q for q in range(1, 20, 2) if q * f0 * 1.3 < 15000)
                cache[key] = x * np.minimum(1, t / 0.002) * np.exp(-t / decay) * np.minimum(1, (dur - t) / 0.01)
            return cache[key]

        def crush(dur, hold, bits, decay):
            m = int(dur * sr)
            x = np.repeat(rng.uniform(-1, 1, m // hold + 1), hold)[:m]      # sample-and-hold noise
            x = np.round(x * 2 ** bits) / 2 ** bits                         # coarse quantiser
            return au.bandpass(x, 90, 9000) * np.exp(-tt(dur)[:m] / decay)

        pan_x = lambda px: clamp(((px - AX) / AW - 0.5) * 1.4, -0.8, 0.8)
        for i, d in enumerate(self.hd):
            last = d["death"] if d["death"] is not None else END + 1
            dirs, pts = d["dirs"], d["pts"]
            for j, k in enumerate(range(2 * i, last, PERIOD)):     # staggered: the lines interleave
                turned = any(dirs[q] != dirs[q - 1] for q in range(max(1, k - PERIOD + 1), min(k, len(dirs) - 1) + 1))
                m = VOICE[i] + (12 if j % 8 in (3, 6) else 0)
                au.place(fx, square(m, 3.0 if turned else 0.0), tick_t[k], 0.20, pan_x(pts[min(k, len(pts) - 1)][0]))

        e_ticks = [e[0] for e in sim["elims"]]
        e1, e2 = e_ticks[0], (e_ticks[-2] if len(e_ticks) > 1 else e_ticks[0])
        for b, k in enumerate(range(0, END, 16)):                  # pulse that thickens after every exit
            ph = (k >= e1) + (k >= e2)
            au.place(bed, au.bass(au.hz(50 if b % 4 != 3 else 48), 0.20), tick_t[k], 0.17 + 0.05 * ph)
            if ph >= 1:
                au.place(bed, au.kick(0.25), tick_t[k], 0.42)
            if ph == 2 and k + 8 <= END:
                au.place(bed, au.kick(0.25), tick_t[k + 8], 0.36)
                au.place(bed, au.bass(au.hz(50), 0.12), tick_t[k + 8], 0.20)
        for q, k in enumerate(range(4, END, 4)):
            if k >= e2 or (k >= e1 and q % 2 == 0):
                au.place(bed, au.hat(rng), tick_t[k], 0.10 if q % 2 == 0 else 0.065, 0.3 if q % 4 < 2 else -0.3)
        for m, g, pn in ((38, 0.10, 0.0), (45, 0.07, -0.3), (50, 0.06, 0.3)):          # low drone
            au.place(bed, au.pad(au.hz(m), E + 0.4, 0.8, 0.6), 0.0, g, pn)
        rl = 2.8                                                    # riser into the deciding crash
        au.place(bed, au.swept_noise(rng, rl, 300, 5000) * (tt(rl) / rl) ** 2, E - rl, 0.34)

        for ts in self.speedups:                                    # speed step: rising chirp
            au.place(fx, au.glide(520, 1560, 0.16, 0.09), ts, 0.20)
            au.place(fx, au.tick(2100, 0.05), ts, 0.12)
        for sw in self.swings:                                      # room cut off: falling zap
            pn = pan_x(self.hd[sw["head"]]["pts"][min(sw["tick"], len(self.hd[sw["head"]]["pts"]) - 1)][0])
            au.place(fx, au.glide(1500, 240, 0.24, 0.11), sw["t"], 0.26, pn)
            au.place(fx, crush(0.12, 14, 3, 0.04), sw["t"], 0.12, pn)
        for tc, x, y, i, big, _ in self.crashes:                    # crash: bit-crushed burst, the voice drops out
            au.place(fx, crush(0.5 if big else 0.36, 9, 3, 0.11 if big else 0.08), tc, 0.62 if big else 0.50, pan_x(x))
            au.place(bed, au.boom(rng, 1.3 if big else 0.6, 100, 40), tc, 0.45 if big else 0.26)
            au.place(fx, au.glide(au.hz(VOICE[i]) * 2, au.hz(VOICE[i]) / 4, 0.30, 0.12), tc, 0.22, pan_x(x))

        for j, m in enumerate((65, 69, 72, 77, 81)):                # result: rising F major arpeggio
            pn = (-0.5, -0.25, 0.0, 0.25, 0.5)[j]
            au.place(fx, square(m, 0.0, 0.55, 0.20), self.T_WIN + j * 0.075, 0.26, pn)
            au.place(fx, au.bell(au.hz(m + 12), 1.8), self.T_WIN + j * 0.075, 0.20, pn)
        for m, pn in ((41, 0.0), (53, -0.4), (57, 0.4), (60, 0.0)):
            au.place(bed, au.pad(au.hz(m), self.T_RESET + 0.3 - self.T_WIN, 0.05, 1.0), self.T_WIN, 0.10, pn)
        au.place(fx, au.glide(660, 990, 0.08, 0.05), self.T_LINE, 0.20)
        au.place(fx, au.glide(990, 1320, 0.10, 0.06), self.T_LINE + 0.09, 0.20)
        t, q = self.T_LINE + 0.4, 0
        while t < self.T_RESET:
            au.place(fx, au.bell(au.hz(rnd.choice((77, 79, 81, 84, 86, 89))), 0.8), t, 0.14 * 0.93 ** q,
                     rnd.uniform(-0.7, 0.7))
            t += rnd.choice((0.125, 0.125, 0.25))
            q += 1
        wl = DUR - self.T_RESET                                     # wipe whoosh into the loop point
        au.place(bed, au.swept_noise(rng, wl, 500, 6000) * (tt(wl) / wl) ** 2, self.T_RESET, 0.5)

        mix_ = au.reverb(fx, rng, decay=0.8, wet=0.13) + bed
        out = mix_[:n].copy()
        out[:tail] += mix_[n:]                    # tails past the end wrap to the start: seamless loop
        return au.eq(out, ((50, -8.0), (110, -5.0), (220, -4.0), (500, 0.0), (1200, 1.5), (3000, 4.0), (8000, 5.0), (16000, 3.0)))

    # ------------------------------------------------------------ upload metadata
    def metadata(self):
        title = "Which line outlasts the others? 4 trails, 4 rules, 1 grid"
        assert len(title) <= 60
        return dict(
            title=title,
            description=(
                "Four lines grow one cell at a time on a 36 x 64 grid and can never cross a trail or a wall. "
                "Each follows one simple rule: STRAIGHT turns only when blocked, HUGGER keeps a wall on one hand, "
                "SEEKER always heads for the larger open region, RANDOM turns when it feels like it. Every crash "
                "clears the board: all trails disappear and the lines that are left carry on from where they "
                f"are, faster (x{ROUND_K[1]:g} with three left, x{ROUND_K[2]:g} for the last two, plus "
                f"{round(100 * (STEP_K - 1))}% every three seconds). The last line still moving wins.\n"
                "The counters show how many cells each line can still reach. The run is a real simulation picked "
                f"from a seed search (seed {self.seed}) and drawn and scored entirely by code, with no footage, "
                "samples or stock media.\n"
                "Was it your pick?"),
            tags=["simulation", "prediction", "algorithm battle", "trail simulation", "last one standing",
                  "satisfying", "generative animation", "made with code"],
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylised 2D animation of abstract lines on a grid; nothing "
                                              "realistic that could be mistaken for real people, places or events.",
            seed=self.seed,
            drama=self.stats,
        )
