"""TERRITORY BATTLE: four balls paint a tiled arena; most tiles when the timer ends wins.

simulate(seed) is deterministic and needs no rendering, so make.py can score hundreds of
seeds for drama (drama()) and render only the best one (Scene).
"""
import math
import random
from array import array

W, H, FPS = 1080, 1920, 30
PLAY = 24.0                  # battle length, seconds (12 bars at 120 BPM)
T_REVEAL = 25.0              # winner reveal (after a one-second count)
T_LINE = 25.6                # "DID YOU CALL IT?"
T_RESET = 27.3               # board rewinds so the last frame cuts back to the first
DUR = 28.0
NF = int(DUR * FPS)
NPLAY = int(PLAY * FPS)
SUB = 4                      # physics substeps per frame
DT = 1.0 / (FPS * SUB)

A = 810.0                    # arena side in arena units (= pixels at zoom 1)
N = 18                       # tiles per side
TILE = A / N
CORE = TILE * 0.25           # a tile is painted once the ball overlaps its middle half
R0 = 30.0
SPEED = 450.0
GROW_R, GROW_T = 50.0, 5.0
FAST_K, FAST_T = 1.75, 4.5
BOMB_R, BOMB_V = 3.3 * TILE, 700.0   # paint-bomb radius and ripple speed
PICK_R, PICK_LIFE = 26.0, 5.5
KINDS = ("grow", "fast", "bomb")


# ====================================================================== simulation
def simulate(seed, record=False):
    rng = random.Random(seed)
    spots = [(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)]
    rng.shuffle(spots)
    bx = [s[0] * A for s in spots]
    by = [s[1] * A for s in spots]
    vx, vy = [], []
    for _ in range(4):
        a = rng.randrange(4) * math.pi / 2 + rng.uniform(0.35, 1.22)
        vx.append(SPEED * math.cos(a))
        vy.append(SPEED * math.sin(a))
    rad = [R0] * 4
    grow_until = [-1.0] * 4
    fast_until = [-1.0] * 4
    owner = [-1] * (N * N)
    prev = [-1] * (N * N)
    ptime = [-9.0] * (N * N)
    counts = [0] * 4

    kinds = list(KINDS) + [rng.choice(KINDS), rng.choice(KINDS)]
    rng.shuffle(kinds)
    sched = [(3.5 + 4.1 * k + rng.uniform(-0.7, 0.7), kinds[k]) for k in range(5)]
    picks = []        # dict(kind, x, y, t0, t1, by)   t1 = time collected or expired
    live = []
    pending = []      # delayed bomb paints (t, tile, ball)
    walls, hits = [], []
    last_hit = {}
    counts_hist = []
    rec = dict(owner=[], prev=[], ptime=[], pos=[]) if record else None

    def paint(idx, b, t):
        o = owner[idx]
        if o == b:
            return
        if o >= 0:
            counts[o] -= 1
        counts[b] += 1
        prev[idx], owner[idx], ptime[idx] = o, b, t

    def snapshot():
        counts_hist.append(tuple(counts))
        if record:
            rec["owner"].append(bytes(o + 1 for o in owner))
            rec["prev"].append(bytes(o + 1 for o in prev))
            rec["ptime"].append(array("f", ptime))

    def wall(i, nx, ny, t):
        vn = vx[i] * nx + vy[i] * ny
        vx[i] -= 2 * vn * nx
        vy[i] -= 2 * vn * ny
        sp = math.hypot(vx[i], vy[i])
        a = math.atan2(vy[i], vx[i]) + rng.uniform(-0.14, 0.14)   # breaks up periodic orbits
        if abs(math.cos(a)) < 0.24 or abs(math.sin(a)) < 0.24:     # no dull axis-parallel shuttling
            diag = (math.floor(a / (math.pi / 2)) + 0.5) * (math.pi / 2)
            a += 0.22 if diag > a else -0.22
        vx[i], vy[i] = sp * math.cos(a), sp * math.sin(a)
        vn = vx[i] * nx + vy[i] * ny
        if vn < 0:
            vx[i] -= 2 * vn * nx
            vy[i] -= 2 * vn * ny
        walls.append((t, i, bx[i] - nx * rad[i], by[i] - ny * rad[i], nx, ny))

    for step in range(NPLAY * SUB):
        t = step * DT
        if step % SUB == 0:
            snapshot()
        if record:
            rec["pos"].append((bx[0], by[0], rad[0], bx[1], by[1], rad[1],
                               bx[2], by[2], rad[2], bx[3], by[3], rad[3]))
        # pickups appear / expire
        if sched and t >= sched[0][0]:
            _, kind = sched.pop(0)
            for _try in range(60):
                px, py = rng.uniform(0.16, 0.84) * A, rng.uniform(0.16, 0.84) * A
                if all(math.hypot(px - bx[i], py - by[i]) > 150 for i in range(4)) and \
                        all(math.hypot(px - p["x"], py - p["y"]) > 150 for p in live):
                    break
            p = dict(kind=kind, x=px, y=py, t0=t, t1=None, by=-1)
            picks.append(p)
            live.append(p)
        for p in live[:]:
            if t - p["t0"] > PICK_LIFE:
                p["t1"] = t
                live.remove(p)
        while pending and pending[0][0] <= t:
            _, idx, b = pending.pop(0)
            paint(idx, b, t)

        order = [(step + k) % 4 for k in range(4)]      # rotate so no colour has priority
        for i in order:
            rt = GROW_R if t < grow_until[i] else R0
            rad[i] += (rt - rad[i]) * min(1.0, DT * 8)
            sp = math.hypot(vx[i], vy[i])
            target = SPEED * (FAST_K if t < fast_until[i] else 1.0)
            k = 1 + (target / sp - 1) * min(1.0, DT * 5)
            vx[i] *= k
            vy[i] *= k
            bx[i] += vx[i] * DT
            by[i] += vy[i] * DT
            r = rad[i]
            if bx[i] < r:
                bx[i] = r
                if vx[i] < 0:
                    wall(i, 1, 0, t)
            elif bx[i] > A - r:
                bx[i] = A - r
                if vx[i] > 0:
                    wall(i, -1, 0, t)
            if by[i] < r:
                by[i] = r
                if vy[i] < 0:
                    wall(i, 0, 1, t)
            elif by[i] > A - r:
                by[i] = A - r
                if vy[i] > 0:
                    wall(i, 0, -1, t)
        # ball-ball: elastic, mass ~ area
        for i in range(3):
            for j in range(i + 1, 4):
                dx, dy = bx[j] - bx[i], by[j] - by[i]
                d = math.hypot(dx, dy)
                rs = rad[i] + rad[j]
                if d >= rs or d < 1e-6:
                    continue
                nx, ny = dx / d, dy / d
                mi, mj = rad[i] ** 2, rad[j] ** 2
                rel = (vx[i] - vx[j]) * nx + (vy[i] - vy[j]) * ny
                if rel > 0:
                    imp = 2 * rel / (mi + mj)
                    vx[i] -= imp * mj * nx
                    vy[i] -= imp * mj * ny
                    vx[j] += imp * mi * nx
                    vy[j] += imp * mi * ny
                    if t - last_hit.get((i, j), -1) > 0.12:
                        hits.append((t, i, j, bx[i] + nx * rad[i], by[i] + ny * rad[i], rel))
                    last_hit[(i, j)] = t
                push = rs - d
                bx[i] -= nx * push * mj / (mi + mj)
                by[i] -= ny * push * mj / (mi + mj)
                bx[j] += nx * push * mi / (mi + mj)
                by[j] += ny * push * mi / (mi + mj)
        for i in order:
            r = rad[i]
            bx[i] = min(max(bx[i], r), A - r)
            by[i] = min(max(by[i], r), A - r)
            # pickups
            for p in live[:]:
                if math.hypot(bx[i] - p["x"], by[i] - p["y"]) < r + PICK_R:
                    p["t1"], p["by"] = t, i
                    live.remove(p)
                    if p["kind"] == "grow":
                        grow_until[i] = t + GROW_T
                    elif p["kind"] == "fast":
                        fast_until[i] = t + FAST_T
                    else:
                        for ty in range(N):
                            for tx in range(N):
                                dd = math.hypot((tx + 0.5) * TILE - p["x"], (ty + 0.5) * TILE - p["y"])
                                if dd <= BOMB_R:
                                    pending.append((t + dd / BOMB_V, ty * N + tx, i))
                        pending.sort()
            # paint
            x, y = bx[i], by[i]
            r2 = r * r
            for ty in range(max(0, int((y - r) / TILE)), min(N - 1, int((y + r) / TILE)) + 1):
                ddy = abs(y - (ty + 0.5) * TILE) - CORE
                if ddy < 0:
                    ddy = 0.0
                for tx in range(max(0, int((x - r) / TILE)), min(N - 1, int((x + r) / TILE)) + 1):
                    ddx = abs(x - (tx + 0.5) * TILE) - CORE
                    if ddx < 0:
                        ddx = 0.0
                    if ddx * ddx + ddy * ddy <= r2:
                        paint(ty * N + tx, i, t)
    for _, idx, b in pending:            # a bomb ripple still spreading at the buzzer does not count
        pass
    snapshot()
    if record:
        rec["pos"].append((bx[0], by[0], rad[0], bx[1], by[1], rad[1],
                           bx[2], by[2], rad[2], bx[3], by[3], rad[3]))
    for p in picks:
        if p["t1"] is None:
            p["t1"] = PLAY
    out = dict(seed=seed, counts=counts_hist, picks=picks, walls=walls, hits=hits,
               start=[(s[0] * A, s[1] * A) for s in spots],
               grow=[], fast=[])
    for p in picks:
        if p["by"] >= 0 and p["kind"] in ("grow", "fast"):
            out[p["kind"]].append((p["t1"], p["by"]))
    if record:
        out.update(rec)
    return out


# ====================================================================== drama scoring
def drama(sim):
    """Score one simulated run. Higher = more worth watching."""
    counts = sim["counts"]
    hold = int(0.4 * FPS)                 # a lead only counts once it has lasted 0.4 s
    leaders = []
    cur = -1
    for c in counts:
        m = max(c)
        if m > 0 and (cur < 0 or c[cur] < m):
            cur = c.index(m)
        leaders.append(cur)
    runs = []                             # [leader, first_frame, length]
    for f, l in enumerate(leaders):
        if runs and runs[-1][0] == l:
            runs[-1][2] += 1
        else:
            runs.append([l, f, 1])
    solid = []
    for r in runs:
        if r[0] < 0 or (r[2] < hold and r is not runs[-1]):
            continue
        if solid and solid[-1][0] == r[0]:
            continue
        solid.append(r)
    final = counts[-1]
    rank = sorted(range(4), key=lambda i: -final[i])
    winner = rank[0]
    margin = final[rank[0]] - final[rank[1]]
    lead_changes = max(0, len(solid) - 1)
    half_leader = leaders[NPLAY // 2]
    last_take = solid[-1][1] / FPS if solid and solid[-1][0] == winner else PLAY
    low = min(min(c) for c in counts[6 * FPS:]) / (N * N)     # weakest share after 6 s
    biggest_rival_lead = max(max(c[i] for i in range(4) if i != winner) - c[winner] for c in counts[4 * FPS:])
    score = 0.0
    score += 1.0 * min(lead_changes, 8)
    score += 4.0 * math.exp(-(margin - 1) / 5.0) if margin >= 1 else -50.0
    score += 3.0 if half_leader != winner else 0.0
    score += 2.5 * min(1.0, max(0.0, (last_take - 15.0) / 7.0))        # winner takes over late
    score += 1.0 * (len({r[0] for r in solid}) - 1)                    # how many colours ever led
    score += 2.0 * min(1.0, max(0, biggest_rival_lead) / 30.0)         # winner came from behind
    score -= 40.0 * max(0.0, 0.10 - low)                               # nobody gets wiped out
    score += 0.5 * sum(1 for p in sim["picks"] if p["by"] >= 0)        # pickups actually matter
    return dict(
        seed=sim["seed"], score=round(score, 3), winner=winner, final_tiles=list(final),
        final_percent=[round(100 * c / (N * N), 2) for c in final],
        margin_tiles=margin, lead_changes=lead_changes,
        leader_at_half=half_leader, winner_led_at_half=half_leader == winner,
        winner_took_final_lead_at_s=round(last_take, 2),
        colours_that_led=len({r[0] for r in solid}),
        lowest_share_after_6s_percent=round(100 * low, 2),
        biggest_deficit_winner_overcame_tiles=max(0, biggest_rival_lead),
        pickups_collected=[(p["kind"], p["by"], round(p["t1"], 2)) for p in sim["picks"] if p["by"] >= 0],
    )


def score_seed(seed):
    return drama(simulate(seed))
