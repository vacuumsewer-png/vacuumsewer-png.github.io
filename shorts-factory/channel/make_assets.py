#!/usr/bin/env python3
"""Render the Predixel channel identity assets with code.

    python3 make_assets.py                    # writes avatar.png, banner.png, watermark.png here
    python3 make_assets.py --preview-dir DIR  # also writes small review images into DIR

Outputs (all next to this file):
    avatar.png     800x800    profile picture; YouTube shows it as a circle
    banner.png     2560x1440  everything essential is inside the central 1546x423 safe area
    watermark.png  150x150    transparent video watermark

Needs pycairo and fonttools (Pillow only for --preview-dir). Text is drawn from the glyph
outlines of the two OFL fonts in ./fonts, so the result does not depend on installed fonts.
Everything is deterministic: same code, same pixels.

The idea behind the pictures: coral is the ball and where it has been (solid), teal is
where you think it will go (dotted). The avatar is a letter p whose bowl is that path.
"""
import argparse
import math
import os

import cairo
from fontTools.pens.basePen import BasePen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.ttLib import TTFont

HERE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(HERE, "fonts")
TAU = math.tau


def rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def mix(a, b, k):
    return tuple(a[i] + (b[i] - a[i]) * k for i in range(3))


# ------------------------------------------------------------------ palette
INK = rgb("#0A0E1F")      # background
NAVY = rgb("#182048")     # lit part of the background
PAPER = rgb("#F5F2EA")    # letters and the mark
CORAL = rgb("#FF5C4D")    # the ball, the past
TEAL = rgb("#2FE0C8")     # the prediction, the future
AMBER = rgb("#FFC247")    # rare highlight (kept for thumbnails and end cards; unused in these three files)
SLATE = rgb("#7C86A8")    # grid, quiet text


# ------------------------------------------------------------------ text from glyph outlines
class _CairoPen(BasePen):
    def __init__(self, glyphset, ctx, ox, oy, scale):
        super().__init__(glyphset)
        self.ctx, self.ox, self.oy, self.s = ctx, ox, oy, scale

    def _p(self, p):
        return self.ox + p[0] * self.s, self.oy - p[1] * self.s

    def _moveTo(self, p):
        self.ctx.move_to(*self._p(p))

    def _lineTo(self, p):
        self.ctx.line_to(*self._p(p))

    def _curveToOne(self, a, b, c):
        self.ctx.curve_to(*self._p(a), *self._p(b), *self._p(c))

    def _closePath(self):
        self.ctx.close_path()

    def _endPath(self):
        pass


class Font:
    def __init__(self, filename):
        self.tt = TTFont(os.path.join(FONT_DIR, filename))
        self.gs = self.tt.getGlyphSet()
        self.cmap = self.tt.getBestCmap()
        self.upem = self.tt["head"].unitsPerEm
        self.xheight = self.tt["OS/2"].sxHeight
        self.ascender = self.bounds("l")[3]

    def bounds(self, glyph_or_char):
        g = self.cmap.get(ord(glyph_or_char), glyph_or_char) if len(glyph_or_char) == 1 else glyph_or_char
        bp = BoundsPen(self.gs)
        self.gs[g].draw(bp)
        return bp.bounds

    def layout(self, text, size, tracking=0.0, kern=None):
        """[(glyph name, x offset in px)], total width in px. tracking and kern are in em."""
        s = size / self.upem
        x, out = 0.0, []
        for i, ch in enumerate(text):
            if kern and i:
                x += kern.get(text[i - 1] + ch, 0.0) * size
            g = self.cmap[ord(ch)]
            out.append((g, x))
            x += self.gs[g].width * s + tracking * size
        return out, x - tracking * size

    def add_path(self, ctx, glyphs, x, y, size):
        s = size / self.upem
        for g, gx in glyphs:
            self.gs[g].draw(_CairoPen(self.gs, ctx, x + gx, y, s))

    def draw(self, ctx, text, x, y, size, color, alpha=1.0, tracking=0.0, align="left"):
        glyphs, w = self.layout(text, size, tracking)
        if align == "center":
            x -= w / 2
        elif align == "right":
            x -= w
        ctx.new_path()
        self.add_path(ctx, glyphs, x, y, size)
        ctx.set_source_rgba(*color, alpha)
        ctx.fill()
        return w


# ------------------------------------------------------------------ shared pieces
def background(ctx, w, h, cx, cy, radius):
    """Ink with one soft navy light, so flat areas never look empty or banded."""
    ctx.set_source_rgb(*INK)
    ctx.paint()
    g = cairo.RadialGradient(cx, cy, 0, cx, cy, radius)
    g.add_color_stop_rgba(0.0, *NAVY, 1.0)
    g.add_color_stop_rgba(0.55, *mix(NAVY, INK, 0.6), 1.0)
    g.add_color_stop_rgba(1.0, *INK, 1.0)
    ctx.set_source(g)
    ctx.paint()


def dot_grid(ctx, w, h, step, r, alpha, cx, cy, fade_radius, hole=None):
    """Graph-paper dots that fade away from (cx, cy). hole=(x0, y0, x1, y1, soft) keeps an area calm."""
    y = (h % step) / 2
    while y < h:
        x = (w % step) / 2
        while x < w:
            a = alpha * max(0.0, 1.0 - math.hypot(x - cx, y - cy) / fade_radius)
            if hole:
                x0, y0, x1, y1, soft = hole
                dx = max(x0 - x, 0.0, x - x1)
                dy = max(y0 - y, 0.0, y - y1)
                a *= min(1.0, math.hypot(dx, dy) / soft)
            if a > 0.004:
                ctx.arc(x, y, r, 0, TAU)
                ctx.set_source_rgba(*SLATE, a)
                ctx.fill()
            x += step
        y += step


def glow(ctx, x, y, r, color, alpha):
    g = cairo.RadialGradient(x, y, r * 0.6, x, y, r * 3.2)
    g.add_color_stop_rgba(0.0, *color, alpha)
    g.add_color_stop_rgba(1.0, *color, 0.0)
    ctx.set_source(g)
    ctx.arc(x, y, r * 3.2, 0, TAU)
    ctx.fill()


def ball(ctx, x, y, r, color=CORAL):
    g = cairo.RadialGradient(x - r * 0.35, y - r * 0.4, r * 0.05, x, y, r)
    g.add_color_stop_rgb(0.0, *mix(color, (1, 1, 1), 0.38))
    g.add_color_stop_rgb(0.55, *color)
    g.add_color_stop_rgb(1.0, *mix(color, (0, 0, 0), 0.14))
    ctx.set_source(g)
    ctx.arc(x, y, r, 0, TAU)
    ctx.fill()


# Mark geometry in a 1000-unit design box. The bowl of the p is a circular path: solid
# where the ball has been, three shrinking dots where it is predicted to go.
MARK_W = 150            # stroke width
MARK_R = 215            # bowl radius (to the middle of the stroke)
MARK_C = (560, 380)     # bowl centre
MARK_FOOT = 860         # bottom of the stem
MARK_BALL_DEG = 60      # ball position, degrees clockwise from 3 o'clock
MARK_BALL_R = 98
MARK_GAP = 28           # clear ring around the ball
MARK_DOTS = ((104, 33), (127, 27), (148, 21))  # (degrees, radius) of the predicted dots
MARK_BOX = (270, 90, 850, 860)                 # ink bounds, used to centre the mark


def draw_mark(ctx, cx, cy, size, fg=PAPER, ball_color=CORAL, future=TEAL, glow_alpha=0.0):
    """Draw the p-with-a-path mark centred on (cx, cy); size is the height of the 1000-unit box."""
    u = size / 1000
    bx = (MARK_BOX[0] + MARK_BOX[2]) / 2
    by = (MARK_BOX[1] + MARK_BOX[3]) / 2
    ox, oy = cx - bx * u, cy - by * u
    ccx, ccy = ox + MARK_C[0] * u, oy + MARK_C[1] * u
    r, w = MARK_R * u, MARK_W * u
    sx = ccx - r
    a_ball = math.radians(MARK_BALL_DEG)
    bxp, byp = ccx + r * math.cos(a_ball), ccy + r * math.sin(a_ball)

    if glow_alpha > 0:
        glow(ctx, bxp, byp, MARK_BALL_R * u, ball_color, glow_alpha)

    ctx.push_group()
    ctx.set_line_width(w)
    ctx.set_source_rgb(*fg)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.move_to(sx, ccy - r)                       # stem
    ctx.line_to(sx, oy + MARK_FOOT * u - w / 2)
    ctx.stroke()
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    ctx.move_to(sx, ccy - r)                       # flat shoulder: stem and bowl share one top edge
    ctx.line_to(ccx, ccy - r)
    ctx.stroke()
    ctx.arc(ccx, ccy, r, math.pi, TAU + a_ball)    # the path travelled so far
    ctx.stroke()
    ctx.set_operator(cairo.OPERATOR_DEST_OUT)      # clear ring around the ball
    ctx.arc(bxp, byp, (MARK_BALL_R + MARK_GAP) * u, 0, TAU)
    ctx.fill()
    ctx.set_operator(cairo.OPERATOR_OVER)
    ctx.pop_group_to_source()
    ctx.paint()

    for deg, dr in MARK_DOTS:                      # the predicted rest of the path
        a = math.radians(deg)
        ctx.arc(ccx + r * math.cos(a), ccy + r * math.sin(a), dr * u, 0, TAU)
        ctx.set_source_rgb(*future)
        ctx.fill()
    ball(ctx, bxp, byp, MARK_BALL_R * u, ball_color)


# ------------------------------------------------------------------ avatar
def make_avatar(path, size=800):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
    ctx = cairo.Context(surf)
    c = size / 2
    background(ctx, size, size, c * 0.92, c * 0.80, size * 0.78)
    dot_grid(ctx, size, size, step=size / 20, r=size / 380, alpha=0.30, cx=c, cy=c, fade_radius=size * 0.56)
    draw_mark(ctx, c + size * 0.010, c + size * 0.004, size * 0.74, glow_alpha=0.26)
    surf.write_to_png(path)


# ------------------------------------------------------------------ watermark
def make_watermark(path, size=150):
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
    ctx = cairo.Context(surf)
    c = size / 2
    # Soft dark halo first, so the light mark still reads over bright video frames.
    halo = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
    hctx = cairo.Context(halo)
    draw_mark(hctx, c, c, size * 1.02, fg=INK, ball_color=INK, future=INK)
    for k in range(16):                            # even ring of copies = a soft outline
        a = TAU * k / 16
        ctx.set_source_surface(halo, 2.6 * math.cos(a), 2.6 * math.sin(a))
        ctx.paint_with_alpha(0.085)
    ctx.set_source_surface(halo, 0, 3.2)           # a little more weight underneath
    ctx.paint_with_alpha(0.14)
    draw_mark(ctx, c, c, size * 1.02)
    surf.write_to_png(path)


# ------------------------------------------------------------------ banner
BANNER_W, BANNER_H = 2560, 1440
SAFE_W, SAFE_H = 1546, 423
SAFE = ((BANNER_W - SAFE_W) / 2, (BANNER_H - SAFE_H) / 2, (BANNER_W + SAFE_W) / 2, (BANNER_H + SAFE_H) / 2)


def hop(p0, p1, apex_y, n=56):
    """One ballistic hop from p0 to p1 whose highest point is exactly apex_y (y grows downward).

    Constant horizontal speed and constant gravity give y = y0 + d*t - 4*L*t*(1-t); the lift L that
    puts the top of that parabola at apex_y is (sqrt(h0) + sqrt(h1))^2 / 4, h being each end's drop.
    """
    h0, h1 = max(1e-6, p0[1] - apex_y), max(1e-6, p1[1] - apex_y)
    lift = (math.sqrt(h0) + math.sqrt(h1)) ** 2 / 4
    pts = []
    for i in range(n + 1):
        t = i / n
        pts.append((p0[0] + (p1[0] - p0[0]) * t,
                    p0[1] + (p1[1] - p0[1]) * t - 4 * lift * t * (1 - t)))
    return pts


def chain(stops, apexes):
    pts = []
    for (a, b), apex_y in zip(zip(stops, stops[1:]), apexes):
        seg = hop(a, b, apex_y)
        pts.extend(seg if not pts else seg[1:])
    return pts


def arclen(pts):
    out = [0.0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        out.append(out[-1] + math.hypot(x1 - x0, y1 - y0))
    return out


def dots_along(ctx, pts, color, spacing, radius, alpha_at, skip_start=0.0):
    s = arclen(pts)
    d, i = skip_start, 0
    while d <= s[-1]:
        while s[i + 1] < d:
            i += 1
        k = (d - s[i]) / max(1e-9, s[i + 1] - s[i])
        x = pts[i][0] + (pts[i + 1][0] - pts[i][0]) * k
        y = pts[i][1] + (pts[i + 1][1] - pts[i][1]) * k
        a = alpha_at(x)
        if a > 0.003:
            ctx.arc(x, y, radius, 0, TAU)
            ctx.set_source_rgba(*color, a)
            ctx.fill()
        d += spacing


def make_banner(path):
    W, H = BANNER_W, BANNER_H
    surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    ctx = cairo.Context(surf)
    cx, cy = W / 2, H / 2
    background(ctx, W, H, cx, cy - 40, W * 0.62)

    display = Font("InterDisplay-ExtraBold.otf")
    mono = Font("JetBrainsMono-Medium.ttf")

    # ---- wordmark layout (lowercase, dotless i: the ball is the dot)
    size = 236
    glyphs, ww = display.layout("predıxel", size, tracking=-0.012)
    s = size / display.upem
    base = cy + 60                       # baseline
    x0 = cx - ww / 2
    xh = base - display.xheight * s      # top of the small letters
    asc = base - display.ascender * s    # top of d and l
    ceiling = SAFE[1] + 30               # nothing rises above this, so no crop cuts an arc

    def box(i):                          # left and right ink edge of letter i, in pixels
        g, gx = glyphs[i]
        b = display.bounds(g)
        return x0 + gx + b[0] * s, x0 + gx + b[2] * s

    mid = [sum(box(i)) / 2 for i in range(len(glyphs))]
    ball_r = 0.118 * size
    ball_x = mid[4]
    ball_y = xh - 0.085 * size - ball_r
    lp, rl = box(0)[0], box(7)[1]

    # ---- quiet graph-paper field, kept away from the lockup
    dot_grid(ctx, W, H, step=40, r=2.3, alpha=0.34, cx=cx, cy=cy, fade_radius=W * 0.60,
             hole=(lp - 40, asc - 120, rl + 170, base + 150, 240))

    # ---- floor line with ruler ticks, interrupted by the word
    def edge_fade(x):                    # 0 at the canvas edges, 1 across the middle
        return max(0.0, min(1.0, x / 420, (W - x) / 420))

    def under_word(x):
        return lp - 34 < x < rl + 34

    ctx.set_line_width(2)
    ctx.set_line_cap(cairo.LINE_CAP_BUTT)
    for x in range(0, W, 8):
        if not under_word(x + 4):
            ctx.move_to(x, base)
            ctx.line_to(x + 8, base)
            ctx.set_source_rgba(*SLATE, 0.30 * edge_fade(x))
            ctx.stroke()
    for x in range(20, W, 40):
        if under_word(x):
            continue
        tall = (x - 20) % 200 == 0
        ctx.move_to(x, base)
        ctx.line_to(x, base + (17 if tall else 8))
        ctx.set_source_rgba(*SLATE, (0.42 if tall else 0.24) * edge_fade(x))
        ctx.stroke()

    # ---- the path so far: in along the floor, up onto the letters, to rest as the dot of the i.
    # Every point is the centre of a ball of radius ball_r that touches a surface.
    floor_y = base - ball_r
    top_y = xh - ball_r
    over_tall = asc - ball_r - 16        # apex that clears d and l
    step = 292
    start = lp - 150
    past = chain(
        [(start - 3 * step, floor_y), (start - 2 * step, floor_y), (start - step, floor_y), (start, floor_y),
         (mid[0] + 4, top_y), (mid[1] + 6, top_y), (mid[2], top_y), (ball_x, ball_y)],
        [ceiling + 2, ceiling + 5, ceiling + 9, ceiling + 6, top_y - 50, top_y - 44, over_tall])

    # ---- the prediction: across x and e, over the l, down to the floor and away
    land = (rl + 132, floor_y)
    future = chain(
        [(ball_x, ball_y), (mid[5], top_y), (mid[6], top_y), land,
         (land[0] + 262, floor_y), (land[0] + 262 + 232, floor_y), (land[0] + 262 + 232 + 206, floor_y)],
        [ball_y - 26, top_y - 40, over_tall, floor_y - 132, floor_y - 100, floor_y - 76])

    grad = cairo.LinearGradient(-60, 0, ball_x, 0)
    grad.add_color_stop_rgba(0.00, *CORAL, 0.0)
    grad.add_color_stop_rgba(0.30, *CORAL, 0.30)
    grad.add_color_stop_rgba(0.62, *CORAL, 0.62)
    grad.add_color_stop_rgba(1.00, *CORAL, 0.96)
    ctx.set_source(grad)
    ctx.set_line_width(6.5)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    ctx.move_to(*past[0])
    for pt in past[1:]:
        ctx.line_to(*pt)
    ctx.stroke()

    ctx.push_group()
    dots_along(ctx, future, TEAL, spacing=26, radius=5.4,
               alpha_at=lambda x: 0.28 + 0.68 * max(0.0, min(1.0, (W - x) / (W - ball_x))) ** 1.6,
               skip_start=ball_r + 22)
    ctx.set_operator(cairo.OPERATOR_DEST_OUT)      # keep the landing ring clear of dots
    ctx.arc(land[0], land[1], ball_r + 13, 0, TAU)
    ctx.fill()
    ctx.set_operator(cairo.OPERATOR_OVER)
    ctx.pop_group_to_source()
    ctx.paint()

    # predicted landing spot: a hollow ball where the dotted path first meets the floor
    ctx.set_line_width(5)
    ctx.set_source_rgba(*TEAL, 0.96)
    ctx.arc(land[0], land[1], ball_r - 2.5, 0, TAU)
    ctx.stroke()
    q = "?"
    qg, qw = mono.layout(q, 34)
    qb = mono.bounds(q)
    qs = 34 / mono.upem
    ctx.new_path()
    mono.add_path(ctx, qg, land[0] - (qb[0] + qb[2]) / 2 * qs, land[1] + (qb[1] + qb[3]) / 2 * qs, 34)
    ctx.set_source_rgba(*TEAL, 0.96)
    ctx.fill()

    # ---- wordmark
    glow(ctx, ball_x, ball_y, ball_r, CORAL, 0.30)
    ctx.new_path()
    display.add_path(ctx, glyphs, x0, base, size)
    ctx.set_source_rgb(*PAPER)
    ctx.fill()
    ball(ctx, ball_x, ball_y, ball_r, CORAL)

    # ---- tagline, centred as one block under the word
    tsize, track, gap, dot_r = 38, 0.17, 30, 5.2
    left, right = "YOU PREDICT", "THE CODE DECIDES"
    _, wl = mono.layout(left, tsize, track)
    _, wr = mono.layout(right, tsize, track)
    tx = cx - (wl + wr + 2 * gap + 2 * dot_r) / 2
    tag_y = base + 0.21 * size + 64
    mono.draw(ctx, left, tx, tag_y, tsize, TEAL, 0.96, tracking=track)
    ctx.arc(tx + wl + gap + dot_r, tag_y - 0.365 * tsize, dot_r, 0, TAU)
    ctx.set_source_rgb(*CORAL)
    ctx.fill()
    mono.draw(ctx, right, tx + wl + 2 * gap + 2 * dot_r, tag_y, tsize, PAPER, 0.88, tracking=track)

    # ---- soft vignette for the TV crop, which shows the whole canvas
    g = cairo.RadialGradient(cx, cy, H * 0.55, cx, cy, W * 0.72)
    g.add_color_stop_rgba(0.0, *INK, 0.0)
    g.add_color_stop_rgba(1.0, 0.02, 0.03, 0.07, 0.55)
    ctx.set_source(g)
    ctx.paint()
    surf.write_to_png(path)


# ------------------------------------------------------------------ previews (review aid only)
def make_previews(out_dir):
    from PIL import Image, ImageDraw
    os.makedirs(out_dir, exist_ok=True)
    av = Image.open(os.path.join(HERE, "avatar.png")).convert("RGBA")
    sheet = Image.new("RGB", (860, 420), (15, 15, 15))
    ImageDraw.Draw(sheet).rectangle([430, 0, 860, 420], fill=(255, 255, 255))
    for col, bg in ((0, (15, 15, 15)), (430, (255, 255, 255))):
        x = col + 24
        for d in (176, 98, 48, 32, 24):
            im = av.resize((d, d), Image.LANCZOS)
            mask = Image.new("L", (d * 4, d * 4), 0)
            ImageDraw.Draw(mask).ellipse([0, 0, d * 4 - 1, d * 4 - 1], fill=255)
            sheet.paste(im, (x, 40), mask.resize((d, d), Image.LANCZOS))
            if d == 48:                   # the 48 px avatar again, enlarged 4x without smoothing
                big = im.resize((192, 192), Image.NEAREST)
                sheet.paste(big, (col + 24, 224), mask.resize((192, 192), Image.NEAREST))
            x += d + 14
    sheet.save(os.path.join(out_dir, "avatar_sizes.png"))
    av.resize((48, 48), Image.LANCZOS).save(os.path.join(out_dir, "avatar_48.png"))

    bn = Image.open(os.path.join(HERE, "banner.png")).convert("RGB")
    x0, y0, x1, y1 = (round(v) for v in SAFE)
    bn.crop((x0, y0, x1, y1)).save(os.path.join(out_dir, "banner_safe_1546x423.png"))
    bn.crop((0, y0, BANNER_W, y1)).resize((1280, 212), Image.LANCZOS).save(os.path.join(out_dir, "banner_desktop_strip.png"))
    tv = bn.resize((1280, 720), Image.LANCZOS)
    d = ImageDraw.Draw(tv)
    d.rectangle([x0 / 2, y0 / 2, x1 / 2, y1 / 2], outline=(255, 255, 0))
    tv.save(os.path.join(out_dir, "banner_tv_with_safe_box.png"))

    wm = Image.open(os.path.join(HERE, "watermark.png")).convert("RGBA")
    sheet = Image.new("RGB", (640, 190), (0, 0, 0))
    for i, bg in enumerate(((250, 250, 250), (18, 18, 18), (255, 92, 77), (47, 224, 200))):
        tile = Image.new("RGBA", (160, 190), bg + (255,))
        tile.alpha_composite(wm, (5, 5))
        small = wm.resize((40, 40), Image.LANCZOS)
        tile.alpha_composite(small, (112, 146))
        sheet.paste(tile.convert("RGB"), (i * 160, 0))
    sheet.save(os.path.join(out_dir, "watermark_on_backgrounds.png"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--preview-dir", help="also write small review images into this folder")
    args = ap.parse_args()
    make_avatar(os.path.join(HERE, "avatar.png"))
    make_banner(os.path.join(HERE, "banner.png"))
    make_watermark(os.path.join(HERE, "watermark.png"))
    if args.preview_dir:
        make_previews(args.preview_dir)
    for name in ("avatar.png", "banner.png", "watermark.png"):
        p = os.path.join(HERE, name)
        print(f"{name}: {os.path.getsize(p):,} bytes")


if __name__ == "__main__":
    main()
