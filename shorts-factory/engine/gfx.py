"""Drawing helpers for pycairo: easing, colour, shapes, and text set straight from a TTF.

Text is drawn from glyph outlines read with fontTools, so the result does not depend on
which fonts are installed on the machine (no fontconfig lookup, no fallback surprises).
"""
import math

import cairo
from fontTools.pens.basePen import BasePen
from fontTools.ttLib import TTFont

TAU = math.tau


# ------------------------------------------------------------------ easing
def clamp(x, lo=0.0, hi=1.0):
    return lo if x < lo else hi if x > hi else x


def lerp(a, b, k):
    return a + (b - a) * k


def ramp(t, t0, t1):
    """0 before t0, 1 after t1, linear in between."""
    if t1 <= t0:
        return 1.0 if t >= t0 else 0.0
    return clamp((t - t0) / (t1 - t0))


def ease_out(x, p=3):
    return 1 - (1 - clamp(x)) ** p


def ease_in(x, p=3):
    return clamp(x) ** p


def ease_in_out(x):
    x = clamp(x)
    return 4 * x * x * x if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_back(x, s=1.70158):
    x = clamp(x) - 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def ease_out_elastic(x):
    x = clamp(x)
    if x in (0.0, 1.0):
        return x
    return 2 ** (-10 * x) * math.sin((x * 10 - 0.75) * TAU / 3) + 1


# ------------------------------------------------------------------ colour
def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


def mix(a, b, k):
    return tuple(a[i] + (b[i] - a[i]) * k for i in range(3))


def lighten(c, k):
    return mix(c, (1, 1, 1), k)


def darken(c, k):
    return mix(c, (0, 0, 0), k)


# ------------------------------------------------------------------ shapes
def rounded_rect(ctx, x, y, w, h, r):
    r = max(0.0, min(r, w / 2, h / 2))
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -TAU / 4, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, TAU / 4)
    ctx.arc(x + r, y + h - r, r, TAU / 4, TAU / 2)
    ctx.arc(x + r, y + r, r, TAU / 2, TAU * 3 / 4)
    ctx.close_path()


def polygon(ctx, pts):
    ctx.move_to(*pts[0])
    for p in pts[1:]:
        ctx.line_to(*p)
    ctx.close_path()


def glow(ctx, x, y, r, col, a=0.5, inner=0.0):
    """Additive radial glow. Caller is expected to have OPERATOR_ADD set or accept plain blending."""
    g = cairo.RadialGradient(x, y, inner, x, y, r)
    g.add_color_stop_rgba(0, col[0], col[1], col[2], a)
    g.add_color_stop_rgba(0.45, col[0], col[1], col[2], a * 0.32)
    g.add_color_stop_rgba(1, col[0], col[1], col[2], 0)
    ctx.set_source(g)
    ctx.arc(x, y, r, 0, TAU)
    ctx.fill()


# ------------------------------------------------------------------ text
class _Recorder(BasePen):
    """Records a glyph outline as cairo-ready ops (y flipped, quadratics promoted to cubics)."""

    def __init__(self, glyph_set):
        super().__init__(glyph_set)
        self.ops = []

    def _moveTo(self, p):
        self.ops.append((0, p[0], -p[1]))

    def _lineTo(self, p):
        self.ops.append((1, p[0], -p[1]))

    def _curveToOne(self, a, b, c):
        self.ops.append((2, a[0], -a[1], b[0], -b[1], c[0], -c[1]))

    def _closePath(self):
        self.ops.append((3,))


class Font:
    def __init__(self, path):
        tt = TTFont(path)
        self.upm = tt["head"].unitsPerEm
        self.cmap = tt.getBestCmap()
        self.glyph_set = tt.getGlyphSet()
        self.metrics = tt["hmtx"].metrics
        os2 = tt["OS/2"]
        self.cap_height = getattr(os2, "sCapHeight", 0) / self.upm or 0.7
        self._pair_tables = self._load_kerning(tt)
        self._glyphs = {}
        self._kern = {}

    # -- kerning: GPOS pair adjustment (formats 1 and 2), enough for display Latin text
    @staticmethod
    def _load_kerning(tt):
        if "GPOS" not in tt:
            return []
        gpos = tt["GPOS"].table
        if not gpos.FeatureList or not gpos.LookupList:
            return []
        idx = []
        for fr in gpos.FeatureList.FeatureRecord:
            if fr.FeatureTag == "kern":
                idx += [i for i in fr.Feature.LookupListIndex if i not in idx]
        subs = []
        for i in idx:
            for st in gpos.LookupList.Lookup[i].SubTable:
                if st.LookupType == 9:
                    st = st.ExtSubTable
                if st.LookupType == 2:
                    subs.append((st, set(st.Coverage.glyphs)))
        return subs

    def kern(self, left, right):
        key = (left, right)
        if key in self._kern:
            return self._kern[key]
        val = 0
        for st, cov in self._pair_tables:
            if left not in cov:
                continue
            if st.Format == 1:
                ps = st.PairSet[st.Coverage.glyphs.index(left)]
                hit = [r for r in ps.PairValueRecord if r.SecondGlyph == right]
                if hit:
                    v = hit[0].Value1
                    val = (v.XAdvance or 0) if v is not None and hasattr(v, "XAdvance") else 0
                    break
            else:
                c1 = st.ClassDef1.classDefs.get(left, 0)
                c2 = st.ClassDef2.classDefs.get(right, 0)
                v = st.Class1Record[c1].Class2Record[c2].Value1
                val = (getattr(v, "XAdvance", 0) or 0) if v is not None else 0
                break
        self._kern[key] = val
        return val

    def _glyph(self, ch):
        g = self._glyphs.get(ch)
        if g is None:
            name = self.cmap.get(ord(ch)) or self.cmap.get(ord("?"))
            pen = _Recorder(self.glyph_set)
            self.glyph_set[name].draw(pen)
            g = self._glyphs[ch] = (name, pen.ops, self.metrics[name][0])
        return g

    def _layout(self, text, size, tracking):
        """Yield (ops, x_offset_px) per glyph and return total width via .width()."""
        k = size / self.upm
        x = 0.0
        prev = None
        out = []
        for ch in text:
            name, ops, adv = self._glyph(ch)
            if prev is not None:
                x += self.kern(prev, name) * k + tracking
            out.append((ops, x))
            x += adv * k
            prev = name
        return out, x

    def width(self, text, size, tracking=0.0):
        return self._layout(text, size, tracking)[1]

    def path(self, ctx, text, x, y, size, anchor="l", tracking=0.0):
        """Append the outline of `text` to the current path; (x, y) is the baseline anchor."""
        lay, w = self._layout(text, size, tracking)
        if anchor == "c":
            x -= w / 2
        elif anchor == "r":
            x -= w
        k = size / self.upm
        for ops, gx in lay:
            ox = x + gx
            for op in ops:
                c = op[0]
                if c == 0:
                    ctx.move_to(ox + op[1] * k, y + op[2] * k)
                elif c == 1:
                    ctx.line_to(ox + op[1] * k, y + op[2] * k)
                elif c == 2:
                    ctx.curve_to(ox + op[1] * k, y + op[2] * k, ox + op[3] * k, y + op[4] * k,
                                 ox + op[5] * k, y + op[6] * k)
                else:
                    ctx.close_path()
        return w

    def draw(self, ctx, text, x, y, size, fill, anchor="l", alpha=1.0, stroke=None, stroke_w=0.0,
             tracking=0.0, shadow=0.0):
        """Filled text with an optional outline drawn underneath (so the letter shapes stay full weight)."""
        if alpha <= 0.003:
            return 0.0
        if shadow:
            ctx.new_path()
            self.path(ctx, text, x, y + shadow, size, anchor, tracking)
            ctx.set_source_rgba(0, 0, 0, 0.45 * alpha)
            if stroke_w:
                ctx.set_line_width(stroke_w)
                ctx.set_line_join(cairo.LINE_JOIN_ROUND)
                ctx.stroke_preserve()
            ctx.fill()
        ctx.new_path()
        w = self.path(ctx, text, x, y, size, anchor, tracking)
        if stroke is not None and stroke_w:
            ctx.set_source_rgba(stroke[0], stroke[1], stroke[2], alpha)
            ctx.set_line_width(stroke_w)
            ctx.set_line_join(cairo.LINE_JOIN_ROUND)
            ctx.stroke_preserve()
        if isinstance(fill, cairo.Pattern):
            ctx.set_source(fill)
        else:
            ctx.set_source_rgba(fill[0], fill[1], fill[2], alpha)
        ctx.fill()
        return w
