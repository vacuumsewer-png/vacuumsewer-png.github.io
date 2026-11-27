"""ICE CRUSH (CRUSH family, episode 1): a hydraulic press crushes one big block of ice. Into how many pieces?

Uses engine/crush.py (fracture cascade, press timing, slam, physics, sound). This file sets the ice look and the
metadata. On screen: one question line and one counter. The answer is the model's final piece count, shown only
in the last ~2 s, then the why line, then a cross-fade back to the intact block (frame 0).
"""
import math

import cairo
import numpy as np

from engine import crush
from engine.crush import CrushScene
from engine.gfx import TAU


class Scene(CrushScene):
    COOL = (0.30, 0.78, 1.00)
    HOT = (1.00, 0.56, 0.24)
    EDGE_LIGHT = (0.92, 0.98, 1.00)
    EDGE_DARK = (0.03, 0.16, 0.34)
    CHIP_COL = (0.86, 0.97, 1.00)
    QUESTION = ("HOW MANY PIECES", "WILL IT BREAK INTO?")
    ROOT = 43

    def make_texture(self):
        """Ice: translucent blue-white body, cloudy core, frost lines, trapped bubbles, bevelled edges, a sheen."""
        w, h = int(self.WC), int(self.HC)
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        ctx = cairo.Context(surf)
        rng = np.random.default_rng(42)
        g = cairo.LinearGradient(0, 0, w * 0.35, h)
        g.add_color_stop_rgba(0, 0.82, 0.95, 1.00, 0.86)
        g.add_color_stop_rgba(0.55, 0.52, 0.80, 0.97, 0.80)
        g.add_color_stop_rgba(1, 0.26, 0.56, 0.86, 0.90)
        ctx.set_source(g)
        ctx.paint()
        g = cairo.RadialGradient(w * 0.5, h * 0.55, 10, w * 0.5, h * 0.55, w * 0.45)     # cloudy white core
        g.add_color_stop_rgba(0, 1, 1, 1, 0.42)
        g.add_color_stop_rgba(1, 1, 1, 1, 0.0)
        ctx.set_source(g)
        ctx.paint()
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        for _ in range(26):                                                            # frost lines
            x, y = rng.uniform(0.1, 0.9) * w, rng.uniform(0.15, 0.9) * h
            ang = rng.uniform(0, TAU)
            ctx.move_to(x, y)
            for _ in range(int(rng.integers(3, 7))):
                ang += rng.normal(0, 0.6)
                ln = rng.uniform(12, 40)
                x, y = x + math.cos(ang) * ln, y + math.sin(ang) * ln
                ctx.line_to(x, y)
            ctx.set_source_rgba(1, 1, 1, rng.uniform(0.18, 0.40))
            ctx.set_line_width(rng.uniform(1.2, 2.6))
            ctx.stroke()
        for _ in range(110):                                                           # bubbles
            x = rng.normal(0.5, 0.2) * w
            y = rng.normal(0.58, 0.2) * h
            r = rng.uniform(2.5, 11) if rng.random() < 0.8 else rng.uniform(11, 18)
            ctx.new_path()
            ctx.arc(x, y, r, 0, TAU)
            ctx.set_source_rgba(0.95, 0.99, 1.0, 0.16)
            ctx.fill_preserve()
            ctx.set_source_rgba(1, 1, 1, 0.55)
            ctx.set_line_width(1.4)
            ctx.stroke()
            ctx.new_path()
            ctx.arc(x - r * 0.35, y - r * 0.35, max(1.0, r * 0.28), 0, TAU)
            ctx.set_source_rgba(1, 1, 1, 0.85)
            ctx.fill()
        g = cairo.LinearGradient(0, 0, w, h)                                           # diagonal sheen
        for s, a in ((0.20, 0.0), (0.28, 0.32), (0.33, 0.0), (0.40, 0.0), (0.43, 0.18), (0.47, 0.0)):
            g.add_color_stop_rgba(s, 1, 1, 1, a)
        ctx.set_source(g)
        ctx.paint()
        for (x0, y0, x1, y1, a) in ((0, 0, 0, 34, 0.55), (0, 0, 30, 0, 0.40)):          # bevel: lit top and left
            g = cairo.LinearGradient(x0, y0, x1, y1)
            g.add_color_stop_rgba(0, 1, 1, 1, a)
            g.add_color_stop_rgba(1, 1, 1, 1, 0)
            ctx.set_source(g)
            ctx.paint()
        for (x0, y0, x1, y1, a) in ((0, h, 0, h - 40, 0.45), (w, 0, w - 34, 0, 0.35)):  # shade: bottom and right
            g = cairo.LinearGradient(x0, y0, x1, y1)
            g.add_color_stop_rgba(0, 0.02, 0.18, 0.40, a)
            g.add_color_stop_rgba(1, 0.02, 0.18, 0.40, 0)
            ctx.set_source(g)
            ctx.paint()
        surf.flush()
        return surf

    def metadata(self):
        st = self.stats
        title = "The Press Crushed One Ice Block… Into How Many Pieces?"
        return dict(
            title=title,
            description=(
                "Comment your number \U0001F447\n"
                "A hydraulic press squeezes one block of ice and the force keeps climbing. A piece cracks in two when "
                "the force passes its strength, and smaller pieces are stronger, so every crack needs more force than "
                "the last.\n"
                "At the end the press slams down at twice its force, and every piece weaker than the slam breaks "
                "again, generation after generation. The counter shows the real count the whole time.\n"
                "No footage, samples or stock media.\n"
                "#simulation #satisfying #physics"),
            tags=["simulation", "hydraulic press", "ice", "ice block", "press crush", "how many pieces", "satisfying",
                  "physics", "crush", "shatter", "made with code", "oddly satisfying"],
            pinned_comment="Comment your number \U0001F447 How many pieces did the press crush this ice block into?",
            made_for_kids=False,
            synthetic_media_disclosure=False,
            synthetic_media_disclosure_reason="Clearly stylized 2D animation of a press and an ice block, generated by "
                                              "a toy fracture model; not realistic footage of people, places or events.",
            seed=self.seed,
            duration_seconds=round(self.nframes / self.FPS, 2),
            answer=int(self.n_final),
            drama=st,
        )


def score_seed(seed):
    return crush.drama(crush.simulate(seed, Scene), Scene)
