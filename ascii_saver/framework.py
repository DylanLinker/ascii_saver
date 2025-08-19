# screen_saver/framework.py
from __future__ import annotations

import sys as _sys
import os
import time
import argparse
import io
import cProfile
import pstats
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Type, TypeVar

import numpy as np
from numpy.typing import NDArray
import pygame as pg
import moderngl as mgl
from PIL import Image, ImageDraw, ImageFont
import pkgutil
import importlib
import ascii_saver.effects

# =================== Config Defaults ===================

DEFAULT_CHARS: NDArray[np.str_] = np.array(
    list("@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/|?-_+~<>i!lI;:,\"^`'. "),
    dtype="<U1",
)
DEFAULT_PALETTE: list[str] = ["#100954", "#7303c0", "#ec38bc", "#fdeff9"]
DEFAULT_FONT: str = "Consolas"
DEFAULT_SIZE: int = 20
DEFAULT_FPS: int = 30
DEFAULT_MIN_SHOW: float = 0.01
DEFAULT_GAMMA: float = 1.6

# =================== Small Utilities ===================


def _hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = s.lstrip("#")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


def make_palette256(stops: Sequence[str]) -> NDArray[np.uint8]:
    if not stops:
        return np.tile(np.array([[255, 255, 255]], np.uint8), (256, 1))
    rgb = np.array([_hex_to_rgb(s) for s in stops], dtype=np.float32)
    if len(rgb) == 1:
        rgb = np.vstack([rgb, rgb])
    t = np.linspace(0, 1, 256, dtype=np.float32)
    idx = t * (len(rgb) - 1)
    k = np.minimum(idx.astype(np.int32), len(rgb) - 2)
    u = idx - k.astype(np.float32)
    out = rgb[k] * (1.0 - u)[:, None] + rgb[k + 1] * u[:, None]
    return out.astype(np.uint8)


def find_font_path(family: str) -> Optional[str]:
    candidates = [
        rf"C:\Windows\Fonts\{family}.ttf",
        rf"C:\Windows\Fonts\{family}.TTF",
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\Consolas.ttf",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _load_font(family: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    path = find_font_path(family)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    try:
        return ImageFont.truetype(f"{family}.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _text_bbox(
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont, ch: str
) -> tuple[int, int]:
    tmp = Image.new("L", (1, 1), 0)
    drw = ImageDraw.Draw(tmp)
    left, top, right, bottom = drw.textbbox((0, 0), ch, font=font)
    w = max(1, int(right - left))
    h = max(1, int(bottom - top))
    return w, h



def _measure_cell(
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont, chars: Sequence[str]
) -> tuple[int, int]:
    max_w = 1
    max_h = 1
    for ch in chars:
        w, h = _text_bbox(font, ch or " ")
        if w > max_w: max_w = w
        if h > max_h: max_h = h
    return max_w, max_h



def build_colorized_atlas(
    font_family: str,
    size_pt: int,
    chars: Sequence[str],
    oversample: int,
    palette256: NDArray[np.uint8],
) -> tuple[Image.Image, int, int, int, int]:
    base = _load_font(font_family, size_pt)
    hi = _load_font(font_family, max(1, size_pt * oversample))

    base_w, base_h = _measure_cell(base, chars)
    atlas_w_cell, atlas_h = _measure_cell(hi, chars)

    atlas = Image.new("RGBA", (atlas_w_cell * len(chars), atlas_h), (0, 0, 0, 0))

    # char[0] reserved for space (transparent)
    gmax = max(1, len(chars) - 2)
    for i, ch in enumerate(chars):
        if i == 0:
            continue
        rank = i - 1
        pal_u8 = int(round(255 * (1.0 - rank / gmax)))
        r, g, b = map(int, palette256[pal_u8])
        glyph_L = Image.new("L", (atlas_w_cell, atlas_h), 0)
        ImageDraw.Draw(glyph_L).text((0, 0), ch, fill=255, font=hi)
        tile = Image.new("RGBA", (atlas_w_cell, atlas_h), (r, g, b, 0))
        tile.putalpha(glyph_L)
        atlas.paste(tile, (i * atlas_w_cell, 0), tile)

    return atlas, base_w, base_h, atlas_w_cell, atlas_h


# =================== App/Renderer Core ===================


@dataclass(frozen=True)
class Grid:
    screen_w: int
    screen_h: int
    cols: int
    rows: int
    cell_w: int
    cell_h: int


class GLRenderer:
    def __init__(self, grid: Grid, atlas_rgba: Image.Image, glyph_count: int) -> None:
        self.ctx: mgl.Context = mgl.create_context()
        self.ctx.enable(mgl.BLEND)
        self.ctx.blend_func = (mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA)

        self.grid: Grid = grid
        self.glyph_count: int = glyph_count
        quad = np.array([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype="f4")
        self.vbo: mgl.Buffer = self.ctx.buffer(quad.tobytes())
        self._atlas: mgl.Texture = self._tex_from_pil(atlas_rgba, 4, nearest=False)

    def _tex_from_pil(self, img: Image.Image, components: int, nearest: bool) -> mgl.Texture:
        want = {1: "L", 3: "RGB", 4: "RGBA"}[components]
        if img.mode != want:
            img = img.convert(want)
        w, h = img.size
        tex = self.ctx.texture((w, h), components=components, data=img.tobytes())
        tex.filter = (mgl.NEAREST, mgl.NEAREST) if nearest else (mgl.LINEAR, mgl.LINEAR)
        tex.repeat_x = False
        tex.repeat_y = False
        return tex

    def make_program(self, vert_src: str, frag_src: str) -> mgl.Program:
        return self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

    def draw_fullscreen(self, prog: mgl.Program, binds: Optional[Mapping[int, mgl.Texture]] = None) -> None:
        vao = self.ctx.simple_vertex_array(prog, self.vbo, "in_pos")
        if binds:
            for unit, tex in binds.items():
                tex.use(location=unit)
        vao.render(mgl.TRIANGLE_STRIP)

    @property
    def atlas_tex(self) -> mgl.Texture:
        return self._atlas


# =================== Base Effect API ===================


class Effect:
    NAME: str = "base"
    CONFIG: dict[str, Any] = {}

    def __init__(self, app: "App", **cfg: Any) -> None:
        self.app = app
        merged = {**self.CONFIG, **cfg}
        for k, v in merged.items():
            setattr(self, k, v)

    # --- NEW: shader API ---
    def shader_sources(self) -> tuple[str, str]:
        """Return (vert_src, frag_src). Override in your effect to swap shaders."""
        return (VERT_SRC, FRAG_GLYPH_AGE)

    def build_program(self, vert_src: str | None = None, frag_src: str | None = None) -> "mgl.Program":
        """Compile/link a program and bind common glyph uniforms."""
        v, f = self.shader_sources()
        v = vert_src or v
        f = frag_src or f
        prog = self.app.renderer.make_program(v, f)
        self.app.bind_common_glyph_uniforms(prog)
        self.on_program_link(prog)
        return prog

    def on_program_link(self, prog: "mgl.Program") -> None:
        """Hook for effect-specific uniform defaults after link."""
        pass

    def init(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def update(self, dt: float) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def render(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


# =================== Shared Shaders ===================

VERT_SRC: str = """
#version 330
in vec2 in_pos;
void main(){ gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

FRAG_GLYPH_AGE: str = """
#version 330
uniform int  u_cols, u_rows, u_glyph_count, u_thresh_u8;
uniform float u_cell_w, u_cell_h, u_atlas_cell_w, u_atlas_cell_h, u_gamma;
uniform sampler2D u_age;
uniform sampler2D u_glyph_atlas;
out vec4 fragColor;
void main(){
    vec2 pix = gl_FragCoord.xy;
    vec2 cell = pix / vec2(u_cell_w, u_cell_h);
    vec2 ci = floor(cell);
    vec2 frac = cell - ci;
    float u = (ci.x + 0.5) / float(u_cols);
    float v = 1.0 - (ci.y + 0.5) / float(u_rows);
    float age = texture(u_age, vec2(u, v)).r;
    float tg   = pow(age, u_gamma);
    float fidx = (1.0 - tg) * float(u_glyph_count - 2);
    int mapped = int(round(fidx));
    mapped = clamp(mapped, 0, u_glyph_count - 2);
    int ch = (int(round(age * 255.0)) < u_thresh_u8) ? 0 : (mapped + 1);
    float atlas_w = float(u_glyph_count) * u_atlas_cell_w;
    float atlas_h = u_atlas_cell_h;
    float U = (float(ch) * u_atlas_cell_w + frac.x * u_atlas_cell_w + 0.5) / atlas_w;
    float V = 1.0 - ((frac.y * u_atlas_cell_h + 0.5) / atlas_h);
    fragColor = texture(u_glyph_atlas, vec2(U, V));
}
"""

FRAG_GLYPH_FIELD: str = """
#version 330
uniform int  u_cols, u_rows, u_num_nodes, u_glyph_count, u_thresh_u8;
uniform float u_cell_w, u_cell_h, u_atlas_cell_w, u_atlas_cell_h, u_sigma, u_mult, u_scale, u_gamma;
uniform vec2 u_nodes[256];
uniform sampler2D u_glyph_atlas;
out vec4 fragColor;
void main(){
    vec2 pix = gl_FragCoord.xy;
    vec2 cell = pix / vec2(u_cell_w, u_cell_h);
    vec2 ci = floor(cell);
    vec2 frac = cell - ci;
    vec2 p = ci + vec2(0.5);
    float acc = 0.0;
    for(int i=0;i<u_num_nodes;++i){
        vec2 d = abs(p - u_nodes[i]);
        d = min(d, vec2(u_cols, u_rows) - d);
        acc += exp(-0.5 * dot(d,d) / (u_sigma*u_sigma)) * u_mult;
    }
    float t = clamp(acc * u_scale, 0.0, 1.0);
    float tg   = pow(t, u_gamma);
    float fidx = (1.0 - tg) * float(u_glyph_count - 2);
    int mapped = int(round(fidx));
    mapped = clamp(mapped, 0, u_glyph_count - 2);
    int ch = (int(round(t * 255.0)) < u_thresh_u8) ? 0 : (mapped + 1);
    float atlas_w = float(u_glyph_count) * u_atlas_cell_w;
    float atlas_h = u_atlas_cell_h;
    float U = (float(ch) * u_atlas_cell_w + frac.x * u_atlas_cell_w + 0.5) / atlas_w;
    float V = 1.0 - ((frac.y * u_atlas_cell_h + 0.5) / atlas_h);
    fragColor = texture(u_glyph_atlas, vec2(U, V));
}
"""

# =================== App Harness & Registry ===================

TEffect = TypeVar("TEffect", bound=Effect)


class App:
    def __init__(
        self,
        effect_cls: type[Effect],
        *,
        font: str = DEFAULT_FONT,
        size: int = DEFAULT_SIZE,
        fps: int = DEFAULT_FPS,
        gamma: float = DEFAULT_GAMMA,
        min_show: float = DEFAULT_MIN_SHOW,
        palette: Sequence[str] = DEFAULT_PALETTE,
        chars: NDArray[np.str_] = DEFAULT_CHARS,
        profile: bool = False,
        frames: int = 0,
    ) -> None:
        self.fps: int = max(1, int(fps))
        self.gamma: float = float(gamma)
        self.min_show: float = float(min_show)
        self.min_show_u8: int = int(max(0, min(255, int(self.min_show * 255))))
        self.frames_left: int = max(0, int(frames))
        self.profile: bool = bool(profile)
        self.prof: Optional[cProfile.Profile] = cProfile.Profile() if self.profile else None

        os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"
        pg.init()
        info = pg.display.Info()
        self.screen_w, self.screen_h = info.current_w, info.current_h
        pg.display.set_mode((self.screen_w, self.screen_h), pg.OPENGL | pg.DOUBLEBUF | pg.NOFRAME)
        pg.mouse.set_visible(False)

        pal256: NDArray[np.uint8] = make_palette256(palette)

        # normalize chars to list[str] and drop last (space already injected at 0)
        chars_list = [str(c) for c in chars.tolist()]

        glyphs: list[str] = [" "] + chars_list[:-1]

        atlas, cell_w, cell_h, atlas_w_cell, atlas_h = build_colorized_atlas(
            font, size, glyphs, 2, pal256
        )
        cols = max(1, self.screen_w // cell_w)
        rows = max(1, self.screen_h // cell_h)
        self.grid: Grid = Grid(self.screen_w, self.screen_h, cols, rows, cell_w, cell_h)
        self.atlas_cell_w: int = atlas_w_cell
        self.atlas_cell_h: int = atlas_h
        self.glyph_count: int = len(glyphs)
        self.renderer: GLRenderer = GLRenderer(self.grid, atlas, self.glyph_count)
        self.effect: Effect = effect_cls(self)

    def bind_common_glyph_uniforms(self, prog: mgl.Program) -> None:
        # Use 'Any' to avoid strict stub issues with moderngl Uniform types.
        g = self.grid
        p: Any = prog
        p["u_cols"].value = int(g.cols)
        p["u_rows"].value = int(g.rows)
        p["u_glyph_count"].value = int(self.glyph_count)
        p["u_cell_w"].value = float(g.cell_w)
        p["u_cell_h"].value = float(g.cell_h)
        p["u_atlas_cell_w"].value = float(self.atlas_cell_w)
        p["u_atlas_cell_h"].value = float(self.atlas_cell_h)
        p["u_gamma"].value = float(self.gamma)
        p["u_thresh_u8"].value = int(self.min_show_u8)
        self.renderer.atlas_tex.use(location=0)
        if "u_glyph_atlas" in prog:
            p["u_glyph_atlas"].value = 0

    def run(self) -> None:
        if self.prof:
            self.prof.enable()
        self.effect.init()
        clock = pg.time.Clock()
        last = time.perf_counter()
        running = True
        fps_ema = 0.0
        while running:
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key == pg.K_ESCAPE):
                    running = False
                elif e.type == pg.KEYDOWN and e.key == pg.K_r:
                    # reset effect
                    self.effect = type(self.effect)(self)
                    self.effect.init()
            now = time.perf_counter()
            dt = now - last
            last = now
            self.effect.update(dt)
            self.renderer.ctx.clear(0, 0, 0, 1)
            self.effect.render()
            pg.display.flip()
            clock.tick(self.fps)
            inst = 1.0 / dt if dt > 0 else float(self.fps)
            fps_ema = inst if fps_ema == 0.0 else 0.85 * fps_ema + 0.15 * inst
            pg.display.set_caption(f"ASCII — {fps_ema:5.1f} FPS")
            if self.frames_left > 0:
                self.frames_left -= 1
                if self.frames_left == 0:
                    running = False
        if self.prof:
            self.prof.disable()
            s = io.StringIO()
            pstats.Stats(self.prof, stream=s).sort_stats("cumtime").print_stats(30)
            print(s.getvalue())
        pg.quit()


# -------- Effect Registry --------

_EFFECTS: Dict[str, Type[Effect]] = {}


def register_effect(cls: Type[TEffect]) -> Type[TEffect]:
    if not hasattr(cls, "NAME"):
        raise ValueError("Effect must define NAME")
    _EFFECTS[cls.NAME] = cls
    return cls


def get_effect(name: str) -> Optional[Type[Effect]]:
    return _EFFECTS.get(name)


def list_effects() -> list[str]:
    return sorted(_EFFECTS.keys())


# -------- CLI --------


def main() -> None:
    _sys.modules.setdefault("screen_saver.framework", _sys.modules[__name__])

    # Import effects before argparse (dynamic registry)
    for _, modname, _ in pkgutil.iter_modules(ascii_saver.effects.__path__):
        importlib.import_module(f"ascii_saver.effects.{modname}")

    p = argparse.ArgumentParser()
    p.add_argument("--effect", required=True)
    p.add_argument("--font", default=DEFAULT_FONT)
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--fps", type=int, default=DEFAULT_FPS)
    p.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    p.add_argument("--min-show", type=float, default=DEFAULT_MIN_SHOW)
    p.add_argument("--palette", nargs="*", default=DEFAULT_PALETTE)
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--next", action="store_true", help="Show tunables for effect")
    p.add_argument("--set", action="append", help="Override tunables, format KEY=VALUE")
    args = p.parse_args()

    cls = get_effect(args.effect)
    if not cls:
        raise SystemExit(f"Unknown effect {args.effect}, available: {list_effects()}")

    if args.next:
        print(f"Effect '{cls.NAME}' tunables:")
        for k, v in cls.CONFIG.items():
            print(f"  {k} = {v}")
        return
    
    cfg_overrides: dict[str, Any] = {}
    if args.set:
        for item in args.set:
            if "=" not in item:
                raise SystemExit(f"Invalid --set {item}, must be KEY=VALUE")
            k, v = item.split("=", 1)
            try:
                v = eval(v, {}, {})  # interpret numbers/bools
            except Exception:
                pass
            cfg_overrides[k] = v

    App(
        cls,
        font=args.font,
        size=args.size,
        fps=args.fps,
        gamma=args.gamma,
        min_show=args.min_show,
        palette=args.palette,
        profile=args.profile,
        frames=args.frames,
        **cfg_overrides
    ).run()


if __name__ == "__main__":
    main()
