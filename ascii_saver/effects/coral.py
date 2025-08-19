from __future__ import annotations

import moderngl as mgl
from typing import Any, Final, Optional, Sequence, TypedDict
import numpy as np
from numpy.typing import NDArray, ArrayLike

from ascii_saver.framework import (
    Effect,
    register_effect,
    VERT_SRC,
    FRAG_GLYPH_AGE,
)

# -------------------- Typed config --------------------

class CoralConfig(TypedDict):
    START_NODES: int
    START_RADIUS: float
    MAX_POINTS: int
    MAX_EDGE_LEN: float
    MAX_INSERTS_PER_FRAME: int

    MAX_SPEED: float
    MAX_FORCE: float
    DESIRED_SEP: float
    SEP_COH_RATIO: float

    WIND_CELL_SIZE: int
    WIND_TAU: float
    WIND_NOISE: float
    WIND_MAX: float
    WIND_GAIN: float
    WIND_SMOOTHING: float

    DECAY_LOW: float
    DECAY_HIGH: float
    FRESH_AGE: float

DEFAULTS: Final[CoralConfig] = {
    "START_NODES": 28,
    "START_RADIUS": 12.0,
    "MAX_POINTS": 1500,
    "MAX_EDGE_LEN": 5.0,
    "MAX_INSERTS_PER_FRAME": 24,
    "MAX_SPEED": 1.5,
    "MAX_FORCE": 1.0,
    "DESIRED_SEP": 10.0,
    "SEP_COH_RATIO": 0.9,
    "WIND_CELL_SIZE": 10,
    "WIND_TAU": 3.0,
    "WIND_NOISE": 0.9,
    "WIND_MAX": 1.0,
    "WIND_GAIN": 0.50,
    "WIND_SMOOTHING": 0.20,
    "DECAY_LOW": 0.88,
    "DECAY_HIGH": 0.985,
    "FRESH_AGE": 1.0,
}

# -------------------- dtype helpers (Pylance-friendly) --------------------

F32 = np.float32

def as32(a: ArrayLike) -> NDArray[np.float32]:
    """Convert any array-like (including scalars) to a float32 ndarray."""
    return np.asarray(a, dtype=np.float32)

def zeros32(shape: int | tuple[int, ...]) -> NDArray[np.float32]:
    """float32 zeros with precise typing."""
    return np.zeros(shape, dtype=np.float32)

def stack32(parts: Sequence[ArrayLike], axis: int = 0) -> NDArray[np.float32]:
    """Stack arbitrary array-likes as float32, returning float32."""
    if not parts:
        return np.empty((0,), dtype=np.float32)
    arrs: list[NDArray[np.float32]] = [np.asarray(p, dtype=np.float32) for p in parts]
    return np.stack(arrs, axis=axis).astype(np.float32, copy=False)

def sqrt32(x: ArrayLike) -> NDArray[np.float32]:
    """sqrt that accepts scalars/arrays and returns float32 ndarray."""
    arr = np.asarray(x, dtype=np.float32)
    return np.sqrt(arr, dtype=np.float32)

def norm2_rows(v: NDArray[np.float32]) -> NDArray[np.float32]:
    """Rowwise L2 norm for [N,2] arrays as float32 column vector [N,1]."""
    return sqrt32(np.sum(v * v, axis=1, keepdims=True, dtype=np.float32))

# -------------------- geometry helpers --------------------

def _clip_segment_to_rect(
    x0: float, y0: float, x1: float, y1: float, w: int, h: int
) -> Optional[tuple[float, float, float, float]]:
    """Liang–Barsky clip. Returns clipped segment or None if outside."""
    xmin, ymin = 0.0, 0.0
    xmax, ymax = float(w) - 1e-6, float(h) - 1e-6
    dx, dy = x1 - x0, y1 - y0

    p = [-dx, dx, -dy, dy]
    q = [x0 - xmin, xmax - x0, y0 - ymin, ymax - y0]

    u1 = 0.0
    u2 = 1.0

    for pi, qi in zip(p, q):
        if pi == 0.0:
            if qi < 0.0:
                return None
            continue
        r = qi / pi
        if pi < 0.0:
            if r > u2:
                return None
            if r > u1:
                u1 = r
        else:
            if r < u1:
                return None
            if r < u2:
                u2 = r

    nx0 = x0 + u1 * dx
    ny0 = y0 + u1 * dy
    nx1 = x0 + u2 * dx
    ny1 = y0 + u2 * dy
    return (nx0, ny0, nx1, ny1)


def _supercover_cells_clamped_aw(
    x0: float, y0: float, x1: float, y1: float, cols: int, rows: int
) -> list[tuple[int, int]]:
    """Grid traversal with proper corner handling (Amanatides & Woo)."""
    out: list[tuple[int, int]] = []

    # Clamp slightly inside so floor() is stable on borders
    eps = 1e-6
    x0 = float(np.clip(x0, 0.0, cols - eps))
    y0 = float(np.clip(y0, 0.0, rows - eps))
    x1 = float(np.clip(x1, 0.0, cols - eps))
    y1 = float(np.clip(y1, 0.0, rows - eps))

    xi = int(np.floor(x0))
    yi = int(np.floor(y0))
    xe = int(np.floor(x1))
    ye = int(np.floor(y1))

    dx = x1 - x0
    dy = y1 - y0

    step_x = 1 if dx > 0.0 else (-1 if dx < 0.0 else 0)
    step_y = 1 if dy > 0.0 else (-1 if dy < 0.0 else 0)

    if step_x == 0 and step_y == 0:
        if 0 <= xi < cols and 0 <= yi < rows:
            out.append((xi, yi))
        return out

    inv_abs_dx = (1.0 / abs(dx)) if dx != 0.0 else float("inf")
    inv_abs_dy = (1.0 / abs(dy)) if dy != 0.0 else float("inf")

    # Distance (in param t) to the first vertical/horizontal boundary
    t_max_x = ((xi + 1) - x0) * inv_abs_dx if step_x > 0 else ((x0 - xi) * inv_abs_dx if step_x < 0 else float("inf"))
    t_max_y = ((yi + 1) - y0) * inv_abs_dy if step_y > 0 else ((y0 - yi) * inv_abs_dy if step_y < 0 else float("inf"))

    t_delta_x = inv_abs_dx
    t_delta_y = inv_abs_dy

    # Safety cap
    max_steps = (abs(xe - xi) + abs(ye - yi)) + cols + rows + 8
    steps = 0

    while True:
        if 0 <= xi < cols and 0 <= yi < rows:
            out.append((xi, yi))
        if xi == xe and yi == ye:
            break

        # Proper tie handling: step both axes when hitting a corner
        if t_max_x < t_max_y:
            xi += step_x
            t_max_x += t_delta_x
        elif t_max_y < t_max_x:
            yi += step_y
            t_max_y += t_delta_y
        else:
            xi += step_x
            yi += step_y
            t_max_x += t_delta_x
            t_max_y += t_delta_y

        steps += 1
        if steps > max_steps:
            break

    return out


# -------------------- Effect --------------------

@register_effect
class Coral(Effect):
    NAME: str = "coral"
    CONFIG_DEFAULTS: CoralConfig = DEFAULTS

    # config (copied at init)
    start_nodes: int
    start_radius: float
    max_points: int
    max_edge_len: float
    max_inserts_per_frame: int

    max_speed: float
    max_force: float
    desired_sep: float
    sep_coh_ratio: float

    wind_cell_size: int
    wind_tau: float
    wind_noise: float
    wind_max: float
    wind_gain: float
    wind_smoothing: float

    decay_low: float
    decay_high: float
    fresh_age: float

    # runtime
    rng: np.random.Generator
    pos: NDArray[np.float32]           # [N,2]
    vel: NDArray[np.float32]           # [N,2]

    wg_w: int
    wg_h: int
    wind_grid: NDArray[np.float32]     # [wg_h,wg_w,2]

    age: NDArray[np.uint8]             # [rows,cols]
    age_tex: mgl.Texture
    prog: Any

    def init(self) -> None:
        cfg = self.CONFIG_DEFAULTS
        self.start_nodes = int(cfg["START_NODES"])
        self.start_radius = float(cfg["START_RADIUS"])
        self.max_points = int(cfg["MAX_POINTS"])
        self.max_edge_len = float(cfg["MAX_EDGE_LEN"])
        self.max_inserts_per_frame = int(cfg["MAX_INSERTS_PER_FRAME"])

        self.max_speed = float(cfg["MAX_SPEED"])
        self.max_force = float(cfg["MAX_FORCE"])
        self.desired_sep = float(cfg["DESIRED_SEP"])
        self.sep_coh_ratio = float(cfg["SEP_COH_RATIO"])

        self.wind_cell_size = int(cfg["WIND_CELL_SIZE"])
        self.wind_tau = float(cfg["WIND_TAU"])
        self.wind_noise = float(cfg["WIND_NOISE"])
        self.wind_max = float(cfg["WIND_MAX"])
        self.wind_gain = float(cfg["WIND_GAIN"])
        self.wind_smoothing = float(cfg["WIND_SMOOTHING"])

        self.decay_low = float(cfg["DECAY_LOW"])
        self.decay_high = float(cfg["DECAY_HIGH"])
        self.fresh_age = float(cfg["FRESH_AGE"])

        self.rng = np.random.default_rng()

        # geometry
        self.pos = self._init_ring(self.start_nodes, self.start_radius)
        self.vel = as32(self._random_unit(self.pos.shape[0]) * F32(0.01))

        # wind grid
        g = self.app.grid
        self.wg_w = max(8, g.cols // max(2, self.wind_cell_size))
        self.wg_h = max(6, g.rows // max(2, self.wind_cell_size))
        self.wind_grid = zeros32((self.wg_h, self.wg_w, 2))

        # age grid + texture
        self.age = np.zeros((g.rows, g.cols), dtype=np.uint8)

        ctx: mgl.Context = self.app.renderer.ctx
        self.age_tex = ctx.texture(
            (int(g.cols), int(g.rows)),
            components=1,       
        )

        self.age_tex.filter   = (mgl.NEAREST, mgl.NEAREST)
        self.age_tex.repeat_x = False
        self.age_tex.repeat_y = False
        self.age_tex.write(self.age, alignment=1)

        # shader
        self.prog = self.app.renderer.make_program(VERT_SRC, FRAG_GLYPH_AGE)
        self.app.bind_common_glyph_uniforms(self.prog)
        p: Any = self.prog
        p["u_age"].value = 1  # texture unit

    # ------------ frame steps ------------

    def update(self, dt: float) -> None:
        # Match deprecated behavior:
        # - wind grid uses real dt
        # - integration is 1 step per frame (no dt multiplier)
        if dt > 0.0:
            self._update_wind_grid(float(dt))

        # forces
        acc = self._forces()

        # integrate (per-frame step, not time-scaled)
        self.vel = as32(self.vel + acc)

        sp = norm2_rows(self.vel)
        sp_safe = as32(np.maximum(sp, F32(1e-6)))
        scale = as32(np.minimum(F32(1.0), F32(self.max_speed) / sp_safe))
        self.vel *= scale

        # NOTE: deprecated version did not multiply by dt
        self.pos = as32(self.pos + self.vel)

        self._grow()
        self._rasterize_age()


    def render(self) -> None:
        self.app.bind_common_glyph_uniforms(self.prog)
        self.age_tex.use(location=1)
        self.app.renderer.draw_fullscreen(
            self.prog, {0: self.app.renderer.atlas_tex, 1: self.age_tex}
        )

    # ------------ internals ------------

    def _init_ring(self, n: int, radius: float) -> NDArray[np.float32]:
        g = self.app.grid
        ang = np.linspace(0.0, 2.0 * np.pi, num=max(1, n), endpoint=False, dtype=np.float32)
        cx = float(g.cols) * 0.5
        cy = float(g.rows) * 0.5
        c = np.cos(ang).astype(np.float32, copy=False)
        s = np.sin(ang).astype(np.float32, copy=False)
        x = c * F32(radius) + F32(cx)
        y = s * F32(radius) + F32(cy)
        return stack32([x, y], axis=1)

    def _random_unit(self, n: int) -> NDArray[np.float32]:
        a = self.rng.random(n, dtype=np.float32) * F32(2.0 * np.pi)
        return stack32([np.cos(a).astype(np.float32, copy=False),
                        np.sin(a).astype(np.float32, copy=False)], axis=1)

    def _seek_limit(
        self, desired: NDArray[np.float32], vel: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        mag = norm2_rows(desired)
        desired_n = as32(desired / np.maximum(mag, F32(1e-6)) * F32(self.max_speed))
        steer = as32(desired_n - vel)
        smag = norm2_rows(steer)
        scl = as32(np.minimum(F32(1.0), F32(self.max_force) / np.maximum(smag, F32(1e-6))))
        return as32(steer * scl)

    def _update_wind_grid(self, dt: float) -> None:
        if dt <= 0.0:
            return
        theta = 1.0 / max(1e-3, self.wind_tau)
        sigma = float(self.wind_noise)

        # noise as float32
        noise = self.rng.standard_normal(size=self.wind_grid.shape).astype(np.float32, copy=False)

        self.wind_grid = as32(
            self.wind_grid
            + F32(theta) * (-self.wind_grid) * F32(dt)
            + F32(sigma) * F32(np.sqrt(dt)) * noise
        )

        # clamp magnitude
        mag = sqrt32(np.sum(self.wind_grid * self.wind_grid, axis=2, dtype=np.float32))[..., None]
        over = (mag > F32(self.wind_max))
        if np.any(over):
            self.wind_grid[over[:, :, 0]] = as32(
                self.wind_grid[over[:, :, 0]] * (F32(self.wind_max) / np.maximum(mag[over[:, :, 0]], F32(1e-6)))
            )

        # spatial smoothing (9-tap)
        if self.wind_smoothing > 0.0:
            W = self.wind_grid
            Wn = as32(
                (
                    W
                    + np.roll(W, 1, axis=0)
                    + np.roll(W, -1, axis=0)
                    + np.roll(W, 1, axis=1)
                    + np.roll(W, -1, axis=1)
                    + np.roll(np.roll(W, 1, axis=0), 1, axis=1)
                    + np.roll(np.roll(W, 1, axis=0), -1, axis=1)
                    + np.roll(np.roll(W, -1, axis=0), 1, axis=1)
                    + np.roll(np.roll(W, -1, axis=0), -1, axis=1)
                )
                / F32(9.0)
            )
            s = F32(self.wind_smoothing)
            self.wind_grid = as32((F32(1.0) - s) * W + s * Wn)

    def _sample_wind(self, xy: NDArray[np.float32]) -> NDArray[np.float32]:
        g = self.app.grid
        gx = as32(xy[:, 0] / F32(max(1, g.cols - 1)) * F32(self.wg_w - 1))
        gy = as32(xy[:, 1] / F32(max(1, g.rows - 1)) * F32(self.wg_h - 1))

        x0 = np.clip(np.floor(gx).astype(np.int32, copy=False), 0, self.wg_w - 1)
        y0 = np.clip(np.floor(gy).astype(np.int32, copy=False), 0, self.wg_h - 1)
        x1 = np.clip(x0 + 1, 0, self.wg_w - 1)
        y1 = np.clip(y0 + 1, 0, self.wg_h - 1)

        tx = (gx - x0.astype(np.float32, copy=False)).reshape((-1, 1))
        ty = (gy - y0.astype(np.float32, copy=False)).reshape((-1, 1))

        W = self.wind_grid
        v00 = W[y0, x0]
        v10 = W[y0, x1]
        v01 = W[y1, x0]
        v11 = W[y1, x1]
        v0 = as32(v00 * (F32(1.0) - tx) + v10 * tx)
        v1 = as32(v01 * (F32(1.0) - tx) + v11 * tx)
        return as32(v0 * (F32(1.0) - ty) + v1 * ty)

    def _forces(self) -> NDArray[np.float32]:
        prev = np.roll(self.pos, 1, axis=0)
        nextp = np.roll(self.pos, -1, axis=0)
        center = as32((prev + nextp) * F32(0.5))
        desired: NDArray[np.float32] = as32(center - self.pos)
        coh = self._seek_limit(desired, self.vel)

        # separation
        px = self.pos[:, 0]
        py = self.pos[:, 1]
        dx = as32(px[:, None] - px[None, :])
        dy = as32(py[:, None] - py[None, :])
        d2 = as32(dx * dx + dy * dy)

        n = self.pos.shape[0]
        d2[np.arange(n), np.arange(n)] = F32(np.inf)

        rad2 = F32(self.desired_sep * self.desired_sep)
        mask = d2 < rad2

        invd = zeros32(d2.shape)
        valid = (mask & (d2 > F32(0.0)))
        if np.any(valid):
            invd[valid] = as32(1.0 / sqrt32(d2[valid]))

        sep = zeros32(self.pos.shape)
        if np.any(valid):
            fx = as32(dx[valid] * invd[valid] * invd[valid])
            fy = as32(dy[valid] * invd[valid] * invd[valid])
            ii, _ = np.nonzero(valid)
            np.add.at(sep[:, 0], ii, fx)
            np.add.at(sep[:, 1], ii, fy)
            mag = norm2_rows(sep)
            sep = as32(sep / np.maximum(mag, F32(1e-6)) * F32(self.max_speed))
            sep = as32(sep - self.vel)
            smag = norm2_rows(sep)
            scl = as32(np.minimum(F32(1.0), F32(self.max_force) / np.maximum(smag, F32(1e-6))))
            sep *= scl

        wind_force = as32(F32(self.wind_gain) * self._sample_wind(self.pos))
        return as32(coh + F32(self.sep_coh_ratio) * sep + wind_force)

    def _grow(self) -> None:
        if self.pos.shape[0] >= self.max_points:
            return
        p0 = self.pos
        p1 = np.roll(self.pos, -1, axis=0)
        dx = as32(p1[:, 0] - p0[:, 0])
        dy = as32(p1[:, 1] - p0[:, 1])
        d = sqrt32(dx * dx + dy * dy)
        idx = np.nonzero(d > F32(self.max_edge_len))[0]
        if idx.size == 0:
            return
        if idx.size > self.max_inserts_per_frame:
            idx = idx[: self.max_inserts_per_frame]

        mids = stack32(
            [
                as32((p0[idx, 0] + p1[idx, 0]) * F32(0.5)),
                as32((p0[idx, 1] + p1[idx, 1]) * F32(0.5)),
            ],
            axis=1,
        )

        n = self.pos.shape[0]
        mark = np.zeros(n, dtype=np.bool_)
        mark[idx] = True

        out_p: list[NDArray[np.float32]] = []
        out_v: list[NDArray[np.float32]] = []
        for i in range(n):
            out_p.append(self.pos[i])
            out_v.append(self.vel[i])
            if mark[i]:
                mid_i = np.nonzero(idx == i)[0]
                if mid_i.size > 0:
                    out_p.append(mids[mid_i[0]])
                    out_v.append(zeros32((2,)))  # new point at rest

        self.pos = np.vstack(out_p).astype(np.float32, copy=False)
        self.vel = np.vstack(out_v).astype(np.float32, copy=False)

    def _rasterize_age(self) -> None:
        g = self.app.grid

        # decay ages
        age_f = self.age.astype(np.float32) / F32(255.0)
        factor = as32(F32(self.decay_low) + (F32(self.decay_high) - F32(self.decay_low)) * age_f)
        age_f = as32(age_f * factor)

        # draw segments
        p0 = self.pos
        p1 = np.roll(self.pos, -1, axis=0)
        cols: int = g.cols
        rows: int = g.rows
        fresh = F32(self.fresh_age)

        for i in range(self.pos.shape[0]):
            x0 = float(p0[i, 0]); y0 = float(p0[i, 1])
            x1 = float(p1[i, 0]); y1 = float(p1[i, 1])

            clipped = _clip_segment_to_rect(x0, y0, x1, y1, cols, rows)
            if clipped is None:
                continue
            cx0, cy0, cx1, cy1 = clipped

            # 1) robust grid traversal
            cells = _supercover_cells_clamped_aw(cx0, cy0, cx1, cy1, cols, rows)
            for cx, cy in cells:
                age_f[cy, cx] = fresh

            # 2) endpoints (covers exact-boundary corner cases)
            ex0 = int(np.floor(np.clip(cx0, 0, cols - 1e-6)))
            ey0 = int(np.floor(np.clip(cy0, 0, rows - 1e-6)))
            ex1 = int(np.floor(np.clip(cx1, 0, cols - 1e-6)))
            ey1 = int(np.floor(np.clip(cy1, 0, rows - 1e-6)))
            if 0 <= ex0 < cols and 0 <= ey0 < rows: age_f[ey0, ex0] = fresh
            if 0 <= ex1 < cols and 0 <= ey1 < rows: age_f[ey1, ex1] = fresh

            # 3) densification fallback for long segments / rare FP ties:
            #    ensure we touch at least 1 cell per grid unit along the longest axis.
            seg_len = max(abs(cx1 - cx0), abs(cy1 - cy0))
            min_samples = int(np.ceil(seg_len)) + 1
            if len(cells) < min_samples:
                n = max(min_samples, 2)
                ts = np.linspace(0.0, 1.0, n, dtype=np.float32)
                xs = cx0 + (cx1 - cx0) * ts
                ys = cy0 + (cy1 - cy0) * ts
                xi = np.clip(np.floor(xs).astype(np.int32, copy=False), 0, cols - 1)
                yi = np.clip(np.floor(ys).astype(np.int32, copy=False), 0, rows - 1)
                age_f[yi, xi] = fresh

        # Clamp → quantize like shader
        np.clip(age_f, F32(0.0), F32(1.0), out=age_f)
        age_u8 = np.rint(age_f * F32(255.0)).astype(np.uint8, copy=False)

        # HARD CLEAR: anything that would render as hidden (or the first dot) → space
        thresh = int(getattr(self.app, "min_show_u8", int(self.app.min_show * 255.0)))
        age_u8[age_u8 <= thresh] = 0

        # Upload (contiguous, byte-aligned)
        self.age[...] = age_u8
        self.age_tex.write(self.age.tobytes(order="C"), alignment=1)


