from __future__ import annotations

from typing import Any, Final, TypedDict, cast
import numpy as np
from numpy.typing import NDArray

from ascii_saver.framework import (
    Effect,
    register_effect,
    VERT_SRC,
    FRAG_GLYPH_FIELD,
)

# ---- Typed config for this effect (UI/launcher can introspect) ---------------

class TopographyConfig(TypedDict):
    GRADIENT_GAMMA: float   # not used directly (framework has global gamma), kept for UI parity
    NODE_RADIUS: float
    NUM_NODES: int
    MAX_VELOCITY: float     # cells / frame at 60fps baseline
    MIN_SHOW: float         # handled by framework; kept for parity
    GAUSS_SIGMA: float      # if <= 0, recomputed as NODE_RADIUS * 0.8 at init
    GAUSS_MULT: float
    EXPOSE_ALPHA: float     # EMA for auto-exposure

DEFAULT_CONFIG: Final[TopographyConfig] = {
    "GRADIENT_GAMMA": 1.6,
    "NODE_RADIUS": 10.0,
    "NUM_NODES": 25,
    "MAX_VELOCITY": 1.0,
    "MIN_SHOW": 0.01,
    "GAUSS_SIGMA": 8.0,     # 0.8 * 10.0; recomputed from NODE_RADIUS anyway
    "GAUSS_MULT": 2.5,
    "EXPOSE_ALPHA": 0.20,
}


@register_effect
class Topography(Effect):
    NAME: str = "topography"

    # Expose defaults so launchers can prefill forms
    CONFIG_DEFAULTS: TopographyConfig = DEFAULT_CONFIG

    # ---- Instance attributes (typed) -----------------------------------------
    # Config (copied from CONFIG_DEFAULTS at init; GUI may override before init)
    gradient_gamma: float
    node_radius: float
    num_nodes: int
    max_velocity: float
    min_show: float
    gauss_sigma: float
    gauss_mult: float
    expose_alpha: float

    # Runtime state
    nodes: NDArray[np.float32]           # [N,2] positions (cell coords)
    vels: NDArray[np.float32]            # [N,2] velocities (cells per 1/60s)
    _u_nodes_buf: NDArray[np.float32]    # [256,2] upload buffer
    scale_ema: float
    prog: Any                            # moderngl.Program (typed Any to satisfy strict stubs)

    # ---------------- Effect lifecycle ----------------

    def init(self) -> None:
        # Copy defaults (your Tk pane can modify attributes before init if desired)
        cfg = self.CONFIG_DEFAULTS
        self.gradient_gamma = float(cfg["GRADIENT_GAMMA"])
        self.node_radius    = float(cfg["NODE_RADIUS"])
        self.num_nodes      = int(cfg["NUM_NODES"])
        self.max_velocity   = float(cfg["MAX_VELOCITY"])
        self.min_show       = float(cfg["MIN_SHOW"])
        gs = float(cfg["GAUSS_SIGMA"])
        self.gauss_sigma    = float(self.node_radius * 0.8) if gs <= 0.0 else gs
        self.gauss_mult     = float(cfg["GAUSS_MULT"])
        self.expose_alpha   = float(cfg["EXPOSE_ALPHA"])

        g = self.app.grid
        rng = np.random.default_rng()

        # Allocate state arrays
        n = self.num_nodes
        self.nodes = np.empty((n, 2), dtype=np.float32)
        self.vels  = np.empty((n, 2), dtype=np.float32)

        # positions in cell coordinates
        self.nodes[:, 0] = rng.random(n, dtype=np.float32) * np.float32(g.cols)
        self.nodes[:, 1] = rng.random(n, dtype=np.float32) * np.float32(g.rows)

        # random headings with speed in [0, max_velocity)
        ang: NDArray[np.float32] = rng.random(n, dtype=np.float32) * np.float32(2.0 * np.pi)
        spd: NDArray[np.float32] = rng.random(n, dtype=np.float32) * np.float32(self.max_velocity)
        # cos/sin preserve float32 when input is float32, but be explicit for the checker
        cos_ang: NDArray[np.float32] = np.cos(ang).astype(np.float32, copy=False)
        sin_ang: NDArray[np.float32] = np.sin(ang).astype(np.float32, copy=False)
        self.vels[:, 0] = cos_ang * spd
        self.vels[:, 1] = sin_ang * spd

        # GL program & common uniforms
        self.prog = self.app.renderer.make_program(VERT_SRC, FRAG_GLYPH_FIELD)
        self.app.bind_common_glyph_uniforms(self.prog)

        # Effect-specific static uniforms
        p: Any = self.prog
        p["u_sigma"].value = float(self.gauss_sigma)
        p["u_mult"].value  = float(self.gauss_mult)

        # Uniform array buffer (vec2[256])
        self._u_nodes_buf = np.zeros((256, 2), dtype=np.float32)

        # Initial auto-exposure
        self.scale_ema = self._compute_initial_scale()

    def update(self, dt: float) -> None:
        # Integrate using a 60fps baseline
        step = max(0.0, float(dt)) * 60.0
        if step == 0.0:
            return

        g = self.app.grid
        step32 = np.float32(step)
        self.nodes[:, 0] = (self.nodes[:, 0] + self.vels[:, 0] * step32) % np.float32(g.cols)
        self.nodes[:, 1] = (self.nodes[:, 1] + self.vels[:, 1] * step32) % np.float32(g.rows)

        # Auto-exposure using coarse estimate + EMA
        m = self._approx_field_max(self.nodes)
        target = 1.0 / max(1e-6, m)
        self.scale_ema = (1.0 - self.expose_alpha) * self.scale_ema + self.expose_alpha * target

    def render(self) -> None:
        # Rebind common uniforms in case grid/app changed
        self.app.bind_common_glyph_uniforms(self.prog)
        p: Any = self.prog

        # Dynamic uniforms
        p["u_scale"].value = float(self.scale_ema)
        p["u_num_nodes"].value = int(self.nodes.shape[0])

        # Pack vec2[256] and upload once
        n = self.nodes.shape[0]
        self._u_nodes_buf[:n] = self.nodes
        if n < self._u_nodes_buf.shape[0]:
            self._u_nodes_buf[n:] = 0.0

        # Some drivers expose arrays as "u_nodes[0]"
        try:
            p["u_nodes"].write(self._u_nodes_buf.tobytes())
        except KeyError:
            p["u_nodes[0]"].write(self._u_nodes_buf.tobytes())

        self.app.renderer.draw_fullscreen(self.prog, {0: self.app.renderer.atlas_tex})

    # ---------------- Internals ----------------

    def _compute_initial_scale(self) -> float:
        m = self._approx_field_max(self.nodes)
        return 1.0 / max(1e-6, m)

    def _approx_field_max(self, nodes_xy: NDArray[np.float32]) -> float:
        """Fast approximate max of the summed Gaussians on a coarse grid (toroidal)."""
        g = self.app.grid
        samp_cols = max(2, min(g.cols, 64))
        samp_rows = max(2, min(g.rows, 36))

        xs: NDArray[np.float32] = np.linspace(
            0.5, g.cols - 0.5, samp_cols, dtype=np.float32
        )
        ys: NDArray[np.float32] = np.linspace(
            0.5, g.rows - 0.5, samp_rows, dtype=np.float32
        )

        # meshgrid return dtype can confuse stubs; make it explicit and avoid reassigning names
        gx_arr, gy_arr = np.meshgrid(xs, ys, indexing="xy")
        gx32: NDArray[np.float32] = np.asarray(gx_arr, dtype=np.float32)
        gy32: NDArray[np.float32] = np.asarray(gy_arr, dtype=np.float32)
        gx3: NDArray[np.float32] = gx32[..., None]
        gy3: NDArray[np.float32] = gy32[..., None]

        nx: NDArray[np.float32] = nodes_xy[:, 0][None, None, :]
        ny: NDArray[np.float32] = nodes_xy[:, 1][None, None, :]

        dx: NDArray[np.float32] = np.minimum(np.abs(gx3 - nx),
                                             np.float32(g.cols) - np.abs(gx3 - nx)).astype(np.float32, copy=False)
        dy: NDArray[np.float32] = np.minimum(np.abs(gy3 - ny),
                                             np.float32(g.rows) - np.abs(gy3 - ny)).astype(np.float32, copy=False)
        d2: NDArray[np.float32] = (dx * dx + dy * dy).astype(np.float32, copy=False)

        sigma2 = np.float32(self.gauss_sigma * self.gauss_sigma)
        w_f64 = np.exp(-0.5 * (d2 / sigma2))  # exp returns float64 by default
        w: NDArray[np.float32] = (w_f64.astype(np.float32, copy=False)
                                  * np.float32(self.gauss_mult))

        # np.sum stub returns scalar | NDArray, cast to keep the checker happy
        acc = cast(NDArray[np.float32], np.sum(w, axis=-1, dtype=np.float32, keepdims=False))
        return float(acc.max())
