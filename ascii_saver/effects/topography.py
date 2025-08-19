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


class TopographyConfig(TypedDict):
    GRADIENT_GAMMA: float
    NODE_RADIUS: float
    NUM_NODES: int
    MAX_VELOCITY: float
    MIN_SHOW: float
    GAUSS_SIGMA: float
    GAUSS_MULT: float
    EXPOSE_ALPHA: float


DEFAULT_CONFIG: Final[TopographyConfig] = {
    "GRADIENT_GAMMA": 1.6,
    "NODE_RADIUS": 10.0,
    "NUM_NODES": 25,
    "MAX_VELOCITY": 1.0,
    "MIN_SHOW": 0.01,
    "GAUSS_SIGMA": 8.0,
    "GAUSS_MULT": 2.5,
    "EXPOSE_ALPHA": 0.20,
}


@register_effect
class Topography(Effect):
    NAME: str = "topography"
    CONFIG_DEFAULTS: TopographyConfig = DEFAULT_CONFIG

    gradient_gamma: float
    node_radius: float
    num_nodes: int
    max_velocity: float
    min_show: float
    gauss_sigma: float
    gauss_mult: float
    expose_alpha: float

    nodes: NDArray[np.float32]
    velocities: NDArray[np.float32]
    _upload_nodes: NDArray[np.float32]
    scale_ema: float
    prog: Any

    def init(self) -> None:
        cfg = self.CONFIG_DEFAULTS
        self.gradient_gamma = float(cfg["GRADIENT_GAMMA"])
        self.node_radius = float(cfg["NODE_RADIUS"])
        self.num_nodes = int(cfg["NUM_NODES"])
        self.max_velocity = float(cfg["MAX_VELOCITY"])
        self.min_show = float(cfg["MIN_SHOW"])
        sigma_val = float(cfg["GAUSS_SIGMA"])
        self.gauss_sigma = float(self.node_radius * 0.8) if sigma_val <= 0.0 else sigma_val
        self.gauss_mult = float(cfg["GAUSS_MULT"])
        self.expose_alpha = float(cfg["EXPOSE_ALPHA"])

        grid = self.app.grid
        rng = np.random.default_rng()

        node_count = self.num_nodes
        self.nodes = np.empty((node_count, 2), dtype=np.float32)
        self.velocities = np.empty((node_count, 2), dtype=np.float32)

        self.nodes[:, 0] = rng.random(node_count, dtype=np.float32) * np.float32(grid.cols)
        self.nodes[:, 1] = rng.random(node_count, dtype=np.float32) * np.float32(grid.rows)

        angle: NDArray[np.float32] = rng.random(node_count, dtype=np.float32) * np.float32(2.0 * np.pi)
        speed: NDArray[np.float32] = rng.random(node_count, dtype=np.float32) * np.float32(self.max_velocity)
        cos_angle: NDArray[np.float32] = np.cos(angle).astype(np.float32, copy=False)
        sin_angle: NDArray[np.float32] = np.sin(angle).astype(np.float32, copy=False)
        self.velocities[:, 0] = cos_angle * speed
        self.velocities[:, 1] = sin_angle * speed

        self.prog = self.app.renderer.make_program(VERT_SRC, FRAG_GLYPH_FIELD)
        self.app.bind_common_glyph_uniforms(self.prog)

        prog_ref: Any = self.prog
        prog_ref["u_sigma"].value = float(self.gauss_sigma)
        prog_ref["u_mult"].value = float(self.gauss_mult)

        self._upload_nodes = np.zeros((256, 2), dtype=np.float32)
        self.scale_ema = self._compute_initial_scale()

    def update(self, dt: float) -> None:
        step = max(0.0, float(dt)) * 60.0
        if step == 0.0:
            return

        grid = self.app.grid
        step32 = np.float32(step)
        self.nodes[:, 0] = (self.nodes[:, 0] + self.velocities[:, 0] * step32) % np.float32(grid.cols)
        self.nodes[:, 1] = (self.nodes[:, 1] + self.velocities[:, 1] * step32) % np.float32(grid.rows)

        field_max = self._approx_field_max(self.nodes)
        target = 1.0 / max(1e-6, field_max)
        self.scale_ema = (1.0 - self.expose_alpha) * self.scale_ema + self.expose_alpha * target

    def render(self) -> None:
        self.app.bind_common_glyph_uniforms(self.prog)
        prog_ref: Any = self.prog

        prog_ref["u_scale"].value = float(self.scale_ema)
        prog_ref["u_num_nodes"].value = int(self.nodes.shape[0])

        node_count = self.nodes.shape[0]
        self._upload_nodes[:node_count] = self.nodes
        if node_count < self._upload_nodes.shape[0]:
            self._upload_nodes[node_count:] = 0.0

        try:
            prog_ref["u_nodes"].write(self._upload_nodes.tobytes())
        except KeyError:
            prog_ref["u_nodes[0]"].write(self._upload_nodes.tobytes())

        self.app.renderer.draw_fullscreen(self.prog, {0: self.app.renderer.atlas_tex})

    def _compute_initial_scale(self) -> float:
        field_max = self._approx_field_max(self.nodes)
        return 1.0 / max(1e-6, field_max)

    def _approx_field_max(self, nodes_xy: NDArray[np.float32]) -> float:
        grid = self.app.grid
        samp_cols = max(2, min(grid.cols, 64))
        samp_rows = max(2, min(grid.rows, 36))

        xs: NDArray[np.float32] = np.linspace(0.5, grid.cols - 0.5, samp_cols, dtype=np.float32)
        ys: NDArray[np.float32] = np.linspace(0.5, grid.rows - 0.5, samp_rows, dtype=np.float32)

        gx_arr, gy_arr = np.meshgrid(xs, ys, indexing="xy")
        gx32: NDArray[np.float32] = np.asarray(gx_arr, dtype=np.float32)
        gy32: NDArray[np.float32] = np.asarray(gy_arr, dtype=np.float32)
        gx3: NDArray[np.float32] = gx32[..., None]
        gy3: NDArray[np.float32] = gy32[..., None]

        nx: NDArray[np.float32] = nodes_xy[:, 0][None, None, :]
        ny: NDArray[np.float32] = nodes_xy[:, 1][None, None, :]

        dx: NDArray[np.float32] = np.minimum(
            np.abs(gx3 - nx), np.float32(grid.cols) - np.abs(gx3 - nx)
        ).astype(np.float32, copy=False)
        dy: NDArray[np.float32] = np.minimum(
            np.abs(gy3 - ny), np.float32(grid.rows) - np.abs(gy3 - ny)
        ).astype(np.float32, copy=False)
        d2: NDArray[np.float32] = (dx * dx + dy * dy).astype(np.float32, copy=False)

        sigma2 = np.float32(self.gauss_sigma * self.gauss_sigma)
        w_f64 = np.exp(-0.5 * (d2 / sigma2))
        weights: NDArray[np.float32] = (
            w_f64.astype(np.float32, copy=False) * np.float32(self.gauss_mult)
        )

        acc = cast(NDArray[np.float32], np.sum(weights, axis=-1, dtype=np.float32))
        return float(acc.max())
