from __future__ import annotations

from typing import Any, Dict, List, Mapping, Type, TypeVar, cast
import tkinter as tk
from tkinter import ttk, messagebox

import pkgutil
import importlib

from ascii_saver.framework import (
    App,
    Effect,
    get_effect,
    list_effects,
    DEFAULT_FONT,
    DEFAULT_SIZE,
    DEFAULT_FPS,
    DEFAULT_GAMMA,
    DEFAULT_MIN_SHOW,
    DEFAULT_PALETTE,
)

# ---------------------------------------------------------------------------

def _ensure_effects_imported() -> None:
    import ascii_saver.effects
    for _, modname, _ in pkgutil.iter_modules(ascii_saver.effects.__path__):
        importlib.import_module(f"ascii_saver.effects.{modname}")


def _parse_palette(s: str) -> List[str]:
    """
    Accept comma or space separated color strings like '#ff00ff'.
    Return a non-empty list; fall back to DEFAULT_PALETTE if none found.
    """
    raw = [t.strip() for t in s.replace(",", " ").split()]
    out: List[str] = []
    for it in raw:
        if len(it) == 7 and it.startswith("#"):
            out.append(it)
    return out or list(DEFAULT_PALETTE)

# Typed helper: grid the widget and return it so one-liners don't drop the ref.
TWidget = TypeVar("TWidget", bound=tk.Misc)
def grid_(widget: TWidget, **kw: Any) -> TWidget:
    """Place widget with grid and return it (Pylance-friendly)."""
    cast(Any, widget).grid(**kw)
    return widget

def _effect_config_defaults(eff_cls: Type[Effect]) -> Dict[str, Any]:
    """
    Return a shallow copy of CONFIG_DEFAULTS if present, else {}.
    The GUI uses these as fields for the per-effect config page.
    """
    cfg = getattr(eff_cls, "CONFIG_DEFAULTS", None)
    if isinstance(cfg, Mapping):
        return dict(cast(Mapping[str, Any], cfg))  # ensure known key/value types
    return {}

def _coerce_value(text: str, default_value: Any) -> Any:
    """
    Convert string input to the type of default_value (int/float/bool/str).
    Falls back to the original default_value on parse error.
    """
    try:
        if isinstance(default_value, bool):
            t = text.strip().lower()
            if t in {"1", "true", "yes", "y", "on"}:
                return True
            if t in {"0", "false", "no", "n", "off"}:
                return False
            return bool(int(text))
        if isinstance(default_value, int) and not isinstance(default_value, bool):
            return int(float(text))  # allow "10.0" as "10"
        if isinstance(default_value, float):
            return float(text)
        # str or other types → leave as string
        return text
    except Exception:
        return default_value

# ---------------------------------------------------------------------------

class LauncherGUI:
    def __init__(self) -> None:
        _ensure_effects_imported()

        self.root: tk.Tk = tk.Tk()
        self.root.title("ASCII ASCIISaver Launcher")
        self.root.resizable(False, False)
        self.root.geometry("+300+200")  # center-ish

        # Tk variables (global/common config with sensible defaults)
        self.effect_var  = tk.StringVar(value="")
        self.font_var    = tk.StringVar(value=DEFAULT_FONT)
        self.size_var    = tk.StringVar(value=str(DEFAULT_SIZE))
        self.fps_var     = tk.StringVar(value=str(DEFAULT_FPS))
        self.gamma_var   = tk.StringVar(value=str(DEFAULT_GAMMA))
        self.minshow_var = tk.StringVar(value=str(DEFAULT_MIN_SHOW))
        self.palette_var = tk.StringVar(value=" ".join(DEFAULT_PALETTE))
        self.frames_var  = tk.StringVar(value="0")
        self.profile_var = tk.BooleanVar(value=False)

        # Built/assigned in _build
        self.effect_combo: ttk.Combobox

        # Per-effect config window state (created in _on_next)
        self._cfg_win: tk.Toplevel | None = None
        self._cfg_vars: Dict[str, tk.StringVar] = {}

        # Build main form
        self._build()

        # Populate effects
        effects = list_effects()
        if not effects:
            messagebox.showerror("No effects found", "No ascii_saver effects were discovered.")
        else:
            self.effect_combo["values"] = tuple(effects)
            self.effect_var.set(effects[0])

    def _build(self) -> None:
        pad = {"padx": 10, "pady": 6}

        frm = grid_(ttk.Frame(self.root), row=0, column=0, sticky="nsew")

        # Effect row
        grid_(ttk.Label(frm, text="Effect:"), row=0, column=0, sticky="w", **pad)
        self.effect_combo = grid_(
            ttk.Combobox(frm, textvariable=self.effect_var, state="readonly", width=24),
            row=0, column=1, sticky="ew", **pad
        )

        # Font
        grid_(ttk.Label(frm, text="Font family:"), row=1, column=0, sticky="w", **pad)
        grid_(ttk.Entry(frm, textvariable=self.font_var, width=26), row=1, column=1, sticky="ew", **pad)

        # Size
        grid_(ttk.Label(frm, text="Font size (pt):"), row=2, column=0, sticky="w", **pad)
        grid_(ttk.Entry(frm, textvariable=self.size_var, width=26), row=2, column=1, sticky="ew", **pad)

        # FPS
        grid_(ttk.Label(frm, text="Target FPS:"), row=3, column=0, sticky="w", **pad)
        grid_(ttk.Entry(frm, textvariable=self.fps_var, width=26), row=3, column=1, sticky="ew", **pad)

        # Gamma
        grid_(ttk.Label(frm, text="Gamma:"), row=4, column=0, sticky="w", **pad)
        grid_(ttk.Entry(frm, textvariable=self.gamma_var, width=26), row=4, column=1, sticky="ew", **pad)

        # Min show
        grid_(ttk.Label(frm, text="Min show (0..1):"), row=5, column=0, sticky="w", **pad)
        grid_(ttk.Entry(frm, textvariable=self.minshow_var, width=26), row=5, column=1, sticky="ew", **pad)

        # Palette
        grid_(ttk.Label(frm, text="Palette (#RRGGBB):"), row=6, column=0, sticky="w", **pad)
        grid_(ttk.Entry(frm, textvariable=self.palette_var, width=26), row=6, column=1, sticky="ew", **pad)
        grid_(ttk.Label(frm, text="(space or comma separated)"), row=7, column=1, sticky="w", padx=10)

        # Frames
        grid_(ttk.Label(frm, text="Frames (0=infinite):"), row=8, column=0, sticky="w", **pad)
        grid_(ttk.Entry(frm, textvariable=self.frames_var, width=26), row=8, column=1, sticky="ew", **pad)

        # Profile
        grid_(ttk.Checkbutton(frm, text="Enable profiling", variable=self.profile_var),
              row=9, column=1, sticky="w", **pad)

        # Buttons
        btns = grid_(ttk.Frame(frm), row=10, column=0, columnspan=2, sticky="ew", pady=(10, 12))
        ttk.Button(btns, text="Quit", command=self.root.destroy).pack(side="right", padx=8)
        ttk.Button(btns, text="Next", command=self._on_next).pack(side="right")

    def _on_next(self) -> None:
        eff_name = self.effect_var.get().strip()
        eff_cls = get_effect(eff_name)
        if eff_cls is None:
            messagebox.showerror("Invalid effect", f"Effect '{eff_name}' was not found.")
            return

        # Create / reset config window
        if self._cfg_win is not None and tk.Toplevel.winfo_exists(self._cfg_win):
            self._cfg_win.destroy()

        cfg_win = tk.Toplevel(self.root)
        self._cfg_win = cfg_win
        cfg_win.title(f"{eff_name} – Configuration")
        cfg_win.resizable(False, False)

        frm = ttk.Frame(cfg_win, padding=10)
        frm.grid(row=0, column=0, sticky="nsew")

        # Build inputs from CONFIG_DEFAULTS if present
        defaults = _effect_config_defaults(eff_cls)
        self._cfg_vars = {}

        if not defaults:
            # No extra config for this effect
            ttk.Label(frm, text="No effect-specific settings.").grid(row=0, column=0, sticky="w")
            ttk.Button(frm, text="Start", command=lambda e=eff_cls: self._on_start(e)).grid(
                row=1, column=0, sticky="e", pady=(10, 4)
            )
            return

        # Render labeled entries
        row_idx = 0
        for key in sorted(defaults.keys()):
            val = defaults[key]
            ttk.Label(frm, text=key).grid(row=row_idx, column=0, sticky="w", padx=6, pady=4)
            var = tk.StringVar(value=str(val))
            self._cfg_vars[key] = var
            ttk.Entry(frm, textvariable=var, width=22).grid(row=row_idx, column=1, sticky="ew", padx=6, pady=4)
            row_idx += 1

        ttk.Button(frm, text="Start", command=lambda e=eff_cls: self._on_start(e)).grid(
            row=row_idx, column=1, sticky="e", pady=(10, 4)
        )

    def _on_start(self, eff_cls: Type[Effect]) -> None:
        # Validate global inputs
        try:
            size = int(self.size_var.get().strip())
            fps = int(self.fps_var.get().strip())
            gamma = float(self.gamma_var.get().strip())
            min_show = float(self.minshow_var.get().strip())
            frames = int(self.frames_var.get().strip())
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Size/FPS/Frames must be integers; Gamma/Min show must be numbers.",
            )
            return

        if size <= 0 or fps <= 0:
            messagebox.showerror("Invalid input", "Font size and FPS must be positive.")
            return

        palette_list = _parse_palette(self.palette_var.get())

        # Apply per-effect overrides by mutating CONFIG_DEFAULTS on the class
        # so the effect reads the updated values during its init().
        defaults = _effect_config_defaults(eff_cls)
        if defaults and self._cfg_vars:
            new_defaults: Dict[str, Any] = dict(defaults)
            for k, var in self._cfg_vars.items():
                new_defaults[k] = _coerce_value(var.get(), defaults.get(k))
            setattr(eff_cls, "CONFIG_DEFAULTS", new_defaults)

        # Close windows before creating the fullscreen GL window
        if self._cfg_win is not None and tk.Toplevel.winfo_exists(self._cfg_win):
            self._cfg_win.destroy()
        self.root.withdraw()
        self.root.update_idletasks()
        self.root.destroy()

        # Launch the app
        App(
            eff_cls,
            font=self.font_var.get().strip() or DEFAULT_FONT,
            size=size,
            fps=fps,
            gamma=gamma,
            min_show=min_show,
            palette=palette_list,
            frames=frames,
            profile=bool(self.profile_var.get()),
        ).run()

    def run(self) -> None:
        self.root.mainloop()

# ---------------------------------------------------------------------------

def main() -> None:
    LauncherGUI().run()

if __name__ == "__main__":
    main()
