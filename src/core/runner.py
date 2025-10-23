import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from src.core.geometry import DomainGeometry
from src.parameters.config import ExperimentConfig


@dataclass
class SimulationState:
    u: np.ndarray
    sf: np.ndarray
    w: np.ndarray
    v_x: Optional[np.ndarray] = None
    v_y: Optional[np.ndarray] = None

    n: int = 0
    t: float = 0.0

    def snapshot(self) -> Dict:
        return {
            "n": int(self.n),
            "t": float(self.t),
            "u": np.array(self.u, copy=True),
            "sf": np.array(self.sf, copy=True),
            "w": np.array(self.w, copy=True),
            "v_x": None if self.v_x is None else np.array(self.v_x, copy=True),
            "v_y": None if self.v_y is None else np.array(self.v_y, copy=True),
        }

    def restore(self, data: Dict) -> None:
        self.n = int(data["n"])
        self.t = float(data["t"])

        for name in ("u", "sf", "w"):
            arr = data[name]
            target = getattr(self, name)
            if target.shape != arr.shape:
                raise ValueError(f"Shape mismatch restoring {name}: {arr.shape} != {target.shape}")
            target[:] = arr
        if data.get("v_x") is not None and self.v_x is not None:
            self.v_x[:] = data["v_x"]
        if data.get("v_y") is not None and self.v_y is not None:
            self.v_y[:] = data["v_y"]


class FileManager:
    def __init__(self, out_dir: str | Path = "./checkpoints", prefix: str = "checkpoint"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

    def checkpoint_path(self, step: int) -> Path:
        return self.out_dir / f"{self.prefix}_step{step:06d}.npz"

    def save(self, state: SimulationState, cfg: Optional[dict] = None) -> Path:
        data = state.snapshot()
        if cfg is not None:
           data["cfg"] = dict(cfg)

        path = self.checkpoint_path(state.n)
        tmp = path.with_suffix(path.suffix + ".tmp")
        # write to temp file and rename
        np.savez_compressed(tmp, **data)
        tmp.replace(path)
        return path

    def load(self, path: str | Path) -> Dict:
        with np.load(path, allow_pickle=True) as f:
            return dict(f)


class ExperimentRunner:
    def __init__(
            self,
            cfg: ExperimentConfig,
            state: SimulationState,
            heat_solver,
            navier_solver,
    ) -> None:
        self.cfg: ExperimentConfig = cfg
        self.geometry: DomainGeometry = cfg.geometry
        self.state: SimulationState = state
        self.file_manager: FileManager = FileManager()
        self.heat_solver = heat_solver
        self.navier_solver = navier_solver

        self._callbacks: Dict[str, list[Callable[[SimulationState], None]]] = {}

    def register_callback(self, event: str, fn: Callable[[SimulationState], None]) -> None:
        self._callbacks.setdefault(event, []).append(fn)

    def run(self) -> SimulationState:
        self._call_event("on_start")
        start_time = time.perf_counter()

        try:
            for step in range(1, self.geometry.n_t):
                self.step()

                if self.state.n % getattr(self.cfg, "save_interval", 1) == 0:
                    try:
                        path = self.file_manager.save(self.state, cfg=getattr(self.cfg, "dict", None))
                        self._call_event("on_save")
                        self.logger.info(f"Saved checkpoint: {path}")
                    except Exception as exc:
                        self.logger.exception("Failed to save checkpoint: %s", exc)

                    try:
                        self.plot_manager.plot_temperature(
                            u=self.state.u * getattr(self.cfg, "delta_u", 1) + getattr(self.cfg, "u_ref", 0),
                            time=self.state.t,
                            graph_id=self.state.n,
                            show_graph=False)
                        self._call_event("on_plot")
                    except Exception:
                        self.logger.exception("Plotting failed; continuing")

                # log diagnostics
                if self.state.n % getattr(self.cfg, "log_interval", 1) == 0:
                    elapsed = time.perf_counter() - start_time
                    steps_done = self.state.n
                    avg = elapsed / max(1, steps_done)
                    remaining = avg * (total - steps_done)
                    self.logger.info(
                        "Step %d / %d: t=%.3f s, elapsed=%.2f s, est remaining=%.2f s",
                        steps_done,
                        total,
                        self.state.t,
                        elapsed,
                        remaining,
                    )

        except Exception as exc:
            self._call_event("on_error")
            self.logger.exception("Uncaught exception during run: %s", exc)

            try:
                self.file_manager.save(self.state, cfg=getattr(self.cfg, "dict", None))
                self.logger.info("Saved checkpoint after exception")
            except Exception:
                self.logger.exception("Failed to save checkpoint after exception")
            raise

        finally:
            self._call_event("on_finish")

        return self.state

    def step(self) -> None:
        self._call_event("pre_step")

        n = self.state.n + 1
        t = n * self.geometry.dt

        try:
            delta = getattr(self.cfg, "delta", (0.01, 0.01))
            self.state.u[:, :] = self.heat_solver.solve(u=self.state.u, sf=self.state.sf, delta=delta, time=t)

            sf_new, w_new = self.navier_solver.solve(w=self.state.w, sf=self.state.sf, u=self.state.u,
                                                     delta=getattr(self.cfg, "navier_delta", 0.008), time=t)
            self.state.sf[:, :] = sf_new
            self.state.w[:, :] = w_new

        except Exception:
            self.logger.exception("Error during solvers step")
            raise

        self.state.n = n
        self.state.t = t

        self._call_event("post_step")

    def save_checkpoint(self, path: Optional[str | Path] = None) -> Path:
        if path is None:
            return self.file_manager.save(self.state, cfg=getattr(self.cfg, "dict", None))
        else:
            data = self.state.snapshot()
            np.savez_compressed(path, **data)
            return Path(path)

    def load_checkpoint(self, path: str | Path) -> None:
        data = self.file_manager.load(path)
        self.state.restore(data)
        self.logger.info("Loaded checkpoint: %s", path)

    def _call_event(self, event: str) -> None:
        for fn in self._callbacks.get(event, []):
            try:
                fn(self.state)
            except Exception:
                self.logger.exception("Callback for %s failed", event)
