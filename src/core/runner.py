import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from src.core.constants import ABS_ZERO
from src.core.geometry import DomainGeometry
from src.fluid_dynamics.utils import calculate_velocity_from_sf
from src.heat_transfer.coefficient_smoothing.mushy_zone import (
    get_mushy_zone_temperature_range,
)
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
            "n": self.n,
            "t": self.t,
            "u": self.u.copy(),
            "sf": self.sf.copy(),
            "w": self.w.copy(),
            "v_x": None if self.v_x is None else self.v_x.copy(),
            "v_y": None if self.v_y is None else self.v_y.copy(),
        }

    def restore(self, data: Dict) -> None:
        # Validate required keys
        required_keys = {"n", "t", "u", "sf", "w"}
        missing_keys = required_keys - data.keys()
        if missing_keys:
            raise ValueError(
                f"Missing required keys in checkpoint data: {missing_keys}"
            )

        self.n = data["n"]
        self.t = data["t"]

        for name in ("u", "sf", "w"):
            arr = data[name]
            target = getattr(self, name)
            if target.shape != arr.shape:
                raise ValueError(
                    f"Shape mismatch restoring {name}: {arr.shape} != {target.shape}"
                )
            target[:] = arr

        if data.get("v_x") is not None and self.v_x is not None:
            self.v_x[:] = data["v_x"]
        if data.get("v_y") is not None and self.v_y is not None:
            self.v_y[:] = data["v_y"]


class FileManager:
    def __init__(self, out_dir: str | Path = "../data", prefix: str = "checkpoint"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix

    def checkpoint_path(self, step: int) -> Path:
        return self.out_dir / f"{self.prefix}_{step}.npz"

    def save(self, state: SimulationState, cfg: Optional[dict] = None) -> Path:
        data = state.snapshot()
        if cfg is not None:
            data["cfg"] = dict(cfg)

        path = self.checkpoint_path(state.n)
        np.savez_compressed(path, **data)
        return path

    @staticmethod
    def load(path: str | Path) -> Dict:
        with np.load(path, allow_pickle=True) as f:
            return dict(f)


class PlotManager:
    def __init__(self, cfg):
        self.cfg = cfg

    def plot_temperature(
        self, *, u: np.ndarray, graph_id: int, show_graph: bool = False
    ) -> None:
        try:
            from src.heat_transfer.plotting import plot_temperature as _plot_temperature
        except Exception:
            logging.getLogger(__name__).warning(
                "plot_temperature not found; skipping plot"
            )
            return

        _plot_temperature(
            u=u,
            cfg=self.cfg,
            graph_id=graph_id,
            plot_boundary=True,
            show_graph=show_graph,
        )

    def plot_stream_function(
        self, *, sf: np.ndarray, graph_id: int, show_graph: bool = False
    ) -> None:
        try:
            from src.fluid_dynamics.plotting import (
                plot_stream_function as _plot_stream_function,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "plot_stream_function not found; skipping plot"
            )
            return

        _plot_stream_function(
            stream_function=sf,
            cfg=self.cfg,
            graph_id=graph_id,
            show_graph=show_graph,
        )

    def plot_velocity(
        self,
        *,
        v_x: np.ndarray,
        v_y: np.ndarray,
        u: np.ndarray,
        graph_id: int,
        show_graph: bool = False,
    ) -> None:
        try:
            from src.fluid_dynamics.plotting import (
                plot_velocity_field as _plot_velocity_field,
            )
        except Exception:
            logging.getLogger(__name__).warning(
                "plot_velocity_field not found; skipping plot"
            )
            return

        _plot_velocity_field(
            v_x=v_x,
            v_y=v_y,
            u_dim=u,
            cfg=self.cfg,
            graph_id=graph_id,
            show_graph=show_graph,
        )


class ExperimentRunner:
    def __init__(
        self,
        cfg: ExperimentConfig,
        state: SimulationState,
        heat_solver,
        navier_solver,
        file_manager: Optional[FileManager] = None,
        plot_manager: Optional[PlotManager] = None,
        logger: Optional[logging.Logger] = None,
        checkpoints_dir: Optional[str | Path] = "../data",
        calculate_velocity: Optional[bool] = None,
        save_at: Optional[set[int]] = None,
        plot_at: Optional[set[int]] = None,
        log_at: Optional[set[int]] = None,
    ) -> None:
        self.cfg: ExperimentConfig = cfg
        self.geometry: DomainGeometry = cfg.geometry
        self.state: SimulationState = state
        self.file_manager: FileManager = file_manager or FileManager(
            out_dir=checkpoints_dir
        )
        self.plot_manager = plot_manager or PlotManager(cfg)
        self.logger = logger or logging.getLogger("experiment")
        self.heat_solver = heat_solver
        self.navier_solver = navier_solver
        self.calculate_velocity = (
            calculate_velocity if calculate_velocity is not None else False
        )
        self.save_at = set() if save_at is None else save_at
        self.plot_at = set() if plot_at is None else plot_at
        self.log_at = set() if log_at is None else log_at
        self._callbacks: Dict[str, list[Callable[[SimulationState], None]]] = {}

        if self.calculate_velocity:
            if self.state.v_x is None or self.state.v_y is None:
                raise ValueError(
                    "calculate_velocity=True requires v_x and v_y to be initialized in SimulationState"
                )

    def register_callback(
        self, event: str, fn: Callable[[SimulationState], None]
    ) -> None:
        self._callbacks.setdefault(event, []).append(fn)

    def run(self) -> SimulationState:
        self._call_event("on_start")
        start_time = time.perf_counter()

        try:
            for n in range(1, self.geometry.n_t + 1):
                self.step()

                if self.state.n in self.save_at:
                    path = self.file_manager.save(self.state, cfg=dict(self.cfg))
                    self._call_event("on_save")
                    self.logger.info(f"Saved checkpoint: {path}")

                if self.state.n in self.plot_at:
                    try:
                        u_dim = self.state.u * self.cfg.delta_u + self.cfg.u_ref
                        self.plot_manager.plot_temperature(
                            u=u_dim,
                            graph_id=self.state.n,
                            show_graph=False,
                        )
                        self.plot_manager.plot_stream_function(
                            sf=self.state.sf, graph_id=self.state.n, show_graph=False
                        )
                        if (
                            self.calculate_velocity
                            and self.state.v_x is not None
                            and self.state.v_y is not None
                        ):
                            self.plot_manager.plot_velocity(
                                v_x=self.state.v_x,
                                v_y=self.state.v_y,
                                u=u_dim,
                                graph_id=self.state.n,
                                show_graph=False,
                            )
                        self._call_event("on_plot")
                    except Exception:
                        self.logger.exception("Plotting failed; continuing")

                if self.state.n in self.log_at:
                    elapsed = time.perf_counter() - start_time
                    steps_done = self.state.n
                    total = self.geometry.n_t
                    avg = elapsed / max(1, steps_done)
                    remaining = avg * (total - steps_done)
                    self.logger.info(
                        "Step %d / %d: t = %.1f s, elapsed = %.2f s, est remaining = %.2f s",
                        steps_done,
                        total,
                        self.state.t,
                        elapsed,
                        remaining,
                    )
                    u_dim_c = (
                        self.state.u * self.cfg.delta_u + self.cfg.u_ref + ABS_ZERO
                    )
                    self.logger.info(
                        f"Maximum temperature value: {np.max(u_dim_c):.2f} C"
                    )
                    self.logger.info(
                        f"Minimum temperature value: {np.min(u_dim_c):.2f} C"
                    )

        except Exception as exc:
            self._call_event("on_error")
            self.logger.exception("Exception during simulation run")

            try:
                path = self.file_manager.save(self.state, cfg=dict(self.cfg))
                self.logger.info(f"Saved error checkpoint: {path}")
            except Exception:
                self.logger.exception("Failed to save error checkpoint")

            raise

        finally:
            self._call_event("on_finish")

        return self.state

    def step(self) -> None:
        self._call_event("pre_step")

        n = self.state.n + 1
        t = n * self.geometry.dt

        try:
            delta = self.cfg.delta_nd
            if delta is None:
                delta = get_mushy_zone_temperature_range(
                    self.state.u, self.cfg.u_pt_nd, n_nodes=1
                )
            self.state.u[:, :] = self.heat_solver.solve(
                u=self.state.u, sf=self.state.sf, delta=delta, time=t
            )

            delta_flow = self.cfg.delta_flow_nd
            if delta_flow is None:
                delta_flow = get_mushy_zone_temperature_range(
                    self.state.u, self.cfg.u_pt_nd, n_nodes=1
                )

            sf_new, w_new = self.navier_solver.solve(
                w=self.state.w,
                sf=self.state.sf,
                u=self.state.u,
                delta=delta_flow,
                time=t,
            )
            self.state.sf[:, :] = sf_new
            self.state.w[:, :] = w_new

            if self.calculate_velocity:
                calculate_velocity_from_sf(
                    self.state.sf, self.state.v_x, self.state.v_y, self.cfg
                )

            self.state.n = n
            self.state.t = t

        except Exception:
            self.logger.exception("Error during solver step")
            raise

        self._call_event("post_step")

    def save_checkpoint(self, path: Optional[str | Path] = None) -> Path:
        if path is None:
            return self.file_manager.save(self.state, cfg=dict(self.cfg))
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
