import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from src.core.geometry import DomainGeometry
from src.fluid_dynamics.utils import calculate_velocity_from_sf
from src.heat_transfer.coefficient_smoothing.mushy_zone import (
    get_mushy_zone_temperature_range,
)
from src.parameters.config import ExperimentConfig

MetricFn = Callable[["SimulationState"], float | str]
StepCallback = Callable[["SimulationState"], None]


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


def load_checkpoint(path: str | Path) -> Dict:
    with np.load(path, allow_pickle=True) as f:
        return dict(f)


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


class PlotManager:
    def __init__(self, cfg):
        self.cfg = cfg
        log = logging.getLogger(__name__)

        try:
            from src.heat_transfer.plotting import plot_temperature

            self._plot_temperature = plot_temperature
        except ImportError:
            log.warning("plot_temperature not found; temperature plots will be skipped")
            self._plot_temperature = None

        try:
            from src.fluid_dynamics.plotting import plot_stream_function

            self._plot_stream_function = plot_stream_function
        except ImportError:
            log.warning(
                "plot_stream_function not found; stream function plots will be skipped"
            )
            self._plot_stream_function = None

        try:
            from src.fluid_dynamics.plotting import plot_velocity_field

            self._plot_velocity_field = plot_velocity_field
        except ImportError:
            log.warning("plot_velocity_field not found; velocity plots will be skipped")
            self._plot_velocity_field = None

    def plot_temperature(
        self, *, u: np.ndarray, graph_id: int, show_graph: bool = False
    ) -> None:
        if self._plot_temperature is None:
            return
        self._plot_temperature(
            u=u,
            cfg=self.cfg,
            graph_id=graph_id,
            plot_boundary=True,
            show_graph=show_graph,
        )

    def plot_stream_function(
        self, *, sf: np.ndarray, graph_id: int, show_graph: bool = False
    ) -> None:
        if self._plot_stream_function is None:
            return
        self._plot_stream_function(
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
        if self._plot_velocity_field is None:
            return
        self._plot_velocity_field(
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
        calculate_velocity: bool = False,
        save_at: Optional[set[int]] = None,
        save_final: bool = False,
        plot_at: Optional[set[int]] = None,
        log_at: Optional[set[int]] = None,
        metrics: Optional[Dict[str, MetricFn]] = None,
        step_callback: Optional[StepCallback] = None,
        stop_criterion: Optional[Callable[[SimulationState], bool]] = None,
    ) -> None:
        self.cfg = cfg
        self.geometry: DomainGeometry = cfg.geometry
        self.state = state
        self.file_manager = file_manager or FileManager(out_dir=checkpoints_dir)
        self.plot_manager = plot_manager or PlotManager(cfg)
        self.logger = logger or logging.getLogger("experiment")
        self.heat_solver = heat_solver
        self.navier_solver = navier_solver
        self.calculate_velocity = calculate_velocity
        self.save_at = save_at or set()
        self.save_final = save_final
        self.plot_at = plot_at or set()
        self.log_at = log_at or set()
        self.metrics = metrics or {}
        self.step_callback = step_callback
        self.stop_criterion = stop_criterion

        if self.calculate_velocity and (
            self.state.v_x is None or self.state.v_y is None
        ):
            raise ValueError(
                "calculate_velocity=True requires v_x and v_y to be initialized in SimulationState"
            )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        cfg: ExperimentConfig,
        heat_solver,
        navier_solver,
        file_manager: Optional[FileManager] = None,
        plot_manager: Optional[PlotManager] = None,
        logger: Optional[logging.Logger] = None,
        checkpoints_dir: Optional[str | Path] = "../data",
        calculate_velocity: bool = False,
        save_at: Optional[set[int]] = None,
        save_final: bool = False,
        plot_at: Optional[set[int]] = None,
        log_at: Optional[set[int]] = None,
        metrics: Optional[Dict[str, MetricFn]] = None,
        step_callback: Optional[StepCallback] = None,
        stop_criterion: Optional[Callable[[SimulationState], bool]] = None,
    ) -> "ExperimentRunner":
        data = load_checkpoint(checkpoint_path)

        state = SimulationState(
            u=np.zeros_like(data["u"]),
            sf=np.zeros_like(data["sf"]),
            w=np.zeros_like(data["w"]),
            v_x=np.zeros_like(data["v_x"]) if data.get("v_x") is not None else None,
            v_y=np.zeros_like(data["v_y"]) if data.get("v_y") is not None else None,
        )
        state.restore(data)

        runner = cls(
            cfg=cfg,
            state=state,
            heat_solver=heat_solver,
            navier_solver=navier_solver,
            file_manager=file_manager,
            plot_manager=plot_manager,
            logger=logger,
            checkpoints_dir=checkpoints_dir,
            calculate_velocity=calculate_velocity,
            save_at=save_at,
            save_final=save_final,
            plot_at=plot_at,
            log_at=log_at,
            metrics=metrics,
            step_callback=step_callback,
            stop_criterion=stop_criterion,
        )
        runner.logger.info(f"Initialized runner from checkpoint: {checkpoint_path}")
        return runner

    def run(self) -> SimulationState:
        start_time = time.perf_counter()

        try:
            for _ in range(1, self.geometry.n_t + 1):
                self.step()
                self._handle_save()
                self._handle_plot()
                self._handle_log(start_time)

                if self.stop_criterion is not None:
                    try:
                        if self.stop_criterion(self.state):
                            self.logger.info(
                                "Stop criterion met at step %d (t = %.2f s); terminating early",
                                self.state.n,
                                self.state.t,
                            )
                            break
                    except Exception:
                        self.logger.exception("stop_criterion failed; continuing")

        except Exception as exc:
            self.logger.exception("Exception during simulation run: %s", exc)
            self._try_save_error_checkpoint()
            raise

        if self.save_final:
            path = self.file_manager.save(self.state, cfg=dict(self.cfg))
            self.logger.info(f"Saved final checkpoint: {path}")

        return self.state

    def step(self) -> None:
        n = self.state.n + 1
        t = n * self.geometry.dt

        try:
            s = self.state
            cfg = self.cfg
            delta = cfg.delta_nd or get_mushy_zone_temperature_range(
                s.u, cfg.u_pt_nd, n_nodes=1
            )
            s.u[:, :] = self.heat_solver.solve(u=s.u, sf=s.sf, delta=delta, time=t)

            delta_flow = cfg.delta_flow_nd or get_mushy_zone_temperature_range(
                s.u, cfg.u_pt_nd, n_nodes=1
            )
            sf_new, w_new = self.navier_solver.solve(
                w=s.w,
                sf=s.sf,
                u=s.u,
                delta=delta_flow,
                time=t,
            )
            s.sf[:, :] = sf_new
            s.w[:, :] = w_new

            if self.calculate_velocity:
                calculate_velocity_from_sf(s.sf, s.v_x, s.v_y, cfg)

            s.n = n
            s.t = t

        except Exception:
            self.logger.exception("Error during solver step")
            raise

        if self.step_callback is not None:
            try:
                self.step_callback(self.state)
            except Exception:
                self.logger.exception("step_callback failed")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _handle_save(self) -> None:
        if self.state.n not in self.save_at:
            return
        path = self.file_manager.save(self.state, cfg=dict(self.cfg))
        self.logger.info(f"Saved checkpoint: {path}")

    def _handle_plot(self) -> None:
        if self.state.n not in self.plot_at:
            return
        try:
            u_dim = self.state.u * self.cfg.delta_u + self.cfg.u_ref
            self.plot_manager.plot_temperature(u=u_dim, graph_id=self.state.n)
            self.plot_manager.plot_stream_function(
                sf=self.state.sf, graph_id=self.state.n
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
                )
        except Exception:
            self.logger.exception("Plotting failed; continuing")

    def _handle_log(self, start_time: float) -> None:
        if self.state.n not in self.log_at:
            return

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

        for name, fn in self.metrics.items():
            try:
                self.logger.info("  %s = %s", name, fn(self.state))
            except Exception:
                self.logger.warning("Metric '%s' failed", name)

    def _try_save_error_checkpoint(self) -> None:
        try:
            path = self.file_manager.save(self.state, cfg=dict(self.cfg))
            self.logger.info(f"Saved error checkpoint: {path}")
        except Exception:
            self.logger.exception("Failed to save error checkpoint")
