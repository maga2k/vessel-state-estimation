"""Monte Carlo evaluation of a filter: many noise realisations over the SAME true trajectory.

Random per run: IMU noise and bias (incl. initial bias), GPS noise, compass noise, initial
estimate error. The data are generated once (``generate_runs``) and can be filtered many times
with different tunings (``evaluate``): a fair comparison on identical measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import Config, EkfConfig, ImuConfig
from ..estimation import PlanarEkf, draw_initial_estimate, initial_covariance, run_filter
from ..rng import make_rngs
from ..sensors import Measurement, build_sensors, simulate_sensors
from ..sim.truth import Trajectory
from .consistency import chi2_bounds, nees, nis_by_sensor, normalized_squared_errors
from .errors import state_errors, true_states


@dataclass
class RunData:
    measurements: list[Measurement]
    x0: np.ndarray  # initial estimate of this run


@dataclass
class McResult:
    t: np.ndarray  # (nt,)
    errors: np.ndarray  # (n_runs, nt, nx)
    sigma: np.ndarray  # (n_runs, nt, nx)  filter 1-sigma = sqrt(diag P)
    nees: np.ndarray  # (n_runs, nt)
    norm_sq: np.ndarray  # (n_runs, nt, nx)  e_i^2 / P_ii
    nis: dict[str, np.ndarray]  # sensor -> (n_runs, n_updates)
    nis_t: dict[str, np.ndarray]  # sensor -> (n_updates,)
    nis_dof: dict[str, int]
    gate: dict[str, int]  # GPS gating counts: outliers / outliers_rejected / nominal / nominal_rejected   

    @property
    def n_runs(self) -> int:
        return self.nees.shape[0]

    @property
    def nx(self) -> int:
        return self.errors.shape[2]

    def anees_t(self) -> np.ndarray:
        """NEES averaged over runs, at every logged time."""
        return self.nees.mean(axis=0)

    def anees(self) -> float:
        """NEES averaged over runs AND time (ideal value: nx)."""
        return float(self.nees.mean())

    def anees_bounds(self, alpha: float = 0.05) -> tuple[float, float]:
        return chi2_bounds(self.nx, self.n_runs, alpha)

    def fraction_anees_inside(self, alpha: float = 0.05) -> float:
        lo, hi = self.anees_bounds(alpha)
        a = self.anees_t()
        return float(np.mean((a >= lo) & (a <= hi)))

    def mean_nis(self, sensor: str) -> float:
        """NIS averaged over runs and updates (ideal value: dof of that sensor)."""
        return float(self.nis[sensor].mean())

    def per_state_ratio(self) -> np.ndarray:
        """Mean of e_i^2 / P_ii per state (ideal: 1; > 1 means overconfident on that state)."""
        return self.norm_sq.mean(axis=(0, 1))

    def rms_error(self) -> np.ndarray:
        return np.sqrt(np.mean(self.errors**2, axis=(0, 1)))

    def position_error(self) -> np.ndarray:
        """Horizontal position error, shape (n_runs, nt)."""
        return np.hypot(self.errors[:, :, 0], self.errors[:, :, 1])

    def window_mask(self, t0: float, t1: float) -> np.ndarray:
        return (self.t >= t0) & (self.t <= t1)


def generate_runs(cfg: Config, traj: Trajectory, n_runs: int, base_seed: int = 1000) -> list[RunData]:
    """Sensor data + initial estimate for ``n_runs`` independent noise realisations."""
    P0 = initial_covariance(cfg.ekf)
    runs = []
    for i in range(n_runs):
        rngs = make_rngs(base_seed + i)
        meas = simulate_sensors(traj, build_sensors(cfg.sensors, rngs))
        runs.append(RunData(meas, draw_initial_estimate(traj[0], P0, rngs["init"])))
    return runs


def evaluate(runs: list[RunData], traj: Trajectory, ekf_cfg: EkfConfig, imu_assumed: ImuConfig,
             log_every: int = 10) -> McResult:
    """Filter every run with the given tuning and collect the consistency statistics."""
    P0 = initial_covariance(ekf_cfg)
    errs, sigs, nees_l, nsq, nis_runs, logs = [], [], [], [], [], []
    t = None
    for run in runs:
        log = run_filter(PlanarEkf(ekf_cfg, imu_assumed, run.x0, P0), run.measurements, log_every)
        err = state_errors(log, true_states(traj, log))
        errs.append(err)
        sigs.append(np.sqrt(np.einsum("ijj->ij", log.P)))
        nees_l.append(nees(err, log.P))
        nsq.append(normalized_squared_errors(err, log.P))
        nis_runs.append(nis_by_sensor(log))
        logs.append(log)
        t = log.t
    gate = {"outliers": 0, "outliers_rejected": 0, "nominal": 0, "nominal_rejected": 0}
    for log in logs:
        for (_, sensor, inn), is_out in zip(log.innovations, log.outlier_flags):
            if sensor != "gps":
                continue
            key = "outliers" if is_out else "nominal"
            gate[key] += 1
            gate[key + "_rejected"] += int(not inn.accepted)
    sensors = nis_runs[0].keys()
    for s in sensors:
        if len({len(r[s][1]) for r in nis_runs}) != 1:
            raise ValueError(f"runs have different numbers of '{s}' updates (random dropouts?)")
    return McResult(
        t=t, errors=np.array(errs), sigma=np.array(sigs), nees=np.array(nees_l), norm_sq=np.array(nsq),
        nis={s: np.array([r[s][1] for r in nis_runs]) for s in sensors},
        nis_t={s: nis_runs[0][s][0] for s in sensors},
        nis_dof={s: nis_runs[0][s][2] for s in sensors},
        gate=gate,
    )