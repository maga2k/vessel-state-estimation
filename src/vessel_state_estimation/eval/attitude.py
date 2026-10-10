"""Attitude estimation errors and consistency against the synthetic truth."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..angles import wrap_angle
from ..config import Config, ImuConfig, MekfConfig
from ..estimation.mekf import Mekf, draw_initial_estimate, initial_covariance
from ..estimation.mekf_runner import MekfLog, run_mekf
from ..quaternion import euler_from_quat, quat_conj, quat_mult, rot_from_quat, rotvec_from_quat
from ..rng import make_rngs
from ..sensors import Gps, Measurement
from ..sensors.imu3d import Imu3d, simulate_imu3d
from ..sim.attitude import AttitudeTruth
from ..sim.truth import Trajectory


def truth_at(truth: AttitudeTruth, log: MekfLog) -> tuple[np.ndarray, np.ndarray]:
    """True quaternions and Euler angles at the logged times (nearest sample)."""
    dt = truth.t[1] - truth.t[0]
    idx = np.clip(np.round(log.t / dt).astype(int), 0, len(truth) - 1)
    return truth.q[idx], truth.euler[idx]


def attitude_error(log: MekfLog, q_true: np.ndarray) -> np.ndarray:
    """Error rotation vector dtheta [rad] in BODY axes:  q_true = q_est (x) exp(dtheta)."""
    dq = quat_mult(quat_conj(log.q), q_true)
    return np.array([rotvec_from_quat(d) for d in dq])


def euler_error(log: MekfLog, euler_true: np.ndarray) -> np.ndarray:
    """Estimated minus true [roll, pitch, yaw] in radians (wrapped)."""
    return wrap_angle(euler_from_quat(log.q) - euler_true)


def nees_full(log: MekfLog, dtheta: np.ndarray) -> np.ndarray:
    """NEES of the 6-dim error [dtheta, db] (ideal mean: 6). Includes the unobservable yaw."""
    err = np.hstack([dtheta, log.true_bias - log.b])
    # same convention as the filter error state: q_true = q_est (x) exp(dtheta), b_true = b_est + db
    z = np.linalg.solve(log.P, err[..., None])[..., 0]
    return np.einsum("ij,ij->i", err, z)


def nees_tilt(log: MekfLog, dtheta: np.ndarray) -> np.ndarray:
    """NEES of the roll/pitch-like error only (2 dof): dtheta rotated to NED, N and E components."""
    out = np.empty(len(log.t))
    for k in range(len(log.t)):
        R = rot_from_quat(log.q[k])
        e = (R @ dtheta[k])[:2]
        P_n = (R @ log.P[k, :3, :3] @ R.T)[:2, :2]
        out[k] = e @ np.linalg.solve(P_n, e)
    return out


def yaw_sigma(log: MekfLog) -> np.ndarray:
    """Filter 1-sigma of the yaw error (rotation about the vertical) at every logged time."""
    out = np.empty(len(log.t))
    for k in range(len(log.t)):
        R = rot_from_quat(log.q[k])
        out[k] = np.sqrt((R @ log.P[k, :3, :3] @ R.T)[2, 2])
    return out


@dataclass
class AttitudeRun:
    measurements: list[Measurement]  # IMU3D + GPS (the GPS only provides the speed), time ordered
    q0: np.ndarray
    b0: np.ndarray


@dataclass
class AttitudeMc:
    t: np.ndarray
    euler_err: np.ndarray  # (n_runs, nt, 3) [rad]  roll, pitch, yaw
    nees_tilt: np.ndarray  # (n_runs, nt)  ideal mean 2
    nees_full: np.ndarray  # (n_runs, nt)  ideal mean 6
    nis: np.ndarray  # (n_runs, n_updates)  accelerometer, ideal mean 3
    yaw_sigma: np.ndarray  # (n_runs, nt)  filter 1-sigma of the yaw error

    def rms_deg(self) -> np.ndarray:
        return np.degrees(np.sqrt(np.mean(self.euler_err**2, axis=(0, 1))))


def generate_attitude_runs(cfg: Config, traj: Trajectory, truth: AttitudeTruth, n_runs: int,
                           base_seed: int = 2000) -> list[AttitudeRun]:
    """Sensor data (3D IMU + GPS speed) and initial estimates for independent noise realisations."""
    P0 = initial_covariance(cfg.mekf)
    runs = []
    for i in range(n_runs):
        rngs = make_rngs(base_seed + i)
        imu = simulate_imu3d(truth, Imu3d(cfg.sensors.imu, rngs["imu3d"]))
        gps_sensor = Gps(cfg.sensors.gps, rngs["gps"], rngs["outlier"])
        gps = [m for s in traj.samples if (m := gps_sensor.measure(s.t, s)) is not None]
        meas = sorted(imu + gps, key=lambda m: m.t)  # stable: IMU first at equal times
        q0, b0 = draw_initial_estimate(truth.q[0], P0, rngs["init"])
        runs.append(AttitudeRun(meas, q0, b0))
    return runs


def evaluate_attitude(runs: list[AttitudeRun], truth: AttitudeTruth, mekf_cfg: MekfConfig,
                      imu_assumed: ImuConfig, imu_rate_hz: float) -> AttitudeMc:
    P0 = initial_covariance(mekf_cfg)
    eul, nt_, nf_, nis, ys = [], [], [], [], []
    t = None
    for run in runs:
        log = run_mekf(Mekf(mekf_cfg, imu_assumed, run.q0, run.b0, P0), run.measurements, imu_rate_hz)
        q_true, e_true = truth_at(truth, log)
        dth = attitude_error(log, q_true)
        eul.append(euler_error(log, e_true))
        nt_.append(nees_tilt(log, dth))
        nf_.append(nees_full(log, dth))
        nis.append([inn.nis for _, inn in log.innovations])
        ys.append(yaw_sigma(log))
        t = log.t
    return AttitudeMc(t, np.array(eul), np.array(nt_), np.array(nf_), np.array(nis), np.array(ys))