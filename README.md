# vessel-state-estimation

Modular Python playground for marine sensor fusion: 3-DOF vessel simulator,
noisy IMU/GPS models, EKF and quaternion MEKF with NEES/NIS consistency analysis.

> Status: work in progress (evening side project).

## Layout

| Package         | Purpose                                                  |
|-----------------|----------------------------------------------------------|
| `sim/`          | Vessel dynamics, maneuvers, environment (current, wind)  |
| `sensors/`      | Noisy sensor models behind a common interface            |
| `estimation/`   | EKF and MEKF                                             |
| `eval/`         | Metrics, plots, NEES/NIS consistency checks              |

## Vessels

Three parameter sets live in `configs/vessels.yaml`: `merchant_ship`, `sailboat`, `small_boat`.
The base vessel is the `vessel:` key in `configs/default.yaml`; override it per run with `--vessel`.

## Quick start

```powershell
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
python scripts\main.py --vessel sailboat --maneuver zigzag
```

```bash
# Linux / macOS
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/main.py --vessel sailboat --maneuver zigzag
```

Plots are saved in `results/`. See `docs/conventions.md` for frames and signs, and
`docs/theory/` for the derivations behind each module.
