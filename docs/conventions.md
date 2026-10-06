# Conventions

- Units: SI only (m, s, rad, m/s, rad/s). Degrees only in YAML rudder fields, plots and CLI output.
- Navigation frame: NED (North-East-Down), flat-earth approximation.
- Body frame: FRD (Forward-Right-Down). Surge `u` forward, sway `v` starboard, yaw rate `r` about Down.
- Heading `psi`: angle from North, positive clockwise (towards East), wrapped to [-pi, pi).
- Rudder `delta`: positive = turn to starboard (positive yaw rate), regardless of the physical
  sign convention of a real steering gear.
- Always wrap angle residuals (innovations, errors) before using them.
- Time: float seconds from simulation start.
- Reproducibility: every random draw goes through the named streams in `rng.py`.
- Vessel selection: `vessel:` key in `configs/default.yaml` (base vessel), overridable with
  `--vessel` on the command line. Parameters live in `configs/vessels.yaml`.
- Current direction = where it flows TOWARDS; wind direction = where it blows FROM (degrees from North, clockwise).