# Session 1 - 3-DOF vessel model

## 1. Kinematics (exact)
Body velocities `(u, v)` are rotated into NED with the heading `psi`, and a current `Vc` is added:

    N_dot = u cos(psi) - v sin(psi) + Vc_N
    E_dot = u sin(psi) + v cos(psi) + Vc_E
    psi_dot = r

The vessel moves through the *water* with `(u, v)`; the water moves over the ground with `Vc`.
Ground velocity (what a GPS sees) is the sum. This is why course over ground differs from heading.

## 2. Yaw dynamics: first-order Nomoto
Linearised yaw dynamics around straight-ahead motion, driven by the rudder `delta`:

    T r_dot + r = K delta

- `K` [1/s]: steady yaw rate per radian of rudder. `r_ss = K delta`.
- `T` [s]: how fast the yaw rate gets there. After a rudder step, `r(T) = 63.2 % r_ss`.

Both scale with speed and size. With non-dimensional indices `K'`, `T'`, length `L`, speed `U`:

    K = K' U / L        T = T' L / U

Consequence: in a steady turn `R = U / r_ss = L / (K' delta)`; the radius is independent of speed
and is set by the rudder angle and the ship's "agility" `K'`.

## 3. Sway from the pivot point
A turning hull drifts: it pivots around a point about `x_p = 0.25-0.3 L` ahead of the CG, where the
lateral velocity is zero. So `v + x_p r = 0` in steady state, and with a lag for the transient:

    v_dot = (-x_p r - v) / T_sway

In a turn to starboard `v < 0`: the velocity vector points slightly outside of the bow direction,
i.e. there is a drift angle. The CG path radius becomes `R_cg = hypot(u, v_ss) / |r_ss|`.

## 4. Surge and actuator
`u_dot = (u_cmd - u) / T_surge` (no speed loss in turns yet). The rudder is saturated and rate
limited outside the ODE, in `Vessel.step`.

## 5. Integration
Classic RK4, with the rudder held at its mid-step value. Position error of a Nomoto system is
dominated by model error, not by the integrator, at `dt = 0.01 s`.

## 6. Validation (tests/test_vessel.py)
1. Straight course stays straight; a current adds exactly `Vc * t` to the position.
2. Yaw-rate step response matches `K delta (1 - exp(-t/T))` to numerical precision.
3. Turning circle fitted from `(N, E)` matches `R_cg` from the formulas above (< 0.5 %).
4. Rudder saturation / rate limit and heading wrap are respected.

## 7. Known simplifications
Linear yaw dynamics (real ships saturate: tight turns are over-predicted), no speed loss in turns,
no roll / heel / leeway, no wind force, sway is kinematic. Later sessions add current and wind;
the model is good enough to generate realistic trajectories for filter testing, not for ship design.
