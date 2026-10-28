"""
STEP 0 — MODELLING: a brushed DC motor from first principles.

Run:   python motor_model.py        (open-loop study, saves model_open_loop.png)
Import from the control scripts:     from motor_model import Motor

The modelling checklist, filled in for this motor
-------------------------------------------------
Purpose       : design a speed controller
Assumptions   : rigid shaft, constant magnetic field, viscous friction only, no heating
States   x    : [i, w]   armature current [A], shaft speed [rad/s]
Input    u    : V        armature voltage [V], limited to ±12 V
Disturbance d : T_load   load torque [N·m]
Output   y    : w        speed, as an encoder would measure it
Parameters    : R, L, K, J, b

Equations (Kirchhoff + Newton):
    L di/dt + R i = V - K w            electrical  (K w = back-EMF)
    J dw/dt + b w = K i - T_load       mechanical  (K i = motor torque)

Steady state (set d/dt = 0) recovers the textbook algebra:
    V = R i + K w     (V = E + I R)  and  K i = b w   ->   w_ss = K_m V

Simplified model (L -> 0):  tau dw/dt + w = K_m V
    K_m = K / (R b + K^2),   tau = R J / (R b + K^2)
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class Motor:
    R: float = 1.0      # armature resistance        [ohm]
    L: float = 1e-3     # armature inductance        [H]
    K: float = 0.05     # torque / back-EMF constant [N·m/A] = [V·s/rad]
    J: float = 5e-4     # rotor + load inertia       [kg·m^2]
    b: float = 1e-4     # viscous friction           [N·m·s/rad]
    V_max: float = 12.0  # supply limit               [V]

    # ---- derived quantities of the simplified (L = 0) model ----
    @property
    def K_m(self):
        """Steady-state gain: rad/s per volt."""
        return self.K / (self.R * self.b + self.K**2)

    @property
    def tau(self):
        """Mechanical time constant [s]."""
        return self.R * self.J / (self.R * self.b + self.K**2)

    @property
    def tau_e(self):
        """Electrical time constant [s]."""
        return self.L / self.R

    # ---- dynamics: dx/dt = f(x, u, d) ----
    def f(self, x, V, T_load=0.0):
        """Full 2-state model. x = [i, w]."""
        i, w = x
        di = (V - self.R * i - self.K * w) / self.L
        dw = (self.K * i - self.b * w - T_load) / self.J
        return np.array([di, dw])

    def f_first_order(self, w, V):
        """Simplified 1-state model: tau dw/dt + w = K_m V."""
        return (self.K_m * V - w) / self.tau

    def output(self, x):
        """y = h(x): we measure speed only."""
        return x[1]

    # ---- one Euler step (18.03 lecture 2): x[k+1] = x[k] + dt * f(x[k]) ----
    def step(self, x, V, dt, T_load=0.0):
        V = float(np.clip(V, -self.V_max, self.V_max))  # the actuator can't exceed its supply
        return x + dt * self.f(x, V, T_load), V


# =============================================================================
# Open-loop study (runs only when this file is executed directly)
# =============================================================================
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = Motor()
    print("=== Motor model ===")
    print(f"K_m   = {m.K_m:7.2f} rad/s per V   (12 V -> {m.K_m*12:6.1f} rad/s ≈ {m.K_m*12*60/(2*np.pi):5.0f} rpm)")
    print(f"tau   = {m.tau:7.3f} s           (mechanical time constant)")
    print(f"tau_e = {m.tau_e*1e3:7.3f} ms          (electrical, << tau so L≈0 is a fair simplification)")

    V_step, T_end = 6.0, 1.5

    # (a) full model, small dt because tau_e is only 1 ms
    dt_full = 2e-5
    t_full = np.arange(0, T_end, dt_full)
    x = np.zeros(2)
    w_full = np.zeros_like(t_full)
    for k in range(len(t_full)):
        w_full[k] = m.output(x)
        x, _ = m.step(x, V_step, dt_full)

    # (b) simplified model with Euler, (c) its exact solution
    dt1 = 1e-3
    t1 = np.arange(0, T_end, dt1)
    w1 = np.zeros_like(t1)
    for k in range(len(t1) - 1):
        w1[k + 1] = w1[k] + dt1 * m.f_first_order(w1[k], V_step)
    w_exact = m.K_m * V_step * (1 - np.exp(-t1 / m.tau))

    # (d) Euler step-size experiment — stable only if dt < 2*tau
    runs = []
    for dt in [0.02, 0.15, 0.40]:
        t = np.arange(0, T_end, dt)
        w = np.zeros_like(t)
        for k in range(len(t) - 1):
            w[k + 1] = w[k] + dt * m.f_first_order(w[k], V_step)
        runs.append((dt, t, w))

    # (e) steady-state check against the textbook algebra V = E + I R
    w_ss = m.K_m * V_step
    i_ss = m.b * w_ss / m.K
    print(f"\nSteady state at {V_step:.0f} V: w = {w_ss:.1f} rad/s, i = {i_ss:.3f} A")
    print(f"Check V = E + I R : {m.K*w_ss:.3f} + {i_ss*m.R:.3f} = {m.K*w_ss + i_ss*m.R:.3f} V")
    print(f"Simulated final speed (full model): {w_full[-1]:.1f} rad/s")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    ax.plot(t_full, w_full, lw=2, label="Full model (i, ω), Euler dt = 20 µs")
    ax.plot(t1, w1, "--", lw=2, label="First-order model, Euler dt = 1 ms")
    ax.plot(t1, w_exact, ":", lw=2.5, color="k", label="First-order, exact")
    ax.axhline(w_ss, color="gray", lw=0.8)
    ax.axvline(m.tau, color="gray", lw=0.8, ls="--")
    ax.annotate(f"τ = {m.tau:.3f} s\n(63% of final speed)", (m.tau, 0.63 * w_ss),
                xytext=(m.tau + 0.15, 0.4 * w_ss), arrowprops=dict(arrowstyle="->"))
    ax.set(title=f"Model: open-loop step, V = {V_step:.0f} V", xlabel="time [s]", ylabel="speed ω [rad/s]")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t1, w_exact, color="k", lw=2, label="exact")
    for dt, t, w in runs:
        tag = "stable" if dt < 2 * m.tau else "UNSTABLE (dt > 2τ)"
        ax.plot(t, w, "o-", ms=3, label=f"Euler dt = {dt} s  ({tag})")
    ax.set(title="Euler step size matters", xlabel="time [s]", ylabel="speed ω [rad/s]",
           ylim=(-50, 3 * w_ss))
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("figures/model_open_loop.png", dpi=140)
    print("\nSaved model_open_loop.png")
