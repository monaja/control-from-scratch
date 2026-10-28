# control-from-scratch

Modelling and control of dynamical systems, built up one step at a time in plain Python (numpy + matplotlib).

## System 1: DC motor

Run the files in order. Each step imports the previous one and adds one idea.

| Step | File | Adds | Output |
|---|---|---|---|
| 0 | `motor_model.py` | First-principles model: states, input, disturbance, output, parameters; Euler simulation; steady-state check (V = E + IR) | `model_open_loop.png` |
| 1 | `control_1_p.py` | P control; steady-state error 1/(1 + K_m·Kp) | `control_1_p.png` |
| 2 | `control_2_pi.py` | Integral action (zero error, load rejection), pole-zero cancellation design, anti-windup | `control_2_pi.png` |
| 3 | `control_3_pid.py` | Derivative on measurement + filter; P vs PI vs PID | `control_3_pid.png` |

```bash
conda install numpy matplotlib
python motor_model.py
python control_1_p.py
python control_2_pi.py
python control_3_pid.py
```

## Coming next
- Step 4: identify the motor's parameters from noisy data (least squares)
- Step 5: PID designed from the transfer function (Laplace)
- System 2: pendulum (linearisation, state space, pole placement, LQR)
